#!/usr/bin/env python3
"""XGB Gate Training Script - Train XGBoost model for trade filtering.

Simpler approach:
1. Run backtest to generate trades
2. Extract features at entry time
3. Train XGBoost to predict win/loss
4. Simulate filtered trading results
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import xgboost as xgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def load_ohlcv(symbol: str, start: str, end: str, timeframe: str = "3m"):
    """Load OHLCV data from TimescaleDB."""
    conn = psycopg2.connect(
        host="localhost", port=5432, database="xrp_timeseries",
        user="xrp_user", password="xrp_password_change_me",
    )

    query = """
        SELECT time as ts, open, high, low, close, volume
        FROM ohlcv
        WHERE symbol = %s AND timeframe = %s AND time >= %s AND time < %s
        ORDER BY time
    """

    df = pd.read_sql(query, conn, params=(symbol, timeframe, start, end))
    df['ts'] = pd.to_datetime(df['ts'])
    conn.close()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical features."""
    df = df.copy()

    # Returns
    df['ret'] = df['close'].pct_change()
    df['ret_2'] = df['close'].pct_change(2)
    df['ret_5'] = df['close'].pct_change(5)
    df['ret_10'] = df['close'].pct_change(10)

    # EMAs
    df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    df['ema_50'] = df['close'].ewm(span=50, adjust=False).mean()
    df['ema_diff'] = (df['ema_20'] - df['ema_50']) / df['close']
    df['price_to_ema20'] = (df['close'] - df['ema_20']) / df['ema_20']
    df['price_to_ema50'] = (df['close'] - df['ema_50']) / df['ema_50']

    # Volatility
    df['volatility'] = df['ret'].rolling(20).std()
    df['range_pct'] = (df['high'] - df['low']) / df['close']
    df['atr'] = (df['high'] - df['low']).rolling(14).mean()

    # Volume
    df['volume_ma'] = df['volume'].rolling(20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, 1)

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # EMA slope
    df['ema_slope'] = df['ema_20'].pct_change(5)

    # Momentum
    df['mom_5'] = df['close'] - df['close'].shift(5)
    df['mom_10'] = df['close'] - df['close'].shift(10)

    return df.fillna(0)


def run_simple_backtest(df: pd.DataFrame, hmm_model, state_labels: dict):
    """Run simplified backtest to generate trades with entry features."""
    import pickle
    from xrp4.core.fsm import TradingFSM
    from xrp4.core.decision_engine import DecisionEngine
    from xrp4.core.types import MarketContext, PositionState, ConfirmContext

    # Add HMM-specific features
    df = df.copy()
    df['vol'] = df['volatility']  # alias
    df['box_range'] = df['range_pct']  # simplified

    # Calculate box breakout features
    lookback = 32
    df['HH'] = df['high'].rolling(lookback).max()
    df['LL'] = df['low'].rolling(lookback).min()
    df['atr_14'] = (df['high'] - df['low']).rolling(14).mean()
    df['B_up'] = (df['close'] - df['HH'].shift(1)) / df['atr_14'].replace(0, 1)
    df['B_dn'] = (df['LL'].shift(1) - df['close']) / df['atr_14'].replace(0, 1)
    df = df.fillna(0)

    # Predict HMM states using correct features
    # Feature names: ['ret', 'vol', 'ema_slope', 'box_range', 'B_up', 'B_dn']
    hmm_features = df[['ret', 'vol', 'ema_slope', 'box_range', 'B_up', 'B_dn']].fillna(0).values
    hmm_states = hmm_model.predict(hmm_features)
    df = df.copy()
    df['regime'] = [state_labels.get(s, 'UNKNOWN') for s in hmm_states]

    fsm = TradingFSM()
    de = DecisionEngine()

    trades = []
    fsm_state = None
    engine_state = None
    pos = PositionState(side="FLAT", entry_price=0, size=0, bars_held_3m=0)
    entry_row = None
    entry_idx = None

    for i in range(50, len(df)):  # Start after warmup
        row = df.iloc[i]

        # Simplified context (no full confirm layer)
        ctx = MarketContext(
            symbol="XRPUSDT",
            ts=int(row['ts'].timestamp() * 1000) if hasattr(row['ts'], 'timestamp') else row['ts'],
            price=row['close'],
            row_3m=row.to_dict(),
            row_15m={'ema_slope_15m': row.get('ema_slope', 0)},
            zone={'support': 0, 'resistance': 0, 'strength': 0},
        )

        # Simple confirm context (pass-through regime)
        regime = row['regime']
        if regime in ['HIGH_VOL']:
            regime = 'HIGH_VOL'
        confirm_ctx = ConfirmContext(
            regime_raw=row['regime'],
            regime_confirmed=regime,
            confirm_reason="PASSTHROUGH",
            confirm_metrics={},
        )

        # FSM
        cand, fsm_state = fsm.step(ctx, confirm_ctx, pos, fsm_state)

        # Decision (order: ctx, confirm, pos, cand, state)
        decision, engine_state = de.decide(ctx, confirm_ctx, pos, cand, engine_state)

        # Position management
        if decision.action == "OPEN_LONG":
            pos = PositionState(side="LONG", entry_price=row['close'], size=decision.size, bars_held_3m=0)
            entry_row = row.copy()
            entry_idx = i
        elif decision.action == "OPEN_SHORT":
            pos = PositionState(side="SHORT", entry_price=row['close'], size=decision.size, bars_held_3m=0)
            entry_row = row.copy()
            entry_idx = i
        elif decision.action == "CLOSE" and pos.side != "FLAT":
            # Calculate PnL
            if pos.side == "LONG":
                pnl_pct = (row['close'] - pos.entry_price) / pos.entry_price
            else:
                pnl_pct = (pos.entry_price - row['close']) / pos.entry_price

            pnl = pnl_pct * pos.size

            trade = {
                'entry_idx': entry_idx,
                'exit_idx': i,
                'side': pos.side,
                'entry_price': pos.entry_price,
                'exit_price': row['close'],
                'pnl': pnl,
                'pnl_pct': pnl_pct * 100,
                'bars_held': pos.bars_held_3m,
                'regime': entry_row['regime'] if entry_row is not None else 'UNKNOWN',
                # Entry features
                'ret': entry_row['ret'] if entry_row is not None else 0,
                'ret_2': entry_row['ret_2'] if entry_row is not None else 0,
                'ret_5': entry_row['ret_5'] if entry_row is not None else 0,
                'ema_diff': entry_row['ema_diff'] if entry_row is not None else 0,
                'price_to_ema20': entry_row['price_to_ema20'] if entry_row is not None else 0,
                'price_to_ema50': entry_row['price_to_ema50'] if entry_row is not None else 0,
                'volatility': entry_row['volatility'] if entry_row is not None else 0,
                'range_pct': entry_row['range_pct'] if entry_row is not None else 0,
                'volume_ratio': entry_row['volume_ratio'] if entry_row is not None else 0,
                'rsi': entry_row['rsi'] if entry_row is not None else 50,
                'ema_slope': entry_row['ema_slope'] if entry_row is not None else 0,
            }
            trades.append(trade)

            pos = PositionState(side="FLAT", entry_price=0, size=0, bars_held_3m=0)
            entry_row = None
            entry_idx = None
        elif pos.side != "FLAT":
            pos = PositionState(
                side=pos.side,
                entry_price=pos.entry_price,
                size=pos.size,
                bars_held_3m=pos.bars_held_3m + 1
            )

    return pd.DataFrame(trades)


def main():
    parser = argparse.ArgumentParser(description='Train XGB Gate for trade filtering')
    parser.add_argument('--train-start', default='2024-01-01', help='Training start date')
    parser.add_argument('--train-end', default='2025-01-01', help='Training end date')
    parser.add_argument('--test-start', default='2025-01-01', help='Test start date')
    parser.add_argument('--test-end', default='2026-01-01', help='Test end date')
    parser.add_argument('--output', default='outputs/xgb_gate', help='Output directory')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    logger.info("=" * 70)
    logger.info("XGB GATE TRAINING")
    logger.info("=" * 70)
    logger.info(f"Train period: {args.train_start} ~ {args.train_end}")
    logger.info(f"Test period: {args.test_start} ~ {args.test_end}")

    # Load HMM model
    import pickle
    hmm_path = Path("models/hmm_simple.pkl")
    if not hmm_path.exists():
        logger.error("HMM model not found!")
        return

    with open(hmm_path, "rb") as f:
        hmm_data = pickle.load(f)
    hmm_model = hmm_data["model"]
    state_labels = hmm_data.get("state_labels", {})
    logger.info(f"Loaded HMM from {hmm_path}")
    logger.info(f"State labels: {state_labels}")

    # Load and process training data
    logger.info("\nLoading training data...")
    df_train = load_ohlcv("XRPUSDT", args.train_start, args.train_end)
    logger.info(f"  Bars: {len(df_train)}")
    df_train = add_features(df_train)

    # Load and process test data
    logger.info("\nLoading test data...")
    df_test = load_ohlcv("XRPUSDT", args.test_start, args.test_end)
    logger.info(f"  Bars: {len(df_test)}")
    df_test = add_features(df_test)

    # Generate trades
    logger.info("\nGenerating trades from backtest...")
    trades_train = run_simple_backtest(df_train, hmm_model, state_labels)
    trades_test = run_simple_backtest(df_test, hmm_model, state_labels)

    logger.info(f"  Train trades: {len(trades_train)}")
    logger.info(f"  Test trades: {len(trades_test)}")

    if len(trades_train) < 50:
        logger.error("Not enough training samples!")
        return

    # Feature columns for XGB
    feature_cols = [
        'ret', 'ret_2', 'ret_5',
        'ema_diff', 'price_to_ema20', 'price_to_ema50',
        'volatility', 'range_pct', 'volume_ratio',
        'rsi', 'ema_slope',
    ]

    # Add side and regime as features
    trades_train['side_num'] = trades_train['side'].apply(lambda x: 1 if x == 'LONG' else -1)
    trades_train['regime_trend_up'] = (trades_train['regime'] == 'TREND_UP').astype(int)
    trades_train['regime_trend_down'] = (trades_train['regime'] == 'TREND_DOWN').astype(int)
    trades_train['target'] = (trades_train['pnl'] > 0).astype(int)

    trades_test['side_num'] = trades_test['side'].apply(lambda x: 1 if x == 'LONG' else -1)
    trades_test['regime_trend_up'] = (trades_test['regime'] == 'TREND_UP').astype(int)
    trades_test['regime_trend_down'] = (trades_test['regime'] == 'TREND_DOWN').astype(int)
    trades_test['target'] = (trades_test['pnl'] > 0).astype(int)

    all_features = feature_cols + ['side_num', 'regime_trend_up', 'regime_trend_down']

    # Prepare data
    X_train = trades_train[all_features].fillna(0).values
    y_train = trades_train['target'].values
    X_test = trades_test[all_features].fillna(0).values
    y_test = trades_test['target'].values

    logger.info(f"\nTrain: {len(X_train)} samples, {y_train.sum()} wins ({y_train.mean()*100:.1f}%)")
    logger.info(f"Test: {len(X_test)} samples, {y_test.sum()} wins ({y_test.mean()*100:.1f}%)")

    # Train XGBoost
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING XGBOOST MODEL")
    logger.info("=" * 70)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 4,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42,
    }

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=100,
        evals=[(dtrain, 'train'), (dtest, 'test')],
        early_stopping_rounds=20,
        verbose_eval=20,
    )

    # Predictions
    y_pred_proba = model.predict(dtest)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Evaluation metrics
    logger.info("\n" + "=" * 70)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 70)
    logger.info(f"\nTest Metrics:")
    logger.info(f"  Accuracy: {accuracy_score(y_test, y_pred)*100:.1f}%")
    logger.info(f"  Precision: {precision_score(y_test, y_pred, zero_division=0)*100:.1f}%")
    logger.info(f"  Recall: {recall_score(y_test, y_pred, zero_division=0)*100:.1f}%")
    logger.info(f"  F1 Score: {f1_score(y_test, y_pred, zero_division=0)*100:.1f}%")
    try:
        logger.info(f"  ROC AUC: {roc_auc_score(y_test, y_pred_proba)*100:.1f}%")
    except:
        pass

    # Feature importance
    importance = model.get_score(importance_type='gain')
    logger.info("\nFeature Importance (top 10):")
    sorted_imp = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:10]
    for fname, score in sorted_imp:
        idx = int(fname.replace('f', ''))
        if idx < len(all_features):
            logger.info(f"  {all_features[idx]}: {score:.2f}")

    # Trading simulation with XGB filter
    logger.info("\n" + "=" * 70)
    logger.info("TRADING SIMULATION WITH XGB FILTER")
    logger.info("=" * 70)

    trades_test['p_win'] = y_pred_proba

    # Baseline
    baseline = {
        'n_trades': len(trades_test),
        'win_rate': trades_test['target'].mean() * 100,
        'total_pnl': trades_test['pnl'].sum(),
        'avg_pnl': trades_test['pnl'].mean(),
    }

    logger.info(f"\n{'Threshold':<12} {'Trades':<10} {'WinRate':<10} {'TotalPnL':<12} {'AvgPnL':<12} {'Filter%':<10}")
    logger.info("-" * 70)
    logger.info(f"{'No Filter':<12} {baseline['n_trades']:<10} {baseline['win_rate']:.1f}%{'':<5} ${baseline['total_pnl']:.2f}{'':<5} ${baseline['avg_pnl']:.4f}{'':<5} 0%")

    best_result = None
    best_threshold = 0.5

    for threshold in [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        filtered = trades_test[trades_test['p_win'] >= threshold]
        if len(filtered) == 0:
            continue

        result = {
            'threshold': threshold,
            'n_trades': len(filtered),
            'win_rate': filtered['target'].mean() * 100,
            'total_pnl': filtered['pnl'].sum(),
            'avg_pnl': filtered['pnl'].mean(),
            'filter_pct': (1 - len(filtered) / len(trades_test)) * 100,
        }

        logger.info(f"{threshold:<12.2f} {result['n_trades']:<10} {result['win_rate']:.1f}%{'':<5} ${result['total_pnl']:.2f}{'':<5} ${result['avg_pnl']:.4f}{'':<5} {result['filter_pct']:.0f}%")

        if best_result is None or result['total_pnl'] > best_result['total_pnl']:
            best_result = result
            best_threshold = threshold

    # Best result summary
    logger.info("\n" + "=" * 70)
    logger.info("BEST RESULT")
    logger.info("=" * 70)

    if best_result:
        pnl_improvement = best_result['total_pnl'] - baseline['total_pnl']
        wr_improvement = best_result['win_rate'] - baseline['win_rate']

        logger.info(f"\nOptimal threshold: {best_threshold}")
        logger.info(f"\nImprovement over baseline:")
        logger.info(f"  PnL: ${baseline['total_pnl']:.2f} -> ${best_result['total_pnl']:.2f} ({'+' if pnl_improvement >= 0 else ''}{pnl_improvement:.2f})")
        logger.info(f"  Win Rate: {baseline['win_rate']:.1f}% -> {best_result['win_rate']:.1f}% ({'+' if wr_improvement >= 0 else ''}{wr_improvement:.1f}%)")
        logger.info(f"  Trades filtered: {best_result['filter_pct']:.0f}%")

    # Save model
    model_path = Path(args.output) / "xgb_model.json"
    model.save_model(str(model_path))
    logger.info(f"\nModel saved to: {model_path}")

    # Save results
    results = {
        'train_period': f"{args.train_start} ~ {args.train_end}",
        'test_period': f"{args.test_start} ~ {args.test_end}",
        'train_trades': len(trades_train),
        'test_trades': len(trades_test),
        'baseline': baseline,
        'best_threshold': best_threshold,
        'best_result': best_result,
        'feature_cols': all_features,
    }

    results_path = Path(args.output) / "xgb_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"Results saved to: {results_path}")

    logger.info("\nDone!")


if __name__ == "__main__":
    main()
