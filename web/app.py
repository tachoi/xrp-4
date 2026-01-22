#!/usr/bin/env python
"""Web Trading Dashboard - FastAPI Backend.

Integrates with paper trading system and broadcasts events via WebSocket.
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.requests import Request
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

# Korean timezone (UTC+9)
KST = timezone(timedelta(hours=9))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="XRP Trading Dashboard")

# Static files and templates
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


# ============================================================================
# Data Models
# ============================================================================

class TradingMode(str, Enum):
    PAPER = "paper"
    LIVE = "live"


class TradingConfig(BaseModel):
    mode: TradingMode = TradingMode.PAPER
    initial_capital: float = 10000.0
    leverage: float = 5.0
    symbol: str = "XRPUSDT"


# ============================================================================
# Regime Colors
# ============================================================================

REGIME_COLORS = {
    "TREND_UP": "#26a69a",
    "TREND_DOWN": "#ef5350",
    "RANGE": "#42a5f5",
    "TRANSITION": "#ffca28",
    "HIGH_VOL": "#ab47bc",
    "NO_TRADE": "#9e9e9e",
    "UNKNOWN": "#9e9e9e",
}


# ============================================================================
# Binance Client
# ============================================================================

class BinanceClient:
    """Simple Binance REST API client."""

    BASE_URL = "https://fapi.binance.com"

    def __init__(self):
        self.session = requests.Session()

    def get_klines(
        self,
        symbol: str = "XRPUSDT",
        interval: str = "3m",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch klines data from Binance."""
        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": min(limit, 1500)}

        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(data, columns=[
            "open_time", "open", "high", "low", "close", "volume",
            "close_time", "quote_volume", "trades", "taker_buy_base",
            "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)

        return df[["timestamp", "open", "high", "low", "close", "volume"]]

    def get_ticker_price(self, symbol: str = "XRPUSDT") -> float:
        """Get current ticker price."""
        url = f"{self.BASE_URL}/fapi/v1/ticker/price"
        params = {"symbol": symbol}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        return float(response.json()["price"])

    def get_current_kline(self, symbol: str = "XRPUSDT", interval: str = "3m") -> dict:
        """Get current building kline with volume."""
        url = f"{self.BASE_URL}/fapi/v1/klines"
        params = {"symbol": symbol, "interval": interval, "limit": 1}
        response = self.session.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        if not data:
            return None

        kline = data[0]
        return {
            "open": float(kline[1]),
            "high": float(kline[2]),
            "low": float(kline[3]),
            "close": float(kline[4]),
            "volume": float(kline[5]),
        }


# ============================================================================
# Chart Data Manager
# ============================================================================

class ChartDataManager:
    """Manages chart data independently of trading engine."""

    def __init__(self, symbol: str = "XRPUSDT"):
        self.symbol = symbol
        self.client = BinanceClient()
        self.df_3m: Optional[pd.DataFrame] = None
        self.current_candle: Optional[Dict] = None
        self.current_candle_start: Optional[datetime] = None
        self.current_price: float = 0.0
        self.initialized: bool = False

    def initialize(self):
        """Fetch initial data from Binance."""
        logger.info(f"Fetching initial data for {self.symbol}...")

        try:
            self.df_3m = self.client.get_klines(self.symbol, "3m", 200)
            self.df_3m = self.df_3m.set_index("timestamp")
            self.current_price = self.client.get_ticker_price(self.symbol)
            self.initialized = True
            logger.info(f"Loaded {len(self.df_3m)} candles. Current price: ${self.current_price:.4f}")
        except Exception as e:
            logger.error(f"Failed to initialize chart data: {e}")
            raise

    def update_price(self, price: float, kline: Optional[Dict] = None) -> Dict:
        """Update current price and building candle.

        Args:
            price: Current price
            kline: Optional kline data with OHLCV from Binance API
        """
        self.current_price = price
        timestamp = datetime.now(KST)

        candle_start = timestamp.replace(second=0, microsecond=0)
        candle_start = candle_start - timedelta(minutes=candle_start.minute % 3)

        if self.current_candle_start != candle_start:
            if self.current_candle is not None and self.df_3m is not None:
                candle_ts = pd.Timestamp(self.current_candle_start)
                # Only add if timestamp doesn't already exist
                if candle_ts not in self.df_3m.index:
                    new_row = pd.DataFrame([{
                        "open": self.current_candle["open"],
                        "high": self.current_candle["high"],
                        "low": self.current_candle["low"],
                        "close": self.current_candle["close"],
                        "volume": self.current_candle.get("volume", 0),
                    }], index=pd.Index([candle_ts], name='timestamp'))

                    self.df_3m = pd.concat([self.df_3m, new_row])
                    if len(self.df_3m) > 200:
                        self.df_3m = self.df_3m.iloc[-200:]

            self.current_candle_start = candle_start
            # Use kline data if available, otherwise use price
            if kline:
                self.current_candle = {
                    "timestamp": candle_start.isoformat(),
                    "open": kline["open"],
                    "high": kline["high"],
                    "low": kline["low"],
                    "close": kline["close"],
                    "volume": kline["volume"],
                    "completed": False,
                }
            else:
                self.current_candle = {
                    "timestamp": candle_start.isoformat(),
                    "open": price,
                    "high": price,
                    "low": price,
                    "close": price,
                    "volume": 0,
                    "completed": False,
                }
        else:
            if self.current_candle:
                # Update from kline data if available
                if kline:
                    self.current_candle["open"] = kline["open"]
                    self.current_candle["high"] = kline["high"]
                    self.current_candle["low"] = kline["low"]
                    self.current_candle["close"] = kline["close"]
                    self.current_candle["volume"] = kline["volume"]
                else:
                    self.current_candle["high"] = max(self.current_candle["high"], price)
                    self.current_candle["low"] = min(self.current_candle["low"], price)
                    self.current_candle["close"] = price

        return self.current_candle

    def get_chart_data(self, limit: int = 100) -> Dict[str, List[Dict]]:
        """Get chart data for frontend."""
        if self.df_3m is None:
            return {"candles": [], "volume": []}

        # Remove duplicates and sort by index
        df = self.df_3m[~self.df_3m.index.duplicated(keep='last')].tail(limit).reset_index()
        candles = []
        volume = []

        for _, row in df.iterrows():
            time_val = int(row["timestamp"].timestamp())
            candles.append({
                "time": time_val,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })
            color = "#26a69a80" if row["close"] >= row["open"] else "#ef535080"
            volume.append({
                "time": time_val,
                "value": float(row["volume"]),
                "color": color,
            })

        return {"candles": candles, "volume": volume}


# ============================================================================
# Web Paper Trader (wraps PaperTrader with event callbacks)
# ============================================================================

class WebPaperTrader:
    """Paper trader with event callbacks for web broadcasting."""

    def __init__(
        self,
        symbol: str,
        initial_capital: float,
        leverage: float,
        chart_manager: ChartDataManager,
        on_tick: Optional[Callable] = None,
        on_regime_change: Optional[Callable] = None,
        on_trade: Optional[Callable] = None,
    ):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.leverage = leverage
        self.chart_manager = chart_manager

        # Event callbacks
        self.on_tick = on_tick
        self.on_regime_change = on_regime_change
        self.on_trade = on_trade

        # Import PaperTradingEngine
        from paper_trading import PaperTradingEngine

        # Create actual paper trader
        self.trader = PaperTradingEngine(
            symbol=symbol,
            initial_capital=initial_capital,
            leverage=leverage,
            poll_interval=5,
        )

        # State
        self.running = False
        self.last_regime: Optional[str] = None
        self.trades: List[Dict] = []
        self.regimes: List[Dict] = []

    async def start(self):
        """Start the paper trading loop."""
        logger.info("Starting WebPaperTrader...")

        # Initialize the underlying trader
        self.trader.initialize()
        self.running = True

        # Run initial pipeline to get first regime
        try:
            result = self.trader.run_pipeline()
            if result:
                decision, market_ctx, confirm_ctx, signal = result
                self.last_regime = confirm_ctx.regime_confirmed
                self.trader.last_regime = self.last_regime

                # Add initial regime to history
                self.regimes.append({
                    "timestamp": datetime.now(KST).isoformat(),
                    "regime": self.last_regime,
                    "color": REGIME_COLORS.get(self.last_regime, "#9e9e9e"),
                })
                logger.info(f"Initial regime: {self.last_regime}")
        except Exception as e:
            logger.error(f"Failed to get initial regime: {e}")
            self.last_regime = "UNKNOWN"

        logger.info(f"WebPaperTrader initialized. Last 3m bar: {self.trader.last_3m_ts}")

    async def stop(self):
        """Stop the paper trading loop."""
        self.running = False
        logger.info("WebPaperTrader stopped.")

    async def tick(self) -> Dict:
        """Run one tick of the paper trading loop."""
        if not self.running:
            return {}

        try:
            # Get current kline with volume
            kline = self.chart_manager.client.get_current_kline(self.symbol, "3m")
            current_price = kline["close"] if kline else self.trader.client.get_ticker_price(self.symbol)
            self.trader._update_tick_history(current_price)

            # Update chart manager with kline data (includes volume)
            candle = self.chart_manager.update_price(current_price, kline)

            # Check emergency exit
            if self.trader.position.side != "FLAT":
                emergency_reason = self.trader._check_emergency_exit(current_price)
                if emergency_reason:
                    logger.info(f"[EMERGENCY] {emergency_reason}")
                    self.trader._execute_emergency_exit(current_price, emergency_reason)
                    await self._emit_trade_event("EMERGENCY_EXIT", emergency_reason)

            # Check trailing stop / breakeven
            if self.trader.position.side != "FLAT":
                self.trader._update_max_profit_5s(current_price)

                breakeven_result = self.trader._check_breakeven_stop_5s(current_price)
                if breakeven_result is not None:
                    exit_price, reason = breakeven_result
                    self.trader.trailing_stop_stats["breakeven_triggered"] += 1
                    self.trader._execute_trailing_stop_exit(exit_price, reason)
                    await self._emit_trade_event("BREAKEVEN_STOP", reason)
                else:
                    trailing_result = self.trader._check_trailing_stop_5s(current_price)
                    if trailing_result is not None:
                        exit_price, reason = trailing_result
                        self.trader._execute_trailing_stop_exit(exit_price, reason)
                        await self._emit_trade_event("TRAILING_STOP", reason)

            # Check pending signal
            if self.trader.pending_signal is not None:
                entry_result = self.trader._check_entry_timing(current_price)
                if entry_result and entry_result["execute"]:
                    self.trader._execute_pending_signal(current_price, entry_result["reason"])
                    await self._emit_trade_event("ENTRY", entry_result["reason"])

            # Check for new 3m bar
            new_bar = self.trader.update_data()

            regime = self.trader.last_regime or "UNKNOWN"
            regime_changed = False

            if new_bar:
                # Cancel pending signal on new bar
                if self.trader.pending_signal:
                    self.trader._cancel_pending_signal("NEW_BAR")

                # Run pipeline
                result = self.trader.run_pipeline()

                if result:
                    decision, market_ctx, confirm_ctx, signal = result
                    regime = confirm_ctx.regime_confirmed

                    # Check regime change
                    if self.last_regime is not None and regime != self.last_regime:
                        regime_changed = True
                        await self._emit_regime_change(self.last_regime, regime)

                    self.last_regime = regime
                    self.trader.last_regime = regime

                    # Handle trade actions
                    if decision.action in ["OPEN_LONG", "OPEN_SHORT"]:
                        self.trader._set_pending_signal(signal.signal, decision, market_ctx)
                    elif decision.action == "CLOSE":
                        self.trader.execute_decision(decision, market_ctx)
                        await self._emit_trade_event("CLOSE", decision.reason)

            # Build tick data
            tick_data = {
                "price": current_price,
                "timestamp": datetime.now(KST).isoformat(),
                "candle": candle,
                "regime": regime,
                "regime_color": REGIME_COLORS.get(regime, "#9e9e9e"),
                "regime_changed": regime_changed,
                "stats": self.get_stats(),
            }

            # Emit tick event
            if self.on_tick:
                await self.on_tick(tick_data)

            return tick_data

        except Exception as e:
            logger.error(f"Tick error: {e}")
            return {}

    async def _emit_regime_change(self, old_regime: str, new_regime: str):
        """Emit regime change event."""
        regime_info = {
            "timestamp": datetime.now(KST).isoformat(),
            "old_regime": old_regime,
            "regime": new_regime,
            "color": REGIME_COLORS.get(new_regime, "#9e9e9e"),
        }
        self.regimes.append(regime_info)

        # Keep only last 50 regimes
        if len(self.regimes) > 50:
            self.regimes = self.regimes[-50:]

        logger.info(f"[REGIME CHANGE] {old_regime} -> {new_regime}")

        if self.on_regime_change:
            await self.on_regime_change(regime_info)

    async def _emit_trade_event(self, event_type: str, reason: str):
        """Emit trade event."""
        if not self.trader.trades:
            return

        last_trade = self.trader.trades[-1]
        trade_info = {
            "timestamp": last_trade.get("timestamp", datetime.now(KST).isoformat()),
            "event_type": event_type,
            "side": last_trade.get("side", ""),
            "entry_price": last_trade.get("entry_price", 0),
            "exit_price": last_trade.get("exit_price", 0),
            "size": last_trade.get("size", 0),
            "pnl": last_trade.get("pnl", 0),
            "pnl_pct": last_trade.get("pnl_pct", 0),
            "reason": reason,
        }
        self.trades.append(trade_info)

        # Keep only last 100 trades
        if len(self.trades) > 100:
            self.trades = self.trades[-100:]

        if self.on_trade:
            await self.on_trade(trade_info)

    def get_stats(self) -> Dict:
        """Get current trading statistics."""
        win_rate = (self.trader.winning_trades / self.trader.total_trades * 100) if self.trader.total_trades > 0 else 0
        # Use equity - initial_capital for true net PnL (includes both entry and exit fees)
        # self.trader.total_pnl only tracks exit fees, so it overstates profits
        net_pnl = self.trader.equity - self.initial_capital

        return {
            "equity": self.trader.equity,
            "initial_capital": self.initial_capital,
            "total_pnl": net_pnl,
            "pnl_pct": (self.trader.equity / self.initial_capital - 1) * 100,
            "total_trades": self.trader.total_trades,
            "winning_trades": self.trader.winning_trades,
            "win_rate": win_rate,
            "position": {
                "side": self.trader.position.side,
                "entry_price": self.trader.position.entry_price if self.trader.position.side != "FLAT" else None,
                "size": self.trader.position.size if self.trader.position.side != "FLAT" else None,
            },
            "current_regime": self.last_regime or "UNKNOWN",
        }

    def get_trades(self) -> List[Dict]:
        """Get trade history."""
        return self.trades[-50:]

    def get_regimes(self) -> List[Dict]:
        """Get regime history."""
        return self.regimes[-20:]


# ============================================================================
# Global State
# ============================================================================

chart_manager: Optional[ChartDataManager] = None
web_trader: Optional[WebPaperTrader] = None
trading_task: Optional[asyncio.Task] = None


# ============================================================================
# WebSocket Manager
# ============================================================================

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"Client connected. Total: {len(self.active_connections)}")

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"Client disconnected. Total: {len(self.active_connections)}")

    async def broadcast(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception as e:
                logger.error(f"Failed to send to client: {e}")
                disconnected.append(connection)

        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# ============================================================================
# Event Callbacks for WebSocket Broadcasting
# ============================================================================

async def on_tick_callback(tick_data: Dict):
    """Broadcast tick data to all connected clients."""
    await manager.broadcast({
        "type": "tick",
        "price": tick_data["price"],
        "timestamp": tick_data["timestamp"],
        "candle": tick_data["candle"],
        "stats": tick_data["stats"],
        "regime": tick_data["regime"],
        "regime_color": tick_data["regime_color"],
    })


async def on_regime_change_callback(regime_info: Dict):
    """Broadcast regime change to all connected clients."""
    await manager.broadcast({
        "type": "regime_change",
        "regime": regime_info,
    })


async def on_trade_callback(trade_info: Dict):
    """Broadcast trade event to all connected clients."""
    await manager.broadcast({
        "type": "trade",
        "trade": trade_info,
        "stats": web_trader.get_stats() if web_trader else {},
    })


# ============================================================================
# Background Task - Trading Loop
# ============================================================================

async def trading_loop():
    """Background task to run the paper trading loop."""
    global web_trader

    while web_trader and web_trader.running:
        try:
            await web_trader.tick()
            await asyncio.sleep(5)  # 5 second interval
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            await asyncio.sleep(5)


async def price_update_task():
    """Background task to update price when trading is not active."""
    global chart_manager, web_trader

    client = BinanceClient()

    while True:
        try:
            # Only run price updates if web_trader is not running
            if chart_manager and chart_manager.initialized and len(manager.active_connections) > 0:
                if web_trader is None or not web_trader.running:
                    # Get current kline with volume
                    kline = client.get_current_kline("XRPUSDT", "3m")
                    price = kline["close"] if kline else client.get_ticker_price("XRPUSDT")
                    candle = chart_manager.update_price(price, kline)

                    await manager.broadcast({
                        "type": "tick",
                        "price": price,
                        "timestamp": datetime.now(KST).isoformat(),
                        "candle": candle,
                        "stats": {
                            "equity": 10000.0,
                            "initial_capital": 10000.0,
                            "total_pnl": 0.0,
                            "pnl_pct": 0.0,
                            "total_trades": 0,
                            "winning_trades": 0,
                            "win_rate": 0.0,
                            "position": {"side": "FLAT", "entry_price": None, "size": None},
                            "current_regime": "UNKNOWN",
                        },
                        "regime": "UNKNOWN",
                        "regime_color": "#9e9e9e",
                    })

            await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Price update error: {e}")
            await asyncio.sleep(5)


# ============================================================================
# API Endpoints
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize chart data and start background tasks on startup."""
    global chart_manager

    chart_manager = ChartDataManager(symbol="XRPUSDT")
    chart_manager.initialize()

    asyncio.create_task(price_update_task())
    logger.info("Web service started with initial data loaded")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve main dashboard page."""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/start")
async def start_trading(config: TradingConfig):
    """Start trading engine with config."""
    global web_trader, trading_task, chart_manager

    if chart_manager is None:
        raise HTTPException(status_code=500, detail="Chart manager not initialized")

    # Stop existing trader if running
    if web_trader and web_trader.running:
        logger.info("Stopping existing trader before starting new one...")
        await web_trader.stop()
        if trading_task:
            trading_task.cancel()
            try:
                await trading_task
            except asyncio.CancelledError:
                pass
            trading_task = None
        web_trader = None

    # Create web trader with callbacks
    web_trader = WebPaperTrader(
        symbol=config.symbol,
        initial_capital=config.initial_capital,
        leverage=config.leverage,
        chart_manager=chart_manager,
        on_tick=on_tick_callback,
        on_regime_change=on_regime_change_callback,
        on_trade=on_trade_callback,
    )

    await web_trader.start()

    # Start trading loop
    trading_task = asyncio.create_task(trading_loop())

    return {
        "status": "started",
        "mode": config.mode,
        "initial_capital": config.initial_capital,
        "leverage": config.leverage,
    }


@app.post("/api/stop")
async def stop_trading():
    """Stop trading engine."""
    global web_trader, trading_task

    if web_trader:
        await web_trader.stop()
        stats = web_trader.get_stats()

        if trading_task:
            trading_task.cancel()
            try:
                await trading_task
            except asyncio.CancelledError:
                pass
            trading_task = None

        web_trader = None
        return {"status": "stopped", "final_stats": stats}

    return {"status": "not_running"}


@app.get("/api/stats")
async def get_stats():
    """Get current trading statistics."""
    if web_trader is None:
        return {
            "equity": 10000.0,
            "initial_capital": 10000.0,
            "total_pnl": 0.0,
            "pnl_pct": 0.0,
            "total_trades": 0,
            "winning_trades": 0,
            "win_rate": 0.0,
            "position": {"side": "FLAT", "entry_price": None, "size": None},
            "current_regime": "UNKNOWN",
        }

    return web_trader.get_stats()


@app.get("/api/trades")
async def get_trades():
    """Get trade history."""
    if web_trader is None:
        return []
    return web_trader.get_trades()


@app.get("/api/chart")
async def get_chart_data(limit: int = 100):
    """Get chart data."""
    global chart_manager

    if chart_manager is None or not chart_manager.initialized:
        raise HTTPException(status_code=500, detail="Chart data not initialized")

    result = chart_manager.get_chart_data(limit)
    return result["candles"]


@app.get("/api/regimes")
async def get_regimes():
    """Get regime history."""
    if web_trader is None:
        return []
    return web_trader.get_regimes()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time updates."""
    global chart_manager, web_trader

    await manager.connect(websocket)

    try:
        # Send initial data
        if chart_manager and chart_manager.initialized:
            chart_result = chart_manager.get_chart_data(100)
            chart_data = chart_result["candles"]
            volume_data = chart_result["volume"]
        else:
            chart_data = []
            volume_data = []

        if web_trader:
            stats = web_trader.get_stats()
            trades = web_trader.get_trades()
            regimes = web_trader.get_regimes()
            regime = web_trader.last_regime or "UNKNOWN"
        else:
            stats = {
                "equity": 10000.0,
                "initial_capital": 10000.0,
                "total_pnl": 0.0,
                "pnl_pct": 0.0,
                "total_trades": 0,
                "winning_trades": 0,
                "win_rate": 0.0,
                "position": {"side": "FLAT", "entry_price": None, "size": None},
                "current_regime": "UNKNOWN",
            }
            trades = []
            regimes = []
            regime = "UNKNOWN"

        # Check if trading is running
        is_running = web_trader is not None and web_trader.running

        await websocket.send_json({
            "type": "init",
            "chart_data": chart_data,
            "volume_data": volume_data,
            "stats": stats,
            "trades": trades,
            "regimes": regimes,
            "price": chart_manager.current_price if chart_manager else 0,
            "regime": regime,
            "regime_color": REGIME_COLORS.get(regime, "#9e9e9e"),
            "is_running": is_running,
        })

        while True:
            data = await websocket.receive_text()
            message = json.loads(data)

            if message.get("type") == "ping":
                await websocket.send_json({"type": "pong"})

    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
