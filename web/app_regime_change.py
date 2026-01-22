#!/usr/bin/env python
"""Web Trading Dashboard for Regime Change Entry Strategy.

Separate web service for regime change paper trading.
Runs on port 8081 to avoid conflict with the existing system on port 8080.

Usage:
    python web/app_regime_change.py
    # or with uvicorn:
    uvicorn web.app_regime_change:app --host 0.0.0.0 --port 8081
"""

import asyncio
import json
import logging
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Callable

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

app = FastAPI(title="XRP Regime Change Trading Dashboard")

# Static files and templates (use existing ones)
app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
app.mount("/css", StaticFiles(directory=Path(__file__).parent / "static" / "css"), name="css")
app.mount("/js", StaticFiles(directory=Path(__file__).parent / "static" / "js"), name="js")
templates = Jinja2Templates(directory=Path(__file__).parent / "templates")


# ============================================================================
# Data Models
# ============================================================================

class TradingConfig(BaseModel):
    initial_capital: float = 10000.0
    leverage: float = 1.0
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
        """Update current price and building candle."""
        self.current_price = price
        timestamp = datetime.now(KST)

        candle_start = timestamp.replace(second=0, microsecond=0)
        candle_start = candle_start - timedelta(minutes=candle_start.minute % 3)

        if self.current_candle_start != candle_start:
            if self.current_candle is not None and self.df_3m is not None:
                candle_ts = pd.Timestamp(self.current_candle_start)
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

    def get_chart_data(self, limit: int = 100, current_regime: str = "UNKNOWN") -> Dict[str, List[Dict]]:
        """Get chart data for frontend."""
        if self.df_3m is None:
            return {"candles": [], "volume": [], "regime": []}

        # Semi-transparent regime colors for background
        REGIME_BG_COLORS = {
            "TREND_UP": "rgba(38, 166, 154, 0.25)",
            "TREND_DOWN": "rgba(239, 83, 80, 0.25)",
            "RANGE": "rgba(66, 165, 245, 0.25)",
            "TRANSITION": "rgba(255, 202, 40, 0.25)",
            "HIGH_VOL": "rgba(171, 71, 188, 0.25)",
            "NO_TRADE": "rgba(158, 158, 158, 0.15)",
            "UNKNOWN": "rgba(158, 158, 158, 0.10)",
        }

        df = self.df_3m[~self.df_3m.index.duplicated(keep='last')].tail(limit).reset_index()
        # Drop rows with NaN values
        df = df.dropna(subset=["open", "high", "low", "close"])

        candles = []
        volume = []
        regime = []
        seen_times = set()

        for _, row in df.iterrows():
            # Skip if any OHLC value is NaN or None
            if pd.isna(row["open"]) or pd.isna(row["high"]) or pd.isna(row["low"]) or pd.isna(row["close"]):
                continue

            time_val = int(row["timestamp"].timestamp())

            # Skip duplicate timestamps (lightweight-charts doesn't allow duplicates)
            if time_val in seen_times:
                continue
            seen_times.add(time_val)

            candles.append({
                "time": time_val,
                "open": float(row["open"]),
                "high": float(row["high"]),
                "low": float(row["low"]),
                "close": float(row["close"]),
            })
            vol_val = float(row["volume"]) if not pd.isna(row["volume"]) else 0.0
            color = "#26a69a80" if row["close"] >= row["open"] else "#ef535080"
            volume.append({
                "time": time_val,
                "value": vol_val,
                "color": color,
            })
            # Regime background (use UNKNOWN for historical data)
            regime_color = REGIME_BG_COLORS.get(current_regime, REGIME_BG_COLORS["UNKNOWN"])
            regime.append({
                "time": time_val,
                "value": 100,  # Fixed high value to fill the chart
                "color": regime_color,
            })

        return {"candles": candles, "volume": volume, "regime": regime}


# ============================================================================
# Web Regime Change Trader
# ============================================================================

class WebRegimeChangeTrader:
    """Regime change paper trader with event callbacks for web broadcasting."""

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

        # Import RegimeChangePaperTrader
        from paper_trading_regime_change import RegimeChangePaperTrader

        # Create actual paper trader
        self.trader = RegimeChangePaperTrader(
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
        logger.info("Starting WebRegimeChangeTrader...")

        # Initialize the underlying trader
        self.trader.initialize()
        self.running = True

        # Get initial regime
        try:
            regime_raw, regime_confirmed, reason = self.trader.detect_regime()
            self.last_regime = regime_confirmed
            self.trader.prev_regime_confirmed = regime_confirmed

            self.regimes.append({
                "timestamp": datetime.now(KST).isoformat(),
                "regime": self.last_regime,
                "color": REGIME_COLORS.get(self.last_regime, "#9e9e9e"),
            })
            logger.info(f"Initial regime: {self.last_regime}")
        except Exception as e:
            logger.error(f"Failed to get initial regime: {e}")
            self.last_regime = "UNKNOWN"

        logger.info(f"WebRegimeChangeTrader initialized. Last 3m bar: {self.trader.last_3m_ts}")

    async def stop(self):
        """Stop the paper trading loop."""
        self.running = False
        logger.info("WebRegimeChangeTrader stopped.")

    async def tick(self) -> Dict:
        """Run one tick of the paper trading loop."""
        if not self.running:
            return {}

        try:
            # Get current kline with volume
            kline = self.chart_manager.client.get_current_kline(self.symbol, "3m")
            current_price = kline["close"] if kline else self.trader.client.get_ticker_price(self.symbol)
            self.trader._update_tick_history(current_price)

            # Update chart manager with kline data
            candle = self.chart_manager.update_price(current_price, kline)

            # Check stops if in position
            if self.trader.position.side != "FLAT":
                self.trader._update_max_profit(current_price)

                # Break-even stop
                be_result = self.trader._check_breakeven_stop(current_price)
                if be_result:
                    self.trader._execute_exit(be_result[0], be_result[1])
                    await self._emit_trade_event("BREAKEVEN_STOP", be_result[1])
                else:
                    # Trailing stop
                    ts_result = self.trader._check_trailing_stop(current_price)
                    if ts_result:
                        self.trader._execute_exit(ts_result[0], ts_result[1])
                        await self._emit_trade_event("TRAILING_STOP", ts_result[1])

            # Check pending signal
            if self.trader.pending_signal is not None:
                executed = self.trader._check_pending_entry(current_price)
                if executed:
                    await self._emit_trade_event("ENTRY", "OPTIMIZED_ENTRY")

            # Check for new 3m bar
            new_bar = self.trader.update_data()

            regime = self.last_regime or "UNKNOWN"
            regime_changed = False

            if new_bar:
                # Cancel pending signal on new bar
                if self.trader.pending_signal:
                    logger.info("[CANCELLED] Pending signal - new bar")
                    self.trader.pending_signal = None

                # Detect regime
                regime_raw, regime_confirmed, reason = self.trader.detect_regime()
                price = float(self.trader.df_3m.iloc[-1]["close"])

                # Detect regime change
                regime_signal = self.trader.detect_regime_change(regime_confirmed)

                # Check regime change
                if self.last_regime is not None and regime_confirmed != self.last_regime:
                    regime_changed = True
                    await self._emit_regime_change(self.last_regime, regime_confirmed)

                self.last_regime = regime_confirmed
                self.trader.prev_regime_confirmed = regime_confirmed
                regime = regime_confirmed

                # Handle regime change signal
                if regime_signal:
                    logger.info(
                        f"[REGIME CHANGE] {self.trader.prev_regime_confirmed} -> {regime_confirmed} | "
                        f"Signal: {regime_signal} | Price: ${price:.4f}"
                    )
                    self.trader._save_signal(price, self.trader.prev_regime_confirmed or "NONE", regime_confirmed, regime_signal)

                    # Create entry signal if flat
                    if self.trader.position.side == "FLAT":
                        if "LONG" in regime_signal:
                            self.trader._set_pending_signal("LONG", price, regime_signal, regime_confirmed)
                        elif "SHORT" in regime_signal:
                            self.trader._set_pending_signal("SHORT", price, regime_signal, regime_confirmed)

                # Update position bars held
                if self.trader.position.side != "FLAT":
                    self.trader.position.bars_held_3m += 1

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
            import traceback
            traceback.print_exc()
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
            "signal": last_trade.get("signal", ""),
        }
        self.trades.append(trade_info)

        if len(self.trades) > 100:
            self.trades = self.trades[-100:]

        if self.on_trade:
            await self.on_trade(trade_info)

    def get_stats(self) -> Dict:
        """Get current trading statistics."""
        stats = self.trader.get_stats()
        position = self.trader.get_position()

        return {
            "equity": stats["equity"],
            "initial_capital": stats["initial_capital"],
            "total_pnl": stats["total_pnl"],
            "pnl_pct": stats["pnl_pct"],
            "total_trades": stats["total_trades"],
            "winning_trades": stats["winning_trades"],
            "win_rate": stats["win_rate"],
            "position": position,
            "current_regime": self.last_regime or "UNKNOWN",
            "pending_signal": self.trader.pending_signal.get("signal") if self.trader.pending_signal else None,
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
web_trader: Optional[WebRegimeChangeTrader] = None
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
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Trading loop error: {e}")
            import traceback
            traceback.print_exc()
            await asyncio.sleep(5)


async def price_update_task():
    """Background task to update price when trading is not active."""
    global chart_manager, web_trader

    client = BinanceClient()

    while True:
        try:
            if chart_manager and chart_manager.initialized and len(manager.active_connections) > 0:
                if web_trader is None or not web_trader.running:
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
                            "position": {"side": "FLAT", "entry_price": 0, "size": 0, "unrealized_pnl": 0},
                            "current_regime": "UNKNOWN",
                            "pending_signal": None,
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
    logger.info("Regime Change Trading web service started with initial data loaded")


@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    """Serve main dashboard page."""
    return templates.TemplateResponse("index_regime_change.html", {"request": request})


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
    web_trader = WebRegimeChangeTrader(
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
        "mode": "regime_change",
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
            "position": {"side": "FLAT", "entry_price": 0, "size": 0, "unrealized_pnl": 0},
            "current_regime": "UNKNOWN",
            "pending_signal": None,
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
        # Get current regime first
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
                "position": {"side": "FLAT", "entry_price": 0, "size": 0, "unrealized_pnl": 0},
                "current_regime": "UNKNOWN",
                "pending_signal": None,
            }
            trades = []
            regimes = []
            regime = "UNKNOWN"

        # Send initial data with regime
        if chart_manager and chart_manager.initialized:
            chart_result = chart_manager.get_chart_data(100, regime)
            chart_data = chart_result["candles"]
            volume_data = chart_result["volume"]
            regime_data = chart_result["regime"]
        else:
            chart_data = []
            volume_data = []
            regime_data = []

        is_running = web_trader is not None and web_trader.running

        await websocket.send_json({
            "type": "init",
            "chart_data": chart_data,
            "volume_data": volume_data,
            "regime_data": regime_data,
            "stats": stats,
            "trades": trades,
            "regimes": regimes,
            "price": chart_manager.current_price if chart_manager else 0,
            "regime": regime,
            "regime_color": REGIME_COLORS.get(regime, "#9e9e9e"),
            "is_running": is_running,
            "strategy": "Regime Change Entry",
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
    uvicorn.run(app, host="0.0.0.0", port=8082)
