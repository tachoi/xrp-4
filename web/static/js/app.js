/**
 * XRP Trading Dashboard - Frontend Application
 */

// ============================================================================
// Global State
// ============================================================================

let chart = null;
let candleSeries = null;
let volumeSeries = null;
let regimeSeries = null;
let ws = null;
let isRunning = false;

// Regime colors (solid)
const REGIME_COLORS = {
    'TREND_UP': '#26a69a',
    'TREND_DOWN': '#ef5350',
    'RANGE': '#42a5f5',
    'TRANSITION': '#ffca28',
    'HIGH_VOL': '#ab47bc',
    'NO_TRADE': '#9e9e9e',
    'UNKNOWN': '#9e9e9e',
};

// Regime colors (semi-transparent for background)
const REGIME_BG_COLORS = {
    'TREND_UP': 'rgba(38, 166, 154, 0.15)',
    'TREND_DOWN': 'rgba(239, 83, 80, 0.15)',
    'RANGE': 'rgba(66, 165, 245, 0.15)',
    'TRANSITION': 'rgba(255, 202, 40, 0.15)',
    'HIGH_VOL': 'rgba(171, 71, 188, 0.15)',
    'NO_TRADE': 'rgba(158, 158, 158, 0.1)',
    'UNKNOWN': 'rgba(158, 158, 158, 0.1)',
};

// Current regime for each candle time
let regimeData = {};

// Trade markers storage
let markers = [];

// ============================================================================
// Chart Initialization
// ============================================================================

// Korean timezone offset (UTC+9) in seconds
const KST_OFFSET = 9 * 60 * 60;

function initChart() {
    const chartContainer = document.getElementById('chart');

    chart = LightweightCharts.createChart(chartContainer, {
        layout: {
            background: { type: 'solid', color: '#1e222d' },
            textColor: '#d1d4dc',
        },
        grid: {
            vertLines: { color: '#2a2e39' },
            horzLines: { color: '#2a2e39' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: '#2a2e39',
        },
        timeScale: {
            borderColor: '#2a2e39',
            timeVisible: true,
            secondsVisible: false,
            rightOffset: 5,
            shiftVisibleRangeOnNewBar: true,
        },
        localization: {
            priceFormatter: (price) => price.toFixed(4),
        },
    });

    // Create regime background series (added first to be behind candles)
    regimeSeries = chart.addHistogramSeries({
        color: 'rgba(66, 165, 245, 0.15)',
        priceFormat: {
            type: 'price',
        },
        priceScaleId: 'regime',
        lastValueVisible: false,
        priceLineVisible: false,
    });

    // Configure regime price scale (hidden, full height)
    chart.priceScale('regime').applyOptions({
        scaleMargins: {
            top: 0,
            bottom: 0,
        },
        visible: false,
    });

    // Create candlestick series with price precision
    candleSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderDownColor: '#ef5350',
        borderUpColor: '#26a69a',
        wickDownColor: '#ef5350',
        wickUpColor: '#26a69a',
        priceFormat: {
            type: 'price',
            precision: 4,
            minMove: 0.0001,
        },
    });

    // Create volume series
    volumeSeries = chart.addHistogramSeries({
        color: '#26a69a',
        priceFormat: {
            type: 'volume',
        },
        priceScaleId: '',
        scaleMargins: {
            top: 0.8,
            bottom: 0,
        },
    });

    // Handle resize
    window.addEventListener('resize', () => {
        chart.applyOptions({
            width: chartContainer.clientWidth,
            height: chartContainer.clientHeight,
        });
    });

    // Trigger initial resize
    chart.applyOptions({
        width: chartContainer.clientWidth,
        height: chartContainer.clientHeight,
    });
}

// ============================================================================
// WebSocket Connection
// ============================================================================

function connectWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}/ws`);

    ws.onopen = () => {
        console.log('WebSocket connected');
        updateStatus(true);
    };

    ws.onclose = () => {
        console.log('WebSocket disconnected');
        updateStatus(false);
        // Reconnect after 3 seconds
        setTimeout(connectWebSocket, 3000);
    };

    ws.onerror = (error) => {
        console.error('WebSocket error:', error);
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleMessage(data);
    };

    // Ping every 30 seconds to keep connection alive
    setInterval(() => {
        if (ws && ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
        }
    }, 30000);
}

function handleMessage(data) {
    switch (data.type) {
        case 'init':
            // Initial data load with KST timezone adjustment
            if (data.chart_data) {
                const adjustedCandles = data.chart_data.map(c => ({
                    ...c,
                    time: c.time + KST_OFFSET,
                }));
                candleSeries.setData(adjustedCandles);

                // Set volume data
                if (data.volume_data) {
                    const adjustedVolume = data.volume_data.map(v => ({
                        ...v,
                        time: v.time + KST_OFFSET,
                    }));
                    volumeSeries.setData(adjustedVolume);
                }

                // Set regime background (use current regime for all candles initially)
                const currentRegime = data.regime || 'UNKNOWN';
                setRegimeData(adjustedCandles, currentRegime);

                // Track last candle time
                if (adjustedCandles.length > 0) {
                    lastCandleTime = adjustedCandles[adjustedCandles.length - 1].time;
                }

                // Scroll to latest
                chart.timeScale().scrollToRealTime();
            }
            if (data.price) {
                updatePrice(data.price);
            }
            if (data.regime && data.regime_color) {
                updateRegimeBadge(data.regime, data.regime_color);
            }
            if (data.stats) {
                updateStats(data.stats);
            }
            if (data.trades) {
                updateTradeHistory(data.trades);
            }
            if (data.regimes) {
                updateRegimeHistory(data.regimes);
            }
            // Sync running state from server
            if (data.is_running !== undefined) {
                isRunning = data.is_running;
                document.getElementById('start-btn').disabled = isRunning;
                document.getElementById('stop-btn').disabled = !isRunning;
                updateStatus(true);
            }
            break;

        case 'tick':
            // Real-time price update
            updatePrice(data.price);
            updateRegimeBadge(data.regime, data.regime_color);

            if (data.candle) {
                updateCandle(data.candle, data.regime);
            }
            if (data.stats) {
                updateStats(data.stats);
            }
            break;

        case 'trade':
            // New trade executed
            addTrade(data.trade);
            addTradeMarker(data.trade);
            if (data.stats) {
                updateStats(data.stats);
            }
            break;

        case 'regime_change':
            // Regime changed
            addRegime(data.regime);
            updateRegimeBadge(data.regime.regime, REGIME_COLORS[data.regime.regime]);
            break;

        case 'pong':
            // Keepalive response
            break;
    }
}

// ============================================================================
// UI Updates
// ============================================================================

function updatePrice(price) {
    const priceEl = document.getElementById('current-price');
    priceEl.textContent = `$${price.toFixed(4)}`;
}

function updateRegimeBadge(regime, color) {
    const badge = document.getElementById('current-regime');
    badge.textContent = regime;
    badge.style.backgroundColor = color;
    badge.style.color = regime === 'TRANSITION' ? '#131722' : '#fff';
}

// Track last candle time to detect new candle
let lastCandleTime = null;

// Update regime background for a specific time
function updateRegimeBackground(time, regime) {
    if (!regimeSeries) return;

    const color = REGIME_BG_COLORS[regime] || REGIME_BG_COLORS['UNKNOWN'];
    regimeData[time] = regime;

    regimeSeries.update({
        time: time,
        value: 100,  // Fixed high value to fill the chart
        color: color,
    });
}

// Set initial regime data for all candles
function setRegimeData(candles, defaultRegime) {
    if (!regimeSeries || !candles || candles.length === 0) return;

    const data = candles.map(c => ({
        time: c.time,
        value: 100,
        color: REGIME_BG_COLORS[defaultRegime] || REGIME_BG_COLORS['UNKNOWN'],
    }));

    regimeSeries.setData(data);
}

function updateCandle(candle, regime) {
    if (!candleSeries) return;

    // Adjust time to KST
    const time = Math.floor(new Date(candle.timestamp).getTime() / 1000) + KST_OFFSET;

    const candleData = {
        time: time,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
    };

    candleSeries.update(candleData);

    // Update volume if available
    if (volumeSeries && candle.volume !== undefined) {
        const color = candle.close >= candle.open ? '#26a69a80' : '#ef535080';
        volumeSeries.update({
            time: time,
            value: candle.volume,
            color: color,
        });
    }

    // Update regime background if regime is provided
    if (regime) {
        updateRegimeBackground(time, regime);
    }

    // Auto-scroll to latest candle when new candle starts
    if (lastCandleTime !== null && time > lastCandleTime) {
        chart.timeScale().scrollToRealTime();
    }
    lastCandleTime = time;
}

function updateStats(stats) {
    // Equity
    document.getElementById('stat-equity').textContent =
        `$${stats.equity.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;

    // Total PnL
    const pnlEl = document.getElementById('stat-pnl');
    const pnlValue = stats.total_pnl;
    pnlEl.textContent = `${pnlValue >= 0 ? '+' : ''}$${pnlValue.toFixed(2)}`;
    pnlEl.className = `stat-value ${pnlValue >= 0 ? 'positive' : 'negative'}`;

    // Return
    const returnEl = document.getElementById('stat-return');
    const returnValue = stats.pnl_pct;
    returnEl.textContent = `${returnValue >= 0 ? '+' : ''}${returnValue.toFixed(2)}%`;
    returnEl.className = `stat-value ${returnValue >= 0 ? 'positive' : 'negative'}`;

    // Trades
    document.getElementById('stat-trades').textContent = stats.total_trades;

    // Win Rate
    document.getElementById('stat-winrate').textContent = `${stats.win_rate.toFixed(1)}%`;

    // Position
    const posEl = document.getElementById('stat-position');
    const position = stats.position;
    if (position.side === 'FLAT') {
        posEl.textContent = 'FLAT';
        posEl.className = 'stat-value';
    } else {
        posEl.textContent = `${position.side} @ $${position.entry_price.toFixed(4)}`;
        posEl.className = `stat-value ${position.side.toLowerCase()}`;
    }
}

function updateTradeHistory(trades) {
    const tbody = document.getElementById('trade-table-body');
    tbody.innerHTML = '';

    trades.slice().reverse().forEach(trade => {
        const time = new Date(trade.timestamp).toLocaleTimeString('ko-KR', {
            hour: '2-digit',
            minute: '2-digit'
        });

        // Exit row (sell for LONG, buy for SHORT)
        const exitRow = document.createElement('tr');
        const exitType = trade.side === 'LONG' ? 'SELL' : 'BUY';
        const exitClass = trade.side === 'LONG' ? 'short' : 'long';
        exitRow.innerHTML = `
            <td>${time}</td>
            <td class="${exitClass}">${exitType}</td>
            <td class="${trade.side.toLowerCase()}">${trade.side}</td>
            <td>$${trade.exit_price.toFixed(4)}</td>
            <td>${trade.size.toFixed(2)}</td>
            <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">
                ${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}
            </td>
        `;
        tbody.appendChild(exitRow);

        // Entry row (buy for LONG, sell for SHORT)
        const entryRow = document.createElement('tr');
        const entryType = trade.side === 'LONG' ? 'BUY' : 'SELL';
        const entryClass = trade.side === 'LONG' ? 'long' : 'short';
        entryRow.innerHTML = `
            <td>${time}</td>
            <td class="${entryClass}">${entryType}</td>
            <td class="${trade.side.toLowerCase()}">${trade.side}</td>
            <td>$${trade.entry_price.toFixed(4)}</td>
            <td>${trade.size.toFixed(2)}</td>
            <td>-</td>
        `;
        tbody.appendChild(entryRow);
    });
}

function addTrade(trade) {
    const tbody = document.getElementById('trade-table-body');
    const time = new Date(trade.timestamp).toLocaleTimeString('ko-KR', {
        hour: '2-digit',
        minute: '2-digit'
    });

    // Exit row (sell for LONG, buy for SHORT) - insert first (most recent)
    const exitRow = document.createElement('tr');
    const exitType = trade.side === 'LONG' ? 'SELL' : 'BUY';
    const exitClass = trade.side === 'LONG' ? 'short' : 'long';
    exitRow.innerHTML = `
        <td>${time}</td>
        <td class="${exitClass}">${exitType}</td>
        <td class="${trade.side.toLowerCase()}">${trade.side}</td>
        <td>$${trade.exit_price.toFixed(4)}</td>
        <td>${trade.size.toFixed(2)}</td>
        <td class="${trade.pnl >= 0 ? 'positive' : 'negative'}">
            ${trade.pnl >= 0 ? '+' : ''}$${trade.pnl.toFixed(2)}
        </td>
    `;

    // Entry row (buy for LONG, sell for SHORT)
    const entryRow = document.createElement('tr');
    const entryType = trade.side === 'LONG' ? 'BUY' : 'SELL';
    const entryClass = trade.side === 'LONG' ? 'long' : 'short';
    entryRow.innerHTML = `
        <td>${time}</td>
        <td class="${entryClass}">${entryType}</td>
        <td class="${trade.side.toLowerCase()}">${trade.side}</td>
        <td>$${trade.entry_price.toFixed(4)}</td>
        <td>${trade.size.toFixed(2)}</td>
        <td>-</td>
    `;

    // Insert entry first, then exit (so exit appears above entry)
    tbody.insertBefore(entryRow, tbody.firstChild);
    tbody.insertBefore(exitRow, tbody.firstChild);

    // Keep only last 100 rows (50 trades x 2 rows)
    while (tbody.children.length > 100) {
        tbody.removeChild(tbody.lastChild);
    }
}

function addTradeMarker(trade) {
    if (!candleSeries) return;

    // Apply KST offset to match candle times
    const time = Math.floor(new Date(trade.timestamp).getTime() / 1000) + KST_OFFSET;
    const isLong = trade.side === 'LONG';

    // Entry marker
    markers.push({
        time: time,
        position: isLong ? 'belowBar' : 'aboveBar',
        color: isLong ? '#26a69a' : '#ef5350',
        shape: isLong ? 'arrowUp' : 'arrowDown',
        text: isLong ? 'L' : 'S',
    });

    candleSeries.setMarkers(markers);
}

function updateRegimeHistory(regimes) {
    const container = document.getElementById('regimes-list');
    container.innerHTML = '';

    regimes.slice().reverse().forEach(regime => {
        addRegimeItem(regime, container);
    });
}

function addRegime(regime) {
    const container = document.getElementById('regimes-list');
    addRegimeItem(regime, container, true);

    // Keep only last 20 items
    while (container.children.length > 20) {
        container.removeChild(container.lastChild);
    }
}

function addRegimeItem(regime, container, prepend = false) {
    const item = document.createElement('div');
    item.className = 'regime-item';

    const time = new Date(regime.timestamp).toLocaleTimeString('ko-KR', {
        hour: '2-digit',
        minute: '2-digit'
    });
    const color = regime.color || REGIME_COLORS[regime.regime] || REGIME_COLORS['UNKNOWN'];

    item.innerHTML = `
        <span class="regime-color" style="background-color: ${color}"></span>
        <span class="regime-time">${time}</span>
        <span class="regime-name">${regime.regime}</span>
    `;

    if (prepend) {
        container.insertBefore(item, container.firstChild);
    } else {
        container.appendChild(item);
    }
}

function updateStatus(connected) {
    const dot = document.getElementById('status-dot');
    const text = document.getElementById('status-text');

    if (isRunning && connected) {
        dot.className = 'status-dot running';
        text.textContent = 'Running';
    } else if (connected) {
        dot.className = 'status-dot';
        text.textContent = 'Connected';
    } else {
        dot.className = 'status-dot';
        text.textContent = 'Disconnected';
    }
}

// ============================================================================
// API Calls
// ============================================================================

async function startTrading() {
    const mode = document.getElementById('trading-mode').value;
    const capital = parseFloat(document.getElementById('initial-capital').value);
    const leverage = parseFloat(document.getElementById('leverage').value);

    try {
        const response = await fetch('/api/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                mode: mode,
                initial_capital: capital,
                leverage: leverage,
                symbol: 'XRPUSDT',
            }),
        });

        if (response.ok) {
            isRunning = true;
            document.getElementById('start-btn').disabled = true;
            document.getElementById('stop-btn').disabled = false;
            updateStatus(true);

            // Clear previous data
            markers = [];
            if (candleSeries) {
                candleSeries.setMarkers([]);
            }
        } else {
            const error = await response.json();
            alert(`Failed to start: ${error.detail}`);
        }
    } catch (err) {
        console.error('Start error:', err);
        alert('Failed to start trading');
    }
}

async function stopTrading() {
    try {
        const response = await fetch('/api/stop', {
            method: 'POST',
        });

        if (response.ok) {
            isRunning = false;
            document.getElementById('start-btn').disabled = false;
            document.getElementById('stop-btn').disabled = true;
            updateStatus(true);

            const result = await response.json();
            if (result.final_stats) {
                updateStats(result.final_stats);
            }
        }
    } catch (err) {
        console.error('Stop error:', err);
        alert('Failed to stop trading');
    }
}

// ============================================================================
// Event Listeners
// ============================================================================

document.addEventListener('DOMContentLoaded', () => {
    // Initialize chart
    initChart();

    // Connect WebSocket
    connectWebSocket();

    // Button handlers
    document.getElementById('start-btn').addEventListener('click', startTrading);
    document.getElementById('stop-btn').addEventListener('click', stopTrading);
});
