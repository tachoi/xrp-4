#!/usr/bin/env python
"""Run Web Trading Dashboard.

Usage:
    python scripts/run_web.py
    python scripts/run_web.py --port 8080
    python scripts/run_web.py --host 0.0.0.0 --port 8080
"""

import argparse
import sys
from pathlib import Path

# Add paths
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "web"))


def main():
    parser = argparse.ArgumentParser(description="Run Web Trading Dashboard")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind (default: 8080)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload for development")
    args = parser.parse_args()

    import uvicorn
    from app import app

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           XRP Trading Dashboard                              ║
╠══════════════════════════════════════════════════════════════╣
║  URL: http://{args.host}:{args.port}                              ║
║  Press Ctrl+C to stop                                        ║
╚══════════════════════════════════════════════════════════════╝
    """)

    uvicorn.run(
        "app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
