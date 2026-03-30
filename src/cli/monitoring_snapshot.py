from __future__ import annotations

import argparse

from src.monitoring.service import build_and_save_monitoring_snapshot


def main() -> None:
    ap = argparse.ArgumentParser(description="Build monitoring snapshot from logs and recommendation artifacts.")
    ap.add_argument("--window-hours", type=int, default=24)
    args = ap.parse_args()

    path = build_and_save_monitoring_snapshot(window_hours=args.window_hours)
    print(f"[monitoring] wrote {path}")


if __name__ == "__main__":
    main()
