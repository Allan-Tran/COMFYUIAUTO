import argparse
import json
import sys
from pathlib import Path

import rp_handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Invoke rp_handler locally with a JSON event file.")
    parser.add_argument(
        "--event",
        type=Path,
        default=Path("test_input_local_smoke.json"),
        help="Path to JSON file containing an event object or input payload.",
    )
    return parser.parse_args()


def load_event(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Event file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("Event JSON must be an object")

    # Accept either full event shape {"input": {...}} or direct input object.
    if "input" in data and isinstance(data.get("input"), dict):
        return data

    return {"input": data}


def main() -> int:
    args = parse_args()
    try:
        event = load_event(args.event)
        result = rp_handler.handler(event)
        print(json.dumps(result, indent=2, ensure_ascii=False))

        if str(result.get("status", "")).lower() == "error":
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"local_invoke failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
