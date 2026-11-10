"""Build the processed dataset. Use --refresh to download fresh raw data first."""
import argparse
import json

from pmo.pipeline import build


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--refresh", action="store_true", help="re-download all raw data")
    args = parser.parse_args()
    manifest = build(refresh=args.refresh).manifest
    summary = {k: v for k, v in manifest.items() if k != "jumps_for_review"}
    summary["jumps_for_review"] = len(manifest["jumps_for_review"])
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
