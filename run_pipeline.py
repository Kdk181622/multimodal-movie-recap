"""Command-line entry point for the multimodal movie-recap pipeline.

Examples
--------
    python run_pipeline.py                              # use config.yaml (Detour, 1945)
    python run_pipeline.py --url https://youtu.be/XXXX --config my_config.yaml
    python run_pipeline.py --fresh summary.json peaks.json   # recompute selected stages
"""

import argparse
import sys

from recap.pipeline import CACHE_FILES, run


def main() -> int:
    ap = argparse.ArgumentParser(description="YouTube movie -> multimodal summary -> peak points -> < 10 min recap")
    ap.add_argument("--config", default=None, help="path to a YAML config (default: config.yaml)")
    ap.add_argument("--url", default=None, help="override the YouTube URL in the config")
    ap.add_argument("--fresh", nargs="*", default=[], choices=CACHE_FILES + ["all"],
                    help="cached stage files to delete before running")
    ap.add_argument("--skip-eval", action="store_true", help="stop after rendering the recap")
    args = ap.parse_args()
    fresh = CACHE_FILES if "all" in args.fresh else args.fresh
    try:
        run(args.config, args.url, fresh, args.skip_eval)
    except Exception as exc:  # surface a clean message; the traceback is still useful for debugging
        import traceback

        traceback.print_exc()
        print(f"\nPipeline failed: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
