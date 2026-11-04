#!/usr/bin/env python3
"""Tonal Fidelity (TF) and TF-u for one source / edited video pair.

  python scripts/eval_tonal_fidelity.py --source SRC.mp4 --edited EDT.mp4             # TF
  python scripts/eval_tonal_fidelity.py --source SRC.mp4 --edited EDT.mp4 \
         --mask MASK.mp4 --category change_attribute_color                           # TF + TF-u
  ... --json      # machine-readable {kl, tf[, kl_u, tf_u, region, category]}

See aredit/metrics/tonal_fidelity.py for the definition (Equation 3).
"""

from __future__ import annotations

import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aredit.metrics.tonal_fidelity import BACKGROUND_EDIT_CATEGORIES, tonal_fidelity  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True, help="source (input) video .mp4")
    parser.add_argument("--edited", required=True, help="edited (output) video .mp4")
    parser.add_argument("--mask", default=None, help="edit-region mask video .mp4 (enables TF-u)")
    parser.add_argument("--category", default="",
                        help="edit category; one of %s uses the mask as the unedited region, "
                             "anything else the mask complement" % sorted(BACKGROUND_EDIT_CATEGORIES))
    parser.add_argument("--json", action="store_true", help="print a JSON dict instead of text")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = tonal_fidelity(args.source, args.edited, args.mask, args.category)
    if args.json:
        print(json.dumps(result.to_dict()))
        return
    print(f"KL   = {result.kl:.6f}   TF   = {result.tf:.6f}")
    if result.has_unedited:
        print(f"KL_u = {result.kl_u:.6f}   TF-u = {result.tf_u:.6f}   "
              f"(region: {result.region}, category: {result.category or 'n/a'})")


if __name__ == "__main__":
    main()
