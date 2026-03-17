#!/usr/bin/env python3
import argparse
import ast
import json
from typing import Iterable, List


def parse_dimensions(value) -> List[str]:
    if isinstance(value, list):
        return [str(x) for x in value]
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = None
        if isinstance(parsed, list):
            return [str(x) for x in parsed]
        return [value]
    return [str(value)]


def pick_one_per_dimension(items: Iterable[dict]) -> dict:
    picked = {}
    for item in items:
        dims = parse_dimensions(item.get("dimensions"))
        for dim in dims:
            if dim not in picked:
                picked[dim] = item
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pick one QA entry for each dimension in ECbench_qa.json."
    )
    parser.add_argument(
        "--input",
        default="data/ECbench_qa.json",
        help="Path to ECbench_qa.json",
    )
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        data = json.load(f)

    picked = pick_one_per_dimension(data)
    print(json.dumps(picked, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
