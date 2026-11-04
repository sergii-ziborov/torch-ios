"""Compares two conformance.py results: compare.py reference.json candidate.json

Exits 1 when any case differs: a case that raised on one side only, a different
structure, shape or dtype, or values outside the tolerance for their dtype.
"""

import json
import math
import sys

TOLERANCE = {
    "torch.float64": (1e-7, 1e-9), "torch.complex128": (1e-7, 1e-9),
    "torch.float32": (2e-4, 2e-5), "torch.complex64": (2e-4, 2e-5),
    "torch.float16": (1e-2, 1e-3), "torch.bfloat16": (2e-2, 1e-2),
}


def close(a, b, rtol, atol, scale):
    if isinstance(a, float) and math.isnan(a):
        return isinstance(b, float) and math.isnan(b)
    if isinstance(a, float) and math.isinf(a):
        return a == b
    return abs(a - b) <= atol * max(1.0, scale) + rtol * abs(a)


def compare(ref, got, where, problems):
    if "error" in ref or "error" in got:
        if "error" in ref and "error" in got:
            return
        problems.append(f"{where}: {'reference' if 'error' in ref else 'this build'} raised: {(ref.get('error') or got.get('error'))}")
        return
    if "items" in ref:
        if len(ref["items"]) != len(got.get("items", [])):
            problems.append(f"{where}: {len(ref['items'])} results against {len(got.get('items', []))}")
            return
        for i, (a, b) in enumerate(zip(ref["items"], got["items"])):
            compare(a, b, f"{where}[{i}]", problems)
        return
    if "values" in ref:
        if ref["dtype"] != got.get("dtype") or ref["shape"] != got.get("shape"):
            problems.append(f"{where}: {ref['dtype']} {ref['shape']} against {got.get('dtype')} {got.get('shape')}")
            return
        rtol, atol = TOLERANCE.get(ref["dtype"], (0.0, 0.0))
        finite = [abs(v) for v in ref["values"] if isinstance(v, float) and math.isfinite(v)]
        scale = max(finite) if finite else 1.0
        bad = [i for i, (a, b) in enumerate(zip(ref["values"], got["values"])) if not close(a, b, rtol, atol, scale)]
        if bad:
            i = bad[0]
            problems.append(f"{where}: {len(bad)} of {len(ref['values'])} values differ, first at {i}: "
                            f"{ref['values'][i]!r} against {got['values'][i]!r} ({ref['dtype']})")
        return
    if "number" in ref:
        if not close(ref["number"], got.get("number", float("nan")), 2e-4, 1e-6, 1.0):
            problems.append(f"{where}: {ref['number']} against {got.get('number')}")
        return
    if ref != got:
        problems.append(f"{where}: {ref} against {got}")


def main(reference_path, candidate_path):
    reference = json.load(open(reference_path))
    candidate = json.load(open(candidate_path))
    problems = []
    for name, ref in reference["cases"].items():
        if name not in candidate["cases"]:
            problems.append(f"{name}: missing")
            continue
        compare(ref, candidate["cases"][name], name, problems)
    for line in problems:
        print("DIFF", line)
    raised = sorted(name for name, value in candidate["cases"].items() if "error" in value)
    print(f"{len(reference['cases'])} cases; {len(problems)} differences; raised in this build: {len(raised)}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:3]))
