#!/usr/bin/env python3
"""Trace every number in RESULTS.md to the output of the check.py of the claims tagged on the same line.

Run after `make check` (which writes claims/<ID>/check_output.txt). For each line of RESULTS.md carrying tags like
[K5.8] or [K5.2, K5.7], every decimal number on that line (unicode minus folded, errors in parentheses dropped,
scientific notation "a × 10⁻ⁿ" normalised) must occur in the tagged outputs, either literally or rounded from a
printed number to the quoted number of decimals. Lines without tags are skipped (prose). Exit 1 on any miss.
"""

import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
SUP = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")
NUM = re.compile(r"(?<![\w.])-?\d+(?:\.\d+)?(?:e-?\d+)?")


def normalise(s):
    s = s.replace("−", "-").replace("–", " ").replace("…", " ")
    s = re.sub(r"(\d(?:\.\d+)?)\s*×\s*10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", lambda m: m.group(1) + "e" + m.group(2).translate(SUP), s)
    s = re.sub(r"\(\d+(?:\.\d+)?\)", "", s)  # drop the error in parentheses
    return s


def numbers(s):
    return [t for t in NUM.findall(normalise(s))]


def found(tok, outputs, printed):
    t = tok.lstrip("-")
    if t in outputs:
        return True
    if "." not in t or "e" in t:
        return False
    nd = len(t.split(".")[1])
    target = float(t)
    return any(abs(abs(v) - target) <= 0.5 * 10**-nd + 1e-12 for v in printed)


def main():
    text = (ROOT / "RESULTS.md").read_text().splitlines()
    misses, n = [], 0
    for ln in text:
        tags = re.findall(r"\[((?:[KIM]\d*\.\d+(?:,\s*)?)+)\]", ln)
        if not tags:
            continue
        ids = [i.strip() for t in tags for i in t.split(",")]
        out = ""
        for i in ids:
            f = ROOT / "claims" / i / "check_output.txt"
            if not f.exists():
                misses.append((i, "no check_output.txt"))
                continue
            out += f.read_text()
        outn = normalise(out)
        printed = []
        for t in NUM.findall(outn):
            try:
                printed.append(float(t))
            except ValueError:
                pass
        body = re.sub(r"\[(?:[KIM]\d*\.\d+(?:,\s*)?)+\]", "", ln)
        body = re.sub(r"\b[KIM]\d(?:\.\d+)?\b", "", body)  # claim names in prose
        for tok in numbers(body):
            if re.fullmatch(r"-?\d", tok):  # single digits (indices, counts in words)
                continue
            n += 1
            if not found(tok, outn, printed):
                misses.append((",".join(ids), tok))
    for m in misses:
        print("MISS", *m)
    print(f"{n} numbers checked, {len(misses)} misses")
    sys.exit(1 if misses else 0)


if __name__ == "__main__":
    main()
