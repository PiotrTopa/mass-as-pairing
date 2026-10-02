"""Helper for claims/<ID>/check.py: numbered items with their numbers and a final PASS/FAIL line.

    from masspairing.claimcheck import Check
    c = Check("K3.1")
    c.item("chi10 exponent (6,8) at (2.41, 0.05, +0.05)", value, value + 2 * err < 0.5, fmt="{:+.3f}")
    c.done()          # prints "PASS K3.1" and exits 0, or "FAIL K3.1" and exits 1

A check passes iff every item holds. ``c.record`` prints a number that is reported but not asserted.
"""

from __future__ import annotations

import sys
import time


class Check:
    def __init__(self, claim_id: str):
        self.id = claim_id
        self.items = []
        self.t0 = time.time()
        print(f"== {claim_id}", flush=True)

    def item(self, name, value, ok, fmt=None):
        ok = bool(ok)
        self.items.append((name, ok))
        v = self._fmt(value, fmt)
        print(f"  [{'ok' if ok else 'FAIL'}] {name}: {v}", flush=True)
        return ok

    def record(self, name, value, fmt=None):
        print(f"  [rec] {name}: {self._fmt(value, fmt)}", flush=True)

    @staticmethod
    def _fmt(value, fmt):
        if fmt is None:
            return repr(value) if not isinstance(value, str) else value
        try:
            return fmt.format(value)
        except (TypeError, ValueError):
            return repr(value)

    def done(self):
        n_fail = sum(1 for _, ok in self.items if not ok)
        dt = time.time() - self.t0
        if n_fail or not self.items:
            print(f"FAIL {self.id} ({n_fail} of {len(self.items)} items failed, {dt:.0f} s)", flush=True)
            sys.exit(1)
        print(f"PASS {self.id} ({len(self.items)} items, {dt:.0f} s)", flush=True)
        sys.exit(0)
