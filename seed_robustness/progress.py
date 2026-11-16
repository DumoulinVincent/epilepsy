"""Shared progress-reporting helpers.

Three different granularities of "how long is this going to take":

1. `brian_progress_reporter(tag)` -- for a single net.run() call whose full
   duration is known in advance (baseline calibration: exactly calib_time_s
   of simulated time). Brian2's `report=` accepts a callback
   `f(elapsed, completed, start, duration)` called at `report_period`
   (always also at 0% and 100%); this builds one that prints a real,
   computed percentage + ETA.

2. `Stage` -- for work where the total isn't knowable up front (the
   phase-snapshot scan runs until it finds a match or exhausts its budget).
   No percentage; just an honest "started X / finished X after Ns" marker.
   Used inside coarse_phase.py.

3. Batch-level "N of M jobs done, elapsed, ETA" comes from joblib's own
   `verbose=` on Parallel(...) in run_seed_check.py, not from here -- that's
   the actual answer to "how long will the whole run take," extrapolated
   from real observed per-job durations rather than guessed.
"""
import time


def brian_progress_reporter(tag):
    def _cb(elapsed, completed, start, duration):
        pct = completed * 100
        elapsed_s = float(elapsed)
        eta_str = (f"{elapsed_s * (1 - completed) / completed:7.0f}s"
                  if completed > 0 else "    n/a")
        print(f"[{tag}] {pct:5.1f}%  elapsed={elapsed_s:7.0f}s wall  "
            f"ETA={eta_str}  ({float(start) + completed * float(duration):.0f}/"
            f"{float(start) + float(duration):.0f}s sim)", flush=True)
    return _cb


class Stage:
    """`with Stage(tag, "characterizing 36Hz cycle"):` prints a start line,
    then an elapsed-wall-time done line on exit."""

    def __init__(self, tag, label):
        self.tag = tag
        self.label = label

    def __enter__(self):
        self.t0 = time.time()
        print(f"[{self.tag}] {self.label} ...", flush=True)
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.time() - self.t0
        status = "FAILED" if exc_type else "done"
        print(f"[{self.tag}] {self.label}: {status} in {dt:.1f}s wall", flush=True)
        return False
