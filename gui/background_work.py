"""Runs and analyses on one background thread, kept for the life of the app.

    run_in_background(work, *args)      work(*args) on that thread, after the jobs before it

Why one thread and not a new one per job: the maths libraries (MKL, OpenMP) give each thread that calls them a pool
of worker threads, one per core, and keep that pool after the thread ends. A new thread per run left about 23 idle
threads behind on every run and analysis of a 24-core computer; one thread builds its pool once and reuses it.

The thread is a daemon, so closing the app never waits for a run. An error in a job goes to threading.excepthook,
as it would from a thread of its own: the crash report and the console get it, and the next job still runs.
"""
import queue
import sys
import threading

_jobs = queue.Queue()
_worker = {"thread": None}
_lock = threading.Lock()


def run_in_background(work, *args):
    """work(*args) on the background thread (started on first use)."""
    with _lock:
        if _worker["thread"] is None or not _worker["thread"].is_alive():
            _worker["thread"] = threading.Thread(target=_serve, name="LSR MMTO background work", daemon=True)
            _worker["thread"].start()
    _jobs.put((work, args))


def _serve():
    while True:
        work, args = _jobs.get()
        try:
            work(*args)
        except BaseException:                                       # noqa: BLE001 (reported as a thread's error)
            threading.excepthook(threading.ExceptHookArgs((*sys.exc_info(), threading.current_thread())))
