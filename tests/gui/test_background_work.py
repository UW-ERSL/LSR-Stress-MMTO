"""gui/background_work.py: jobs run one after another on the same thread; an error in one is reported as a
thread's error would be, and the next job still runs."""
import threading

from gui.background_work import run_in_background


def test_jobs_share_one_thread_and_an_error_does_not_stop_the_next():
    seen, done, reported = [], threading.Event(), []
    hook = threading.excepthook
    threading.excepthook = lambda args: reported.append(args.exc_type)
    try:
        run_in_background(lambda: seen.append(threading.current_thread()))
        run_in_background(lambda: 1 / 0)
        run_in_background(lambda: (seen.append(threading.current_thread()), done.set()))
        assert done.wait(10), "the job after the error did not run"
    finally:
        threading.excepthook = hook
    assert len(seen) == 2 and seen[0] is seen[1] and seen[0] is not threading.main_thread()
    assert seen[0].daemon, "closing the app would wait for a run"
    assert reported == [ZeroDivisionError]
