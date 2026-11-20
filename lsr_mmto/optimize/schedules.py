"""
schedules.py
============
How a number that changes during a run changes: the Heaviside sharpness, and each SIMP exponent with SIMP
continuation on. A Schedule starts at a value and, on every iteration that is a multiple of its every, after
iteration after, grows by its step (multiplied or added, Growth) up to its largest. The loop's continuation
step asks each schedule for the next iteration's value; a new kind of growth is a member of Growth and a
line in Schedule.next, not a change to the loop.
"""
from dataclasses import dataclass
from typing import TYPE_CHECKING

from lsr_mmto.problem.settings import Growth

if TYPE_CHECKING:
    from types import ModuleType

    from lsr_mmto.problem.settings import RunSettings

SIMP_STEP, SIMP_EVERY = 0.25, 10              # SIMP continuation: raised by this, every this many iterations


@dataclass(frozen=True)
class Schedule:
    """start, then on each iteration > after that is a multiple of every, grown by step (times step for
    Growth.MULTIPLY, plus step for Growth.ADD), never past largest."""
    start: float
    largest: float
    growth: Growth
    step: float
    every: int
    after: int = 0

    def next(self, value: float, iteration: int) -> tuple[float, bool]:
        """(the value at iteration, whether it grew there), from the value before it."""
        if iteration <= self.after or iteration % self.every != 0 or value >= self.largest:
            return value, False
        if self.growth is Growth.MULTIPLY:
            grown = value * self.step
        elif self.growth is Growth.ADD:
            grown = value + self.step
        else:
            raise ValueError(f"a schedule grows by one of {', '.join(Growth.__members__)}, not {self.growth!r}")
        return min(grown, self.largest), True


BETA_KEYS = ("beta_init", "beta_max", "beta_growth", "beta_factor", "beta_update_every", "beta_start_iter")


def sharpness_schedule(s: "RunSettings") -> Schedule:
    """The Heaviside sharpness schedule, as the run's settings give it: it starts at Starting sharpness and,
    after iteration Start growing after iteration, grows by Sharpness growth (multiplied or added, Growth by)
    every Grow every iterations, up to Largest sharpness. ValueError, in the settings' own words, for numbers it
    cannot run with."""
    growth = Growth[s.beta_growth] if isinstance(s.beta_growth, str) else s.beta_growth   # a name, or the member
    wrong = [text for bad, text in (
        (s.beta_init <= 0, "Starting sharpness must be above 0"),
        (s.beta_max < s.beta_init, "Largest sharpness must be at least Starting sharpness"),
        (growth is Growth.MULTIPLY and s.beta_factor < 1, "Sharpness growth must be at least 1"),
        (growth is Growth.ADD and s.beta_factor < 0, "Sharpness growth must be at least 0"),
        (s.beta_update_every < 1, "Grow every must be at least 1"),
        (s.beta_start_iter < 0, "Start growing after iteration must be at least 0")) if bad]
    if wrong:
        raise ValueError("; ".join(wrong))
    return Schedule(start=s.beta_init, largest=s.beta_max, growth=growth, step=s.beta_factor,
                    every=s.beta_update_every, after=s.beta_start_iter)


def sharpness_record(s: "RunSettings") -> dict[str, float | str]:
    """The sharpness settings as the run record keeps them: {key: value} of BETA_KEYS."""
    return {k: getattr(getattr(s, k), "name", getattr(s, k)) for k in BETA_KEYS}


def simp_schedules(physics: "ModuleType", continuation: bool) -> dict[str, Schedule]:
    """{name: Schedule} of each SIMP exponent of the physics (its SIMP_PENALTIES: start, start with continuation,
    largest): with SIMP continuation on, from its start with continuation, raised by SIMP_STEP every SIMP_EVERY
    iterations up to its largest; off, its start, never raised. The core names no physics."""
    return {name: (Schedule(start=float(with_continuation), largest=float(largest), growth=Growth.ADD,
                            step=SIMP_STEP, every=SIMP_EVERY) if continuation else
                   Schedule(start=float(start), largest=float(start), growth=Growth.ADD, step=0.0, every=1))
            for name, (start, with_continuation, largest) in physics.SIMP_PENALTIES.items()}
