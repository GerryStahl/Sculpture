"""Lane traffic for the sculpture animation.

Sculptures glide along straight lanes. At a lane end, each one chooses a different, unoccupied
lane and curves over to it, then glides back the other way. The choice is the lane where the
whole turn stays farthest from every other sculpture, so traffic spreads itself out.

Pure Python (no Blender) so the motion can be tested and tuned on its own.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass

Point = tuple[float, float]

CLEARANCE_PER_HEIGHT = 0.3  # required centre distance is this times the sum of the two heights
LANE_CHANGE_BONUS = 2.0  # prefer moving to another lane over turning around in place
DISTANCE_PENALTY = 0.4  # per lane crossed, so short moves win when the options are similar
LOOKAHEAD_SEC = 18.0
LOOKAHEAD_STEP_SEC = 0.5


@dataclass
class Lane:
    y: float
    half: float  # the lane runs from x = -half to x = +half


class Segment:
    """A polyline traveled at constant speed, ending at a lane end."""

    def __init__(self, pts: list[Point], lane: int):
        self.pts = pts
        self.lane = lane
        self.cum = [0.0]
        for a, b in zip(pts, pts[1:]):
            self.cum.append(self.cum[-1] + math.dist(a, b))
        self.length = self.cum[-1]

    def at(self, s: float) -> Point:
        s = min(max(s, 0.0), self.length)
        i = min(bisect.bisect_right(self.cum, s) - 1, len(self.pts) - 2)
        span = self.cum[i + 1] - self.cum[i]
        f = (s - self.cum[i]) / span if span else 0.0
        a, b = self.pts[i], self.pts[i + 1]
        return (a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f)


@dataclass
class Walker:
    name: str
    speed: float
    height: float
    segment: Segment
    s: float = 0.0
    changes: int = 0

    def predict(self, seconds: float) -> Point:
        return self.segment.at(self.s + self.speed * seconds)


def lane_end_x(lane: Lane, side: int) -> float:
    return side * lane.half


def turn_points(lanes: list[Lane], src: int, dst: int, side: int, samples: int = 32) -> list[Point]:
    """Curve out past the end of lane src, over to lane dst, then straight back along dst."""
    a = (lane_end_x(lanes[src], side), lanes[src].y)
    b = (lane_end_x(lanes[dst], side), lanes[dst].y)
    bulge = 1.5 + 0.12 * abs(b[1] - a[1])
    p1 = (a[0] + side * bulge, a[1])
    p2 = (b[0] + side * bulge, b[1])
    pts = []
    for i in range(samples + 1):
        t = i / samples
        u = 1 - t
        pts.append(
            tuple(
                u**3 * a[d] + 3 * u * u * t * p1[d] + 3 * u * t * t * p2[d] + t**3 * b[d]
                for d in (0, 1)
            )
        )
    pts.append((lane_end_x(lanes[dst], -side), lanes[dst].y))
    return pts


def slack(walker: Walker, candidate: Segment, others: list[Walker]) -> float:
    """Smallest margin between the candidate route and every other sculpture's expected position."""
    horizon = min(candidate.length / walker.speed, LOOKAHEAD_SEC)
    worst = math.inf
    t = 0.0
    while t <= horizon:
        mine = candidate.at(walker.speed * t)
        for other in others:
            need = CLEARANCE_PER_HEIGHT * (walker.height + other.height)
            worst = min(worst, math.dist(mine, other.predict(t)) - need)
        t += LOOKAHEAD_STEP_SEC
    return worst


def choose_lane(lanes: list[Lane], walker: Walker, others: list[Walker], side: int) -> Segment:
    current = walker.segment.lane
    taken = {o.segment.lane for o in others}
    best = None
    for k in (k for k in range(len(lanes)) if k not in taken):
        seg = Segment(turn_points(lanes, current, k, side), k)
        bonus = LANE_CHANGE_BONUS if k != current else 0.0
        score = (slack(walker, seg, others) + bonus - DISTANCE_PENALTY * abs(k - current), 0)
        if best is None or score > best[0]:
            best = (score, seg)
    return best[1]


def start_segment(lanes: list[Lane], lane: int, direction: int, fraction: float) -> Segment:
    half = lanes[lane].half
    x0 = -direction * half + direction * fraction * 2 * half
    return Segment([(x0, lanes[lane].y), (direction * half, lanes[lane].y)], lane)


def simulate(
    lanes: list[Lane],
    sculptures: list[dict],
    duration: float,
    dt: float,
    clearance_scale: float = 1.0,
) -> tuple[dict[str, list[Point]], dict[str, int]]:
    """Return ({name: positions every dt seconds}, {name: lane changes}) for 0..duration."""
    walkers = []
    for cfg in sculptures:
        t = cfg["traffic"]
        segment = start_segment(lanes, t["lane"], t["direction"], t.get("start", 0.0))
        height = cfg.get("height", 2.0) * clearance_scale
        walkers.append(Walker(cfg["name"], t["speed"], height, segment))

    tracks: dict[str, list[Point]] = {w.name: [] for w in walkers}
    for _ in range(int(duration / dt) + 1):
        for w in walkers:
            tracks[w.name].append(w.segment.at(w.s))
            w.s += w.speed * dt
        for w in sorted(walkers, key=lambda w: w.segment.length - w.s):
            if w.s < w.segment.length:
                continue
            side = 1 if w.segment.pts[-1][0] > 0 else -1
            overshoot = w.s - w.segment.length
            lane_before = w.segment.lane
            others = [o for o in walkers if o is not w]
            w.segment = choose_lane(lanes, w, others, side)
            w.s = overshoot
            w.changes += w.segment.lane != lane_before
    return tracks, {w.name: w.changes for w in walkers}
