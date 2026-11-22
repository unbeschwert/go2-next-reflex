#!/usr/bin/env python3
"""LiDAR person following for the Go2, switched on by the "follow me" gesture.

The fixed event camera only recognises the gesture. The following itself uses the Go2's
own LiDAR height map (/utlidar/height_map_array, world frame) and pose (/utlidar/robot_pose):

  1. find people: cells standing PERSON_MIN_H..PERSON_MAX_H above the floor, grouped into
     clusters with a person-sized footprint (not the robot itself, not walls)
  2. lock on: at the start, the nearest person within ACQUIRE_RANGE m and ACQUIRE_ANGLE ahead
  3. track: every map, the cluster closest to the last position (within GATE m)
  4. follow: walk toward the person until FOLLOW_DIST m from them, turning to face them;
     stand still while they stand still

Follow mode ends when the person stays still for DONE_S s, is lost for LOST_S s, after
MAX_FOLLOW_S s, or when the caller stops it (another gesture, obstacle, Ctrl-C).
This module has no ROS imports so the logic can be tested offline (see the self-test).

    python3 lidar_follow.py --selftest      # simulated person walking away and stopping
"""
import math
import time

import numpy as np

UNKNOWN_HEIGHT = 200.0     # height map value of unobserved cells
PERSON_MIN_H = 0.5         # m above the floor: legs / torso, not kerbs and clutter
PERSON_MAX_H = 2.2
PERSON_MIN_SIZE = 0.10     # m, footprint of a cluster (bounding box, larger side)
PERSON_MAX_SIZE = 0.90     # m: larger = wall, furniture, a group
SELF_RADIUS = 0.45         # m around the robot centre ignored (its own body)
ACQUIRE_RANGE = 3.0        # m: whom to lock on to at the start
ACQUIRE_ANGLE = math.radians(60)
GATE = 0.6                 # m a person can move between two maps (~10 Hz) and still be the same one

FOLLOW_DIST = 1.0          # m, robot centre to person centre (the nose is ~0.35 m ahead of the centre)
DEAD_BAND = 0.15           # m: no forward motion within FOLLOW_DIST + DEAD_BAND
MAX_VX = 0.5               # m/s (same cap as go2_cmd.py)
MIN_VX = 0.05              # m/s: slower counts as standing (the approach is asymptotic; without this the
                           # robot creeps forever and "the person stopped" never fires)
MAX_YAW = 0.8              # rad/s
K_DIST = 0.8               # m/s per m of extra distance
K_YAW = 1.5                # rad/s per rad of bearing
TURN_ONLY = math.radians(35)   # person further off-axis than this: turn on the spot first

STILL_RADIUS = 0.20        # m: the person counts as standing still while within this circle...
DONE_S = 5.0               # ...for this long while the robot is at its distance -> follow mode ends
START_WAIT_S = 15.0        # the person never walks off after the gesture -> follow mode ends
LOST_S = 1.0               # no matching person for this long -> stop, follow mode ends
MAX_FOLLOW_S = 90.0


def find_people(data, width, height, resolution, origin, robot_xy):
    """Person-sized clusters of tall cells. Returns a list of (x, y) world positions."""
    grid = np.asarray(data, np.float32).reshape(height, width)          # [iy, ix]
    known = grid < UNKNOWN_HEIGHT - 1
    xs = origin[0] + np.arange(width) * resolution
    ys = origin[1] + np.arange(height) * resolution
    dx = xs[None, :] - robot_xy[0]
    dy = ys[:, None] - robot_xy[1]
    r2 = dx * dx + dy * dy
    near = known & (r2 < 2.5 ** 2)
    if not near.any():
        return []
    floor = float(np.percentile(grid[near], 5))                         # lowest ground around the robot
    rel = grid - floor
    tall = known & (rel >= PERSON_MIN_H) & (rel <= PERSON_MAX_H) & (r2 >= SELF_RADIUS ** 2)
    labels, n = _components(tall)
    people = []
    for k in range(1, n + 1):
        iy, ix = np.nonzero(labels == k)
        size = max(ix.max() - ix.min() + 1, iy.max() - iy.min() + 1) * resolution
        if PERSON_MIN_SIZE <= size <= PERSON_MAX_SIZE:
            people.append((float(xs[ix].mean()), float(ys[iy].mean())))
    return people


def _components(mask):
    """8-connected components (cv2 when available, else a small flood fill)."""
    try:
        import cv2
        n, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
        return labels, n - 1
    except ImportError:
        labels = np.zeros(mask.shape, np.int32)
        n = 0
        for y0, x0 in zip(*np.nonzero(mask)):
            if labels[y0, x0]:
                continue
            n += 1
            stack = [(y0, x0)]
            labels[y0, x0] = n
            while stack:
                y, x = stack.pop()
                for yy in (y - 1, y, y + 1):
                    for xx in (x - 1, x, x + 1):
                        if 0 <= yy < mask.shape[0] and 0 <= xx < mask.shape[1] and mask[yy, xx] and not labels[yy, xx]:
                            labels[yy, xx] = n
                            stack.append((yy, xx))
        return labels, n


def to_robot(p, pose):
    """World point -> (distance, bearing) from the robot; bearing + = to the robot's left."""
    x, y, yaw = pose
    dx, dy = p[0] - x, p[1] - y
    fwd = math.cos(yaw) * dx + math.sin(yaw) * dy
    left = -math.sin(yaw) * dx + math.cos(yaw) * dy
    return math.hypot(fwd, left), math.atan2(left, fwd)


class Follower:
    """Lock on to one person and produce walking commands. Call update() for every height map."""

    def __init__(self, now):
        self.target = None
        self.t_start = now
        self.t_seen = now
        self.still_anchor, self.t_still = None, now
        self.moved = False                 # has the person walked off since the lock-on?
        self.state = "acquiring"
        self.reason = ""

    def update(self, people, pose, now):
        """people: list of world (x, y); pose: (x, y, yaw). Returns (vx, yaw_rate, done)."""
        if now - self.t_start > MAX_FOLLOW_S:
            return self._end("time limit")
        if self.target is None:                                  # lock on to the person ahead
            cands = [(d, p) for p in people for d, b in [to_robot(p, pose)]
                     if d <= ACQUIRE_RANGE and abs(b) <= ACQUIRE_ANGLE]
            if not cands:
                if now - self.t_start > 2.0:
                    return self._end("nobody in front of the robot")
                return 0.0, 0.0, False
            self.target = min(cands)[1]
            self.t_seen = now
            self.still_anchor, self.t_still = self.target, now
            self.state = "following"
        else:                                                    # keep track of the same person
            best = min(people, key=lambda p: math.dist(p, self.target), default=None)
            if best is not None and math.dist(best, self.target) <= GATE:
                self.target, self.t_seen = best, now
            elif now - self.t_seen > LOST_S:
                return self._end("lost the person")
            else:                                                # not seen in this map: stand still
                self.state = f"searching ({now - self.t_seen:.1f} s)"
                return 0.0, 0.0, False
        dist, bearing = to_robot(self.target, pose)
        yaw_rate = max(-MAX_YAW, min(MAX_YAW, K_YAW * bearing))
        extra = dist - (FOLLOW_DIST + DEAD_BAND)
        vx = 0.0 if abs(bearing) > TURN_ONLY or extra <= 0 else min(MAX_VX, K_DIST * extra) * math.cos(bearing)
        if vx < MIN_VX:
            vx = 0.0
        # the person standing still (after walking off) while the robot waits at its distance ends
        # the follow mode; a person who never walks off ends it after START_WAIT_S
        if math.dist(self.target, self.still_anchor) > STILL_RADIUS:
            self.still_anchor, self.t_still, self.moved = self.target, now, True
        elif vx == 0.0 and abs(bearing) < TURN_ONLY:
            if self.moved and now - self.t_still > DONE_S:
                return self._end("the person stopped")
            if not self.moved and now - self.t_start > START_WAIT_S:
                return self._end("the person did not walk off")
        self.state = f"following: {dist:.2f} m, {math.degrees(bearing):+.0f} deg"
        return vx, yaw_rate, False

    def _end(self, reason):
        self.state, self.reason = "done", reason
        return 0.0, 0.0, True


# ------------------------------------------------------------------ offline self-test
def _selftest():
    """A simulated person walks away from the robot, turns, stops; a wall and a box stand nearby."""
    res, size = 0.06, 128                                   # Go2-like map: 7.7 m square around the robot
    dt = 0.1
    robot = [0.0, 0.0, 0.0]                                 # x, y, yaw
    person = np.array([1.2, 0.2])
    # person path: stand 2 s, walk +x 2.5 m at 0.8 m/s, turn left and walk 1.5 m, stand still
    path = [(2.0, (0, 0))] + [(2.5 / 0.8, (0.8, 0))] + [(1.5 / 0.6, (0, 0.6))] + [(12.0, (0, 0))]
    f = Follower(0.0)
    t, log, min_gap = 0.0, [], 9.0
    for dur, (vx_p, vy_p) in path:
        for _ in range(int(round(dur / dt))):
            t += dt
            person += np.array([vx_p, vy_p]) * dt
            origin = (robot[0] - size * res / 2, robot[1] - size * res / 2)
            xs = origin[0] + np.arange(size) * res
            ys = origin[1] + np.arange(size) * res
            X, Y = np.meshgrid(xs, ys)                      # [iy, ix]
            grid = np.full((size, size), 0.02, np.float32)  # floor
            grid[(X - person[0]) ** 2 + (Y - person[1]) ** 2 < 0.18 ** 2] = 1.7   # person
            grid[np.abs(Y - 2.6) < 0.06] = 2.0              # long wall (must be ignored)
            grid[(np.abs(X - 1.5) < 0.2) & (np.abs(Y + 1.2) < 0.2)] = 0.45        # low box (ignored)
            grid[(X - robot[0]) ** 2 + (Y - robot[1]) ** 2 < 0.35 ** 2] = 0.6     # own body
            grid[np.hypot(X - robot[0], Y - robot[1]) > 3.5] = UNKNOWN_HEIGHT
            people = find_people(grid.ravel(), size, size, res, origin, robot[:2])
            vx, wz, done = f.update(people, tuple(robot), t)
            robot[2] += wz * dt
            robot[0] += vx * math.cos(robot[2]) * dt
            robot[1] += vx * math.sin(robot[2]) * dt
            gap = math.dist(robot[:2], person)
            min_gap = min(min_gap, gap)
            log.append((t, gap, vx, wz, f.state))
            if done:
                break
        if f.state == "done":
            break
    for row in log[::10]:
        print(f"  t={row[0]:5.1f}s  gap {row[1]:.2f} m  vx {row[2]:.2f}  yaw {row[3]:+.2f}  {row[4]}")
    print(f"ended at t={t:.1f}s: {f.reason!r} | final gap {math.dist(robot[:2], person):.2f} m, "
          f"closest {min_gap:.2f} m | robot at ({robot[0]:.2f}, {robot[1]:.2f}), person at ({person[0]:.2f}, {person[1]:.2f})")


if __name__ == "__main__":
    import sys
    if "--selftest" in sys.argv:
        _selftest()
    else:
        print(__doc__)
