#!/usr/bin/env python3
"""Standalone axis-debug script for the Crazyflie.

Sends a fixed sequence of attitude setpoints — pure +pitch, pure -pitch, pure
+roll, pure -roll — using the same `send_setpoint` API as fly_LQR_new_jax.py.
Logs commanded vs actual angles AND the resulting position drift, then
prints a diagnosis table.

What you learn:
  - whether commanded pitch tilts the drone the expected way (forward/back)
  - whether commanded roll tilts the expected way (left/right)
  - whether the axes are swapped (cmd_pitch shows up in act_roll, etc.)
  - whether the on-board PID actually tracks the commanded angle

Safety: tilt is held at 5° for 1.0 s per axis with 1.5 s zero-attitude recovery
between phases. Run over a clear 2×2 m floor area. Trigger Ctrl-C at any
time — the script catches it and lands.

Edit URI / CONTROLLER_ID at the top to match your setup. Run twice — once with
CONTROLLER_ID = 1 (Mellinger), once with 2 (legacy PID) — and compare the
diagnosis tables.
"""
import os
import sys
import time
import csv
import math
import logging
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

# ─── Config ───────────────────────────────────────────────────────────────
URI               = 'radio://0/80/2M/E7E7E7E710'
# Match fly_LQR_new_jax.py — Mellinger has the position/velocity controller
# that makes send_hover_setpoint(0,0,0,z) actually stationary. Legacy PID
# (controller=2) has no position controller → drifts uncontrolled.
CONTROLLER_ID     = 1          # 1 = Mellinger (working), 2 = PID (no pos ctrl)
DEFAULT_HEIGHT    = 0.35       # m — hover altitude
TILT_DEG          = 5.0        # |angle| held during each axis phase
PHASE_DURATION    = 1.0        # s — duration of each tilt-phase
RECOVERY_DURATION = 1.5        # s — zero-attitude recovery between phases
HOVER_PWM         = 38000      # baseline thrust during attitude phases
LOOP_DT           = 0.025      # 25 ms = 40 Hz, matches deploy

# Safety gates
Z_MIN = 0.18                   # abort if z drops below this
Z_MAX = 0.60                   # abort if z rises above this
XY_LIMIT = 1.2                 # abort if displacement > this from takeoff origin

LOG_DIR = os.path.join(os.getcwd(), "cf_logs",
                        "axis_debug_" + time.strftime("%Y%m%d_%H%M%S"))
logging.basicConfig(level=logging.ERROR)

fullstate = {}


def _cb(ts, data, logconf):
    fullstate.update(data)


def make_logconfs():
    lc1 = LogConfig(name='Pos', period_in_ms=25)
    for v in ('stateEstimateZ.x', 'stateEstimateZ.y', 'stateEstimateZ.z',
              'stateEstimateZ.vx', 'stateEstimateZ.vy', 'stateEstimateZ.vz'):
        lc1.add_variable(v, 'float')
    lc1.data_received_cb.add_callback(_cb)

    lc2 = LogConfig(name='Att', period_in_ms=25)
    for v in ('stabilizer.roll', 'stabilizer.pitch', 'stabilizer.yaw'):
        lc2.add_variable(v, 'float')
    lc2.data_received_cb.add_callback(_cb)

    return [lc1, lc2]


def get_state():
    s = fullstate
    return {
        'x':     float(s.get('stateEstimateZ.x', 0))  * 1e-3,
        'y':     float(s.get('stateEstimateZ.y', 0))  * 1e-3,
        'z':     float(s.get('stateEstimateZ.z', 0))  * 1e-3,
        'vx':    float(s.get('stateEstimateZ.vx', 0)) * 1e-3,
        'vy':    float(s.get('stateEstimateZ.vy', 0)) * 1e-3,
        'vz':    float(s.get('stateEstimateZ.vz', 0)) * 1e-3,
        'roll':  float(s.get('stabilizer.roll', 0)),
        'pitch': float(s.get('stabilizer.pitch', 0)),
        'yaw':   float(s.get('stabilizer.yaw', 0)),
    }


def smooth_send_hover(cf, start, target, duration):
    n = int(duration / LOOP_DT)
    for i in range(n):
        f = (i + 1) / n
        v = [start[k] + (target[k] - start[k]) * f for k in range(4)]
        cf.commander.send_hover_setpoint(v[0], v[1], v[2], v[3])
        time.sleep(LOOP_DT)


def smooth_land(cf, height, duration=2.5):
    n = int(duration / LOOP_DT)
    for i in range(n):
        z = height * (1.0 - (i + 1) / n)
        cf.commander.send_hover_setpoint(0, 0, 0, max(z, 0.02))
        time.sleep(LOOP_DT)
    cf.commander.send_stop_setpoint()
    time.sleep(0.1)


def run_phase(cf, label, target_roll, target_pitch, duration, log,
              t_start, origin):
    n = int(duration / LOOP_DT)
    print(f"  [{label:<14}] roll={target_roll:+.0f}°  pitch={target_pitch:+.0f}°  "
          f"({duration:.1f}s)")
    is_hover = (target_roll == 0 and target_pitch == 0)
    for i in range(n):
        if is_hover:
            # Active position-hold: Mellinger tracks vx=vy=0 using flow-deck
            # feedback. Same call fly_LQR_new_jax.py uses for takeoff hover.
            cf.commander.send_hover_setpoint(0, 0, 0, DEFAULT_HEIGHT)
        else:
            # Tilt test: command roll/pitch directly, altitude held by autopilot.
            # Position-hold is OFF during this phase — drone WILL drift in xy
            # in the direction the commanded tilt actually produces.
            cf.commander.send_zdistance_setpoint(
                target_roll, target_pitch, 0, DEFAULT_HEIGHT
            )
        time.sleep(LOOP_DT)
        s = get_state()
        log.append({
            't':         time.time() - t_start,
            'phase':     label,
            'cmd_roll':  target_roll,
            'cmd_pitch': target_pitch,
            **s,
        })
        # Safety abort
        if (s['z'] < Z_MIN or s['z'] > Z_MAX or
                abs(s['x'] - origin['x']) > XY_LIMIT or
                abs(s['y'] - origin['y']) > XY_LIMIT):
            print(f"    ⚠ SAFETY ABORT (z={s['z']:.2f}, "
                  f"dxy=({s['x']-origin['x']:+.2f},{s['y']-origin['y']:+.2f}))")
            return False
    return True


PHASES = [
    ("hover_baseline", 0.0,        0.0,         RECOVERY_DURATION),
    ("pitch_pos",      0.0,       +TILT_DEG,    PHASE_DURATION),
    ("recover_1",      0.0,        0.0,         RECOVERY_DURATION),
    ("pitch_neg",      0.0,       -TILT_DEG,    PHASE_DURATION),
    ("recover_2",      0.0,        0.0,         RECOVERY_DURATION),
    ("roll_pos",      +TILT_DEG,   0.0,         PHASE_DURATION),
    ("recover_3",      0.0,        0.0,         RECOVERY_DURATION),
    ("roll_neg",      -TILT_DEG,   0.0,         PHASE_DURATION),
    ("recover_4",      0.0,        0.0,         RECOVERY_DURATION),
]


def fly():
    os.makedirs(LOG_DIR, exist_ok=True)
    cflib.crtp.init_drivers()

    scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache'))
    scf.open_link()
    time.sleep(0.2)
    cf = scf.cf

    cf.param.set_value("stabilizer.estimator", 2)            # Kalman
    cf.param.set_value("commander.enHighLevel", 1)
    cf.param.set_value("flightmode.stabModeRoll", 1)         # angle mode
    cf.param.set_value("flightmode.stabModePitch", 1)        # angle mode
    cf.param.set_value("flightmode.stabModeYaw", 0)          # rate mode
    cf.param.set_value("stabilizer.controller", CONTROLLER_ID)
    print(f"[setup] stabilizer.controller={CONTROLLER_ID}  "
          f"(1=Mellinger, 2=PID)")

    for lc in make_logconfs():
        cf.log.add_config(lc)
        lc.start()
    time.sleep(0.5)  # let log streams populate

    cf.commander.send_setpoint(0, 0, 0, 0)  # arm
    time.sleep(0.1)
    print(f"[takeoff] climbing to {DEFAULT_HEIGHT:.2f} m")
    smooth_send_hover(cf, (0, 0, 0, 0),
                      (0, 0, 0, DEFAULT_HEIGHT), 3.0)

    cf.param.set_value("commander.enHighLevel", 0)
    time.sleep(0.05)

    # Drive yaw to ~0 with a short closed-loop yaw-rate command. Without this,
    # the body and world frames are misaligned and Δx/Δy bake in a yaw rotation.
    print("[yaw-zero] driving yaw → 0°")
    for _ in range(int(2.5 / LOOP_DT)):
        s = get_state()
        yaw_err = s['yaw']                          # deg, want 0
        yaw_rate_cmd = float(np.clip(-2.0 * yaw_err, -60.0, 60.0))
        # send_hover_setpoint keeps position-hold ON while we yaw.
        cf.commander.send_hover_setpoint(0, 0, yaw_rate_cmd, DEFAULT_HEIGHT)
        time.sleep(LOOP_DT)
        if abs(yaw_err) < 3.0:
            break

    origin = get_state()
    print(f"[ready] origin: x={origin['x']:.3f} y={origin['y']:.3f} "
          f"z={origin['z']:.3f}  yaw={origin['yaw']:+.1f}°")
    if abs(origin['yaw']) > 10:
        print(f"  ⚠ yaw is {origin['yaw']:+.1f}° — still large; diagnosis "
              f"will rotate Δxy into body frame to compensate.")

    log = []
    t0 = time.time()
    aborted = False
    try:
        for label, cmd_r, cmd_p, dur in PHASES:
            ok = run_phase(cf, label, cmd_r, cmd_p, dur, log, t0, origin)
            if not ok:
                aborted = True
                break
    except KeyboardInterrupt:
        print("\n[interrupt] user requested stop")
        aborted = True

    # Land
    print("[land]")
    cf.param.set_value("commander.enHighLevel", 1)
    time.sleep(0.05)
    s_now = get_state()
    smooth_send_hover(cf, (0, 0, 0, max(s_now['z'], 0.10)),
                      (0, 0, 0, DEFAULT_HEIGHT), 1.0)
    smooth_land(cf, DEFAULT_HEIGHT, duration=2.5)
    scf.close_link()

    # Save CSV
    if log:
        path = os.path.join(LOG_DIR, "axis_debug.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log[0].keys()))
            w.writeheader()
            w.writerows(log)
        print(f"[saved] {len(log)} samples → {path}")

    return log, origin, aborted


def diagnose(log, origin):
    if not log:
        return
    by_phase = defaultdict(list)
    for r in log:
        by_phase[r['phase']].append(r)

    # Rotate world-frame Δxy into BODY frame using the initial yaw, so the
    # diagnosis is independent of how the drone was oriented at takeoff.
    yaw0_rad = math.radians(origin['yaw'])
    c, s = math.cos(yaw0_rad), math.sin(yaw0_rad)
    def to_body(dx, dy):
        # World→body: rotate by -yaw0
        return (c * dx + s * dy, -s * dx + c * dy)

    print("\n" + "=" * 86)
    print(f"PER-PHASE DRIFT TABLE  (Δ rotated to body frame; takeoff yaw={origin['yaw']:+.1f}°)")
    print("=" * 86)
    print(f"{'phase':<16}{'cmd':>14}{'Δx_b (m)':>10}{'Δy_b (m)':>10}"
          f"{'mean act_r':>12}{'mean act_p':>12}{'mean yaw':>10}")
    print("-" * 86)
    for label, *_ in PHASES:
        rows = by_phase.get(label, [])
        if not rows:
            continue
        s_start = rows[0]
        s_end = rows[-1]
        dx_w = s_end['x'] - s_start['x']
        dy_w = s_end['y'] - s_start['y']
        dx_b, dy_b = to_body(dx_w, dy_w)
        cmd = f"r{s_start['cmd_roll']:+.0f}/p{s_start['cmd_pitch']:+.0f}"
        half = rows[len(rows)//2:]
        ar = np.mean([r['roll']  for r in half])
        ap = np.mean([r['pitch'] for r in half])
        ay = np.mean([r['yaw']   for r in half])
        print(f"{label:<16}{cmd:>14}{dx_b:+10.3f}{dy_b:+10.3f}"
              f"{ar:+12.1f}{ap:+12.1f}{ay:+10.1f}")

    # Verdict block — pure axis tests only
    print("\n" + "=" * 78)
    print("DIAGNOSIS — what each axis test should produce vs what it did")
    print("=" * 78)

    def phase_drift(name):
        rows = by_phase.get(name, [])
        if not rows:
            return None
        dx_w = rows[-1]['x'] - rows[0]['x']
        dy_w = rows[-1]['y'] - rows[0]['y']
        return to_body(dx_w, dy_w)   # body-frame Δ

    expected = {
        # CF firmware convention (the one we're testing for):
        # +pitch  → drone tilts forward (nose down) → +x_body
        # -pitch  → -x_body
        # +roll   → drone tilts right → +y_body
        # -roll   → -y_body
        'pitch_pos': "+x", 'pitch_neg': "-x",
        'roll_pos':  "+y", 'roll_neg':  "-y",
    }
    for name, exp in expected.items():
        d = phase_drift(name)
        if d is None:
            continue
        dx, dy = d
        mag = math.hypot(dx, dy)
        if mag < 0.02:
            verdict = "no motion (PID didn't track or thrust off)"
        else:
            ang = math.degrees(math.atan2(dy, dx))
            primary = ('+x' if abs(dx) > abs(dy) and dx > 0 else
                       '-x' if abs(dx) > abs(dy) and dx < 0 else
                       '+y' if dy > 0 else '-y')
            ok = "✓ matches" if primary == exp else f"✗ MISMATCH (expected {exp})"
            verdict = (f"primary={primary}  Δ=({dx:+.3f},{dy:+.3f}) "
                       f"|d|={mag:.3f}  ang={ang:+.0f}°  {ok}")
        print(f"  {name:<10} cmd→{exp:<3}  →  {verdict}")

    # Cross-axis correlation (full flight)
    cmd_r = np.array([r['cmd_roll']  for r in log])
    cmd_p = np.array([r['cmd_pitch'] for r in log])
    act_r = np.array([r['roll']      for r in log])
    act_p = np.array([r['pitch']     for r in log])

    def corr(a, b):
        return float(np.corrcoef(a, b)[0, 1]) if a.std() > 1e-3 and b.std() > 1e-3 else float('nan')

    print("\n" + "=" * 78)
    print("CMD ↔ ACTUAL CROSS-AXIS CORRELATION (lag=0)")
    print("=" * 78)
    print(f"  cmd_roll  ↔ act_roll  :  {corr(cmd_r, act_r):+.3f}   (want close to +1)")
    print(f"  cmd_roll  ↔ act_pitch :  {corr(cmd_r, act_p):+.3f}   (want close to  0)")
    print(f"  cmd_pitch ↔ act_roll  :  {corr(cmd_p, act_r):+.3f}   (want close to  0)")
    print(f"  cmd_pitch ↔ act_pitch :  {corr(cmd_p, act_p):+.3f}   (want close to +1)")

    print("\nINTERPRETATION:")
    print("  - both diag ≈ +1 and both off-diag ≈ 0  →  axes correctly mapped")
    print("  - off-diag > diag                       →  roll/pitch SWAPPED")
    print("  - diag near 0 with no swap              →  PID isn't tracking at all")
    print("  - diag negative                          →  axis is INVERTED-sign")


def plot(log):
    if not log:
        return
    t = np.array([r['t']         for r in log])
    cmd_r = np.array([r['cmd_roll']  for r in log])
    cmd_p = np.array([r['cmd_pitch'] for r in log])
    act_r = np.array([r['roll']      for r in log])
    act_p = np.array([r['pitch']     for r in log])
    act_y = np.array([r['yaw']       for r in log])
    x = np.array([r['x'] for r in log])
    y = np.array([r['y'] for r in log])
    z = np.array([r['z'] for r in log])
    phases = [r['phase'] for r in log]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(t, cmd_r, 'r--', linewidth=1.0, alpha=0.7, label='cmd roll')
    ax.plot(t, act_r, 'r-',  linewidth=1.4, alpha=0.95, label='actual roll')
    ax.plot(t, cmd_p, 'b--', linewidth=1.0, alpha=0.7, label='cmd pitch')
    ax.plot(t, act_p, 'b-',  linewidth=1.4, alpha=0.95, label='actual pitch')
    ax.set_xlabel('t (s)'); ax.set_ylabel('deg')
    ax.set_title('Roll/Pitch: cmd vs actual')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _shade_phases(ax, t, phases)

    ax = axes[0, 1]
    ax.plot(t, x - x[0], label='Δx')
    ax.plot(t, y - y[0], label='Δy')
    ax.set_xlabel('t (s)'); ax.set_ylabel('m')
    ax.set_title('Position drift (relative to takeoff origin)')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    _shade_phases(ax, t, phases)

    ax = axes[1, 0]
    ax.plot(x, y, 'b-', linewidth=1.5, alpha=0.85)
    ax.scatter(x[0], y[0], c='g', s=100, label='Start', zorder=5,
               edgecolors='k', linewidths=0.6)
    ax.scatter(x[-1], y[-1], c='r', s=100, label='End', zorder=5,
               edgecolors='k', linewidths=0.6)
    last_phase = None
    for i, p in enumerate(phases):
        if p != last_phase and not p.startswith('recover') and p != 'hover_baseline':
            ax.scatter(x[i], y[i], marker='o', c='orange', s=40, zorder=4,
                       edgecolors='k', linewidths=0.4)
            ax.annotate(p, (x[i], y[i]), fontsize=7, alpha=0.85,
                        xytext=(4, 4), textcoords='offset points')
            last_phase = p
    ax.set_xlabel('x (m)'); ax.set_ylabel('y (m)')
    ax.set_title('XY trajectory (orange dots = phase starts)')
    ax.set_aspect('equal'); ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.plot(t, z, 'g-', label='z')
    ax2 = ax.twinx()
    ax2.plot(t, act_y, 'm-', linewidth=0.9, alpha=0.7, label='yaw')
    ax2.set_ylabel('yaw (deg)', color='m')
    ax2.tick_params(axis='y', labelcolor='m')
    ax.axhline(Z_MIN, color='gray', linestyle=':', linewidth=0.5)
    ax.axhline(Z_MAX, color='gray', linestyle=':', linewidth=0.5)
    ax.set_xlabel('t (s)'); ax.set_ylabel('z (m)', color='g')
    ax.tick_params(axis='y', labelcolor='g')
    ax.set_title('Altitude & yaw')
    ax.grid(True, alpha=0.3)
    _shade_phases(ax, t, phases)

    plt.suptitle(f"Axis debug — controller={CONTROLLER_ID}, "
                 f"tilt=±{TILT_DEG:.0f}°, phase={PHASE_DURATION:.1f}s")
    plt.tight_layout()
    out = os.path.join(LOG_DIR, "axis_debug.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[saved] plot → {out}")


def _shade_phases(ax, t, phases):
    colors = {
        'pitch_pos': '#ffd0d0', 'pitch_neg': '#ffe0b0',
        'roll_pos':  '#d0d0ff', 'roll_neg':  '#b0e0ff',
    }
    last_p = phases[0]
    last_t = t[0]
    for i in range(1, len(phases)):
        if phases[i] != last_p:
            c = colors.get(last_p)
            if c:
                ax.axvspan(last_t, t[i], alpha=0.25, color=c)
            last_p, last_t = phases[i], t[i]
    c = colors.get(last_p)
    if c:
        ax.axvspan(last_t, t[-1], alpha=0.25, color=c)


if __name__ == '__main__':
    log, origin, aborted = fly()
    if log:
        diagnose(log, origin)
        plot(log)
    if aborted:
        sys.exit(2)
