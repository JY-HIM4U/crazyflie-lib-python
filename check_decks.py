#!/usr/bin/env python3
"""Connect to the Crazyflie and report what positioning hardware is active.

No flight. Tells you which deck is bound and whether you have horizontal
position feedback at all. Run this BEFORE trying to fly — if no flow/
lighthouse/loco deck is reported, send_hover_setpoint(0,0,0,z) and the
LQR's xy state are both garbage, and the drone WILL drift.
"""
import time
import logging

import cflib.crtp
from cflib.crazyflie import Crazyflie
from cflib.crazyflie.syncCrazyflie import SyncCrazyflie
from cflib.crazyflie.log import LogConfig

URI = 'radio://0/80/2M/E7E7E7E710'
logging.basicConfig(level=logging.ERROR)


def main():
    cflib.crtp.init_drivers()
    scf = SyncCrazyflie(URI, cf=Crazyflie(rw_cache='./cache'))
    scf.open_link()
    time.sleep(0.5)
    cf = scf.cf

    print("=" * 64)
    print("DECK / POSITIONING HARDWARE PRESENT")
    print("=" * 64)
    deck_params = [
        ("deck.bcFlow",         "Flow Deck v1     (optical flow + ToF)"),
        ("deck.bcFlow2",        "Flow Deck v2     (optical flow + ToF, current)"),
        ("deck.bcLighthouse4",  "Lighthouse deck  (SteamVR 2.0 base stations)"),
        ("deck.bcLoco",         "Loco deck        (UWB anchors)"),
        ("deck.bcZRanger",      "Z-Ranger deck    (vertical only — does NOT help xy)"),
        ("deck.bcZRanger2",     "Z-Ranger v2      (vertical only)"),
        ("deck.bcMultiranger",  "Multiranger deck (5 ToF sensors)"),
        ("deck.bcUSD",          "uSD card deck    (logging only)"),
    ]
    have_xy = False
    for p, desc in deck_params:
        try:
            v = int(cf.param.get_value(p))
            mark = "✓" if v else " "
            print(f"  [{mark}] {p:<22}  {desc}")
            if v and any(s in p for s in ("Flow", "Lighthouse", "Loco")):
                have_xy = True
        except Exception:
            print(f"  [?] {p:<22}  (param missing on this firmware)")

    print()
    print("=" * 64)
    print("ESTIMATOR & POSITION SOURCE")
    print("=" * 64)
    est = int(cf.param.get_value("stabilizer.estimator"))
    est_name = {0: "Auto", 1: "Complementary", 2: "Kalman"}.get(est, f"Unknown({est})")
    print(f"  stabilizer.estimator = {est}  ({est_name})")

    # Read a few state-estimate values to see if they're moving / sensible.
    seen = {}

    def cb(ts, data, lc):
        seen.update(data)

    lc1 = LogConfig(name='Probe', period_in_ms=100)
    for v in ('stateEstimateZ.x', 'stateEstimateZ.y', 'stateEstimateZ.z',
              'stateEstimateZ.vx', 'stateEstimateZ.vy', 'stateEstimateZ.vz',
              'stateEstimate.x', 'stateEstimate.y', 'stateEstimate.z',
              'stateEstimate.vx', 'stateEstimate.vy', 'stateEstimate.vz'):
        try:
            lc1.add_variable(v, 'float')
        except KeyError:
            pass
    lc1.data_received_cb.add_callback(cb)
    cf.log.add_config(lc1)
    lc1.start()
    time.sleep(2.0)
    lc1.stop()

    print()
    print("Stationary readings (drone on the floor, NOT flying):")
    for k in sorted(seen.keys()):
        v = seen[k]
        print(f"  {k:<28} = {v:+8.3f}")

    print()
    print("=" * 64)
    print("VERDICT")
    print("=" * 64)
    if have_xy:
        print("  ✓ You have a horizontal-position-aware deck.")
        print("    → The Kalman estimator can produce real (x,y) feedback.")
        print("    → If the drone still drifts, check FLOOR (low light, mirror,")
        print("      featureless surface, glass) — flow deck needs ~50 lux and")
        print("      a textured surface to work.")
    else:
        print("  ✗ NO horizontal-position deck detected.")
        print("    → stateEstimateZ.x/y/vx/vy are pure dead-reckoning")
        print("      from the IMU — they will drift unboundedly.")
        print("    → send_hover_setpoint(0,0,…) tracks PHANTOM velocity →")
        print("      real drift in some random direction.")
        print("    → You CANNOT fly the LQR controller without positioning.")
        print("    → Options:")
        print("        1. Install a Flow Deck v2 (optical flow + downward ToF)")
        print("        2. Use a Lighthouse positioning system")
        print("        3. Use external motion capture (Vicon/OptiTrack) and")
        print("           feed pose via cf.extpos.send_extpos / send_extpose")

    scf.close_link()


if __name__ == '__main__':
    main()
