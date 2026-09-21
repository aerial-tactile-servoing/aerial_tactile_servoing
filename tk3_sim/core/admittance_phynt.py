#!/usr/bin/env python3
"""
PHYNT admittance-filter baseline for the local Tilthex simulation.

Pipeline:
    TK3Runtime connect/setup/start/takeoff
        -> configure PHYNT wrench observer + admittance-filter gains
        -> enable PHYNT with wo=1, af=1 internally
        -> publish a live reference to PHYNT
        -> read phynt/desired, sanitize z/roll/pitch, send to uavpos.set_state

This is intended as a comparison baseline for tactile DTS controllers.
It does not use GelSight or DTS. Compliance comes from PHYNT's external-wrench
observer/admittance filter.

The reference publisher keeps altitude at takeoff_height and propagates the
horizontal/yaw reference from the current measured motion. PHYNT is not
connected directly to uavpos in the controller loop: its desired output is used
only as an internal admittance proposal for x, y and yaw.
"""

from __future__ import annotations

import argparse
import code
import json
import pathlib
import readline
import rlcompleter
import time
from typing import Iterable

import numpy as np
from scipy.spatial.transform import Rotation as R

import tk3_sim.core.utils as utils
from tk3_sim.core.log_paths import make_log_paths, print_log_paths
from tk3_sim.core.tk3_runtime import TK3Runtime

DEFAULT_J = [0.011549, 0.0, 0.0, 0.0, 0.011368, 0.0, 0.0, 0.0, 0.019444]
DEFAULT_WO_K = [1.0, 1.0, 5.0, 2.0, 2.0, 1.0]
DEFAULT_WO_THRESH = [0.35, 0.35, 0.35, 0.35, 0.35, 0.35]
DEFAULT_WO_FC = [20.0, 20.0, 20.0, 0.0, 0.0, 0.0]
DEFAULT_AF_K = [0.0, 0.0, 120.0, 200.0, 200.0, 0.0]
DEFAULT_AF_B = [
    6.32455532034,
    6.32455532034,
    37.051315766110115,
    22.0,
    22.0,
    6.32455532034,
]


def parse_float_list(text: str, expected: int, name: str) -> list[float]:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != expected:
        raise argparse.ArgumentTypeError(
            f"{name} must contain {expected} comma-separated values"
        )
    return vals


def compute_default_admittance_gains(*, mass: float, mass_scale: float):
    J_af = list(DEFAULT_J)
    return {
        "m_af": float(mass) * float(mass_scale),
        "J_af": J_af,
        "K_af": list(DEFAULT_AF_K),
        "B_af": list(DEFAULT_AF_B),
    }


def write_gains_file(
    log_path: pathlib.Path, args, *, relative_log_path: str, gains: dict
):
    config = {
        "controller": "admittance_phynt",
        "relative_log_path": relative_log_path,
        "runtime": {
            "takeoff_height": args.takeoff_height,
            "land_height": args.land_height,
            "mass": args.mass,
            "phynt_enabled": True,
        },
        "phynt": {
            "wo": args.wo,
            "af": args.af,
            "wo_K": args.wo_K,
            "wo_thresh": args.wo_thresh,
            "wo_fc": args.wo_fc,
            "m_af": gains["m_af"],
            "J_af": gains["J_af"],
            "K_af": args.af_K,
            "B_af": args.af_B,
        },
        "controller_loop": {
            "duration": args.duration,
            "rate_hz": args.rate_hz,
            "fixed_altitude_ref": args.takeoff_height,
            "reference_source": "custom PHYNT reference publisher + phynt/desired sanitizer relay",
            "sanitized_axes": "z fixed to takeoff_height; roll/pitch fixed to 0; vz/wx/wy/az/awx/awy fixed to 0",
        },
    }
    with open(log_path / "gains.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(config, indent=2))
        f.write("\n")


def configure_phynt_admittance(runtime: TK3Runtime, args) -> dict:
    if not runtime.phynt_enabled or runtime.phynt is None:
        raise RuntimeError("The admittance controller requires PHYNT.")

    gains = compute_default_admittance_gains(
        mass=args.mass, mass_scale=args.af_mass_scale
    )
    gains["K_af"] = list(args.af_K)
    gains["B_af"] = list(args.af_B)

    runtime.phynt.set_mass(args.mass)
    runtime.phynt.set_geom(DEFAULT_J)
    runtime.phynt.set_wo_gains({"K": list(args.wo_K)})
    runtime.phynt.set_wo_thresh({"thresh": list(args.wo_thresh)})
    runtime.phynt.set_wo_fc({"fc": list(args.wo_fc)})
    runtime.phynt.set_af_parameters(
        gains["m_af"],
        gains["B_af"],
        gains["K_af"],
        gains["J_af"],
    )

    print("PHYNT admittance parameters configured:")
    print(f"  m_af={gains['m_af']}")
    print(f"  K_af={gains['K_af']}")
    print(f"  B_af={gains['B_af']}")
    return gains


def timestamp_msg() -> dict:
    t_ns = time.time_ns()
    return {
        "sec": int(t_ns // 1_000_000_000),
        "nsec": int(t_ns % 1_000_000_000),
    }


def make_reference_msg(
    *, x: float, y: float, z: float, quat_wxyz: Iterable[float]
) -> dict:
    qw, qx, qy, qz = [float(v) for v in quat_wxyz]
    return {
        "ts": timestamp_msg(),
        "intrinsic": 0,
        "pos": {"x": float(x), "y": float(y), "z": float(z)},
        "att": {"qw": qw, "qx": qx, "qy": qy, "qz": qz},
        "vel": {"vx": 0.0, "vy": 0.0, "vz": 0.0},
        "avel": {"wx": 0.0, "wy": 0.0, "wz": 0.0},
        "acc": {"ax": 0.0, "ay": 0.0, "az": 0.0},
        "aacc": None,
        "jerk": None,
        "snap": None,
    }


def state_yaw_quat_wxyz(state: dict) -> list[float]:
    att = state["att"]
    roll, pitch, yaw = R.from_quat(
        [
            float(att["qx"]),
            float(att["qy"]),
            float(att["qz"]),
            float(att["qw"]),
        ]
    ).as_euler("xyz", degrees=False)
    qx, qy, qz, qw = R.from_euler("xyz", [0.0, 0.0, yaw]).as_quat()
    return [float(qw), float(qx), float(qy), float(qz)]


def _unwrap_phynt_desired(msg: dict) -> dict:
    """Return the state payload from runtime.phynt.desired().

    Genomix posters usually return a dict with one top-level key, e.g.
    {"desired": {...}}.  This helper keeps the controller robust if the exact
    wrapper name differs slightly between builds.
    """
    if not isinstance(msg, dict):
        raise TypeError(f"phynt.desired() returned {type(msg).__name__}, expected dict")

    for key in ("desired", "state", "frame", "reference"):
        if key in msg and isinstance(msg[key], dict):
            msg = msg[key]
            break

    if "pos" not in msg or "att" not in msg:
        raise KeyError(
            f"Could not find a state payload in phynt.desired(): keys={list(msg.keys())}"
        )
    return msg


def _finite_or_default(value, default: float) -> float:
    try:
        value = float(value)
    except Exception:
        return float(default)
    if not np.isfinite(value):
        return float(default)
    return value


def _yaw_from_att(att: dict | None, *, fallback_yaw: float = 0.0) -> float:
    att = att or {}
    qw = _finite_or_default(att.get("qw"), np.nan)
    qx = _finite_or_default(att.get("qx"), np.nan)
    qy = _finite_or_default(att.get("qy"), np.nan)
    qz = _finite_or_default(att.get("qz"), np.nan)
    if not np.all(np.isfinite([qw, qx, qy, qz])):
        return float(fallback_yaw)
    try:
        return float(R.from_quat([qx, qy, qz, qw]).as_euler("xyz", degrees=False)[2])
    except Exception:
        return float(fallback_yaw)


def _yaw_from_state(state: dict) -> float:
    return _yaw_from_att(state.get("att"), fallback_yaw=0.0)


def make_sanitized_uavpos_state(
    phynt_desired: dict,
    *,
    current_state: dict,
    z_ref: float,
) -> dict:
    """Convert phynt/desired into a uavpos.set_state command.

    PHYNT is allowed to modify only x, y and yaw.  The blocked axes are
    overwritten before the low-level controllers see the command:
        z = z_ref, roll = 0, pitch = 0, vz = 0, wx = 0, wy = 0.
    """
    pos_d = phynt_desired.get("pos", {})
    vel_d = phynt_desired.get("vel", {})
    avel_d = phynt_desired.get("avel", {})
    acc_d = phynt_desired.get("acc", {})

    pos_cur = current_state.get("pos", {})
    x = _finite_or_default(pos_d.get("x"), pos_cur.get("x", 0.0))
    y = _finite_or_default(pos_d.get("y"), pos_cur.get("y", 0.0))

    fallback_yaw = _yaw_from_state(current_state)
    yaw = _yaw_from_att(phynt_desired.get("att"), fallback_yaw=fallback_yaw)
    qx, qy, qz, qw = R.from_euler("xyz", [0.0, 0.0, yaw]).as_quat()

    vx = _finite_or_default(vel_d.get("vx"), 0.0)
    vy = _finite_or_default(vel_d.get("vy"), 0.0)
    wz = _finite_or_default(avel_d.get("wz"), 0.0)

    ax = _finite_or_default(acc_d.get("ax"), 0.0)
    ay = _finite_or_default(acc_d.get("ay"), 0.0)

    return {
        "pos": {"x": float(x), "y": float(y), "z": float(z_ref)},
        "att": {"qw": float(qw), "qx": float(qx), "qy": float(qy), "qz": float(qz)},
        "vel": {"vx": float(vx), "vy": float(vy), "vz": 0.0},
        "avel": {"wx": 0.0, "wy": 0.0, "wz": float(wz)},
        "acc": {"ax": float(ax), "ay": float(ay), "az": 0.0},
        "aacc": {"awx": 0.0, "awy": 0.0, "awz": 0.0},
        "jerk": {"jx": 0.0, "jy": 0.0, "jz": 0.0},
        "snap": {"sx": 0.0, "sy": 0.0, "sz": 0.0},
    }


def enable_admittance_filter_internal_only(
    runtime: TK3Runtime, *, wo: int = 1, af: int = 1
):
    """Enable PHYNT WO/AF without connecting phynt/desired to uavpos.

    This is intentionally different from TK3Runtime.enable_phynt(), which
    connects uavpos/reference to phynt/desired when af=1.  Here PHYNT runs only
    as an internal admittance generator.  The controller loop reads
    runtime.phynt.desired(), sanitizes z/roll/pitch, then sends the result with
    uavpos.set_state().
    """
    if not runtime.phynt_enabled or runtime.phynt is None:
        raise RuntimeError("PHYNT is not available")

    if runtime.phynt_active:
        runtime.disable_phynt()

    runtime.phynt.enable({"enable": {"wo": wo, "af": af}})
    runtime.phynt.servo(ack=True)
    runtime.phynt.log(f"{runtime.relative_log_path}/phynt.log")

    # Keep uavpos away from phynt/desired and phynt/external_wrench.  Commands
    # are sent explicitly through uavpos.set_state() below.
    runtime.uavpos.connect_port({"local": "reference", "remote": "maneuver/desired"})

    runtime.phynt_active = True

    if af != 1:
        raise RuntimeError("This admittance baseline needs af=1")

    state = runtime.get_state()
    pos = state["pos"]
    yaw = _yaw_from_state(state)
    runtime.phynt.set_position(
        float(pos["x"]),
        float(pos["y"]),
        float(pos["z"]),
        float(yaw),
    )


def create_phynt_reference_publisher(
    runtime: TK3Runtime, *, pub_path: str, z_ref: float
):
    if runtime.phynt is None:
        raise RuntimeError("PHYNT is not loaded")

    pub = runtime.phynt.reference(pub_path)

    state = runtime.get_state()
    pos = state["pos"]
    ref_msg = make_reference_msg(
        x=float(pos["x"]),
        y=float(pos["y"]),
        z=float(z_ref),
        quat_wxyz=state_yaw_quat_wxyz(state),
    )
    pub(reference=ref_msg)

    print(f"[admittance] Connecting phynt/reference -> {pub_path}")
    runtime.phynt.connect_port({"local": "reference", "remote": pub_path})

    # Re-servo so PHYNT uses the freshly connected reference port.
    try:
        runtime.phynt.stop()
    except Exception:
        pass
    runtime.phynt.servo(ack=True)
    return pub


def reconnect_phynt_to_maneuver(runtime: TK3Runtime):
    if runtime.phynt is None:
        return
    try:
        runtime.phynt.connect_port({"local": "reference", "remote": "maneuver/desired"})
    except Exception as exc:
        print(f"[WARN] Could not reconnect PHYNT reference to maneuver/desired: {exc}")


def run_admittance_controller(
    runtime: TK3Runtime,
    *,
    duration: float,
    rate_hz: float,
    z_ref: float,
    pub_path: str = "/tmp/phynt_reference_admittance",
    reconnect_reference_on_exit: bool = True,
):
    if not runtime.is_flying:
        print("[WARN] Runtime does not report is_flying=True. Continuing anyway.")

    pub = create_phynt_reference_publisher(runtime, pub_path=pub_path, z_ref=z_ref)
    enable_admittance_filter_internal_only(runtime, wo=1, af=1)

    dt = 1.0 / float(rate_hz)
    t0 = time.time()
    next_tick = t0
    print(
        f"[admittance] Running sanitized PHYNT relay for {duration:.1f}s "
        f"at {rate_hz:.1f} Hz, z_ref={z_ref:.3f} m"
    )
    print("[admittance] PHYNT output axes allowed: x, y, yaw. Blocked: z, roll, pitch.")
    t0 = time.time()
    next_countdown_print = t0
    try:
        print("[admittance] Starting -> Press Ctrl-C to stop early.")
        while True:
            now = time.time()
            elapsed = now - t0
            remaining = max(0.0, duration - elapsed)
            if now >= next_countdown_print:
                print(f"\rRemaining trial time: {remaining:5.1f} s", end="", flush=True)
                next_countdown_print = now + 1.0

            state = runtime.get_state()
            vel = state["vel"]
            avel = state["avel"]
            v_w = np.array(
                [
                    float(vel["vx"]),
                    float(vel["vy"]),
                    float(vel["vz"]),
                    float(avel["wx"]),
                    float(avel["wy"]),
                    float(avel["wz"]),
                ],
                dtype=float,
            )

            pos_new, _ = utils.integrate_world_twist_from_state(state, v_w, dt)

            ref_msg = make_reference_msg(
                x=float(pos_new[0]),
                y=float(pos_new[1]),
                z=float(z_ref),
                quat_wxyz=state_yaw_quat_wxyz(
                    state
                ),  # roll=0, pitch=0, yaw=current yaw
            )
            pub(reference=ref_msg)

            phynt_desired = _unwrap_phynt_desired(runtime.phynt.desired())
            state_cmd = make_sanitized_uavpos_state(
                phynt_desired,
                current_state=state,
                z_ref=z_ref,
            )
            runtime.uavpos.set_state(state_cmd, oneway=True)

            next_tick += dt
            sleep_s = next_tick - time.time()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_tick = time.time()

    except KeyboardInterrupt:
        print("\n[admittance] Interrupted by user.")
    finally:
        if reconnect_reference_on_exit:
            reconnect_phynt_to_maneuver(runtime)
            print("[admittance] Reconnected PHYNT reference to maneuver/desired.")
        runtime.land()
        runtime.stop()


def run_from_args(
    *,
    runtime: TK3Runtime,
    args,
    takeoff: bool = False,
    land: bool | None = None,
):
    if land is None:
        land = not args.no_land

    runtime.start()
    time.sleep(2.0)

    if takeoff:
        runtime.takeoff()

    run_admittance_controller(
        runtime,
        duration=args.duration,
        rate_hz=args.rate_hz,
        z_ref=args.takeoff_height,
        pub_path=args.pub_path,
        reconnect_reference_on_exit=not args.keep_phynt_reference,
    )

    if land:
        runtime.land()
        runtime.stop()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="admittance")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--rate-hz", type=float, default=10.0)

    parser.add_argument("--takeoff-height", type=float, default=1.15)
    parser.add_argument("--land-height", type=float, default=0.25)
    parser.add_argument("--mass", type=float, default=3.1)

    parser.add_argument("--wo", type=int, default=1)
    parser.add_argument(
        "--af",
        type=int,
        default=0,
        help="Runtime takeoff PHYNT AF flag. Keep 0; controller enables AF when running.",
    )
    parser.add_argument("--af-mass-scale", type=float, default=0.7)
    parser.add_argument(
        "--wo-K",
        type=lambda s: parse_float_list(s, 6, "--wo-K"),
        default=list(DEFAULT_WO_K),
    )
    parser.add_argument(
        "--wo-thresh",
        type=lambda s: parse_float_list(s, 6, "--wo-thresh"),
        default=list(DEFAULT_WO_THRESH),
    )
    parser.add_argument(
        "--wo-fc",
        type=lambda s: parse_float_list(s, 6, "--wo-fc"),
        default=list(DEFAULT_WO_FC),
    )
    parser.add_argument(
        "--af-K",
        type=lambda s: parse_float_list(s, 6, "--af-K"),
        default=list(DEFAULT_AF_K),
    )
    parser.add_argument(
        "--af-B",
        type=lambda s: parse_float_list(s, 6, "--af-B"),
        default=list(DEFAULT_AF_B),
    )

    parser.add_argument("--pub-path", default="/tmp/phynt_reference_admittance")
    parser.add_argument(
        "--keep-phynt-reference",
        action="store_true",
        help="Do not reconnect PHYNT reference to maneuver/desired at the end.",
    )
    parser.add_argument("--no-land", action="store_true")

    args = parser.parse_args()

    file_path = pathlib.Path(__file__).resolve()
    repo_root = file_path.parents[2]

    relative_log_path, log_path = make_log_paths(repo_root, args.tag, default_tag="adm")

    print_log_paths(repo_root, relative_log_path)

    runtime = TK3Runtime(
        repo_root=repo_root,
        relative_log_path=relative_log_path,
        phynt_enabled=True,
        wo=args.wo,
        af=args.af,
        takeoff_height=args.takeoff_height,
        land_height=args.land_height,
        mass=args.mass,
    )

    def admittance(
        takeoff: bool = False, land: bool = False, duration: float | None = None
    ):
        old_duration = args.duration
        if duration is not None:
            args.duration = float(duration)
        try:
            return run_from_args(runtime=runtime, args=args, takeoff=takeoff, land=land)
        finally:
            args.duration = old_duration

    try:
        print("\nConnecting/setup TeleKyb...")
        runtime.connect()
        runtime.setup()
        gains = configure_phynt_admittance(runtime, args)
        write_gains_file(
            log_path, args, relative_log_path=relative_log_path, gains=gains
        )
        shell = {
            "runtime": runtime,
            "args": args,
            "repo_root": repo_root,
            "log_path": log_path,
            "relative_log_path": relative_log_path,
            "admittance": admittance,
            "configure_phynt_admittance": configure_phynt_admittance,
            "configure_current_phynt_admittance": lambda: configure_phynt_admittance(
                runtime, args
            ),
            "run_admittance_controller": run_admittance_controller,
            "enable_admittance_filter_internal_only": enable_admittance_filter_internal_only,
        }
        readline.set_completer(rlcompleter.Completer(shell).complete)
        readline.parse_and_bind("tab: complete")

        print(
            "\nPHYNT admittance interactive shell ready.\n"
            "Available objects/functions:\n"
            "  runtime    -> TK3Runtime instance\n"
            "  args       -> parsed arguments\n"
            "  admittance() -> start admittance loop without takeoff/land\n"
            "  admittance(takeoff=True) -> start logs, take off, then run admittance\n"
            "  admittance(takeoff=True, land=True) -> take off, run, then land\n"
            "  admittance(duration=20) -> override duration for one run\n  configure_current_phynt_admittance() -> re-apply current PHYNT admittance gains\n"
            "\nExample checks:\n"
            "  runtime.check_pom()\n"
            "  runtime.check_phynt()\n"
            "\nWhen ready, run:\n"
            "  admittance(takeoff=True)\n"
        )
        print_log_paths(repo_root, relative_log_path)

        code.interact(
            banner="PHYNT admittance shell. Exit with Ctrl-D or exit().",
            local=shell,
        )

    finally:
        runtime.stop()
        print_log_paths(repo_root, relative_log_path)


if __name__ == "__main__":
    main()
