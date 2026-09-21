#!/usr/bin/env python3
"""TeleKyb component lifecycle and motion commands for Gazebo simulation."""

import os
import pathlib
import time

import genomix
import numpy as np
from scipy.spatial.transform import Rotation as R

from tk3_sim.core.log_paths import ensure_log_dir


class TK3Runtime:
    """Connect and operate the TeleKyb components of the local Tilthex simulation."""

    def __init__(
        self,
        *,
        repo_root: pathlib.Path,
        relative_log_path: str,
        phynt_enabled: bool = True,
        wo: int = 1,
        af: int = 0,
        takeoff_height: float = 1.55,
        land_height: float = 0.25,
        mass: float = 2.9,
    ):
        self.repo_root = pathlib.Path(repo_root)
        self.relative_log_path = relative_log_path
        self.phynt_enabled = phynt_enabled
        self.phynt_active = False
        self.wo = wo
        self.af = af
        self.is_flying = False
        self.takeoff_height = takeoff_height
        self.land_height = land_height
        self.mass = mass

        print(
            "Using the local Tilthex simulation:",
            f"takeoff_height={takeoff_height}, land_height={land_height},",
            f"mass={mass}, phynt_enabled={phynt_enabled}, wo={wo}, af={af}",
            flush=True,
        )

        self.g = None
        self.mocap = None
        self.rotorcraft = None
        self.pom = None
        self.uavatt = None
        self.uavpos = None
        self.maneuver = None
        self.phynt = None

    def connect(self):
        self.g = genomix.connect("localhost")
        plugin_path = os.environ.get("TK3_GENOM_PLUGIN_PATH")
        if plugin_path:
            self.g.rpath(plugin_path)
        self.mocap = self.g.load("optitrack")
        self.rotorcraft = self.g.load("rotorcraft")
        self.pom = self.g.load("pom")
        self.uavatt = self.g.load("uavatt")
        self.uavpos = self.g.load("uavpos")
        self.maneuver = self.g.load("maneuver")

        if self.phynt_enabled:
            self.phynt = self.g.load("phynt")

    def setup(self):
        J = [0.011549, 0, 0, 0, 0.011368, 0, 0, 0, 0.019444]

        if self.phynt_enabled:
            self.phynt.connect_port({"local": "state", "remote": "pom/frame/robot"})
            self.phynt.connect_port(
                {"local": "reference", "remote": "maneuver/desired"}
            )
            self.phynt.connect_port(
                {"local": "wrench_measure", "remote": "uavatt/wrench_measure"}
            )
            self.phynt.set_mass(self.mass)
            self.phynt.set_geom(J)
            self.phynt.set_wo_gains({"K": [1.0, 1.0, 5.0, 2.0, 2.0, 1.0]})
            self.phynt.set_wo_thresh({"thresh": [0, 0, 0, 0, 0, 0]})
            self.phynt.set_wo_fc({"fc": [20, 20, 20, 0, 0, 0]})

        self.mocap.connect(
            {"host": "localhost", "host_port": "1509", "mcast": "", "mcast_port": "0"}
        )
        self.rotorcraft.connect({"serial": "/tmp/pty-hr6", "baud": 0})
        self.rotorcraft.set_sensor_rate(
            {"rate": {"imu": 1000, "mag": 0, "motor": 20, "battery": 1}}
        )
        self.rotorcraft.set_imu_filter(
            {"gfc": [20, 20, 20], "afc": [5, 5, 5], "mfc": [20, 20, 20]}
        )
        self.rotorcraft.connect_port(
            {"local": "rotor_input", "remote": "uavatt/rotor_input"}
        )

        self.pom.set_prediction_model("::pom::constant_acceleration")
        self.pom.set_process_noise({"max_jerk": 100, "max_dw": 50})
        self.pom.set_history_length({"history_length": 0.25})
        self.pom.connect_port({"local": "measure/imu", "remote": "rotorcraft/imu"})
        self.pom.add_measurement("imu", x=0, y=0, z=0, roll=0, pitch=0, yaw=0)
        self._wait_for_mocap()
        # These must finish successfully before setup can continue.
        self.pom.connect_port(
            {"local": "measure/mocap", "remote": "optitrack/bodies/HR_6"}
        )
        self.pom.add_measurement("mocap", x=0, y=0, z=0, roll=0, pitch=0, yaw=0)

        self.uavpos.set_saturation({"sat": {"x": 1, "v": 1, "ix": 0}})
        self.uavpos.set_servo_gain(
            {"gain": {"Kpxy": 1, "Kpz": 10, "Kvxy": 8, "Kvz": 10, "Kixy": 0, "Kiz": 0}}
        )
        self.uavpos.set_mass({"mass": self.mass})
        self.uavpos.set_xyradius({"rxy": 2})
        self.uavpos.connect_port({"local": "state", "remote": "pom/frame/robot"})
        self.uavpos.connect_port({"local": "reference", "remote": "maneuver/desired"})

        self.uavatt.set_gtmrp_geom(
            {
                "rotors": 6,
                "cx": 0,
                "cy": 0,
                "cz": 0,
                "armlen": 0.39,
                "mass": self.mass,
                "rx": -20,
                "ry": -20,
                "rz": -1,
                "cf": 12.5e-4,
                "ct": 2.4e-5,
            }
        )
        self.uavatt.set_wlimit({"wmin": 16, "wmax": 100})
        self.uavatt.set_servo_gain(
            {"gain": {"Kqxy": 15, "Kqz": 15, "Kwxy": 1.5, "Kwz": 1}}
        )
        self.uavatt.set_emerg({"emerg": {"dq": 5, "dw": 20}})
        self.uavatt.connect_port({"local": "uav_input", "remote": "uavpos/uav_input"})
        self.uavatt.connect_port(
            {"local": "rotor_measure", "remote": "rotorcraft/rotor_measure"}
        )
        self.uavatt.connect_port({"local": "state", "remote": "pom/frame/robot"})

        pi = 3.14
        self.maneuver.set_bounds(
            {
                "xmin": -100,
                "xmax": 100,
                "ymin": -100,
                "ymax": 100,
                "zmin": -10,
                "zmax": 20,
                "yawmin": -2 * pi,
                "yawmax": 2 * pi,
            }
        )
        self.maneuver.connect_port({"local": "state", "remote": "pom/frame/robot"})

    def _wait_for_mocap(self, timeout: float = 10.0):
        """Wait for body discovery and two valid, advancing pose samples."""
        print("Waiting for live OptiTrack data on optitrack/bodies/HR_6...")
        deadline = time.monotonic() + timeout
        previous_stamp = None
        while time.monotonic() < deadline:
            if "HR_6" in self.mocap.body_list()["body_list"]:
                state = self.mocap.bodies("HR_6")["bodies"]
                if state and state["pos"] is not None and state["att"] is not None:
                    pose = [state["pos"][key] for key in ("x", "y", "z")]
                    quat = [state["att"][key] for key in ("qw", "qx", "qy", "qz")]
                    stamp = (state["ts"]["sec"], state["ts"]["nsec"])
                    if np.isfinite(pose + quat).all() and np.linalg.norm(quat) > 0:
                        if previous_stamp is not None and stamp > previous_stamp:
                            return
                        previous_stamp = stamp
            time.sleep(0.05)
        raise TimeoutError(
            f"No live OptiTrack pose for HR_6 after {timeout:g} s. "
            "Check that Gazebo is running and publishing on port 1509, "
            "and that no second native or Docker simulator is running. "
            "POM mocap setup was not completed."
        )

    def logs(self):
        ensure_log_dir(self.repo_root, self.relative_log_path)
        self.mocap.set_logfile(f"{self.relative_log_path}/mocap.log")
        self.rotorcraft.log(f"{self.relative_log_path}/rotorcraft.log")
        self.pom.log_state(f"{self.relative_log_path}/pom.log")
        self.pom.log_measurements(f"{self.relative_log_path}/pom-measurements.log")
        self.uavpos.log(f"{self.relative_log_path}/uavpos.log")
        self.uavatt.log(f"{self.relative_log_path}/uavatt.log")
        self.maneuver.log(f"{self.relative_log_path}/maneuver.log")

    def start(self):
        self.logs()

        self.rotorcraft.start()
        self.rotorcraft.servo(ack=True)

    def enable_phynt(self, wo, af):
        if not self.phynt_enabled:
            return

        self.phynt.enable({"enable": {"wo": wo, "af": af}})
        self.phynt.servo(ack=True)
        self.phynt.log(f"{self.relative_log_path}/phynt.log")

        if af == 1:
            self.uavpos.connect_port(
                {"local": "wrench_measure", "remote": "phynt/external_wrench"}
            )
            self.uavpos.connect_port({"local": "reference", "remote": "phynt/desired"})
        else:
            self.uavpos.connect_port(
                {"local": "reference", "remote": "maneuver/desired"}
            )

        self.phynt_active = True

    def disable_phynt(self):
        if self.phynt_enabled and self.phynt_active:
            self.phynt.enable({"enable": {"wo": 0, "af": 0}})
            self.phynt.stop()
            self.uavpos.connect_port(
                {"local": "reference", "remote": "maneuver/desired"}
            )
        self.phynt_active = False

    def takeoff(self):
        print(f"Taking off to {self.takeoff_height} m...")

        self.maneuver.set_current_state()
        self.maneuver.take_off(self.takeoff_height, 0, ack=True)
        self.uavatt.servo(ack=True)
        self.uavpos.servo(ack=True)
        self.maneuver.wait()

        state = self.get_state()
        x = state["pos"]["x"]
        y = state["pos"]["y"]
        z = state["pos"]["z"]
        att = state["att"]
        rpy = R.from_quat([att["qx"], att["qy"], att["qz"], att["qw"]]).as_euler(
            "xyz", degrees=False
        )

        print(
            f"Current position: x={x:.2f}, y={y:.2f}, z={z:.2f}, yaw={rpy[2]:.2f} rad"
        )

        if self.phynt_enabled and not self.phynt_active:
            self.enable_phynt(self.wo, self.af)
            self.phynt.set_position(x, y, z, rpy[2])

        self.maneuver.wait()
        self.is_flying = True

    def land(self):
        if self.phynt_enabled:
            self.disable_phynt()
            self.uavpos.servo(ack=True)

        print(f"Landing to {self.land_height} m...")

        self.maneuver.set_current_state()
        self.maneuver.take_off(self.land_height, 0, ack=True)
        self.maneuver.wait()
        self.is_flying = False

    def get_state(self):
        return self.pom.frame("robot")["frame"]

    def send_world_velocity(self, v_w, z):
        v_w = np.asarray(v_w, dtype=float).reshape(6)
        state_cmd = {
            "pos": {"x": float("nan"), "y": float("nan"), "z": z},
            "att": {
                "qw": float("nan"),
                "qx": float("nan"),
                "qy": float("nan"),
                "qz": float("nan"),
            },
            "vel": {"vx": float(v_w[0]), "vy": float(v_w[1]), "vz": float(v_w[2])},
            "avel": {"wx": float(v_w[3]), "wy": float(v_w[4]), "wz": float(v_w[5])},
            "acc": {"ax": 0, "ay": 0, "az": 0},
            "aacc": {"awx": 0, "awy": 0, "awz": 0},
            "jerk": {"jx": 0, "jy": 0, "jz": 0},
            "snap": {"sx": 0, "sy": 0, "sz": 0},
        }
        self.uavpos.set_state(state_cmd, oneway=True)

    def stop(self):
        print("Stopping robot and logs...")

        try:
            self.rotorcraft.stop()
            self.maneuver.stop()
            self.uavpos.hover()
            self.uavatt.coast()
        finally:
            try:
                self.mocap.unset_logfile()
                self.rotorcraft.log_stop()
                self.pom.log_stop()
                self.uavpos.log_stop()
                self.uavatt.log_stop()
                self.maneuver.log_stop()
            except Exception:
                pass

            if self.phynt_enabled:
                try:
                    self.phynt.log_stop()
                    self.disable_phynt()
                except Exception:
                    pass

    def check_phynt(self):
        print("Phynt state:")
        print(f"phynt is set to {self.phynt_enabled}")
        print(f"phynt active: {self.phynt_active}")
        print(self.phynt.log_info())

    def check_pom(self):
        print("POM state:")
        position = self.pom.frame("robot")["frame"]["pos"]
        orientation = self.pom.frame("robot")["frame"]["att"]
        print(
            f"Position: x={position['x']:.2f}, y={position['y']:.2f}, z={position['z']:.2f}"
        )
        roll, pitch, yaw = R.from_quat(
            [orientation["qx"], orientation["qy"], orientation["qz"], orientation["qw"]]
        ).as_euler("xyz", degrees=True)
        print(f"Orientation: roll={roll:.2f}, pitch={pitch:.2f}, yaw={yaw:.2f}")
