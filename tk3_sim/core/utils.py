"""Frame transforms and integration for TeleKyb control.

``frame_R_basis`` maps coordinates from ``frame`` into ``basis``.
Twists are ordered [vx, vy, vz, wx, wy, wz]. For an offset p expressed in
basis coordinates, v_basis = R @ v_frame + p x (R @ w_frame).
"""

import math

import numpy as np
from scipy.spatial.transform import Rotation as R


DEFAULT_SENSOR_Z_OFFSET_M = (74.255 + 65.0) / 1000.0


def skew(p: np.ndarray) -> np.ndarray:
    """Return [p]_x so that [p]_x @ q equals p cross q."""
    p = np.asarray(p, dtype=float).reshape(3)
    return np.array(
        [
            [0.0, -p[2], p[1]],
            [p[2], 0.0, -p[0]],
            [-p[1], p[0], 0.0],
        ],
        dtype=float,
    )


def rotation_to_twist_matrix(frame_R_basis: np.ndarray) -> np.ndarray:
    """Build diag(R, R) for frames with coincident origins."""
    frame_R_basis = np.asarray(frame_R_basis, dtype=float).reshape(3, 3)

    return np.block(
        [
            [frame_R_basis, np.zeros((3, 3))],
            [np.zeros((3, 3)), frame_R_basis],
        ]
    )


def pose_to_twist_matrix(
    frame_R_basis: np.ndarray, frame_p_basis: np.ndarray
) -> np.ndarray:
    """Map frame twists to basis twists using rotation R and origin offset p."""
    frame_R_basis = np.asarray(frame_R_basis, dtype=float).reshape(3, 3)
    frame_p_basis = np.asarray(frame_p_basis, dtype=float).reshape(3)

    return np.block(
        [
            [frame_R_basis, skew(frame_p_basis) @ frame_R_basis],
            [np.zeros((3, 3)), frame_R_basis],
        ]
    )


def state_to_body_R_world(state) -> np.ndarray:
    """Return the body-to-world rotation from a POM quaternion."""
    att = state["att"]
    qw = float(att["qw"])
    qx = float(att["qx"])
    qy = float(att["qy"])
    qz = float(att["qz"])

    return R.from_quat([qx, qy, qz, qw]).as_matrix()


def state_to_world_R_body(state) -> np.ndarray:
    """Return the world-to-body rotation from a POM quaternion."""
    return state_to_body_R_world(state).T


def camera_R_sensor() -> np.ndarray:
    """Map camera axes to sensor axes: x and y are reversed; z is shared."""
    return np.diag([-1.0, -1.0, 1.0])


SENSOR_YAW_WRT_BODY_DEG = -30.0


def sensor_R_body() -> np.ndarray:
    """Rotate sensor axes into body axes using the mounting yaw."""
    return R.from_euler(
        "z",
        SENSOR_YAW_WRT_BODY_DEG,
        degrees=True,
    ).as_matrix()


def body_R_sensor() -> np.ndarray:
    """Return the inverse of the sensor mounting rotation."""
    return sensor_R_body().T


def camera_to_sensor_twist_matrix() -> np.ndarray:
    """Map camera twists to sensor twists."""
    return rotation_to_twist_matrix(camera_R_sensor())


def sensor_to_camera_twist_matrix() -> np.ndarray:
    """Map sensor twists to camera twists."""
    return np.linalg.inv(camera_to_sensor_twist_matrix())


def sensor_u4_to_twist_matrix() -> np.ndarray:
    """Embed [vx, vy, vz, wz] in a sensor twist with zero roll/pitch rates."""
    sensor_S_u4 = np.zeros((6, 4), dtype=float)

    sensor_S_u4[0, 0] = 1.0  # vx
    sensor_S_u4[1, 1] = 1.0  # vy
    sensor_S_u4[2, 2] = 1.0  # vz
    sensor_S_u4[5, 3] = 1.0  # yaw -> sensor/body wz

    return sensor_S_u4


def u4_to_sensor_twist(u4: np.ndarray) -> np.ndarray:
    """Convert [vx, vy, vz, wz] into a six-axis sensor twist."""
    u4 = np.asarray(u4, dtype=float).reshape(4)
    return sensor_u4_to_twist_matrix() @ u4


def u4_to_camera_twist_matrix() -> np.ndarray:
    """Map [vx, vy, vz, wz] in sensor coordinates into a camera twist."""
    return sensor_to_camera_twist_matrix() @ sensor_u4_to_twist_matrix()


def sensor_to_body_twist_matrix(d: float = DEFAULT_SENSOR_Z_OFFSET_M) -> np.ndarray:
    """Map sensor twists to the body origin, with sensor offset [0, 0, -d]."""
    sensor_p_body = np.array([0.0, 0.0, -float(d)], dtype=float)

    return pose_to_twist_matrix(
        frame_R_basis=sensor_R_body(),
        frame_p_basis=sensor_p_body,
    )


def body_to_world_twist_matrix(state) -> np.ndarray:
    """Rotate body-origin twists from body into world coordinates."""
    return rotation_to_twist_matrix(state_to_body_R_world(state))


def world_to_body_twist_matrix(state) -> np.ndarray:
    """Rotate body-origin twists from world into body coordinates."""
    return rotation_to_twist_matrix(state_to_world_R_body(state))


def camera_to_body_twist_matrix(d: float = DEFAULT_SENSOR_Z_OFFSET_M) -> np.ndarray:
    """Map camera twists to body-origin twists, including the sensor offset."""
    return sensor_to_body_twist_matrix(d) @ camera_to_sensor_twist_matrix()


def camera_to_world_twist_matrix(
    state,
    d: float = DEFAULT_SENSOR_Z_OFFSET_M,
) -> np.ndarray:
    """Map camera twists through sensor and body frames into world coordinates."""
    return (
        body_to_world_twist_matrix(state)
        @ sensor_to_body_twist_matrix(d)
        @ camera_to_sensor_twist_matrix()
    )


def camera_twist_to_world_twist(
    V_camera: np.ndarray,
    state,
    d: float = DEFAULT_SENSOR_Z_OFFSET_M,
) -> np.ndarray:
    """
    Transform a camera-frame twist into a world-frame body-origin twist.
    """
    V_camera = np.asarray(V_camera, dtype=float).reshape(6)
    return camera_to_world_twist_matrix(state, d) @ V_camera


def world_twist_to_body_twist(V_world: np.ndarray, state) -> np.ndarray:
    """
    Transform a world-frame body-origin twist into body-frame coordinates.
    """
    V_world = np.asarray(V_world, dtype=float).reshape(6)
    return world_to_body_twist_matrix(state) @ V_world


def camera_twist_to_body_twist(
    V_camera: np.ndarray,
    d: float = DEFAULT_SENSOR_Z_OFFSET_M,
) -> np.ndarray:
    """
    Transform a camera-frame twist into a body-frame body-origin twist.
    """
    V_camera = np.asarray(V_camera, dtype=float).reshape(6)
    return camera_to_body_twist_matrix(d) @ V_camera


def body_angular_velocity_to_euler_rates_matrix(
    roll: float, pitch: float
) -> np.ndarray:
    """Map body angular velocity to XYZ Euler rates; singular at pitch +/- pi/2."""
    sr = math.sin(roll)
    cr = math.cos(roll)
    tp = math.tan(pitch)
    cp = math.cos(pitch)

    if abs(cp) < 1e-6:
        raise ValueError("Pitch too close to +/- pi/2; Euler-rate matrix is singular.")

    return np.array(
        [
            [1.0, sr * tp, cr * tp],
            [0.0, cr, -sr],
            [0.0, sr / cp, cr / cp],
        ],
        dtype=float,
    )


def euler_rates_to_body_angular_velocity_matrix(
    roll: float, pitch: float
) -> np.ndarray:
    """Map XYZ Euler rates to body angular velocity."""
    sr = math.sin(roll)
    cr = math.cos(roll)
    sp = math.sin(pitch)
    cp = math.cos(pitch)

    return np.array(
        [
            [1.0, 0.0, -sp],
            [0.0, cr, sr * cp],
            [0.0, -sr, cr * cp],
        ],
        dtype=float,
    )


def world_angular_velocity_to_euler_rates_matrix(state) -> np.ndarray:
    """Map world angular velocity to XYZ Euler rates."""
    att = state["att"]
    qw = float(att["qw"])
    qx = float(att["qx"])
    qy = float(att["qy"])
    qz = float(att["qz"])

    roll, pitch = quat_to_roll_pitch(qw, qx, qy, qz)

    E_body_to_euler = body_angular_velocity_to_euler_rates_matrix(roll, pitch)
    world_R_body = state_to_world_R_body(state)

    return E_body_to_euler @ world_R_body


def wrap_to_pi(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def quat_to_roll_pitch(qw, qx, qy, qz):
    """
    Return roll, pitch from quaternion [qw, qx, qy, qz].
    """
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    return roll, pitch


def integrate_world_twist_from_state(state, V_world, dt):
    """Integrate a world twist over dt; return position and [qw, qx, qy, qz]."""
    V_world = np.asarray(V_world, dtype=float).reshape(6)

    p = np.array(
        [
            state["pos"]["x"],
            state["pos"]["y"],
            state["pos"]["z"],
        ],
        dtype=float,
    )

    att = state["att"]
    qw = float(att["qw"])
    qx = float(att["qx"])
    qy = float(att["qy"])
    qz = float(att["qz"])

    body_R_world = R.from_quat([qx, qy, qz, qw])

    p_new = p + V_world[:3] * dt

    omega_world = V_world[3:6]
    if np.linalg.norm(omega_world) < 1e-12:
        inc_R_world = R.identity()
    else:
        inc_R_world = R.from_rotvec(omega_world * dt)

    body_R_world_new = inc_R_world * body_R_world

    qx_n, qy_n, qz_n, qw_n = body_R_world_new.as_quat()
    q_new = np.array([qw_n, qx_n, qy_n, qz_n], dtype=float)

    return p_new, q_new
