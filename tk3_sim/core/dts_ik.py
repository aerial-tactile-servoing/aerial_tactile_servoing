#!/usr/bin/env python3
"""Incremental GelSight DTS control in sensor x, y and yaw.

Consecutive neural grayscale frames define the image error. A damped solve
using their mean luminance interaction matrix produces a velocity increment;
the accumulated command is clipped and transformed into world coordinates.
The uavpos command holds altitude separately. ``--no-motors`` records images
and computed velocities without takeoff or controller motion commands.
"""

from __future__ import annotations

import argparse
import code
import json
import pathlib
import readline
import rlcompleter
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import cv2
import numpy as np
import torch
import torch_tensorrt  # Registers TensorRT operators used by torch.jit.load.

import gelsight_interface.gelsight as gs
import mylogger
import tk3_sim.core.utils as utils
from tk3_sim.core.log_paths import make_log_paths, print_log_paths
from tk3_sim.core.tk3_runtime import TK3Runtime


FRAME_HEIGHT = 480
FRAME_WIDTH = 640

TACTILE_DEPTH_M = 0.024
SN_DEFAULT = "2DDW-JXDC"
FEATURE_BORDER = 10


def write_gains_file(log_path: pathlib.Path, args, *, relative_log_path: str):
    gains = {
        "controller": "dts_ik_3dof_incremental_fixed_altitude",
        "relative_log_path": relative_log_path,
        "sensor": {
            "sn": args.sn,
            "threshold_first_touch": args.threshold_first_touch,
            "initial_delay": args.desired_delay,
        },
        "preprocessing": {
            "grayscale_method": "network",
        },
        "dts_ik": {
            "reference_mode": "incremental",
            "decision_variable": "delta_u3_sensor=[delta_vx_sensor, delta_vy_sensor, delta_wz_sensor]",
            "command_update": "u3_k = u3_{k-1} + delta_u3",
            "error": "current_gray_valid - previous_gray_valid",
            "interaction_matrix_mode": "mean",
            "lambda_x": args.lambda_x,
            "lambda_y": args.lambda_y,
            "lambda_yaw": args.lambda_yaw,
            "mu_ik": args.mu_ik,
            "sign": args.sign,
            "controlled_dofs": "[vx_sensor, vy_sensor, wz_sensor]",
            "disabled_dofs": "[vz_sensor, wx_sensor, wy_sensor]",
        },
        "safety_clipping": {
            "vmax_translation_xy": args.vmax_translation,
            "wmax_yaw": args.wmax_rotation,
            "altitude_ref_m": args.takeoff_height,
            "where": (
                "final integrated u3_sensor is clipped. This is equivalent to shifted "
                "delta constraints: lower-u_prev <= delta_u <= upper-u_prev"
            ),
        },
        "run": {
            "duration": args.duration,
            "command_transmission": not args.no_motors,
            "motors_enabled": not args.no_motors,
            "display": False,
            "phynt": True,
            "land": not args.no_motors,
            "no_motors": args.no_motors,
        },
    }

    with open(log_path / "gains.txt", "w", encoding="utf-8") as f:
        f.write(json.dumps(gains, indent=2))
        f.write("\n")


def load_grayscale_model(engine_path: pathlib.Path, device: torch.device):
    if not engine_path.exists():
        raise FileNotFoundError(f"TensorRT engine not found: {engine_path}")
    model = torch.jit.load(str(engine_path), map_location=device)
    model.eval()
    return model


def parse_camera_xml(camera_xml: pathlib.Path) -> tuple[float, float, float, float]:
    root = ET.parse(camera_xml).getroot()
    for cam in root.findall("camera"):
        name = (cam.findtext("name") or "").strip()
        if name != "Camera":
            continue
        model = cam.find("model")
        if model is None:
            continue
        typ = (model.findtext("type") or "").strip()
        if typ == "perspectiveProjWithoutDistortion":
            px = float(model.findtext("px"))
            py = float(model.findtext("py"))
            u0 = float(model.findtext("u0"))
            v0 = float(model.findtext("v0"))
            return px, py, u0, v0
    raise RuntimeError(
        f"Could not find perspectiveProjWithoutDistortion Camera model in {camera_xml}"
    )


class GrayInputBuffer:
    """Reuse pinned CPU and GPU buffers to convert BGR uint8 HWC to float32 NCHW."""

    def __init__(self, device: torch.device):
        self.device = device
        self.hwc_cpu_u8 = torch.empty(
            (FRAME_HEIGHT, FRAME_WIDTH, 3),
            dtype=torch.uint8,
            pin_memory=(device.type == "cuda"),
        )
        self.hwc_gpu_u8 = torch.empty(
            (FRAME_HEIGHT, FRAME_WIDTH, 3), device=device, dtype=torch.uint8
        )
        self.x_gpu = torch.empty(
            (1, 3, FRAME_HEIGHT, FRAME_WIDTH), device=device, dtype=torch.float32
        )

    def prepare(self, frame_bgr_u8: np.ndarray) -> torch.Tensor:
        if frame_bgr_u8 is None:
            raise ValueError("Cannot grayscale a None frame")
        frame_c = np.ascontiguousarray(frame_bgr_u8)
        np.copyto(self.hwc_cpu_u8.numpy(), frame_c)

        self.hwc_gpu_u8.copy_(self.hwc_cpu_u8, non_blocking=True)
        self.x_gpu[0, 0].copy_(self.hwc_gpu_u8[:, :, 0])
        self.x_gpu[0, 1].copy_(self.hwc_gpu_u8[:, :, 1])
        self.x_gpu[0, 2].copy_(self.hwc_gpu_u8[:, :, 2])
        self.x_gpu.mul_(1.0 / 255.0)
        return self.x_gpu


@torch.inference_mode()
def grayscale_from_model_gpu_u8(
    frame_bgr_u8: np.ndarray,
    model,
    buffer: GrayInputBuffer,
) -> torch.Tensor:
    x = buffer.prepare(frame_bgr_u8)
    y = model(x)
    if isinstance(y, (tuple, list)):
        y = y[0]
    return (y[0, 0].clamp(0, 1).mul(255.0)).to(torch.uint8).contiguous()


@dataclass
class TorchFeatureIKContext:
    x: torch.Tensor
    y: torch.Tensor
    T: torch.Tensor
    dtype: torch.dtype
    device: torch.device
    border: int
    px: float
    py: float


def scalar_tensor(x: float, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    return torch.tensor(float(x), dtype=dtype, device=device)


def torch_luminance_interaction(
    gray_u8: torch.Tensor,
    *,
    x: torch.Tensor,
    y: torch.Tensor,
    px: float,
    py: float,
    z_depth: float,
    dtype: torch.dtype,
    border: int,
) -> torch.Tensor:
    if gray_u8.ndim != 2:
        raise ValueError(f"Expected gray HxW tensor, got {tuple(gray_u8.shape)}")

    I = gray_u8.to(dtype=dtype)
    b = int(border)

    # Same derivative stencil used by ViSP FeatureLuminance.
    dfx = (
        2047.0 * (I[b:-b, b + 1 : -b + 1] - I[b:-b, b - 1 : -b - 1])
        + 913.0 * (I[b:-b, b + 2 : -b + 2] - I[b:-b, b - 2 : -b - 2])
        + 112.0 * (I[b:-b, b + 3 : -b + 3] - I[b:-b, b - 3 : -b - 3])
    ) / 8418.0
    dfy = (
        2047.0 * (I[b + 1 : -b + 1, b:-b] - I[b - 1 : -b - 1, b:-b])
        + 913.0 * (I[b + 2 : -b + 2, b:-b] - I[b - 2 : -b - 2, b:-b])
        + 112.0 * (I[b + 3 : -b + 3, b:-b] - I[b - 3 : -b - 3, b:-b])
    ) / 8418.0

    Ix = scalar_tensor(px, dtype, gray_u8.device) * dfx
    Iy = scalar_tensor(py, dtype, gray_u8.device) * dfy
    Zinv = scalar_tensor(1.0 / z_depth, dtype, gray_u8.device)

    L0 = Ix * Zinv
    L1 = Iy * Zinv
    L2 = -(x * Ix + y * Iy) * Zinv
    L3 = -Ix * x * y - (1.0 + y * y) * Iy
    L4 = (1.0 + x * x) * Ix + Iy * x * y
    L5 = Iy * x - Ix * y
    return torch.stack([L0, L1, L2, L3, L4, L5], dim=-1).reshape(-1, 6)


def make_gpu_feature_ik_context(
    example_gray_u8: torch.Tensor,
    *,
    px: float,
    py: float,
    u0: float,
    v0: float,
    dtype: torch.dtype,
    device: torch.device,
    border: int = FEATURE_BORDER,
) -> TorchFeatureIKContext:
    h, w = int(example_gray_u8.shape[0]), int(example_gray_u8.shape[1])
    rows = torch.arange(border, h - border, device=device, dtype=dtype)
    cols = torch.arange(border, w - border, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(rows, cols, indexing="ij")
    x = (xx - scalar_tensor(u0, dtype, device)) / scalar_tensor(px, dtype, device)
    y = (yy - scalar_tensor(v0, dtype, device)) / scalar_tensor(py, dtype, device)

    # 3-DoF control vector is u3 = [vx_sensor, vy_sensor, wz_sensor].
    # Start from the existing safe4D mapping u4=[vx, vy, vz, wz] and remove
    # the tactile z column. This gives V_camera = T @ u3.
    T4_np = np.asarray(utils.u4_to_camera_twist_matrix(), dtype=np.float32)
    T3_np = T4_np[:, [0, 1, 3]]
    T = torch.as_tensor(T3_np, dtype=dtype, device=device)

    return TorchFeatureIKContext(
        x=x,
        y=y,
        T=T,
        dtype=dtype,
        device=device,
        border=border,
        px=px,
        py=py,
    )


@torch.inference_mode()
def compute_frame_feature_gpu(
    gray_u8: torch.Tensor,
    ctx: TorchFeatureIKContext,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cropped grayscale vector and 3-DoF luminance interaction matrix."""
    b = ctx.border
    h, w = int(gray_u8.shape[0]), int(gray_u8.shape[1])
    valid = gray_u8[b : h - b, b : w - b].to(dtype=ctx.dtype).reshape(-1)

    L = torch_luminance_interaction(
        gray_u8,
        x=ctx.x,
        y=ctx.y,
        px=ctx.px,
        py=ctx.py,
        z_depth=TACTILE_DEPTH_M,
        dtype=ctx.dtype,
        border=b,
    )
    Lu = L @ ctx.T
    return valid, Lu


def exact_delta_u3_from_H_rhs_cpu(
    H: np.ndarray,
    rhs: np.ndarray,
    *,
    lambda_x: float,
    lambda_y: float,
    lambda_yaw: float,
    mu_ik: float,
    sign: float,
) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    rhs = np.asarray(rhs, dtype=np.float64).reshape(3)

    H_damped = (
        H + float(mu_ik) * np.diag(np.diag(H)) + 1e-12 * np.eye(3, dtype=np.float64)
    )

    try:
        delta_raw = np.linalg.solve(H_damped, rhs)
    except np.linalg.LinAlgError:
        delta_raw = np.linalg.pinv(H_damped) @ rhs

    lambda_vec = np.array([lambda_x, lambda_y, lambda_yaw], dtype=np.float64)
    return np.asarray(float(sign) * lambda_vec * delta_raw, dtype=np.float64).reshape(3)


@torch.inference_mode()
def gpu_incremental_feature_ik_output_u3(
    current_gray_u8: torch.Tensor,
    *,
    reference_valid: torch.Tensor,
    reference_Lu: torch.Tensor,
    ctx: TorchFeatureIKContext,
    lambda_x: float,
    lambda_y: float,
    lambda_yaw: float,
    mu_ik: float,
    sign: float,
) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
    """Solve one incremental 3-DoF output using the mean interaction matrix."""
    current_valid, current_Lu = compute_frame_feature_gpu(current_gray_u8, ctx)
    Lu = 0.5 * (reference_Lu + current_Lu)

    err = current_valid - reference_valid
    rhs = Lu.transpose(0, 1) @ err
    H = Lu.transpose(0, 1) @ Lu

    # This copy is the natural synchronization point for all previous GPU work.
    H_np = H.detach().cpu().numpy().astype(np.float64, copy=False)
    rhs_np = rhs.detach().cpu().numpy().astype(np.float64, copy=False)
    output_u3 = exact_delta_u3_from_H_rhs_cpu(
        H_np,
        rhs_np,
        lambda_x=lambda_x,
        lambda_y=lambda_y,
        lambda_yaw=lambda_yaw,
        mu_ik=mu_ik,
        sign=sign,
    )
    return output_u3, current_valid.detach(), current_Lu.detach()


def compute_contact_norm(
    frame_bgr_u8: np.ndarray, background_bgr_u8: np.ndarray
) -> float:
    return float(np.linalg.norm(cv2.absdiff(frame_bgr_u8, background_bgr_u8)))


def fill_sensor_twist_from_u3(out: np.ndarray, u3: np.ndarray) -> None:
    """Fill a preallocated 6D sensor twist from u3=[vx, vy, wz]."""
    out[:] = 0.0
    out[0] = float(u3[0])  # vx_sensor
    out[1] = float(u3[1])  # vy_sensor
    out[5] = float(u3[2])  # wz_sensor / yaw


def apply_shifted_delta_constraints_inplace(
    delta_u3_raw: np.ndarray,
    u3_prev: np.ndarray,
    delta_out: np.ndarray,
    u3_out: np.ndarray,
    *,
    vmax_translation: float,
    wmax_yaw: float,
) -> None:
    """In-place version of shifted box constraints for the realtime loop."""
    u3_out[0] = np.clip(
        u3_prev[0] + delta_u3_raw[0], -float(vmax_translation), +float(vmax_translation)
    )
    u3_out[1] = np.clip(
        u3_prev[1] + delta_u3_raw[1], -float(vmax_translation), +float(vmax_translation)
    )
    u3_out[2] = np.clip(
        u3_prev[2] + delta_u3_raw[2], -float(wmax_yaw), +float(wmax_yaw)
    )
    delta_out[:] = u3_out - u3_prev


def apply_incremental_command_inplace(
    solver_output_u3: np.ndarray,
    u3_prev: np.ndarray,
    applied_change_out: np.ndarray,
    u3_out: np.ndarray,
    *,
    vmax_translation: float,
    wmax_yaw: float,
) -> None:
    """Accumulate and bound one LM velocity increment."""
    apply_shifted_delta_constraints_inplace(
        solver_output_u3,
        u3_prev,
        applied_change_out,
        u3_out,
        vmax_translation=vmax_translation,
        wmax_yaw=wmax_yaw,
    )


def send_world_velocity_at_altitude(
    runtime: TK3Runtime, v_w: np.ndarray, z_ref: float
) -> None:
    """Send a world-frame velocity command together with a fixed altitude reference."""
    runtime.send_world_velocity(v_w, float(z_ref))


class PreparedDTSTrial:
    """Resources and mutable controller state prepared before takeoff/logging."""

    def __init__(
        self,
        *,
        repo_root: pathlib.Path,
        log_path: pathlib.Path,
        sn: str,
    ):
        self.log_path = log_path
        self.logger = None
        self.gelsight = None
        self.closed = False

        try:
            self.logger = mylogger.TactileLogger(
                log_dir=log_path,
                image_shape=(FRAME_HEIGHT, FRAME_WIDTH, 3),
                world_dim=6,
            )

            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            engine_path = repo_root / "dts/models/compiled_model.trt"
            self.model = load_grayscale_model(engine_path, self.device)
            self.gray_buffer = GrayInputBuffer(self.device)
            self.feature_dtype = torch.float32

            camera_file_path = (
                repo_root / f"dts/models/camera_{sn.replace('-', '_')}.xml"
            )
            self.px, self.py, self.u0, self.v0 = parse_camera_xml(camera_file_path)
            example_gray_u8 = torch.empty(
                (FRAME_HEIGHT, FRAME_WIDTH),
                dtype=torch.uint8,
                device=self.device,
            )
            self.feature_ctx = make_gpu_feature_ik_context(
                example_gray_u8,
                px=self.px,
                py=self.py,
                u0=self.u0,
                v0=self.v0,
                dtype=self.feature_dtype,
                device=self.device,
                border=FEATURE_BORDER,
            )

            self.gelsight = gs.Gelsight(sn, (FRAME_HEIGHT, FRAME_WIDTH))
            self.gelsight.connect()

            background_path = repo_root / "dts/models/background.png"
            self.background_bgr = cv2.imread(str(background_path), cv2.IMREAD_COLOR)
            if self.background_bgr is None:
                raise FileNotFoundError(
                    f"Could not read background image: {background_path}"
                )
            cv2.imwrite(str(log_path / "background_bgr.png"), self.background_bgr)

            self.flag_initialized = False
            self.reference_gray_gpu: torch.Tensor | None = None
            self.reference_valid: torch.Tensor | None = None
            self.reference_Lu: torch.Tensor | None = None
            self.u3_cmd_prev = np.zeros(3, dtype=np.float64)
            self.delta_u3 = np.zeros(3, dtype=np.float64)
            self.u3_cmd = np.zeros(3, dtype=np.float64)

            self.V_zero = np.zeros(6, dtype=float)
            self.V_sensor = np.zeros(6, dtype=float)
            self.V_body_dts = np.zeros(6, dtype=float)
            self.V_world = np.zeros(6, dtype=float)
            self.sensor_to_body = utils.sensor_to_body_twist_matrix()
            self.last_ctr_seen = -1
            self.skip_first_flag = True
        except BaseException:
            self.close()
            raise

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True

        if self.logger is not None:
            try:
                self.logger._q.join()
            except Exception:
                pass
            try:
                self.logger.close(timeout_s=30.0)
            except Exception:
                pass

        if self.gelsight is not None:
            try:
                self.gelsight.release()
            except Exception:
                pass


def dts_ik_loop(
    runtime: TK3Runtime,
    *,
    trial: PreparedDTSTrial,
    duration: float,
    threshold_first_touch: float,
    desired_delay: float,
    lambda_x: float,
    lambda_y: float,
    lambda_yaw: float,
    mu_ik: float,
    sign: float,
    vmax_translation: float,
    wmax_rotation: float,
    altitude_ref: float,
    send_commands: bool,
):
    logger = trial.logger
    gelsight = trial.gelsight
    if logger is None or gelsight is None:
        raise RuntimeError("DTS trial resources were not prepared.")

    model = trial.model
    gray_buffer = trial.gray_buffer
    background_bgr = trial.background_bgr
    flag_initialized = trial.flag_initialized
    feature_ctx = trial.feature_ctx
    reference_gray_gpu = trial.reference_gray_gpu
    reference_valid = trial.reference_valid
    reference_Lu = trial.reference_Lu
    u3_cmd_prev = trial.u3_cmd_prev
    delta_u3 = trial.delta_u3
    u3_cmd = trial.u3_cmd
    V_zero = trial.V_zero
    V_sensor = trial.V_sensor
    V_body_dts = trial.V_body_dts
    V_world = trial.V_world
    sensor_to_body = trial.sensor_to_body
    last_ctr_seen = trial.last_ctr_seen
    skip_first_flag = trial.skip_first_flag
    t0 = time.time()
    next_countdown_print = t0

    print(
        "DTS IK 3DoF clipped controller started. "
        "reference_mode=incremental, grayscale=network, interaction_matrix=mean"
    )

    try:
        while True:
            now = time.time()
            elapsed = now - t0
            remaining = max(0.0, duration - elapsed)
            if now >= next_countdown_print:
                print(f"\rRemaining trial time: {remaining:5.1f} s", end="", flush=True)
                next_countdown_print = now + 1.0

            if elapsed >= duration:
                print("\nReached duration.")
                break

            frame_bgr, ctr = gelsight.get_frame_with_counter(copy=True)
            if frame_bgr is None:
                continue
            if ctr == last_ctr_seen:
                continue

            last_ctr_seen = ctr
            logger.log_image(frame_bgr, time.time())

            contact_active = (
                compute_contact_norm(frame_bgr, background_bgr) > threshold_first_touch
            )

            if not contact_active:
                if flag_initialized:
                    print("\nContact lost. Resetting tactile state.")
                    flag_initialized = False
                    reference_gray_gpu = None
                    reference_valid = None
                    reference_Lu = None
                    u3_cmd_prev[:] = 0.0
                    delta_u3[:] = 0.0
                    u3_cmd[:] = 0.0
                    break

                if send_commands:
                    send_world_velocity_at_altitude(runtime, V_zero, altitude_ref)
                logger.log_world_velocities(V_zero, time.time())
                logger.log_body_velocities(V_zero, time.time())
                continue

            just_initialized = False
            if not flag_initialized:
                print(
                    "\nFirst contact detected. Waiting before tactile reference initialization."
                )
                if desired_delay > 0:
                    time.sleep(desired_delay)
                    frame_bgr, ctr = gelsight.get_frame_with_counter(copy=True)
                    if frame_bgr is None:
                        continue
                    last_ctr_seen = ctr
                    logger.log_image(frame_bgr, time.time())

                initial_frame_bgr = frame_bgr
                reference_gray_gpu = grayscale_from_model_gpu_u8(
                    initial_frame_bgr,
                    model,
                    gray_buffer,
                )

                reference_valid, reference_Lu = compute_frame_feature_gpu(
                    reference_gray_gpu, feature_ctx
                )
                reference_valid = reference_valid.detach()
                reference_Lu = reference_Lu.detach()
                u3_cmd_prev[:] = 0.0
                delta_u3[:] = 0.0
                u3_cmd[:] = 0.0

                flag_initialized = True
                just_initialized = True
                skip_first_flag = True

                print("Incremental tactile reference initialized.")

            if (
                reference_gray_gpu is None
                or reference_valid is None
                or reference_Lu is None
            ):
                raise RuntimeError("Tactile reference state was not initialized.")

            if just_initialized:
                current_gray_gpu = reference_gray_gpu
            else:
                current_gray_gpu = grayscale_from_model_gpu_u8(
                    frame_bgr,
                    model,
                    gray_buffer,
                )

            if skip_first_flag:
                skip_first_flag = False
                print("skipped first frame after initialization")
                continue

            solver_output_u3, current_valid, current_Lu = (
                gpu_incremental_feature_ik_output_u3(
                    current_gray_gpu,
                    reference_valid=reference_valid,
                    reference_Lu=reference_Lu,
                    ctx=feature_ctx,
                    lambda_x=lambda_x,
                    lambda_y=lambda_y,
                    lambda_yaw=lambda_yaw,
                    mu_ik=mu_ik,
                    sign=sign,
                )
            )

            apply_incremental_command_inplace(
                solver_output_u3,
                u3_cmd_prev,
                delta_u3,
                u3_cmd,
                vmax_translation=vmax_translation,
                wmax_yaw=wmax_rotation,
            )

            fill_sensor_twist_from_u3(V_sensor, u3_cmd)

            state = runtime.get_state()

            V_body_dts[:] = sensor_to_body @ V_sensor
            V_body_dts[2] = 0.0
            V_body_dts[3] = 0.0
            V_body_dts[4] = 0.0

            W_body = utils.body_to_world_twist_matrix(state)
            V_world[:] = W_body @ V_body_dts

            # Hold altitude through the position reference, with zero vertical velocity.
            V_world[2] = 0.0

            if send_commands:
                send_world_velocity_at_altitude(runtime, V_world, altitude_ref)

            logger.log_world_velocities(V_world, time.time())
            logger.log_body_velocities(V_body_dts, time.time())

            reference_gray_gpu = current_gray_gpu.detach()
            reference_valid = current_valid.detach()
            reference_Lu = current_Lu.detach()
            u3_cmd_prev[:] = u3_cmd

    finally:
        if send_commands:
            try:
                send_world_velocity_at_altitude(runtime, V_zero, altitude_ref)
            except Exception:
                pass
        trial.flag_initialized = flag_initialized
        trial.feature_ctx = feature_ctx
        trial.reference_gray_gpu = reference_gray_gpu
        trial.reference_valid = reference_valid
        trial.reference_Lu = reference_Lu
        trial.last_ctr_seen = last_ctr_seen
        trial.skip_first_flag = skip_first_flag


def run_dts_from_args(
    *,
    runtime: TK3Runtime,
    args,
    repo_root: pathlib.Path,
    takeoff: bool = False,
):
    relative_log_path, log_path = make_log_paths(
        repo_root, args.tag, default_tag="dts"
    )
    write_gains_file(log_path, args, relative_log_path=relative_log_path)
    runtime.relative_log_path = relative_log_path

    print_log_paths(repo_root, relative_log_path)

    trial = None
    runtime_start_attempted = False
    runtime_started = False
    logs_started = False
    interrupted = False

    try:
        print("Preparing DTS resources before takeoff/logging...")
        trial = PreparedDTSTrial(
            repo_root=repo_root,
            log_path=log_path,
            sn=args.sn,
        )
        print("DTS preflight initialization complete.")

        if args.no_motors:
            if takeoff:
                print(
                    "--no-motors active: takeoff=True ignored; motors remain stopped."
                )
            runtime.logs()
            logs_started = True
        elif takeoff:
            runtime_start_attempted = True
            runtime.start()
            runtime.takeoff()
            runtime_started = True

        dts_ik_loop(
            runtime,
            trial=trial,
            duration=args.duration,
            threshold_first_touch=args.threshold_first_touch,
            desired_delay=args.desired_delay,
            lambda_x=args.lambda_x,
            lambda_y=args.lambda_y,
            lambda_yaw=args.lambda_yaw,
            mu_ik=args.mu_ik,
            sign=args.sign,
            vmax_translation=args.vmax_translation,
            wmax_rotation=args.wmax_rotation,
            altitude_ref=args.takeoff_height,
            send_commands=not args.no_motors,
        )
    except KeyboardInterrupt:
        interrupted = True
        if args.no_motors:
            print("\nCtrl-C received. Stopping the controller and logs...")
        else:
            print("\nCtrl-C received. Stopping the controller and landing...")
    finally:
        if runtime_started:
            try:
                send_world_velocity_at_altitude(
                    runtime,
                    np.zeros(6, dtype=float),
                    args.takeoff_height,
                )
            except Exception:
                pass

            try:
                runtime.land()
            except BaseException as exc:
                print(f"Landing reported an error: {exc!r}")

        if runtime_start_attempted or logs_started:
            try:
                runtime.stop()
            except BaseException as exc:
                print(f"Runtime shutdown reported an error: {exc!r}")

        if trial is not None:
            try:
                trial.close()
            except BaseException as exc:
                print(f"Tactile resource shutdown reported an error: {exc!r}")

        print_log_paths(repo_root, relative_log_path)

    return {
        "interrupted": interrupted,
        "relative_log_path": relative_log_path,
        "log_path": log_path,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="")
    parser.add_argument("--duration", type=float, default=30.0)

    parser.add_argument("--sn", default=SN_DEFAULT)
    parser.add_argument("--threshold-first-touch", type=float, default=1000.0)
    parser.add_argument("--desired-delay", type=float, default=2.0)
    parser.add_argument("--mass", type=float, default=3.1)

    parser.add_argument("--lambda-x", type=float, default=1000.0)
    parser.add_argument("--lambda-y", type=float, default=1000.0)
    parser.add_argument("--lambda-yaw", type=float, default=20.0)
    parser.add_argument("--mu-ik", type=float, default=0.2)
    parser.add_argument("--sign", type=float, choices=[-1.0, 1.0], default=-1.0)

    parser.add_argument("--vmax-translation", type=float, default=0.4)
    parser.add_argument("--wmax-rotation", type=float, default=0.2)
    parser.add_argument("--takeoff-height", type=float, default=1.35)
    parser.add_argument("--land-height", type=float, default=0.25)

    parser.add_argument(
        "--no-motors",
        action="store_true",
        help="Run the controller and logs without takeoff, landing, or motion commands.",
    )

    args = parser.parse_args()

    file_path = pathlib.Path(__file__).resolve()
    repo_root = file_path.parents[2]

    runtime = TK3Runtime(
        repo_root=repo_root,
        relative_log_path="results/logs_pending_dts",
        phynt_enabled=True,
        takeoff_height=args.takeoff_height,
        mass=args.mass,
        land_height=args.land_height,
    )

    last_trial = {}

    def dts(takeoff: bool = False):
        result = run_dts_from_args(
            runtime=runtime,
            args=args,
            repo_root=repo_root,
            takeoff=takeoff,
        )
        last_trial.clear()
        last_trial.update(result)
        shell.update(
            log_path=result["log_path"], relative_log_path=result["relative_log_path"]
        )
        return result

    shell = {
        "runtime": runtime,
        "args": args,
        "repo_root": repo_root,
        "last_trial": last_trial,
        "dts": dts,
        "run_dts_from_args": run_dts_from_args,
    }

    try:
        print("\nConnecting to the local TeleKyb simulation...")
        runtime.connect()
        runtime.setup()
        readline.set_completer(rlcompleter.Completer(shell).complete)
        readline.parse_and_bind("tab: complete")

        mode_instructions = (
            "  dts()      -> start DTS processing and logs; motors remain stopped\n"
            "  dts(takeoff=True) -> same; takeoff is ignored\n"
            if args.no_motors
            else "  dts()      -> start DTS with current args\n"
            "  dts(takeoff=True) -> take off, then start DTS\n"
        )
        run_instruction = "  dts()\n" if args.no_motors else "  dts(takeoff=True)\n"

        print(
            "\nSimulation controller shell ready.\n"
            "Available objects/functions:\n"
            "  runtime    -> TK3Runtime instance\n"
            "  args       -> parsed arguments\n"
            "  last_trial -> paths/status from the latest run\n"
            f"{mode_instructions}"
            "\nExample checks:\n"
            "  runtime.check_pom()\n"
            "  runtime.check_phynt()\n"
            "\nWhen ready, run:\n"
            f"{run_instruction}"
        )

        code.interact(
            banner="Aerial incremental DTS shell. Exit with Ctrl-D or exit().",
            local=shell,
        )

    finally:
        runtime.stop()


if __name__ == "__main__":
    main()
