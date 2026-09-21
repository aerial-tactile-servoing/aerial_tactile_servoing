from pathlib import Path
from datetime import datetime
import time
import threading
import queue
import numpy as np
import h5py


class TactileLogger:
    """
    Non-blocking logger:
      - log_image(img, t), log_world_velocities(v_W, t), and
        log_body_velocities(v_B, t) only enqueue
      - a background writer thread batches + writes to HDF5

    This prevents periodic multi-second stalls in the control loop.
    """

    def __init__(
        self,
        log_dir="logs",
        image_shape=(480, 640, 3),       # (height, width) or (height, width, channels)
        world_dim=6,
        image_batch=200,
        vel_batch=200,
        flush_every_s=1.0,
        compression="lzf",           # "lzf" or None
        queue_max_items=2000,        # capacity for combined items (images+vels)
        drop_policy="drop_new",      # "drop_new" (recommended) or "block"
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.filename = self.log_dir / f"tactile_log_{ts}.h5"

        self.image_shape = tuple(image_shape)
        self.world_dim = int(world_dim)
        self.image_batch = int(image_batch)
        self.vel_batch = int(vel_batch)
        self.flush_every_s = float(flush_every_s)
        self.drop_policy = str(drop_policy)
        self.compression = compression
        self.bounds_dim = int(world_dim*2)  # Assuming bounds dimension is the same as world dimension

        # Stats
        self.dropped_images = 0
        self.dropped_vels = 0
        self.dropped_body_vels = 0
        self.dropped_qp_bounds = 0
        self.dropped_prediction_events = 0

        # Queue for writer thread
        self._q = queue.Queue(maxsize=int(queue_max_items))
        self._stop = threading.Event()

        # Open file in the writer thread (safer with HDF5 thread-safety)
        self._writer_thread = threading.Thread(target=self._writer_main, daemon=True)
        self._writer_thread.start()

        print(f"[LOGGER] Async writer started. Writing to {self.filename}")

    # ---------------- public API (realtime-safe) ----------------

    def log_image(self, image, timestamp: float) -> None:
        img = np.asarray(image, dtype=np.uint8, order="C")
        if img.shape != self.image_shape:
            raise ValueError(f"Expected image shape {self.image_shape}, got {img.shape}")

        item = ("img", float(timestamp), img)

        if self.drop_policy == "block":
            self._q.put(item)  # may block (NOT recommended for drones)
        else:
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self.dropped_images += 1

    def log_world_velocities(self, v_W, timestamp: float) -> None:
        v = np.asarray(v_W, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.world_dim:
            raise ValueError(f"Expected v_W dim {self.world_dim}, got {v.shape[0]}")

        item = ("vw", float(timestamp), v)

        if self.drop_policy == "block":
            self._q.put(item)
        else:
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self.dropped_vels += 1

    def log_body_velocities(self, v_B, timestamp: float) -> None:
        v = np.asarray(v_B, dtype=np.float32).reshape(-1)
        if v.shape[0] != self.world_dim:
            raise ValueError(f"Expected v_B dim {self.world_dim}, got {v.shape[0]}")

        item = ("vb", float(timestamp), v)

        if self.drop_policy == "block":
            self._q.put(item)
        else:
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self.dropped_body_vels += 1
                
    def log_qp_bounds(self, lb, ub, timestamp: float) -> None:
        lb = np.asarray(lb, dtype=np.float32).reshape(-1)
        ub = np.asarray(ub, dtype=np.float32).reshape(-1)

        if lb.shape[0] != self.bounds_dim or ub.shape[0] != self.bounds_dim:
            raise ValueError(f"QP bounds must be {self.bounds_dim}-element vectors")

        item = ("qp_bounds", float(timestamp), (lb, ub))

        if self.drop_policy == "block":
            self._q.put(item)
        else:
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self.dropped_qp_bounds += 1

    def log_prediction_detector_event(
        self,
        *,
        timestamp: float,
        frame_i: int = -1,
        reduced_velocity: bool = False,
        reset: bool = False,
        bad_run: int = 0,
        command_scale: float = 1.0,
        prediction_residual: float = np.nan,
        prediction_cosine: float = np.nan,
        obs_delta_rms: float = np.nan,
        pred_delta_rms: float = np.nan,
        pred_error_rms: float = np.nan,
        e_ref: float = np.nan,
        d_step: float = np.nan,
        u3=None,
    ) -> None:
        """Log only prediction-detector intervention frames to HDF5.

        This is intentionally event-based rather than per-frame: call it only
        when a velocity reduction is applied or when the desired image is reset.
        The timestamp matches the image timestamp, so the triggering frame can be
        recovered later from images/timestamp.
        """
        if u3 is None:
            u3_arr = np.zeros(3, dtype=np.float32)
        else:
            u3_arr = np.asarray(u3, dtype=np.float32).reshape(3)

        payload = {
            "frame_i": int(frame_i),
            "reduced_velocity": bool(reduced_velocity),
            "reset": bool(reset),
            "bad_run": int(bad_run),
            "command_scale": float(command_scale),
            "prediction_residual": float(prediction_residual),
            "prediction_cosine": float(prediction_cosine),
            "obs_delta_rms": float(obs_delta_rms),
            "pred_delta_rms": float(pred_delta_rms),
            "pred_error_rms": float(pred_error_rms),
            "E_ref": float(e_ref),
            "D_step": float(d_step),
            "u3": u3_arr,
        }
        item = ("pred_event", float(timestamp), payload)

        if self.drop_policy == "block":
            self._q.put(item)
        else:
            try:
                self._q.put_nowait(item)
            except queue.Full:
                self.dropped_prediction_events += 1

    def close(self, timeout_s: float = 5.0) -> None:
        """
        Request shutdown and wait for writer to finish draining.
        """
        self._stop.set()
        self._writer_thread.join(timeout=timeout_s)
        if self._writer_thread.is_alive():
            # If we get here, the disk is likely stalled; we avoid hanging forever.
            print("[LOGGER] Warning: writer did not exit before timeout.")

    # ---------------- internal: writer thread ----------------

    def _writer_main(self):
        file = h5py.File(self.filename, "w", libver="latest")

        # datasets: images
        img_ds_kwargs = dict(
            dtype=np.uint8,
            maxshape=(None, *self.image_shape),
            chunks=(min(self.image_batch, 256), *self.image_shape),
        )
        if self.compression is not None:
            img_ds_kwargs["compression"] = self.compression

        img_ds = file.create_dataset("images/data", shape=(0, *self.image_shape), **img_ds_kwargs)
        img_ts = file.create_dataset(
            "images/timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(min(self.image_batch, 2048),),
        )

        # datasets: world velocities
        vw_ds = file.create_dataset(
            "world_velocities/data",
            shape=(0, self.world_dim),
            maxshape=(None, self.world_dim),
            dtype=np.float32,
            chunks=(min(self.vel_batch, 4096), self.world_dim),
        )
        vw_ts = file.create_dataset(
            "world_velocities/timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(min(self.vel_batch, 4096),),
        )

        # datasets: body velocities
        vb_ds = file.create_dataset(
            "body_velocities/data",
            shape=(0, self.world_dim),
            maxshape=(None, self.world_dim),
            dtype=np.float32,
            chunks=(min(self.vel_batch, 4096), self.world_dim),
        )
        vb_ts = file.create_dataset(
            "body_velocities/timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(min(self.vel_batch, 4096),),
        )
        
        # datasets: QP bounds
        qp_lb_ds = file.create_dataset(
            "qp_lower_bounds/data",
            shape=(0, self.bounds_dim),
            maxshape=(None, self.bounds_dim),
            dtype=np.float32,
            chunks=(min(self.vel_batch, 4096), self.bounds_dim),
        )

        qp_lb_ts = file.create_dataset(
            "qp_lower_bounds/timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(min(self.vel_batch, 4096),),
        )

        qp_ub_ds = file.create_dataset(
            "qp_upper_bounds/data",
            shape=(0, self.bounds_dim),
            maxshape=(None, self.bounds_dim),
            dtype=np.float32,
            chunks=(min(self.vel_batch, 4096), self.bounds_dim),
        )

        qp_ub_ts = file.create_dataset(
            "qp_upper_bounds/timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(min(self.vel_batch, 4096),),
        )

        # datasets: prediction detector intervention events only
        pred_chunk = min(self.vel_batch, 4096)
        pred_ts = file.create_dataset(
            "prediction_detector_events/timestamp",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float64,
            chunks=(pred_chunk,),
        )
        pred_frame_i = file.create_dataset(
            "prediction_detector_events/frame_i",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int64,
            chunks=(pred_chunk,),
        )
        pred_reduced = file.create_dataset(
            "prediction_detector_events/reduced_velocity",
            shape=(0,),
            maxshape=(None,),
            dtype=np.bool_,
            chunks=(pred_chunk,),
        )
        pred_reset = file.create_dataset(
            "prediction_detector_events/reset",
            shape=(0,),
            maxshape=(None,),
            dtype=np.bool_,
            chunks=(pred_chunk,),
        )
        pred_bad_run = file.create_dataset(
            "prediction_detector_events/bad_run",
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            chunks=(pred_chunk,),
        )
        pred_scale = file.create_dataset(
            "prediction_detector_events/command_scale",
            shape=(0,),
            maxshape=(None,),
            dtype=np.float32,
            chunks=(pred_chunk,),
        )
        pred_metrics = file.create_dataset(
            "prediction_detector_events/metrics",
            shape=(0, 7),
            maxshape=(None, 7),
            dtype=np.float32,
            chunks=(pred_chunk, 7),
        )
        pred_metrics.attrs["columns"] = np.array([
            "prediction_residual",
            "prediction_cosine",
            "obs_delta_rms",
            "pred_delta_rms",
            "pred_error_rms",
            "E_ref",
            "D_step",
        ], dtype="S")
        pred_u3 = file.create_dataset(
            "prediction_detector_events/u3",
            shape=(0, 3),
            maxshape=(None, 3),
            dtype=np.float32,
            chunks=(pred_chunk, 3),
        )
        pred_u3.attrs["columns"] = np.array(["u3x", "u3y", "u3yaw"], dtype="S")

        img_count = 0
        vw_count = 0
        vb_count = 0
        qp_count = 0
        pred_count = 0

        img_buf = []
        imgts_buf = []
        vw_buf = []
        vwts_buf = []
        vb_buf = []
        vbts_buf = []
        qp_lb_buf = []
        qp_ub_buf = []
        qp_ts_buf = []
        pred_ts_buf = []
        pred_frame_i_buf = []
        pred_reduced_buf = []
        pred_reset_buf = []
        pred_bad_run_buf = []
        pred_scale_buf = []
        pred_metrics_buf = []
        pred_u3_buf = []

        last_flush_t = time.time()

        def append_block(ds, ts_ds, data_block, ts_block, count):
            n = int(len(ts_block))
            if n == 0:
                return count
            ds.resize(count + n, axis=0)
            ts_ds.resize(count + n, axis=0)
            ds[count:count + n] = data_block
            ts_ds[count:count + n] = ts_block
            return count + n

        def append_array(ds, data_block, count):
            n = int(len(data_block))
            if n == 0:
                return count
            ds.resize(count + n, axis=0)
            ds[count:count + n] = data_block
            return count + n

        def flush_pred_events():
            nonlocal pred_count
            if not pred_ts_buf:
                return
            n0 = pred_count
            tsb = np.asarray(pred_ts_buf, dtype=np.float64)
            pred_ts.resize(n0 + len(tsb), axis=0)
            pred_ts[n0:n0 + len(tsb)] = tsb
            append_array(pred_frame_i, np.asarray(pred_frame_i_buf, dtype=np.int64), n0)
            append_array(pred_reduced, np.asarray(pred_reduced_buf, dtype=np.bool_), n0)
            append_array(pred_reset, np.asarray(pred_reset_buf, dtype=np.bool_), n0)
            append_array(pred_bad_run, np.asarray(pred_bad_run_buf, dtype=np.int32), n0)
            append_array(pred_scale, np.asarray(pred_scale_buf, dtype=np.float32), n0)
            append_array(pred_metrics, np.stack(pred_metrics_buf, axis=0).astype(np.float32), n0)
            append_array(pred_u3, np.stack(pred_u3_buf, axis=0).astype(np.float32), n0)
            pred_count = n0 + len(tsb)
            pred_ts_buf.clear()
            pred_frame_i_buf.clear()
            pred_reduced_buf.clear()
            pred_reset_buf.clear()
            pred_bad_run_buf.clear()
            pred_scale_buf.clear()
            pred_metrics_buf.clear()
            pred_u3_buf.clear()

        def maybe_flush(force=False):
            nonlocal last_flush_t
            now = time.time()
            if force or (now - last_flush_t) >= self.flush_every_s:
                file.flush()
                last_flush_t = now

        # Main consume loop
        while not self._stop.is_set() or not self._q.empty():
            try:
                kind, ts, payload = self._q.get(timeout=0.05)
            except queue.Empty:
                # Periodic flush even if idle
                maybe_flush(force=False)
                continue

            if kind == "img":
                img_buf.append(payload)
                imgts_buf.append(ts)
                if len(imgts_buf) >= self.image_batch:
                    block = np.stack(img_buf, axis=0)
                    tsb = np.asarray(imgts_buf, dtype=np.float64)
                    img_count = append_block(img_ds, img_ts, block, tsb, img_count)
                    img_buf.clear()
                    imgts_buf.clear()
                    maybe_flush(force=False)

            elif kind == "vw":
                vw_buf.append(payload)
                vwts_buf.append(ts)
                if len(vwts_buf) >= self.vel_batch:
                    block = np.stack(vw_buf, axis=0)
                    tsb = np.asarray(vwts_buf, dtype=np.float64)
                    vw_count = append_block(vw_ds, vw_ts, block, tsb, vw_count)
                    vw_buf.clear()
                    vwts_buf.clear()
                    maybe_flush(force=False)

            elif kind == "vb":
                vb_buf.append(payload)
                vbts_buf.append(ts)
                if len(vbts_buf) >= self.vel_batch:
                    block = np.stack(vb_buf, axis=0)
                    tsb = np.asarray(vbts_buf, dtype=np.float64)
                    vb_count = append_block(vb_ds, vb_ts, block, tsb, vb_count)
                    vb_buf.clear()
                    vbts_buf.clear()
                    maybe_flush(force=False)
                    
            elif kind == "qp_bounds":
                lb, ub = payload
                qp_lb_buf.append(lb)
                qp_ub_buf.append(ub)
                qp_ts_buf.append(ts)

                if len(qp_ts_buf) >= self.vel_batch:
                    lb_block = np.stack(qp_lb_buf, axis=0)
                    ub_block = np.stack(qp_ub_buf, axis=0)
                    tsb = np.asarray(qp_ts_buf, dtype=np.float64)

                    qp_count = append_block(qp_lb_ds, qp_lb_ts, lb_block, tsb, qp_count)
                    append_block(qp_ub_ds, qp_ub_ts, ub_block, tsb, qp_count - len(tsb))

                    qp_lb_buf.clear()
                    qp_ub_buf.clear()
                    qp_ts_buf.clear()
                    maybe_flush(force=False)

            elif kind == "pred_event":
                payload = dict(payload)
                pred_ts_buf.append(ts)
                pred_frame_i_buf.append(int(payload.get("frame_i", -1)))
                pred_reduced_buf.append(bool(payload.get("reduced_velocity", False)))
                pred_reset_buf.append(bool(payload.get("reset", False)))
                pred_bad_run_buf.append(int(payload.get("bad_run", 0)))
                pred_scale_buf.append(float(payload.get("command_scale", 1.0)))
                pred_metrics_buf.append(np.asarray([
                    payload.get("prediction_residual", np.nan),
                    payload.get("prediction_cosine", np.nan),
                    payload.get("obs_delta_rms", np.nan),
                    payload.get("pred_delta_rms", np.nan),
                    payload.get("pred_error_rms", np.nan),
                    payload.get("E_ref", np.nan),
                    payload.get("D_step", np.nan),
                ], dtype=np.float32))
                pred_u3_buf.append(np.asarray(payload.get("u3", np.zeros(3, dtype=np.float32)), dtype=np.float32).reshape(3))

                if len(pred_ts_buf) >= self.vel_batch:
                    flush_pred_events()
                    maybe_flush(force=False)

            self._q.task_done()

        # Drain remaining buffers
        if imgts_buf:
            block = np.stack(img_buf, axis=0)
            tsb = np.asarray(imgts_buf, dtype=np.float64)
            img_count = append_block(img_ds, img_ts, block, tsb, img_count)

        if vwts_buf:
            block = np.stack(vw_buf, axis=0)
            tsb = np.asarray(vwts_buf, dtype=np.float64)
            vw_count = append_block(vw_ds, vw_ts, block, tsb, vw_count)

        if vbts_buf:
            block = np.stack(vb_buf, axis=0)
            tsb = np.asarray(vbts_buf, dtype=np.float64)
            vb_count = append_block(vb_ds, vb_ts, block, tsb, vb_count)
        
        
        if qp_ts_buf:
            lb_block = np.stack(qp_lb_buf, axis=0)
            ub_block = np.stack(qp_ub_buf, axis=0)
            tsb = np.asarray(qp_ts_buf, dtype=np.float64)

            qp_count = append_block(qp_lb_ds, qp_lb_ts, lb_block, tsb, qp_count)
            append_block(qp_ub_ds, qp_ub_ts, ub_block, tsb, qp_count - len(tsb))

        flush_pred_events()

        maybe_flush(force=True)
        file.close()
