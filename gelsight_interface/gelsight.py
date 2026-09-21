import cv2
import os
import re
import logging
import numpy as np
import subprocess
import tempfile
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.FileHandler("gelsight.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("Gelsight")
_DISPLAY_WINDOW_SIZES = {}

class camera:
    def get_camera_name(self, camera_sn):
        if camera_sn is None:
            logger.warning("No camera serial number provided")
            return None
        else:
            name = "GelSight Mini R0B " + camera_sn
            logger.info("Camera SN: {}".format(name))
            return name

    def get_camera_id(self, camera_name):
        cam_num = None
        if camera_name is None:
            logger.warning("No camera name provided")
            return None
        for file in os.listdir("/sys/class/video4linux"):
            real_file = os.path.realpath("/sys/class/video4linux/" + file + "/name")
            with open(real_file, "rt") as name_file:
                name = name_file.read().rstrip()
            if camera_name in name:
                cam_num = int(re.search(r"\d+$", file).group(0))
                found = "FOUND!"
            else:
                found = "      "
            logger.info("{} {} -> {}".format(found, file, name))
        return cam_num
    
    def _build_ffmpeg_cmd(
        self,
        device_path: str,
        raw_w: int,
        raw_h: int,
        fps: int,
        video_filter: str | None = None,
    ):
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel", "error",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-analyzeduration", "0",
            "-probesize", "32",
            "-f", "v4l2",
            "-input_format", "mjpeg",
            "-framerate", str(fps),
            "-video_size", f"{raw_w}x{raw_h}",
            "-thread_queue_size", "1",
            "-i", device_path,
        ]
        if video_filter is not None:
            cmd.extend(["-vf", video_filter])
        cmd.extend([
            "-pix_fmt", "bgr24",
            "-threads", "1",
            "-f", "rawvideo",
            "pipe:1",
        ])
        return cmd

    def _get_ffmpeg_output_config(self, raw_w: int, raw_h: int):
        crop_ratio = self.crop_to_resize_ratio
        if not 0 < crop_ratio <= 1:
            raise ValueError("Crop to resize ratio must be between 0 and 1")

        out_w, out_h = self.width, self.height
        crop_w = max(1, int(raw_w * crop_ratio))
        crop_h = max(1, int(raw_h * crop_ratio))
        if crop_w < raw_w and crop_w % 2:
            crop_w = max(1, crop_w - 1)
        if crop_h < raw_h and crop_h % 2:
            crop_h = max(1, crop_h - 1)

        left = max(0, (raw_w - crop_w) // 2)
        top = max(0, (raw_h - crop_h) // 2)
        if crop_w == raw_w and crop_h == raw_h and out_w == raw_w and out_h == raw_h:
            return None, out_w, out_h

        video_filter = (
            f"crop={crop_w}:{crop_h}:{left}:{top},"
            f"scale={out_w}:{out_h}:flags=fast_bilinear"
        )
        return video_filter, out_w, out_h
        
    def _read_exact_into(self, buf: memoryview) -> bool:
        stdout = self._ffmpeg_proc.stdout
        if stdout is None:
            return False

        mv = buf
        frame_t0 = time.monotonic()
        while mv and self._run:
            t0 = time.monotonic()
            n = stdout.readinto(mv)
            t1 = time.monotonic()

            if not n:
                return False

            dt = t1 - t0
            self._stall_time_s += dt
            if dt > self._stall_max_s:
                self._stall_max_s = dt

            mv = mv[n:]

        frame_dt = time.monotonic() - frame_t0
        if frame_dt > self._frame_read_max_s:
            self._frame_read_max_s = frame_dt

        if mv:
            return False
        self._bytes_read_total += len(buf)
        return True

    def _ffmpeg_capture_loop(self):
        while self._run:
            buf = self._buf_a if self._use_a else self._buf_b
            mv = memoryview(buf)
            if not self._read_exact_into(mv):
                break
            t_cap = time.monotonic()

            frame = np.frombuffer(buf, np.uint8).reshape((self._ffmpeg_out_h, self._ffmpeg_out_w, 3))

            with self._latest_lock:
                self._latest = frame
                self._latest_seq += 1
                self._latest_t_cap = t_cap

            self._use_a = not self._use_a
            self._frame_counter += 1
        
class Gelsight(camera):
    def __init__(self, camera_sn, size=(480 // 2, 640 // 2), crop_to_resize_ratio=0.75):
        self.camera_sn = camera_sn
        self.camera_name = self.get_camera_name(self.camera_sn)
        self.camera_id = self.get_camera_id(self.camera_name)
        self.max_size = (3280, 2464)
        frame_height, frame_width = size
        if frame_height is None:
            logger.warning("No height provided, using default value of 2464")
            self.height = self.max_size[1]
        else:
            self.height = int(frame_height)
        if frame_width is None:
            logger.warning("No width provided, using default value of 3280")
            self.width = self.max_size[0]
        else:
            self.width = int(frame_width)
        
        self.crop_to_resize_ratio = crop_to_resize_ratio
        
        self._is_connected = False
        self.background = None
        
        self._use_ffmpeg = False
        self._ffmpeg_proc = None
        self._ffmpeg_stderr = None
        self._last_ffmpeg_error = ""
        self._ffmpeg_raw_w = 3280
        self._ffmpeg_raw_h = 2464
        self._ffmpeg_out_w = self._ffmpeg_raw_w
        self._ffmpeg_out_h = self._ffmpeg_raw_h
        self._ffmpeg_fps = 25
        self._ffmpeg_frame_bytes = self._ffmpeg_raw_w * self._ffmpeg_raw_h * 3
        self._latest_seq = 0 
        self._latest_t_cap = 0.0
        
    def connect(self, use_ffmpeg=True, raw_w=3280, raw_h=2464, fps=25, warmup_frames=20):
        self.release()
        self._last_ffmpeg_error = ""
        self._is_connected = False
        self._use_ffmpeg = bool(use_ffmpeg)

        if not self._use_ffmpeg:
            # --- your existing OpenCV connect path ---
            self.cap = cv2.VideoCapture(self.camera_id)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not self.cap.isOpened():
                raise IOError("Cannot open webcam")
            self._is_connected = True
            return

        # --- ffmpeg path ---
        self._ffmpeg_raw_w = int(raw_w)
        self._ffmpeg_raw_h = int(raw_h)
        ffmpeg_filter, self._ffmpeg_out_w, self._ffmpeg_out_h = self._get_ffmpeg_output_config(
            self._ffmpeg_raw_w,
            self._ffmpeg_raw_h,
        )
        self._ffmpeg_fps = int(fps)
        self._ffmpeg_frame_bytes = self._ffmpeg_out_w * self._ffmpeg_out_h * 3
        self._buf_a = bytearray(self._ffmpeg_frame_bytes)
        self._buf_b = bytearray(self._ffmpeg_frame_bytes)
        self._use_a = True

        device_path = f"/dev/video{self.camera_id}"
        if not os.path.exists(device_path):
            raise IOError(f"Device not found: {device_path}")

        cmd = self._build_ffmpeg_cmd(
            device_path,
            self._ffmpeg_raw_w,
            self._ffmpeg_raw_h,
            self._ffmpeg_fps,
            ffmpeg_filter,
        )

        # Use a file for diagnostics so an unread stderr pipe cannot stall FFmpeg.
        self._ffmpeg_stderr = tempfile.TemporaryFile()
        try:
            self._ffmpeg_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=self._ffmpeg_stderr,
                bufsize=0,
            )
            if self._ffmpeg_proc.stdout is None:
                raise IOError("Failed to open ffmpeg stdout pipe")

            self._latest = None
            self._latest_seq = 0
            self._latest_lock = threading.Lock()
            self._run = True
            self._frame_counter = 0
            self._bytes_read_total = 0
            self._last_read_time = time.monotonic()
            self._stall_count = 0
            self._stall_time_s = 0.0
            self._stall_max_s = 0.0
            self._frame_read_max_s = 0.0

            self._t = threading.Thread(target=self._ffmpeg_capture_loop, daemon=True)
            self._t.start()

            deadline = time.monotonic() + 2.0
            while self._latest is None and time.monotonic() < deadline:
                if self._ffmpeg_proc.poll() is not None:
                    break
                time.sleep(0.005)

            if self._latest is not None:
                for _ in range(max(0, int(warmup_frames))):
                    if self._ffmpeg_proc.poll() is not None:
                        break
                    time.sleep(1.0 / max(1, self._ffmpeg_fps))

            returncode = self._ffmpeg_proc.poll()
            if self._latest is None or returncode is not None:
                self.release()
                reason = self._last_ffmpeg_error or "No frames arrived before the startup timeout."
                status = f" (exit {returncode})" if returncode is not None else ""
                raise IOError(f"FFmpeg capture failed for {device_path}{status}: {reason}")

            self._is_connected = True
        except BaseException:
            self.release()
            raise

        logger.info("Connected to the sensor with ID: {}".format(self.camera_id))
        logger.info("Camera resolution: {} x {}".format(self.get_width(), self.get_height()))
        
    def get_height(self):
        if self._is_connected:
            logger.info("Height: {}".format(self.height))
        return self.height
    
    def get_width(self):
        if self._is_connected:
            logger.info("Width: {}".format(self.width))
        return self.width
    
    def set_height(self, height):
        if self._is_connected:
            self.height = height
            logger.info("New height: {}".format(self.height))

    def set_width(self, width):
        if self._is_connected:
            self.width = width
            logger.info("New width: {}".format(self.width))
    
    def get_frame(self):
        frame, _ = self.get_frame_with_counter(copy=True)
        return frame
    
    def get_frame_with_meta(self, copy=False):
        with self._latest_lock:
            f = self._latest
            if f is None:
                return None, 0, 0.0
            seq = self._latest_seq
            t_cap = self._latest_t_cap

        # copy OUTSIDE lock
        return (f.copy() if copy else f), seq, t_cap
        
    def get_frame_with_counter(self, copy: bool = True):
        """
        Returns (frame, counter).
        - counter increments only when a truly new frame is captured (ffmpeg thread).
        - copy=True keeps existing safe behavior.
        - copy=False is faster but returns a reference; treat it as READ-ONLY.
        """
        if not self._is_connected:
            logger.error("Not connected to the sensor")
            raise IOError("Not connected to the sensor")

        if not self._use_ffmpeg:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                return None, getattr(self, "_frame_counter", 0)

            if (self.width, self.height) != self.max_size:
                frame = self.resize_frame(frame, (self.width, self.height))

            # Maintain a counter for the OpenCV path too (so caller logic is uniform)
            self._frame_counter = getattr(self, "_frame_counter", 0) + 1
            return (frame.copy() if copy else frame), self._frame_counter

        with self._latest_lock:
            f = self._latest
            c = self._frame_counter
        # copy OUTSIDE lock
        if f is None:
            return None, c
        return (f.copy() if copy else f), c
        
    def resize_frame(self, frame, size=None, crop_to_resize_ratio=None):
        if size is None:
            size = (self.width, self.height)
        if crop_to_resize_ratio is None:
            crop_to_resize_ratio = self.crop_to_resize_ratio
        
        h, w = frame.shape[:2]
        new_h, new_w = int(h * crop_to_resize_ratio), int(w * crop_to_resize_ratio)
        top = (h - new_h) // 2
        left = (w - new_w) // 2
        frame = frame[top:top + new_h, left:left + new_w]
        frame = cv2.resize(frame, size)
        #print(f"Resized frame to {size} with crop ratio {crop_to_resize_ratio}")
        #print(f"Original frame shape: {frame.shape}, Resized frame shape: {frame.shape}")
        return frame
    
    def set_crop_to_resize_ratio(self, ratio):
        if ratio < 0 or ratio > 1:
            logger.error("Invalid crop to resize ratio: {}".format(ratio))
            raise ValueError("Crop to resize ratio must be between 0 and 1")
        self.crop_to_resize_ratio = ratio
        logger.info("New crop to resize ratio: {}".format(self.crop_to_resize_ratio))

    def get_crop_to_resize_ratio(self):
        return self.crop_to_resize_ratio
    
    def show_frame(self, record=False, record_path=None):
        if record:
            logger.info("Video will be recorded")
            frames = []
            size = (self.get_width(), self.get_height())
            out = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*'DIVX'), 25, size)
            if record_path is None:
                logger.error("No path specified to save the video")
                raise IOError("Please specify the path to save the video")

        window_name = 'Gelsight Camera'
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, self.get_width(), self.get_height())

        last_ctr = -1
        try:
            while True:
                frame, ctr = self.get_frame_with_counter(copy=False)
                if frame is None or ctr == last_ctr:
                    if (cv2.waitKey(1) & 0xFF) == ord('q'):
                        logger.info("User pressed 'q' to quit")
                        break
                    time.sleep(0.001)
                    continue
                last_ctr = ctr
                cv2.imshow(window_name, frame)
                if record:
                    frames.append(frame.copy())
                if (cv2.waitKey(1) & 0xFF) == ord('q'):
                    logger.info("User pressed 'q' to quit")
                    break
        finally:   
            cv2.destroyAllWindows()
            _DISPLAY_WINDOW_SIZES.clear()
        
            if record:
                logger.info("Saving the video to {}".format(record_path))
                for f in frames:
                    out.write(f)
                out.release()
                logger.info("Video saved successfully")

    def release(self):
        self._run = False

        # Stop and reap FFmpeg before joining the reader: this unblocks readinto
        # and ensures the device is released before another trial opens it.
        proc = self._ffmpeg_proc
        if proc is not None:
            if proc.poll() is None:
                try:
                    proc.terminate()
                except ProcessLookupError:
                    pass
            try:
                proc.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait(timeout=1.0)

        if getattr(self, "_t", None) is not None:
            self._t.join(timeout=1.0)
            self._t = None

        if proc is not None and proc.stdout is not None:
            proc.stdout.close()
        self._ffmpeg_proc = None

        if self._ffmpeg_stderr is not None:
            self._ffmpeg_stderr.seek(0, os.SEEK_END)
            size = self._ffmpeg_stderr.tell()
            self._ffmpeg_stderr.seek(max(0, size - 8192))
            self._last_ffmpeg_error = self._ffmpeg_stderr.read().decode(errors="replace").strip()
            self._ffmpeg_stderr.close()
            self._ffmpeg_stderr = None

        # Close OpenCV
        if hasattr(self, "cap") and self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

        self._is_connected = False

    @staticmethod
    def display_image(image, name='Gelsight Camera'):
        cv2.namedWindow(name, cv2.WINDOW_NORMAL)
        size = (image.shape[1], image.shape[0])
        if _DISPLAY_WINDOW_SIZES.get(name) != size:
            cv2.resizeWindow(name, size[0], size[1])
            _DISPLAY_WINDOW_SIZES[name] = size
        cv2.imshow(name, image)

    @staticmethod
    def save_image(image, path):
        cv2.imwrite(path, image)
        logger.info("Image saved to {}".format(path))
        
    def get_frame_counter(self):
        return getattr(self, "_frame_counter", 0)
        
    def get_background(self, num_frames=10):
        logger.info("Getting background. Do not touch the sensor...")
        frames = []
        last = self.get_frame_counter()

        while len(frames) < num_frames:
            # wait for a new frame
            while self.get_frame_counter() == last:
                time.sleep(0.001)
            last = self.get_frame_counter()

            f = self.get_frame()
            if f is not None:
                frames.append(f.copy())  # copy so later updates don't affect stored frames

        self.background = np.mean(np.stack(frames, axis=0), axis=0).astype(np.uint8)
        logger.info("Background computed successfully")
        return self.background
