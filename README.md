# Aerial tactile servoing

Tilthex simulation and TeleKyb3 controllers for aerial tactile interaction.

- **Gazebo Ionic + TeleKyb3:** a Docker image containing the robot world, robot
  plugins, GenoM components, and a constant-wrench plugin.
- **Incremental DTS:** a host Python controller that processes a physical
  GelSight camera through a TensorRT grayscale network and commands sensor x, y,
  and yaw while holding altitude.
- **PHYNT admittance:** a comparison controller using the wrench observer and
  admittance filter, without a GelSight camera or neural network.

The simulator runs in Docker; the Python controllers run on the host and connect
through GenoMix. Only the simulated robot is supported. DTS still requires a
physical GelSight camera; the PHYNT baseline needs no camera.
All commands below are run from the repository root unless stated otherwise.

## 1. Get the repository

Install Git LFS before cloning; the inference models are stored in LFS.

```sh
git lfs install
git clone --branch cleanup/aerial-dts-baseline --recurse-submodules https://github.com/aerial-tactile-servoing/aerial_tactile_servoing.git
cd aerial_tactile_servoing
```

Access to the repository and the `gelsight_interface` submodule is required.

## 2. Start the Docker simulator

Requirements: Linux x86_64, Docker accessible to your user, a graphical
X11/XWayland session, and `xauth`. The image build needs internet access. On Ubuntu:

```sh
sudo apt install xauth
sh tk3_sim/simulation/build.sh
sh tk3_sim/simulation/run.sh
```

The image is named `aerial-robotics-laboratory:ionic`. The launcher opens the
Tilthex world and starts `genomixd`, `rotorcraft`, `uavpos`, `uavatt`, `pom`,
`optitrack`, `maneuver`, `phynt`, and `nhfc`. This starts simulation and component
services; takeoff is requested separately from a controller shell.

The container uses host networking, your user ID, and your X11 authorization. It
mounts your home and repository at their host paths, and starts components from
the repository root so `results/...` resolves identically on both sides.
Run one simulator instance at a time; a native TeleKyb session can use the same
GenoMix and OptiTrack ports.

To open a container shell in another terminal:

```sh
docker exec -it aerial-robotics-laboratory bash
```

To close Gazebo or stop the container cleanly:

```sh
docker stop aerial-robotics-laboratory
```

The stopped container is removed automatically. Host log files remain. A forced
kill can lose the last buffered samples, so use normal shutdown when possible.

## 3. Prepare the host Python environment

DTS requires an NVIDIA GPU, a suitable driver, PyTorch with CUDA, Torch-TensorRT,
and a TensorRT runtime compatible with the supplied engine. Install a matching
combination using the [Torch-TensorRT installation guide](https://docs.pytorch.org/TensorRT/v2.9.0/getting_started/installation.html).
The development workstation currently uses PyTorch `2.9.0+cu128`, Torch-TensorRT
`2.9.0+cu128`, and TensorRT `10.13.3.9`; these are a reference, not a guarantee that
an engine will load on another GPU.

The desktop Conda environment snapshot is an optional starting point:

```sh
conda env create -f dts/environments/visp-gpu-ws-environment.yaml
conda activate visp-gpu-ws
```

### Models and calibration

DTS reads these paths relative to the repository:

| File in `dts/models/` | Purpose |
| --- | --- |
| `compiled_model.trt` | Desktop TorchScript/TensorRT grayscale model |
| `camera_<SERIAL_WITH_UNDERSCORES>.xml` | Camera intrinsics; for example `camera_2DDW_JXDC.xml` |
| `background.png` | Untouched-sensor background for contact detection |
| `gs_model_concat_501.pth` | Network weights; not loaded directly by the DTS entry point |

The `.trt` model is loaded with `torch.jit.load`. TensorRT engine compatibility
depends on the runtime and GPU; see [NVIDIA's compatibility documentation](https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/version-compatibility.html).
Obtain or rebuild a matching engine if deserialization fails. This repository
currently has no command-line engine export script.

The controller expects a 640 × 480 BGR image after a centered 0.75 crop of the
camera's 3280 × 2464 MJPEG stream at 25 fps. Camera intrinsics and background must
match this processing and the sensor being used. Mounting transforms are described
in [dts/mounting.txt](dts/mounting.txt).

To capture an untouched-sensor background with the same processing, set your
serial and run on the host:

```sh
python - <<'PY'
import cv2
from gelsight_interface.gelsight_interface.gelsight import Gelsight

sensor = Gelsight("2DDW-JXDC", (480, 640))
try:
    sensor.connect()
    background = sensor.get_background(num_frames=10)
    if not cv2.imwrite("dts/models/background.png", background):
        raise OSError("Could not save background.png")
finally:
    sensor.release()
PY
```

## 4. Run incremental DTS

Keep the simulator running in its terminal. In a host terminal with the Python
and inference dependencies available:

```sh
python -m tk3_sim.core.dts_ik \
  --sn 2DDW-JXDC --tag demo
```

An interactive Python shell opens after component setup. To run a flight trial:

```python
runtime.check_pom()
dts(takeoff=True)
```

A trial prepares camera/inference resources, starts logs and motors, takes off,
and runs until its duration expires or contact is lost after initialization.
It then requests landing and component shutdown. Ctrl-C interrupts a trial and
runs its cleanup; Ctrl-D exits the shell.

For camera processing and logging without starting motors or transmitting DTS
motion commands, launch with `--no-motors` and call `dts()`:

```sh
python -m tk3_sim.core.dts_ik \
  --sn 2DDW-JXDC --no-motors --tag camera_only
```

Without `--no-motors`, `dts()` assumes flight/component state is already managed
by the caller: it can send commands but does not perform the takeoff sequence.
Use `dts(takeoff=True)` for a complete trial. `last_trial` in the shell contains
its paths and interruption status.

Common options:

| Option | Default | Meaning |
| --- | --- | --- |
| `--duration` | `30` | Trial time in seconds, including initial contact waiting |
| `--takeoff-height` / `--land-height` | `1.35` / `0.25` | Altitude references in meters |
| `--mass` | `3.1` | Robot mass in kg |
| `--threshold-first-touch` | `1000` | Norm threshold against the BGR background |
| `--desired-delay` | `2` | Delay after first contact before setting the reference |
| `--lambda-x`, `--lambda-y`, `--lambda-yaw` | `1000`, `1000`, `20` | Incremental controller gains |
| `--mu-ik` / `--sign` | `0.2` / `-1` | Damping and command sign |
| `--vmax-translation` / `--wmax-rotation` | `0.4` / `0.2` | Sensor XY and yaw command limits |

Each `dts()` call creates a new trial folder. You can edit `args` in the shell
before the next trial; runtime configuration values set during setup require a
new controller session.

## 5. Run the PHYNT admittance baseline

This controller needs the simulator and the host NumPy/SciPy/GenoMix dependencies,
but no GelSight, PyTorch, or TensorRT:

```sh
python -m tk3_sim.core.admittance_phynt --tag admittance
```

In its interactive shell:

```python
runtime.check_pom()
admittance(takeoff=True)
```

PHYNT proposes x/y/yaw motion; the relay fixes altitude and removes roll/pitch
commands. The shell exposes `args`, `runtime`, and
`configure_current_phynt_admittance()` for adjusting and reapplying gains.

**Current behavior:** stop the loop with Ctrl-C. Its `--duration` value drives the
countdown but currently does not terminate the loop. The inner loop always lands
and stops components on exit; the outer `land` argument and `--no-land` option do
not override that cleanup. These existing flight semantics have been retained.

## 6. Apply a simulated external wrench

Inside the running container, set a 1 N force along world X with zero torque:

```sh
gz topic -t /model/hr6/external_wrench_cmd \
  -m gz.msgs.Wrench \
  -p 'force: {x: 1, y: 0, z: 0}, torque: {x: 0, y: 0, z: 0}'
gz topic -t /model/hr6/external_wrench_enable \
  -m gz.msgs.Boolean -p 'data: true'
```

The plugin starts disabled. It applies the last wrench continuously while enabled
to the robot's `base` link in world coordinates. Force is in N and torque in N·m.
Disable it with:

```sh
gz topic -t /model/hr6/external_wrench_enable \
  -m gz.msgs.Boolean -p 'data: false'
```

## 7. Logs

Trial folders are `results/logs_<timestamp>_<tag>/`. Tags are sanitized and shortened
to fit GenoM's 64-character filename limit. Keep component paths relative.

- `gains.txt`: controller settings as JSON.
- `mocap.log`, `rotorcraft.log`, `pom*.log`, `uavpos.log`, `uavatt.log`,
  `maneuver.log`, `phynt.log`: component output; `phynt.log` is written when PHYNT is enabled.
- `tactile_log_*.h5`: DTS camera images, world/body command velocities, and timestamps.
- `background_bgr.png`: background used by the DTS trial.

The HDF5 datasets used by DTS are `images/data`, `images/timestamp`,
`world_velocities/data`, `world_velocities/timestamp`, `body_velocities/data`, and
`body_velocities/timestamp`. Images are BGR uint8; velocities have six components.
Legacy logger datasets may exist without samples. The admittance baseline writes
component logs and gains, not tactile HDF5 recordings.

Docker logs are written directly into the host checkout and survive container
removal. Let the trial finish cleanup before closing Python to flush queued HDF5
records. No log-copy step is needed.

## 8. Native simulation and plugin paths

Install Gazebo Ionic and TeleKyb3 locally, including development packages for
`gz-sim9`, `gz-plugin3`, and `gz-transport14`, a C++ compiler, CMake, and cppzmq
headers. Load your usual TeleKyb environment so `gz`, `h2`, and component binaries
are on `PATH`. Then:

```sh
export ROBOTPKG_BASE="$HOME/openrobots"  # your local installation
cmake -S tk3_sim/simulation/gazebo_plugin \
  -B tk3_sim/simulation/gazebo_plugin/build \
  -DCMAKE_PREFIX_PATH="$ROBOTPKG_BASE" -DCMAKE_BUILD_TYPE=Release
cmake --build tk3_sim/simulation/gazebo_plugin/build --parallel 2
sh tk3_sim/simulation/tk3-wrench-sim.sh
```

`GZ_SIM_SYSTEM_PLUGIN_PATH` contains installed plugins and `gazebo_plugin/build`;
`GZ_SIM_RESOURCE_PATH` contains repository and installed robot models. The prefix
comes from `OR_INSTALL_PREFIX`, then `ROBOTPKG_BASE`, then `/opt/openrobots`.
Existing search paths are preserved. Docker uses the plugin built inside its image;
native simulation uses libraries built for the host. These are Gazebo paths,
separate from the GenoM component-client search path.

The controllers connect to GenoMix on `localhost`. The simulation supplies the
OptiTrack stream on port `1509`, the tracked body `HR_6`, and the rotorcraft device
`/tmp/pty-hr6`. Launch the native components from the repository root so their
relative log paths resolve to the same `results/` folder as Python.

GenoMix normally uses its installation's component-client search path. If your
native installation needs an override, set `TK3_GENOM_PLUGIN_PATH` to the plugin
directory on the **GenoMix server**. With Docker, this must be a path inside the
container; a host-only library path will not work.

## Troubleshooting

Controller setup waits for live `HR_6` mocap data before connecting it to POM.
If discovery fails, setup stops with an error instead of continuing with only IMU
data. In a successful trial, `pom-measurements.log` includes both `imu` and `mocap`
measurements. After updating the Python runtime, restart the controller session;
this change does not require a Docker image rebuild.

| Symptom | Action |
| --- | --- |
| `DISPLAY` or X11 authorization error | Launch `run.sh` from a graphical host terminal with `xauth` installed. |
| Container name already in use | Stop the old simulator before starting another. |
| `ModuleNotFoundError: genomix` | Install/copy the Python client into the interpreter used for the controller. |
| Component plugin cannot load | Check the server's plugin installation and `TK3_GENOM_PLUGIN_PATH`. |
| Missing `libconstant_wrench_system.so` | Rebuild the Docker image, or compile the native plugin and use the matching launcher. |
| Log `e_access: No such file or directory` | Restart using the current `run.sh`; host and components must share the repository working directory. |
| GelSight is busy | Use `fuser /dev/videoN` to identify the owner; release/close that capture session before retrying. |
| GelSight gives no frames | Read the FFmpeg error in the exception; check camera serial, device permissions, and MJPEG resolution/fps with `v4l2-ctl`. |
| TensorRT model does not load | Match the engine's GPU/runtime and Torch-TensorRT stack; obtain/rebuild a compatible model. |
| Wrong contact detection or image shape | Recapture an untouched background with the controller's crop and resolution. |

Only one process should capture a GelSight device. The camera interface terminates
and waits for FFmpeg during release, and includes capture errors in exceptions.
A Python process killed without cleanup can still leave a capture process behind;
identify the exact process before stopping it. Gazebo plugin paths do not control
USB camera access.

## Code layout and editing

| Path | Responsibility |
| --- | --- |
| `tk3_sim/core/tk3_runtime.py` | GenoM connection, robot configuration, component lifecycle, motion commands |
| `tk3_sim/core/dts_ik.py` | Tactile preprocessing, incremental solver, interactive DTS trials |
| `tk3_sim/core/admittance_phynt.py` | PHYNT reference publishing and admittance relay |
| `tk3_sim/core/utils.py` | Explicit frame transforms and twist integration |
| `tk3_sim/core/log_paths.py` | Shared trial naming, local directory creation, log locations |
| `tk3_sim/simulation/` | Docker build, launch scripts, world, models, wrench-plugin source |
| `gelsight_interface/` | Camera submodule |
| `mylogger.py` | Asynchronous HDF5 writer |
| `dts/models/`, `dts/environments/` | Models/calibration and environment snapshots |

Use the explicit transform names in `utils.py`. Restart the Python session after
editing core modules. Rebuild the image after changes to its Dockerfile, plugins,
models, world, or bundled startup script. Changes to the host `run.sh` only require
restarting the container. Native plugin changes require a new CMake build.
