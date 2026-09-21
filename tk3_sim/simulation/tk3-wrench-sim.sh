#!/bin/sh

# settings
middleware=pocolibs # or ros
components="
  rotorcraft
  uavpos
  uavatt
  pom
  optitrack
  maneuver
  phynt
  nhfc
"
SIM_REPO_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
gzworld="$SIM_REPO_DIR/worlds/tilthex-wrench.world"

# Docker sets OR_INSTALL_PREFIX; native robotpkg setups use ROBOTPKG_BASE.
openrobots_prefix=${OR_INSTALL_PREFIX:-${ROBOTPKG_BASE:-/opt/openrobots}}
export GZ_SIM_RESOURCE_PATH="${SIM_REPO_DIR}/models:${openrobots_prefix}/share/gazebo/models${GZ_SIM_RESOURCE_PATH:+:$GZ_SIM_RESOURCE_PATH}"
# Prefer installed libraries (including Docker's own build) over local binaries.
export GZ_SIM_SYSTEM_PLUGIN_PATH="${openrobots_prefix}/lib/gazebo:${SIM_REPO_DIR}/gazebo_plugin/build${GZ_SIM_SYSTEM_PLUGIN_PATH:+:$GZ_SIM_SYSTEM_PLUGIN_PATH}"

printf 'Gazebo models: %s\n' "$GZ_SIM_RESOURCE_PATH"
printf 'Gazebo plugins: %s\n' "$GZ_SIM_SYSTEM_PLUGIN_PATH"

# list of process ids to clean, populated after each spawn
pids=

# cleanup, called after ctrl-C
atexit() {
    status=$?
    trap - 0 INT TERM
    set +e

    [ -z "$pids" ] || kill $pids
    wait
    case $middleware in
        pocolibs) h2 end;;
    esac
    exit "$status"
}
trap atexit 0 INT
trap 'exit 143' TERM
set -e

# init middleware
mkdir -p "$HOME"
case $middleware in
    pocolibs) h2 init;;
    ros) roscore & pids="$pids $!";;
esac

# optionally run a genomix server for remote control
genomixd & pids="$pids $!"

# spawn required components
for c in $components; do
    $c-$middleware & pids="$pids $!"
done

# start gazebo
gz sim -r "$gzworld" "$@" &
gz_pid=$!
pids="$pids $gz_pid"
# ---- Move Gazebo window (X11) ----
if [ -n "${DISPLAY:-}" ] && command -v wmctrl >/dev/null 2>&1; then
set +e

# Wait until Gazebo GUI window appears (match class)
GZ_WIN=""
i=0
while [ -z "$GZ_WIN" ] && [ $i -lt 200 ]; do
  GZ_WIN=$(wmctrl -lx | awk '$3=="gz-sim-gui.Gazebo" {print $1; exit}')
  [ -z "$GZ_WIN" ] && GZ_WIN=$(wmctrl -lx | awk '$3=="gz-sim-gui.Gazebo\ GUI" {print $1; exit}')  # fallback
  i=$((i+1))
  sleep 0.1
done

if [ -n "$GZ_WIN" ]; then
  wmctrl -i -r "$GZ_WIN" -e 0,1920,0,-1,-1
  wmctrl -i -r "$GZ_WIN" -b add,maximized_vert,maximized_horz
else
  echo "Warning: Gazebo GUI window not found; skipping move."
fi

set -e
fi
# ----------------------------------

# Stop the components when Gazebo exits, preserving its exit status.
wait "$gz_pid"
