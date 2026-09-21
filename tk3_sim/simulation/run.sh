#!/bin/sh
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
repo_root=$(CDPATH= cd -- "$script_dir/../.." && pwd)
: "${DISPLAY:?Run this command from a terminal on your Linux desktop.}"
command -v xauth >/dev/null || { echo 'Install xauth: sudo apt install xauth' >&2; exit 1; }
docker image inspect aerial-robotics-laboratory:ionic >/dev/null 2>&1 || {
    echo "Build first: sh $(dirname -- "$0")/build.sh" >&2; exit 1;
}

auth_file=$(mktemp)
trap 'rm -f "$auth_file"' EXIT HUP INT TERM
# Authorize only this display without changing the host's X server permissions.
xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$auth_file" nmerge -
[ -s "$auth_file" ] || { echo 'No X11 authorization found for this desktop.' >&2; exit 1; }

docker run --rm --init --name aerial-robotics-laboratory --shm-size=1g --network=host \
    --user "$(id -u):$(id -g)" \
    --workdir "$repo_root" \
    --mount "type=bind,src=$repo_root,dst=$repo_root" \
    -e DISPLAY -e XAUTHORITY=/tmp/display.xauth \
    -e HOME \
    --mount type=bind,src=/tmp/.X11-unix,dst=/tmp/.X11-unix,readonly \
    --mount "type=bind,src=$auth_file,dst=/tmp/display.xauth,readonly" \
    --mount "type=bind,src=$HOME,dst=$HOME" \
    aerial-robotics-laboratory:ionic "$@"
