FROM ubuntu:24.04 AS runtime

ARG UBUNTU_MIRROR=http://fr.archive.ubuntu.com/ubuntu
ENV DEBIAN_FRONTEND=noninteractive

RUN sed -i -E "s#http://(archive|security).ubuntu.com/ubuntu#${UBUNTU_MIRROR}#g" \
      /etc/apt/sources.list.d/ubuntu.sources && \
    apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && \
    curl -fsSL https://packages.osrfoundation.org/gazebo.gpg \
      -o /usr/share/keyrings/gazebo.gpg && \
    echo 'deb [signed-by=/usr/share/keyrings/gazebo.gpg] https://packages.osrfoundation.org/gazebo/ubuntu-stable noble main' \
      > /etc/apt/sources.list.d/gazebo.list && \
    rm -rf /var/lib/apt/lists/*

# Ionic runtime only: no class desktop, tutorials, editors or TeleKyb3 tools.
RUN apt-get update && apt-get install -y --no-install-recommends \
      gz-tools2 gz-sim9-cli libgz-sim9-plugins \
      libgz-physics8-dartsim libgz-rendering9-ogre2 \
      libgl1-mesa-dri fonts-dejavu-core \
      qml-module-qtquick-controls qml-module-qtquick-controls2 \
      qml-module-qtquick-dialogs qml-module-qtquick-layouts && \
    rm -rf /var/lib/apt/lists/*

# Keep the full TeleKyb3 binary stack and its declared dependencies.
FROM runtime AS telekyb
SHELL ["/bin/sh", "-ec"]
RUN apt-get update && apt-get install -y --no-install-recommends curl python3 tcl tk && \
    curl -fsSL http://robotpkg.openrobots.org/packages/debian/robotpkg.asc \
      -o /usr/share/keyrings/robotpkg.asc && \
    echo 'deb [arch=amd64 signed-by=/usr/share/keyrings/robotpkg.asc] http://robotpkg.openrobots.org/packages/debian/pub noble robotpkg' \
      > /etc/apt/sources.list.d/robotpkg.list && \
    apt-get update && \
    set -- libgz-physics8-dartsim-dev libgz-rendering9-ogre2-dev; \
    for component in arduio dynamixel joystick nhfc maneuver mrsim optitrack phynt pom rotorcraft uavatt uavpos; do \
      set -- "$@" "robotpkg-${component}-genom3+codels+openprs+pocolibs-client-c+pocolibs-server"; \
    done; \
    for module in dxsim-gazebo mrsim-gazebo optitrack-gazebo eltclsh genomix \
                  py312-python-genomix tcl-genomix tk3-flash tk3-mikrokopter tk3-paparazzi; do \
      set -- "$@" "robotpkg-${module}"; \
    done; \
    apt-get install -y --no-install-recommends "$@" && \
    apt-get purge -y --auto-remove curl && rm -rf /var/lib/apt/lists/*

# Build the OptiTrack shutdown fix separately.
FROM telekyb AS optitrack-build
RUN apt-get update && apt-get install -y --no-install-recommends \
      git build-essential autoconf automake libtool pkg-config patch
COPY optitrack-shutdown.patch /tmp/optitrack-shutdown.patch
RUN git clone --no-checkout git://git.openrobots.org/robots/optitrack-gazebo.git /tmp/optitrack && \
    cd /tmp/optitrack && git checkout 0c1887cf9e2c1401211c41d7bb264aa20e52e056 && \
    patch -p1 < /tmp/optitrack-shutdown.patch && autoreconf -vi && \
    ./configure --prefix=/opt/openrobots --with-gz-sim=gz-sim9 && \
    make -j2 optitrack-gazebo.la && make install-pluginLTLIBRARIES

# Compile the custom system against the same Ionic libraries as the runtime.
RUN apt-get update && apt-get install -y --no-install-recommends cmake cppzmq-dev
COPY gazebo_plugin/ /tmp/constant-wrench/
RUN cmake -S /tmp/constant-wrench -B /tmp/constant-wrench-build \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/openrobots && \
    cmake --build /tmp/constant-wrench-build --parallel 2 && \
    cmake --install /tmp/constant-wrench-build

# Find the system runtime libraries used by every installed TeleKyb3 ELF file.
# Keep the complete /opt/openrobots tree, but not the build-stage toolchains.
RUN python3 - <<'PYTHON'
import os
from pathlib import Path
import re
import subprocess

packages = set()
for executable in Path('/opt/openrobots').rglob('*'):
    if not executable.is_file() or executable.is_symlink():
        continue
    with executable.open('rb') as stream:
        if stream.read(4) != b'\x7fELF':
            continue
    result = subprocess.run(['ldd', str(executable)], capture_output=True, text=True)
    if '=> not found' in result.stdout:
        raise RuntimeError(f'Missing library for {executable}: {result.stdout}')
    for library in re.findall(r'(?:=>\s+)?(/[^\s]+)', result.stdout):
        if library.startswith('/opt/openrobots/'):
            continue
        for path in dict.fromkeys([os.path.realpath(library), library]):
            owner = subprocess.run(['dpkg-query', '-S', path], capture_output=True, text=True)
            if owner.returncode == 0:
                for line in owner.stdout.splitlines():
                    packages.update(line.rsplit(': ', 1)[0].split(', '))
                break
        else:
            raise RuntimeError(f'No package owns runtime library {library}')
Path('/runtime-packages.txt').write_text('\n'.join(sorted(packages)) + '\n')
PYTHON

FROM runtime
COPY --from=optitrack-build /runtime-packages.txt /tmp/runtime-packages.txt
RUN apt-get update && \
    xargs apt-get install -y --no-install-recommends python3 tcl tk < /tmp/runtime-packages.txt && \
    apt-get purge -y --auto-remove curl && \
    rm -rf /var/lib/apt/lists/* /tmp/runtime-packages.txt
COPY --from=optitrack-build /opt/openrobots/ /opt/openrobots/
# Required shaders are packaged as development data; copy only runtime assets.
COPY --from=telekyb /usr/share/gz/gz-rendering9/media/ /usr/share/gz/gz-rendering9/media/
COPY --from=telekyb /usr/share/gz/gz-rendering9/ogre2/ /usr/share/gz/gz-rendering9/ogre2/
COPY --from=telekyb /usr/share/doc/libgz-rendering9-core-dev/copyright /usr/share/doc/gazebo-shaders/core-copyright
COPY --from=telekyb /usr/share/doc/libgz-rendering9-ogre2-dev/copyright /usr/share/doc/gazebo-shaders/ogre2-copyright
RUN for dir in /usr/lib/*/gz-physics-8/engine-plugins; do \
      ln -s libgz-physics8-dartsim-plugin.so.8 "$dir/libgz-physics-dartsim-plugin.so"; \
    done && \
    for dir in /usr/lib/*/gz-rendering-9/engine-plugins; do \
      ln -s libgz-rendering9-ogre2.so.9 "$dir/libgz-rendering-ogre2.so"; \
    done
COPY LICENSE /usr/share/doc/aerial-robotics-laboratory/LICENSE
ENV HOME=/tmp/gazebo QT_X11_NO_MITSHM=1 LIBGL_ALWAYS_SOFTWARE=1 \
    OR_INSTALL_PREFIX=/opt/openrobots \
    PATH=/opt/openrobots/bin:/opt/openrobots/sbin:$PATH \
    PYTHONPATH=/opt/openrobots/lib/python3.12/site-packages \
    TCLLIBPATH="/opt/openrobots/lib /opt/openrobots/share" \
    PKG_CONFIG_PATH=/opt/openrobots/lib/pkgconfig \
    GZ_SIM_RESOURCE_PATH=/opt/openrobots/share/gazebo/models:/opt/aerial-robotics-laboratory/models \
    GZ_SIM_SYSTEM_PLUGIN_PATH=/opt/openrobots/lib/gazebo
RUN apt-get update && apt-get install -y --no-install-recommends \
            wmctrl gz-transport14-cli && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /opt/aerial-robotics-laboratory
COPY models/ ./models/
COPY worlds/ ./worlds/
COPY tk3-wrench-sim.sh ./tk3-wrench-sim.sh
ENTRYPOINT ["/bin/sh", "/opt/aerial-robotics-laboratory/tk3-wrench-sim.sh"]
