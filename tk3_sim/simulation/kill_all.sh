#!/bin/bash

# Kill the processes related to the Genom components
#pkill qualisys-pocolibs &

# Kill genomix first
pkill -f genomixd 2>/dev/null || true

# Kill pocolibs modules
for mod in optitrack-pocolibs rotorcraft-pocolibs pom-pocolibs nhfc-pocolibs maneuver-pocolibs uavpos-pocolibs uavatt-pocolibs phynt-pocolibs; do
    pkill -f $mod 2>/dev/null || true
done

# Kill generic module names
for mod in rotorcraft pom nhfc optitrack maneuver uavpos uavatt phynt; do
    pkill $mod 2>/dev/null || true
done

# Clean up h2 if initialized
if h2 list 2>/dev/null | grep -q .; then
    h2 end
fi

# Kill any existing gz sim processes for the specific world file
pids=$(ps aux | grep gz | grep -v grep | awk '{print $2}')

if [ -n "$pids" ]; then
  echo "Killing existing Gazebo processes..."
  for pid in $pids; do
    kill -9 $pid
    echo "Killed process $pid"
  done
else
  echo "No existing Gazebo processes found."
fi

pkill -f genomixd &
pkill -f joystick-pocolibs &
pkill -f maneuver-pocolibs &
pkill -f optitrack-pocolibs &
pkill -f phynt-pocolibs &
pkill -f pom-pocolibs &
pkill -f rotorcraft-pocolibs &
pkill -f uavatt-pocolibs &
pkill -f uavpos-pocolibs &
pkill -f phynt-pocolibs &
pkill -f nhfc-pocolibs &
h2 end 

rm $HOME/.nhfc.pid-lprior
rm $HOME/.optitrack.pid-lprior
rm $HOME/.uavatt.pid-lprior
rm $HOME/.phynt.pid-lprior
rm $HOME/.uavpos.pid-lprior
rm $HOME/.maneuver.pid-lprior
rm $HOME/.pom.pid-lprior
rm $HOME/.rotorcraft.pid-lprior
rm $HOME/.phynt.pid-lprior
rm $HOME/.genomixd.pid-lprior

echo "All cleaned up ✅"
