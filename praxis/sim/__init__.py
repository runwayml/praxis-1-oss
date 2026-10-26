"""MuJoCo simulation of the single-arm rig (a Franka arm with a two-finger gripper).

The simulation renders the two camera views of the rig and executes the robot
commands of the policy at 15 Hz. Use it to check a single-arm checkpoint in
closed loop without a real robot.

Modules:

* :mod:`praxis.sim.gl`: the OpenGL backend of the renderer.
* :mod:`praxis.sim.assets`: download and check of the open-source robot models.
* :mod:`praxis.sim.scene`: the MuJoCo model (arm, gripper, table, objects,
  lights, cameras).
* :mod:`praxis.sim.robot`: frames, inverse kinematics, gripper map and one
  control tick.
* :mod:`praxis.sim.tasks`: the task registry (prompt, object placement,
  success check).
* :mod:`praxis.sim.env`: the environment (reset, step, observe).
* :mod:`praxis.sim.scripted`: a scripted pick policy that sends the same
  commands as the model.
* :mod:`praxis.sim.config`: the settings of a run and the YAML loader.
* :mod:`praxis.sim.loop`: the closed-loop sampling loop and the policies
  (model, scripted, hold).
* :mod:`praxis.sim.video`: the MP4 writer and the frame layout.
* :mod:`praxis.sim.viewer`: the live view in a web browser.

Run it with ``scripts/run_sim.py``. See ``praxis/sim/README.md``.

MuJoCo reads the OpenGL backend when Python imports it. Thus this package sets
the default backend before any sub-module imports MuJoCo. To select a
different backend, call :func:`praxis.sim.gl.prepare` before the first import
of a sub-module that uses MuJoCo.
"""

from praxis.sim import gl

gl.prepare()
