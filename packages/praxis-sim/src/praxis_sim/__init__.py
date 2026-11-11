"""MuJoCo simulation of the single-arm rig (a Franka arm with a two-finger gripper).

The simulation renders the two camera views of the rig and executes the robot
commands of the policy at 15 Hz. Use it to check a single-arm checkpoint in
closed loop without a real robot.

Modules:

* :mod:`praxis_sim.gl`: the OpenGL backend of the renderer.
* :mod:`praxis_sim.assets`: download and check of the open-source robot models.
* :mod:`praxis_sim.scene`: the MuJoCo model (arm, gripper, table, objects,
  lights, cameras).
* :mod:`praxis_sim.robot`: frames, inverse kinematics, gripper map and one
  control tick.
* :mod:`praxis_sim.tasks`: the task registry (prompt, object placement,
  success check).
* :mod:`praxis_sim.env`: the environment (reset, step, observe).
* :mod:`praxis_sim.scripted`: a scripted pick policy that sends the same
  commands as the model.
* :mod:`praxis_sim.config`: the settings of a run and the YAML loader.
* :mod:`praxis.sim.loop`: the closed-loop sampling loop and the policies
  (model, scripted, hold).
* :mod:`praxis.sim.video`: the MP4 writer and the frame layout.
* :mod:`praxis.sim.viewer`: the live view in a web browser.

``praxis_sim.gym_env`` wraps a task as a Gymnasium environment and
``praxis_sim.compare`` pairs a LeRobot evaluation with a run of the praxis
runner. The model loop, the video writer and the live
viewer stay in ``praxis.sim`` of the praxis repository, which runs the sim
through this package. See ``packages/praxis-sim/README.md``.

MuJoCo reads the OpenGL backend when Python imports it. Thus this package sets
the default backend before any sub-module imports MuJoCo. To select a
different backend, call :func:`praxis_sim.gl.prepare` before the first import
of a sub-module that uses MuJoCo.
"""

from praxis_sim import gl

gl.prepare()
