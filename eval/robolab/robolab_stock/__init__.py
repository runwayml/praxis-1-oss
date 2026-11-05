"""The praxis policy through RoboLab's official, unmodified eval runner.

RoboLab's own episode loop (`robolab/eval/runner.py` + `episode.py`) drives the tasks
with its own AbsIK registration (`DroidIKActionCfg`); `praxis_client.PraxisRemoteClient`
is the `InferenceClient` it calls. The client runs positronic's client stack
(`praxis_sidecar.py`, inside positronic's own venv) and converts the
resulting wire command into the 8-dim DroidIK action (`praxis_stock_wire.py`).
"""
