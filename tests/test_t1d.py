"""The vectorised UVA/Padova port reproduces simglucose's T1DPatient."""

import sys
import types
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _import_simglucose_patient():
    # simglucose/__init__ imports gym for its RL environments; stub it
    gym = types.ModuleType("gym")
    gym.envs = types.ModuleType("gym.envs")
    gym.envs.registration = types.ModuleType("gym.envs.registration")
    gym.envs.registration.register = lambda *a, **k: None
    sys.modules.setdefault("gym", gym)
    sys.modules.setdefault("gym.envs", gym.envs)
    sys.modules.setdefault("gym.envs.registration", gym.envs.registration)
    from simglucose.patient.t1dpatient import Action, T1DPatient
    return Action, T1DPatient


def test_port_matches_simglucose():
    from e4g import t1d
    Action, T1DPatient = _import_simglucose_patient()
    for name in ["adult#001", "adolescent#003", "child#002"]:
        ref = T1DPatient.withName(name)
        p = t1d.patient(name)
        b = t1d.Batch.initial(p)
        basal = p.u2ss * p.BW / 6000
        worst = 0.0
        for m in range(24 * 60):
            carb = 60 if m == 120 else (80 if m == 600 else 0)
            ins = basal + (carb / 12 if carb else 0)
            ref.step(Action(CHO=carb, insulin=ins))
            t1d.minute(b, p, carb, ins)
            worst = max(worst, abs(ref.observation.Gsub - t1d.bg(b, p)[0]))
        assert worst < 0.5, (name, worst)
