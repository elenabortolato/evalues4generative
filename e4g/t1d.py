"""Vectorised UVA/Padova type 1 diabetes simulator (port of simglucose's T1DPatient).

`simglucose` (Xie, 2018) implements the 2008 academic version of the UVA/Padova
T1D simulator: a 13-state ODE model of glucose-insulin dynamics with 30 virtual
patients. Its `T1DPatient` integrates one patient with an adaptive solver. Here
the same right-hand side is written for a batch of K trajectories and integrated
with fixed-step RK4 (1-minute steps), so that the real patient and the M twin
draws can be propagated together; `tests/test_t1d.py` checks the port against
simglucose itself.

Stochastic components added for the monitoring application (Section 6.6 of the
paper), identical in the "real" patient under H0 and in the twin:

* intra-interval variability of insulin sensitivity: during each 5-minute
  control interval, the insulin-action parameters (Vmx, kp3) are multiplied by
  exp(sigma_is * eta), eta ~ N(0, 1);
* carbohydrate-counting error: a meal announced as C grams is eaten as
  C * exp(sigma_carb * eta);
* unannounced snacks: with probability p_snack per interval, U(5, 25) grams;
* CGM sensor noise: i.i.d. Johnson-SU noise with simglucose's parameters
  (the AR(1) correlation of simglucose's sensor model is dropped, so that the
  noise is conditionally independent given the state).
"""

from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

# locate simglucose's parameter files without importing the package, whose
# __init__ pulls in gym (only needed for its RL environments)
_SG = Path(importlib.util.find_spec("simglucose").origin).parent
PARAMS = pd.read_csv(_SG / "params" / "vpatient_params.csv")
QUEST = pd.read_csv(_SG / "params" / "Quest.csv")
EAT_RATE = 5.0  # g/min, as in simglucose
# Johnson SU parameters of simglucose's CGM noise model (sensor_params.csv)
J_GAMMA, J_LAMBDA, J_DELTA, J_XI = -0.5444, 15.9574, 1.6898, -5.47


def patient(name: str) -> pd.Series:
    return PARAMS.loc[PARAMS.Name == name].squeeze()


def quest(name: str) -> pd.Series:
    return QUEST.loc[QUEST.Name == name].squeeze()


@dataclass
class Batch:
    """State of K trajectories: ODE state plus simglucose's meal bookkeeping."""
    x: np.ndarray                 # (K, 13)
    last_qsto: np.ndarray         # (K,)
    last_food: np.ndarray         # (K,)  g eaten since the meal started
    eating: np.ndarray            # (K,)  bool
    planned: np.ndarray           # (K,)  g still to eat
    last_cho: np.ndarray = field(default=None)  # (K,) CHO of the previous minute

    @classmethod
    def initial(cls, p: pd.Series, k: int = 1) -> "Batch":
        x0 = np.asarray(p.iloc[2:15].values, dtype=float)
        x = np.tile(x0, (k, 1))
        return cls(x=x, last_qsto=x[:, 0] + x[:, 1], last_food=np.zeros(k),
                   eating=np.zeros(k, bool), planned=np.zeros(k), last_cho=np.zeros(k))

    def copy(self) -> "Batch":
        return Batch(*(np.copy(v) for v in (self.x, self.last_qsto, self.last_food,
                                            self.eating, self.planned, self.last_cho)))

    def repeat(self, k: int) -> "Batch":
        """K copies of a single trajectory (the twin re-initialised at the state)."""
        return Batch(*(np.repeat(v[:1], k, axis=0) for v in (
            self.x, self.last_qsto, self.last_food, self.eating, self.planned,
            self.last_cho)))


def rhs(x, cho, insulin, p, dbar, vmx_mult, kp3_mult):
    """simglucose T1DPatient.model, vectorised over the first axis of x."""
    d = cho * 1000.0
    ins = insulin * 6000.0 / p.BW
    qsto = x[:, 0] + x[:, 1]
    dx = np.empty_like(x)
    dx[:, 0] = -p.kmax * x[:, 0] + d
    with np.errstate(divide="ignore", invalid="ignore"):
        aa = 5.0 / (2.0 * dbar * (1.0 - p.b))
        cc = 5.0 / (2.0 * dbar * p.d)
        kgut_meal = p.kmin + (p.kmax - p.kmin) / 2.0 * (
            np.tanh(aa * (qsto - p.b * dbar)) - np.tanh(cc * (qsto - p.d * dbar)) + 2.0)
    kgut = np.where(dbar > 0, kgut_meal, p.kmax)
    dx[:, 1] = p.kmax * x[:, 0] - x[:, 1] * kgut
    dx[:, 2] = kgut * x[:, 1] - p.kabs * x[:, 2]
    rat = p.f * p.kabs * x[:, 2] / p.BW
    egp = p.kp1 - p.kp2 * x[:, 3] - p.kp3 * kp3_mult * x[:, 8]
    et = np.where(x[:, 3] > p.ke2, p.ke1 * (x[:, 3] - p.ke2), 0.0)
    dx[:, 3] = np.maximum(egp, 0) + rat - p.Fsnc - et - p.k1 * x[:, 3] + p.k2 * x[:, 4]
    dx[:, 3] *= x[:, 3] >= 0
    vmt = p.Vm0 + p.Vmx * vmx_mult * x[:, 6]
    uid = vmt * x[:, 4] / (p.Km0 + x[:, 4])
    dx[:, 4] = (-uid + p.k1 * x[:, 3] - p.k2 * x[:, 4]) * (x[:, 4] >= 0)
    dx[:, 5] = (-(p.m2 + p.m4) * x[:, 5] + p.m1 * x[:, 9] + p.ka1 * x[:, 10]
                + p.ka2 * x[:, 11]) * (x[:, 5] >= 0)
    it = x[:, 5] / p.Vi
    dx[:, 6] = -p.p2u * x[:, 6] + p.p2u * (it - p.Ib)
    dx[:, 7] = -p.ki * (x[:, 7] - it)
    dx[:, 8] = -p.ki * (x[:, 8] - x[:, 7])
    dx[:, 9] = (-(p.m1 + p.m30) * x[:, 9] + p.m2 * x[:, 5]) * (x[:, 9] >= 0)
    dx[:, 10] = (ins - (p.ka1 + p.kd) * x[:, 10]) * (x[:, 10] >= 0)
    dx[:, 11] = (p.kd * x[:, 10] - p.ka2 * x[:, 11]) * (x[:, 11] >= 0)
    dx[:, 12] = (-p.ksc * x[:, 12] + p.ksc * x[:, 3]) * (x[:, 12] >= 0)
    return dx


def minute(b: Batch, p, meal_announce, insulin, vmx_mult=1.0, kp3_mult=1.0):
    """One minute, reproducing simglucose's meal bookkeeping, then RK4."""
    b.planned = b.planned + meal_announce
    to_eat = np.minimum(EAT_RATE, b.planned) * (b.planned > 0)
    b.planned = np.maximum(b.planned - to_eat, 0.0)
    start = (to_eat > 0) & (b.last_cho <= 0)
    b.last_qsto = np.where(start, b.x[:, 0] + b.x[:, 1], b.last_qsto)
    b.last_food = np.where(start, 0.0, b.last_food)
    b.eating = b.eating | start
    b.last_food = b.last_food + np.where(b.eating, to_eat, 0.0)
    b.eating = b.eating & ~((to_eat <= 0) & (b.last_cho > 0))
    b.last_cho = to_eat
    dbar = b.last_qsto + b.last_food * 1000.0
    f = lambda y: rhs(y, to_eat, insulin, p, dbar, vmx_mult, kp3_mult)  # noqa: E731
    k1 = f(b.x)
    k2 = f(b.x + 0.5 * k1)
    k3 = f(b.x + 0.5 * k2)
    k4 = f(b.x + k3)
    b.x = b.x + (k1 + 2 * k2 + 2 * k3 + k4) / 6.0
    return b


def bg(b: Batch, p) -> np.ndarray:
    """Subcutaneous glucose (mg/dL), simglucose's observation Gsub."""
    return b.x[:, 12] / p.Vg


def johnson_noise(rng, shape):
    z = rng.standard_normal(shape)
    return J_XI + J_LAMBDA * np.sinh((z - J_GAMMA) / J_DELTA)


@dataclass
class Variability:
    sigma_is: float = 0.15     # insulin-sensitivity variability per interval
    sigma_carb: float = 0.2    # carbohydrate-counting error (log scale)
    p_snack: float = 0.01      # unannounced snack probability per 5-min interval
    noise_scale: float = 1.0   # multiplies the Johnson CGM noise


def interval(b: Batch, p, rng, insulin, meal_announce, var: Variability,
             vmx_factor=1.0, kp3_factor=1.0, sensor_gain=1.0, deterministic=False,
             minutes=5):
    """Advance K trajectories by one control interval and return the CGM readings.

    `insulin` (U/min) and `meal_announce` (g, announced at the first minute) are
    the action; they may be scalars or (K,) arrays. Returns (K, minutes) readings,
    one per minute, clipped to the sensor range [39, 600] as in simglucose.
    """
    k = b.x.shape[0]
    if deterministic:
        eta = np.zeros(k)
        carbs = np.broadcast_to(meal_announce, (k,)).astype(float)
        snack = np.zeros(k)
    else:
        eta = rng.standard_normal(k)
        carbs = np.broadcast_to(meal_announce, (k,)) * np.exp(
            var.sigma_carb * rng.standard_normal(k))
        snack = np.where(rng.random(k) < var.p_snack, rng.uniform(5, 25, k), 0.0)
    mult = np.exp(var.sigma_is * eta)
    out = np.empty((k, minutes))
    for i in range(minutes):
        announce = carbs + snack if i == 0 else 0.0
        minute(b, p, announce, insulin, vmx_mult=mult * vmx_factor,
               kp3_mult=mult * kp3_factor)
        out[:, i] = bg(b, p)
    if not deterministic:
        out = sensor_gain * out + var.noise_scale * johnson_noise(rng, out.shape)
    return np.clip(out, 39.0, 600.0)


class Controller:
    """Basal-bolus therapy with correction and low-glucose suspend.

    Basal and meal bolus follow simglucose's BBController (u2ss, CR, CF); the
    correction bolus uses the current CGM reading; basal is suspended while the
    CGM reads below 80 mg/dL. Every dose therefore depends on the data.
    """

    def __init__(self, name: str, target: float = 140.0):
        p, q = patient(name), quest(name)
        self.basal = p.u2ss * p.BW / 6000.0  # U/min
        self.cr, self.cf, self.target = q.CR, q.CF, target

    def dose(self, cgm: float, meal_announce: float, minutes: int = 5) -> float:
        rate = 0.0 if cgm < 80 else self.basal
        if meal_announce > 0:
            bolus = meal_announce / self.cr + (cgm > 150) * (cgm - self.target) / self.cf
            rate += max(bolus, 0.0) / minutes
        return rate

    def nominal(self, meal_announce: float, minutes: int = 5) -> float:
        """The open-loop schedule: basal plus carbohydrate bolus, no feedback."""
        return self.basal + (meal_announce / self.cr) / minutes


def meal_plan(rng, days: int, interval_min: int = 5) -> np.ndarray:
    """Announced carbohydrates per interval: three meals a day with jittered times."""
    steps = days * 24 * 60 // interval_min
    plan = np.zeros(steps)
    for d in range(days):
        for hour, grams in ((7.0, 50.0), (12.5, 70.0), (19.0, 80.0)):
            t = (d * 24 + hour + rng.normal(0, 0.5)) * 60 / interval_min
            plan[int(np.clip(round(t), 0, steps - 1))] += grams * np.exp(0.15 * rng.normal())
    return plan
