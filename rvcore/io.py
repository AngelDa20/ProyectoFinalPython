# rvcore/io.py
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from dataclasses import dataclass

@dataclass
class JointLimits:
    q_min: np.ndarray      # rad
    q_max: np.ndarray      # rad
    dq_max: np.ndarray     # rad/s
    ddq_max: np.ndarray    # rad/s^2

@dataclass
class RobotCsvBundle:
    name: str
    dh: np.ndarray         # (n,4) -> [a_mm, alpha_rad, d_mm, theta0_rad]
    base: np.ndarray       # (4,4)
    tool: np.ndarray       # (4,4)
    limits: JointLimits

# ---------- Lectura de CSV ----------
def read_dh_csv(path: str) -> np.ndarray:
    df = pd.read_csv(path)
    required = ["a_mm", "alpha_deg", "d_mm", "theta0_deg"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas en dh.csv: {missing}")
    dh = df[required].to_numpy(dtype=float)  # [a, alpha_deg, d, theta0_deg]
    dh[:, 1] = np.deg2rad(dh[:, 1])  # alpha → rad
    dh[:, 3] = np.deg2rad(dh[:, 3])  # theta0 → rad
    return dh

def read_matrix4_csv(path: str) -> np.ndarray:
    M = pd.read_csv(path, header=None).to_numpy(dtype=float)
    if M.shape != (4, 4):
        raise ValueError(f"Matriz en {path} debe ser 4x4, obtuve {M.shape}")
    return M

def read_limits_wide_csv(path: str) -> JointLimits:
    df = pd.read_csv(path)
    if "type" not in df.columns:
        raise ValueError("limits.csv debe tener columna 'type'")
    df = df.set_index("type")
    # Convertir deg → rad (también vel y acel en deg/s y deg/s^2)
    q_min = np.deg2rad(df.loc["q_min_deg"].to_numpy(dtype=float))
    q_max = np.deg2rad(df.loc["q_max_deg"].to_numpy(dtype=float))
    dq_max = np.deg2rad(df.loc["dq_max_deg_s"].to_numpy(dtype=float))
    ddq_max = np.deg2rad(df.loc["ddq_max_deg_s2"].to_numpy(dtype=float))
    return JointLimits(q_min=q_min, q_max=q_max, dq_max=dq_max, ddq_max=ddq_max)

def load_robot_from_csv_dir(dirpath: str, name: str = "Robot (CSV)") -> RobotCsvBundle:
    dh = read_dh_csv(os.path.join(dirpath, "dh.csv"))
    base = read_matrix4_csv(os.path.join(dirpath, "base.csv"))
    tool = read_matrix4_csv(os.path.join(dirpath, "tool.csv"))
    limits = read_limits_wide_csv(os.path.join(dirpath, "limits.csv"))
    return RobotCsvBundle(name=name, dh=dh, base=base, tool=tool, limits=limits)
