# rvcore/robot_model.py
from dataclasses import dataclass
import numpy as np
from rvcore.io import RobotCsvBundle, JointLimits as CsvJointLimits

@dataclass
class JointLimits:
    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray
    ddq_max: np.ndarray

@dataclass
class RobotModel:
    name: str
    dof: int
    dh: np.ndarray         # [a, alpha(rad), d, theta0(rad)]
    base: np.ndarray       # 4x4
    tool: np.ndarray       # 4x4
    limits: JointLimits

    def clamp(self, q: np.ndarray) -> np.ndarray:
        return np.minimum(np.maximum(q, self.limits.q_min), self.limits.q_max)

def from_csv_bundle(bundle: RobotCsvBundle) -> RobotModel:
    dof = bundle.dh.shape[0]
    lim = bundle.limits
    limits = JointLimits(q_min=lim.q_min, q_max=lim.q_max, dq_max=lim.dq_max, ddq_max=lim.ddq_max)
    return RobotModel(
        name=bundle.name,
        dof=dof,
        dh=bundle.dh,
        base=bundle.base,
        tool=bundle.tool,
        limits=limits
    )
