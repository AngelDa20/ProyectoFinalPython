# rvcore/robot_model.py
from dataclasses import dataclass
import numpy as np
from rvcore.io import RobotCsvBundle, JointLimits
from rvcore.ik_analytic import ik_rvm2_position

@dataclass
class RobotModel:
    name: str
    dof: int
    dh: np.ndarray
    base: np.ndarray
    tool: np.ndarray
    limits: JointLimits
    ik_solver: callable = None  # Campo opcional

def from_csv_bundle(bundle: RobotCsvBundle) -> RobotModel:
    """Convierte un paquete CSV en un modelo de robot utilizable."""
    limits = bundle.limits

    # Crear modelo base
    model = RobotModel(
        name=bundle.name,
        dof=bundle.dh.shape[0],
        dh=bundle.dh,
        base=bundle.base,
        tool=bundle.tool,
        limits=limits
    )

    # --- Asignar solver analítico si es RV-M2 ---
    if "RV-M2" in bundle.name.upper():
        model.ik_solver = ik_rvm2_position

    return model
