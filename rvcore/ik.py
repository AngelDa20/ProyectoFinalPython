# rvcore/ik.py
import numpy as np
from rvcore.kinematics import fk_dh

def ee_position(model, q):
    T, _ = fk_dh(model, q)
    return T[:3, 3].copy()  # x,y,z en mm

def numerical_jacobian_pos(model, q, eps=1e-4):
    """
    Jacobiano de posición 3xN por diferencias finitas:
    J[:,i] = d(x,y,z)/d(q_i)
    q en radianes, posición en mm.
    """
    q = np.asarray(q, float)
    x0 = ee_position(model, q)
    J = np.zeros((3, q.size), dtype=float)
    for i in range(q.size):
        dq = np.zeros_like(q)
        dq[i] = eps
        x1 = ee_position(model, q + dq)
        J[:, i] = (x1 - x0) / eps
    return J

def ik_step_dls(model, q, dx_mm, lam=2.0, step_clip=np.deg2rad(2.0)):
    """
    Un paso de IK DLS para mover la herramienta por delta cartesiano (mm).
    - dx_mm: np.array([dx, dy, dz]) mm
    - lam: amortiguación (λ)
    - step_clip: límite de paso articular por iteración (rad)
    Retorna q_next (clamp a límites).
    """
    q = np.asarray(q, float)
    J = numerical_jacobian_pos(model, q)  # 3xN
    JT = J.T
    A = J @ JT + (lam**2) * np.eye(3)
    dq = JT @ np.linalg.solve(A, dx_mm)
    dq = np.clip(dq, -step_clip, step_clip)
    q_next = q + dq
    if hasattr(model, "limits"):
        q_next = np.minimum(np.maximum(q_next, model.limits.q_min), model.limits.q_max)
    return q_next
