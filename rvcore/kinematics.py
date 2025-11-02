# rvcore/kinematics.py
import numpy as np

def _A(a, alpha, d, theta):
    ca, sa = np.cos(alpha), np.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    return np.array([
        [ct, -st*ca,  st*sa, a*ct],
        [st,  ct*ca, -ct*sa, a*st],
        [0.0,    sa,    ca,    d],
        [0.0,   0.0,   0.0,  1.0]
    ], dtype=float)

def fk_dh(model, q):
    """
    Devuelve:
      T0e: matriz 4x4 base→tool
      joints: (n+2, 3) posiciones de base, juntas y punta de herramienta [mm]
    """
    q = np.asarray(q, dtype=float)
    assert q.size == model.dof

    a, alpha, d, theta0 = model.dh[:,0], model.dh[:,1], model.dh[:,2], model.dh[:,3]
    thetas = theta0 + q

    T = model.base.copy()
    joints = [T[:3,3].copy()]  # base

    for i in range(model.dof):
        T = T @ _A(a[i], alpha[i], d[i], thetas[i])
        joints.append(T[:3,3].copy())

    T = T @ model.tool
    joints.append(T[:3,3].copy())  # tool tip
    return T, np.vstack(joints)
