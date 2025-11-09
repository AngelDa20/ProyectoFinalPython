# rvcore/utils.py
import numpy as np

def wrap_to_pi(q):
    """Envuelve ángulos a [-pi, pi] (vector)."""
    return (q + np.pi) % (2*np.pi) - np.pi

def clip_joints(q, qmin=None, qmax=None):
    """Limita q a [qmin,qmax] si existen."""
    if qmin is not None and qmax is not None:
        return np.minimum(np.maximum(q, qmin), qmax)
    return q
