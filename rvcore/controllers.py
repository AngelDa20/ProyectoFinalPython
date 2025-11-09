# rvcore/controllers.py
import numpy as np

class PID3:
    """
    PID vectorial de 3 ejes (X,Y,Z) con anti-windup y derivada filtrada.
    Unidades: si dt está en s, salida será mm/s; si dt es 1/fps, será mm/tick.
    """
    def __init__(self, kp=(0.6,0.6,0.6), ki=(0.0,0.0,0.0), kd=(0.12,0.12,0.12),
                 umin=None, umax=None, tau=0.05):
        self.kp = np.array(kp, dtype=float)
        self.ki = np.array(ki, dtype=float)
        self.kd = np.array(kd, dtype=float)
        self.tau = float(tau)
        self.umin = None if umin is None else np.array(umin, dtype=float)
        self.umax = None if umax is None else np.array(umax, dtype=float)
        self.i = np.zeros(3, dtype=float)
        self.d = np.zeros(3, dtype=float)
        self.e_prev = np.zeros(3, dtype=float)

    def reset(self):
        self.i[:] = 0.0
        self.d[:] = 0.0
        self.e_prev[:] = 0.0

    def step(self, e, dt):
        # Integral (anti-windup simple por saturación posterior)
        self.i += e * dt

        # Derivada filtrada (Tustin / low-pass)
        de = (e - self.e_prev) / max(dt, 1e-6)
        alpha = self.tau / (self.tau + dt)
        self.d = alpha * self.d + (1 - alpha) * de
        self.e_prev = e.copy()

        u = self.kp*e + self.ki*self.i + self.kd*self.d

        # Saturación
        if self.umin is not None:
            u = np.maximum(u, self.umin)
        if self.umax is not None:
            u = np.minimum(u, self.umax)
        return u
