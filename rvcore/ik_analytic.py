# rvcore/ik_analytic.py
import numpy as np

def ik_rvm2_position(model, target_pos):
    """
    Cinemática inversa analítica simplificada para el Mitsubishi RV-M2.
    Basada en los parámetros DH del modelo (solo posición x, y, z).
    Retorna una lista de posibles soluciones articulares [q1..q5].
    """

    # --- Extraer parámetros desde el modelo DH ---
    a1 = model.dh[1, 0]   # mm (hombro)
    a2 = model.dh[2, 0]   # mm (codo)
    d1 = model.dh[0, 2]   # mm (altura base)

    x, y, z = target_pos

    # --- Articulación J1: base ---
    q1 = np.arctan2(y, x)

    # --- Convertir coordenadas a plano del brazo ---
    r = np.hypot(x, y)
    z_rel = z - d1

    # --- Articulación J3 (codo) mediante ley del coseno ---
    D = (r**2 + z_rel**2 - a1**2 - a2**2) / (2 * a1 * a2)
    D = np.clip(D, -1.0, 1.0)  # proteger contra errores numéricos

    q3_up = np.arctan2(np.sqrt(1 - D**2), D)     # codo arriba
    q3_down = np.arctan2(-np.sqrt(1 - D**2), D)  # codo abajo

    # --- Articulación J2 (hombro) ---
    phi = np.arctan2(z_rel, r)
    psi_up = np.arctan2(a2 * np.sin(q3_up), a1 + a2 * np.cos(q3_up))
    psi_down = np.arctan2(a2 * np.sin(q3_down), a1 + a2 * np.cos(q3_down))

    q2_up = phi - psi_up
    q2_down = phi - psi_down

    # --- Articulaciones J4 y J5 (simplificadas: mantener orientación plana) ---
    q4_up = -(q2_up + q3_up)
    q4_down = -(q2_down + q3_down)
    q5 = 0.0

    # --- Ensamblar las soluciones ---
    sol_up = np.array([q1, q2_up, q3_up, q4_up, q5])
    sol_down = np.array([q1, q2_down, q3_down, q4_down, q5])

    # --- Filtrar por límites articulares si existen ---
    qmin = getattr(model.limits, "q_min", None)
    qmax = getattr(model.limits, "q_max", None)

    valid_solutions = []
    for q_sol in [sol_up, sol_down]:
        if qmin is not None and qmax is not None:
            if np.all(q_sol >= qmin) and np.all(q_sol <= qmax):
                valid_solutions.append(q_sol)
        else:
            valid_solutions.append(q_sol)

    # Si ninguna válida, devolver la más cercana dentro de límites
    if not valid_solutions and qmin is not None:
        clipped = np.clip(sol_up, qmin, qmax)
        valid_solutions.append(clipped)

    return valid_solutions
