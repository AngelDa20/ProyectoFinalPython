# main.py
from pathlib import Path
import numpy as np
from rvcore.io import load_robot_from_csv_dir
from rvcore.robot_model import from_csv_bundle
from rvcore.kinematics import fk_dh
from ui.viz_matplotlib import plot_robot
import matplotlib.pyplot as plt

def main():
    root = Path(__file__).parent
    bundle = load_robot_from_csv_dir(root / "config_csv", name="Mitsubishi RV-M2 (CSV)")
    model = from_csv_bundle(bundle)

    #Pruebas de visualización estática de varias coordenadas del robot.
    # Caso A: Home
    qA = np.deg2rad([0, 0, 0, 0, 0])
    TA, jointsA = fk_dh(model, qA)
    print(model.name)
    print("Home T0e:\n", np.round(TA, 3))
    plot_robot(model, jointsA, show=True, title="A) Home")

    # Caso B: Plano (-30,+30)
    qB = np.deg2rad([0, -30, 30, 0, 0])
    TB, jointsB = fk_dh(model, qB)
    print("Plano T0e:\n", np.round(TB, 3))
    plot_robot(model, jointsB, show=True, title="B) Plano (-30,+30)")

    # Caso C: Giro y Pitch
    qC = np.deg2rad([45, -20, 15, -10, 0])
    TC, jointsC = fk_dh(model, qC)
    print("Giro/Pitch T0e:\n", np.round(TC, 3))
    plot_robot(model, jointsC, show=True, title="C) Giro/Pitch")

if __name__ == "__main__":
    main()
