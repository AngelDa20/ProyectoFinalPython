# ui/gui_tk.py
import tkinter as tk
from tkinter import ttk
import numpy as np
from rvcore.kinematics import fk_dh
from rvcore.ik import ik_step_dls

# Matplotlib embebido
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


class RobotGUI(tk.Tk):
    def __init__(self, model, update_hz=30):
        super().__init__()
        self.title("RV-M2 Sim - Palancas X/Y/Z (Tkinter)")
        self.model = model
        self.dt = 1.0 / update_hz
        self.running = False

        # Estado del robot
        self.q = np.zeros(self.model.dof, dtype=float)     # rad
        self.target_vel = np.zeros(3)                      # [vx, vy, vz] mm/tick

        # Variables ajustables (sliders)
        self.x_scale = tk.DoubleVar(value=1.5)             # mm/tick (palanca X)
        self.y_scale = tk.DoubleVar(value=1.5)             # mm/tick (palanca Y)
        self.z_scale = tk.DoubleVar(value=1.5)             # mm/tick (palanca Z)
        self.x_step  = tk.DoubleVar(value=2.0)             # mm/click (botones finos X)
        self.y_step  = tk.DoubleVar(value=2.0)             # mm/click (botones finos Y)
        self.z_step  = tk.DoubleVar(value=2.0)             # mm/click (botones finos Z)
        self.lam     = tk.DoubleVar(value=2.0)             # Damping λ (DLS)

        # Modo de IK (por defecto DLS)
        self.use_analytic_ik = tk.BooleanVar(value=False)

        # Palancas: estado
        self.lever_active = {"x": False, "y": False, "z": False}  # True si arrastrando
        self.impulse_pending = {"x": 0.0, "y": 0.0, "z": 0.0}     # impulsos por click

        # Layout general
        self.columnconfigure(0, weight=0)
        self.columnconfigure(1, weight=1)
        self._build_controls(side=0)
        self._build_plot(side=1)

        self.after_id = None

    # ==============================================================
    # UI: CONTROLES
    # ==============================================================
    def _build_controls(self, side=0):
        panel = ttk.Frame(self)
        panel.grid(row=0, column=side, sticky="ns", padx=6, pady=6)

        # Tres columnas: X | Y | Z
        for c in range(3):
            panel.columnconfigure(c, weight=1)

        # -------- Palanca X --------
        self._build_lever_column(panel, col=0, axis="x",
                                 title="Palanca X",
                                 minus_text="−X", plus_text="+X",
                                 on_minus=self._btn_x_minus, on_plus=self._btn_x_plus)

        # -------- Palanca Y --------
        self._build_lever_column(panel, col=1, axis="y",
                                 title="Palanca Y",
                                 minus_text="−Y", plus_text="+Y",
                                 on_minus=self._btn_y_minus, on_plus=self._btn_y_plus)

        # -------- Palanca Z --------
        self._build_lever_column(panel, col=2, axis="z",
                                 title="Palanca Z",
                                 minus_text="-Z", plus_text="+Z",
                                 on_minus=self._btn_z_down, on_plus=self._btn_z_up)

        # -------- Parámetros (sliders) --------
        sfrm = ttk.LabelFrame(panel, text="Parámetros")
        sfrm.grid(row=2, column=0, columnspan=3, sticky="we", pady=(10, 6), padx=2)

        # Velocidades continuas
        self._slider_with_value(sfrm, row=0, text="Vel X (mm/tick)", var=self.x_scale, rng=(0.5, 8.0))
        self._slider_with_value(sfrm, row=2, text="Vel Y (mm/tick)", var=self.y_scale, rng=(0.5, 8.0))
        self._slider_with_value(sfrm, row=4, text="Vel Z (mm/tick)", var=self.z_scale, rng=(0.5, 8.0))

        # Pasos finos
        self._slider_with_value(sfrm, row=6, text="Paso X (mm/click)", var=self.x_step, rng=(0.5, 10.0))
        self._slider_with_value(sfrm, row=8, text="Paso Y (mm/click)", var=self.y_step, rng=(0.5, 10.0))
        self._slider_with_value(sfrm, row=10, text="Paso Z (mm/click)", var=self.z_step, rng=(0.5, 10.0))

        # Damping λ
        self._slider_with_value(sfrm, row=12, text="Damping λ (IK)", var=self.lam, rng=(0.5, 8.0))

        # Reset sliders
        ttk.Button(sfrm, text="Reset valores",
                   command=self._reset_sliders).grid(row=14, column=0, columnspan=2, pady=(6, 4))

        # -------- Botones principales --------
        bfrm = ttk.Frame(panel)
        bfrm.grid(row=3, column=0, columnspan=3, pady=8)
        ttk.Button(bfrm, text="Iniciar", command=self.start).grid(row=0, column=0, padx=6)
        ttk.Button(bfrm, text="Pausa", command=self.pause).grid(row=0, column=1, padx=6)
        ttk.Button(bfrm, text="Home", command=self.home).grid(row=0, column=2, padx=6)
        
        # -------- Indicador de estado --------
        ledfrm = ttk.Frame(panel)
        ledfrm.grid(row=3, column=3, padx=(10, 0), pady=8, sticky="e")

        ttk.Label(ledfrm, text="Estado:").grid(row=0, column=0, padx=(0,4))
        self.led_canvas = tk.Canvas(ledfrm, width=20, height=20, highlightthickness=0)
        self.led_canvas.grid(row=0, column=1)
        self.led_circle = self.led_canvas.create_oval(2, 2, 18, 18, fill="red", outline="black")

        # -------- Modo IK --------
        modefrm = ttk.LabelFrame(panel, text="Modo IK")
        modefrm.grid(row=4, column=0, columnspan=3, sticky="we", pady=6)
        ttk.Checkbutton(modefrm, text="Usar IK analítica (RV-M2)",
                        variable=self.use_analytic_ik).grid(row=0, column=0, padx=6, pady=4)

        # -------- Estado de pose --------
        self.pose_var = tk.StringVar(value="EE: (---, ---, ---) mm")
        ttk.Label(panel, textvariable=self.pose_var, font=("Consolas", 10)).grid(
            row=5, column=0, columnspan=3, pady=(8, 6)
        )

    # Helpers UI
    def _build_lever_column(self, parent, col, axis, title, minus_text, plus_text, on_minus, on_plus):
        """Construye una columna con la palanca y los botones finos para un eje."""
        # Marco contenedor vertical para el eje
        cfrm = ttk.Frame(parent)
        cfrm.grid(row=0, column=col, rowspan=2, sticky="n", padx=8, pady=4)

        # Título
        ttk.Label(cfrm, text=title).grid(row=0, column=0, columnspan=2, pady=(0, 4))

        # Canvas de la palanca
        lever_h, lever_w = 180, 60
        canvas = tk.Canvas(cfrm, width=lever_w, height=lever_h, bg="#f7f7f7", highlightthickness=0)
        canvas.grid(row=1, column=0, columnspan=2, padx=4, pady=2)
        if not hasattr(self, "lever"):
            self.lever = {}
            self.knob = {}
            self.lever_top = {}
            self.lever_bot = {}
            self.lever_center_y = {}
            self.lever_radius = {}
        self.lever[axis] = canvas

        pad = 10
        self.lever_top[axis] = pad
        self.lever_bot[axis] = lever_h - pad
        self.lever_center_y[axis] = lever_h // 2
        self.lever_radius[axis] = lever_h//2 - pad

        # Pista vertical
        canvas.create_line(lever_w//2, self.lever_top[axis], lever_w//2, self.lever_bot[axis],
                           fill="#888", width=3)

        # Knob
        r = 10
        self.knob[axis] = canvas.create_oval(0, 0, 0, 0, fill=("#2aa84a" if axis=="z" else "#4c8bf5"), outline="")
        self._move_axis_knob(axis, lever_w//2, self.lever_center_y[axis])

        # Bindings
        canvas.bind("<B1-Motion>", lambda e, ax=axis: self._on_drag_axis(ax, e))
        canvas.bind("<Button-1>",  lambda e, ax=axis: self._on_drag_axis(ax, e))
        canvas.bind("<ButtonRelease-1>", lambda e, ax=axis: self._on_release_axis(ax))

        # --- Botones finos (impulso) ---
        ttk.Button(cfrm, text=minus_text, width=5, command=on_minus).grid(row=2, column=0, padx=3, pady=(6, 2))
        ttk.Button(cfrm, text=plus_text,  width=5, command=on_plus ).grid(row=2, column=1, padx=3, pady=(6, 2))


    def _slider_with_value(self, frame, row, text, var, rng):
        ttk.Label(frame, text=text).grid(row=row, column=0, sticky="w")
        s = ttk.Scale(frame, from_=rng[0], to=rng[1], orient="horizontal", variable=var, length=160)
        s.grid(row=row+1, column=0, padx=6, sticky="we")
        lbl = ttk.Label(frame, text=f"{var.get():.2f}")
        lbl.grid(row=row+1, column=1, sticky="w")
        var.trace_add("write", lambda *args, L=lbl, V=var: L.config(text=f"{V.get():.2f}"))

    def _reset_sliders(self):
        self.x_scale.set(1.5); self.y_scale.set(1.5); self.z_scale.set(1.5)
        self.x_step.set(2.0);  self.y_step.set(2.0);  self.z_step.set(2.0)
        self.lam.set(2.0)

    # ==============================================================
    # PLOT 3D + CONTROLES DE CÁMARA
    # ==============================================================
    def _build_plot(self, side=1):
        plotfrm = ttk.Frame(self)
        plotfrm.grid(row=0, column=side, sticky="nsew")
        self.rowconfigure(0, weight=1)
        plotfrm.rowconfigure(0, weight=1)
        plotfrm.columnconfigure(0, weight=1)

        # Figura y ejes 3D
        self.fig = plt.Figure(figsize=(6,5))
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.ax.set_xlabel("X [mm]")
        self.ax.set_ylabel("Y [mm]")
        self.ax.set_zlabel("Z [mm]")
        self.ax.set_title("RV-M2 - Vista 3D")

        # Vista inicial (igual que la predeterminada anterior)
        self.default_elev = 20
        self.default_azim = -60
        self.ax.view_init(elev=self.default_elev, azim=self.default_azim)

        # Canvas embebido en Tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=plotfrm)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        # Controles de cámara
        camfrm = ttk.Frame(plotfrm)
        camfrm.grid(row=1, column=0, pady=(4, 2))

        ttk.Label(camfrm, text="Controles de vista 3D:").grid(row=0, column=0, columnspan=4, pady=(0,4))

        ttk.Button(camfrm, text="Vista original",
                   command=self._reset_view).grid(row=1, column=0, padx=4)
        ttk.Button(camfrm, text="Frontal",
                   command=lambda: self._set_view(elev=0, azim=0)).grid(row=1, column=1, padx=4)
        ttk.Button(camfrm, text="Lateral",
                   command=lambda: self._set_view(elev=0, azim=90)).grid(row=1, column=2, padx=4)
        ttk.Button(camfrm, text="Superior",
                   command=lambda: self._set_view(elev=90, azim=-90)).grid(row=1, column=3, padx=4)

        # Dibujo inicial del robot
        self._draw_robot()

    def _set_view(self, elev, azim):
        """Aplica un ángulo de cámara específico."""
        self.ax.view_init(elev=elev, azim=azim)
        self.canvas.draw_idle()

    def _reset_view(self):
        """Restaura la vista original predeterminada."""
        self.ax.view_init(elev=self.default_elev, azim=self.default_azim)
        self.canvas.draw_idle()

    # ==============================================================
    # PALANCAS (X/Y/Z)
    # ==============================================================
    def _on_drag_axis(self, axis, event):
        cv = self.lever[axis]
        y = min(max(event.y, self.lever_top[axis]), self.lever_bot[axis])
        x = cv.winfo_width() // 2
        self._move_axis_knob(axis, x, y)

        # Normalizado respecto al centro → [-1, 1], arriba positivo
        dy_norm = (self.lever_center_y[axis] - y) / self.lever_radius[axis]
        dy_norm = max(min(dy_norm, 1.0), -1.0)

        # Velocidad continua en el eje correspondiente
        speed = {"x": self.x_scale.get(), "y": self.y_scale.get(), "z": self.z_scale.get()}[axis]
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        self.target_vel[idx] = speed * dy_norm * 5.0
        self.lever_active[axis] = abs(dy_norm) > 1e-3

    def _on_release_axis(self, axis):
        cv = self.lever[axis]
        x = cv.winfo_width() // 2
        y = self.lever_center_y[axis]
        self._move_axis_knob(axis, x, y)
        idx = {"x": 0, "y": 1, "z": 2}[axis]
        self.target_vel[idx] = 0.0
        self.lever_active[axis] = False

    def _move_axis_knob(self, axis, x, y):
        r = 10
        self.lever[axis].coords(self.knob[axis], x-r, y-r, x+r, y+r)

    # ==============================================================
    # BOTONES FINOS (impulsos por click)
    # ==============================================================
    def _btn_x_minus(self):
        if not self.lever_active["x"]:
            self.impulse_pending["x"] -= self.x_step.get()

    def _btn_x_plus(self):
        if not self.lever_active["x"]:
            self.impulse_pending["x"] += self.x_step.get()

    def _btn_y_minus(self):
        if not self.lever_active["y"]:
            self.impulse_pending["y"] -= self.y_step.get()

    def _btn_y_plus(self):
        if not self.lever_active["y"]:
            self.impulse_pending["y"] += self.y_step.get()

    def _btn_z_down(self):
        if not self.lever_active["z"]:
            self.impulse_pending["z"] -= self.z_step.get()

    def _btn_z_up(self):
        if not self.lever_active["z"]:
            self.impulse_pending["z"] += self.z_step.get()

    # ==============================================================
    # CICLO DE SIMULACIÓN
    # ==============================================================
    def start(self):
        if not self.running:
            self.running = True
            self._update_led_state()
            self._tick()

    def pause(self):
        self.running = False
        self._update_led_state()
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None

    def home(self):
        self.q = np.zeros(self.model.dof, dtype=float)
        # Centrar palancas y detener velocidades
        for ax in ("x", "y", "z"):
            self._on_release_axis(ax)
            self.impulse_pending[ax] = 0.0
        self._draw_robot()
        self._update_led_state()
    
    def _set_led(self, color):
        """Actualiza el color del LED de estado."""
        self.led_canvas.itemconfig(self.led_circle, fill=color)

    def _update_led_state(self):
        """Actualiza automáticamente el LED según el estado."""
        if self.running:
            self._set_led("green")
        else:
            self._set_led("red")

    def _tick(self):
        # Construir dx a partir de velocidades continuas
        dx = self.target_vel.copy()

        # Añadir impulsos de click (un único tick) si la palanca correspondiente está centrada
        for ax, idx in (("x",0), ("y",1), ("z",2)):
            if not self.lever_active[ax] and abs(self.impulse_pending[ax]) > 0.0:
                dx[idx] += self.impulse_pending[ax]
                self.impulse_pending[ax] = 0.0

        if np.any(dx != 0.0):
            if self.use_analytic_ik.get() and hasattr(self.model, "ik_solver") and self.model.ik_solver:
                # IK analítica (suavizada)
                T, _ = fk_dh(self.model, self.q)
                current_pos = T[:3, 3]
                target_pos = current_pos + dx
                sols = self.model.ik_solver(self.model, target_pos)
                if sols:
                    q_new = sols[0]
                    self.q += 0.3 * (q_new - self.q)  # suavizado para evitar vibración
            else:
                # DLS por defecto (estable e incremental)
                self.q = ik_step_dls(self.model, self.q, dx_mm=dx, lam=self.lam.get())

        self._draw_robot()

        if self.running:
            self.after_id = self.after(int(self.dt * 1000), self._tick)

    # ==============================================================
    # DIBUJO DEL ROBOT
    # ==============================================================
    def _draw_robot(self):
        T, joints = fk_dh(self.model, self.q)
        self.ax.cla()
        xs, ys, zs = joints[:,0], joints[:,1], joints[:,2]
        self.ax.plot(xs, ys, zs, marker='o')
        self.ax.scatter([xs[0]],[ys[0]],[zs[0]], s=35, color='black')   # base
        self.ax.scatter([xs[-1]],[ys[-1]],[zs[-1]], s=50, color='red')  # efector
        self.ax.set_xlabel("X [mm]"); self.ax.set_ylabel("Y [mm]"); self.ax.set_zlabel("Z [mm]")
        self.ax.set_title("RV-M2 - Vista 3D")
        self.ax.set_xlim(-50, 500); self.ax.set_ylim(-300, 300); self.ax.set_zlim(0, 600)
        self.canvas.draw()

        p = T[:3,3]
        self.pose_var.set(f"EE: ({p[0]:7.1f}, {p[1]:7.1f}, {p[2]:7.1f}) mm")
