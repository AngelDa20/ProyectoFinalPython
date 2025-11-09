# Proyecto Final Python | Simulador de un brazo robot de 5 GDL (RV-M2)
## Diplomado de Programación en Python Especializado para Ingenieros
Alumno: Angel Gamarra.

Este proyecto integra los conocimientos obtenidos durante el diplomado de Python.
Consiste en un simulador interactivo del brazo robótico **Mitsubishi RV-M2** (5 grados de libertad), desarrollado en **Python** utilizando **NumPy**, **Matplotlib**, **Tkinter** y **Pandas** para el manejo de datos del modelo cinemático.

El sistema permite **visualizar y controlar el movimiento del efector final (tool)** del robot mediante una interfaz gráfica con palancas virtuales y control de parámetros dinámicos.  

---

## Estructura del Proyecto
```bash
ProyectoFinalPython/
│
├── config_csv/ # Archivos de configuración del modelo
│ ├── base.csv # Matriz base del robot
│ ├── dh.csv # Parámetros DH (Denavit-Hartenberg)
│ ├── limits.csv # Límites articulares
│ └── tool.csv # Matriz de herramienta (efector final)
│
├── rvcore/ # Núcleo lógico y matemático del simulador
│ ├── controllers.py # Controladores (PID y modos futuros | aun no implementado)
│ ├── ik_analytic.py # Cinemática inversa analítica del RV-M2
│ ├── ik.py # Cinemática inversa DLS (numérica)
│ ├── io.py # Funciones de entrada/salida
│ ├── kinematics.py # Cinemática directa (FK)
│ ├── robot_model.py # Carga de archivos CSV y creación del modelo
│ └── utils.py # Funciones auxiliares (wrap_to_pi, clip_joints)
│
├── ui/ # Interfaz gráfica y visualización
│ ├── gui_tk.py # Interfaz Tkinter con palancas y control 3D
│ └── viz_matplotlib.py # Funciones de visualización con Matplotlib
│
├── main.py # Punto de entrada principal del programa
├── requirements.txt # Dependencias del entorno (NumPy, Tkinter, etc.)
├── README.md # Documentación del proyecto
└── LICENSE # Licencia del proyecto
```

---

## Fundamentos Técnicos

- **Cinemática directa (FK):**  
  Calcula la posición y orientación del efector final usando la cadena **Denavit-Hartenberg (DH)** cargada desde `dh.csv`.

- **Cinemática inversa (IK):**  
  Implementa dos métodos:
  - **DLS (Damped Least Squares):** método numérico robusto para trayectorias suaves.
  - **Analítica RV-M2:** solución directa de ángulos de las 5 articulaciones (opcional).

- **Límites articulares y recorte:**  
  Se aplican automáticamente al calcular los movimientos, usando los datos de `limits.csv`.

---

## Interfaz de Usuario (Tkinter)

La interfaz gráfica permite controlar el movimiento del efector mediante **palancas virtuales** y **botones de ajuste fino**.  
Cada eje (X, Y, Z) se controla de manera independiente.

---

## Elementos principales:

| Elemento | Descripción |
|-----------|-------------|
| **Palanca X/Y/Z** | Controla el movimiento continuo en el eje respectivo. |
| **−X / +X, −Y / +Y, −Z / +Z** | Movimiento fino por pasos (impulso). |
| **Vel X/Y/Z (mm/tick)** | Escala de velocidad de cada eje durante el arrastre. |
| **Paso X/Y/Z (mm/click)** | Distancia desplazada al pulsar los botones de impulso. |
| **Damping λ (IK)** | Factor de amortiguamiento para la cinemática inversa DLS. |
| **Iniciar / Pausa / Home** | Control del ciclo de simulación y retorno a la posición inicial. |
| **Estado (LED)** | 🔴 detenido / 🟢 en ejecución. |
| **Modo IK** | Activa la solución de cinemática inversa analítica del RV-M2. |
| **EE: (x, y, z)** | EE =  End Effector, es la posición actual del efector final. |
| **Vista 3D (Frontal / Lateral / Superior / Original)** | Controles de cámara para la vista en el gráfico 3D. |

---

## Librerías Utilizadas

| Librería | Uso principal |
|-----------|----------------|
| **NumPy** | Cálculo matricial, rotaciones y transformaciones DH. |
| **Matplotlib** | Visualización 3D del robot en tiempo real. |
| **Tkinter** | Interfaz gráfica con sliders, botones y canvas. |
| **Pandas** | Lectura de los archivos CSV de configuración (DH, límites, base, tool). |

---

## Archivos de configuración CSV

- **`dh.csv`** → Parámetros Denavit-Hartenberg: θ, d, a, α  
- **`limits.csv`** → Límites articulares.  
- **`base.csv`** → Matriz de transformación de la base del robot.  
- **`tool.csv`** → Matriz del efector final (por defecto identidad).  

Estos valores se utilizan para reconstruir el modelo cinemático y graficar el robot con precisión.

---

## Ejecución

```bash
python main.py
