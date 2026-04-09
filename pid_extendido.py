"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   SIMULADOR PID EXTENDIDO — DRON 2D CON MASA VARIABLE                     ║
║   ─────────────────────────────────────────────────────                    ║
║                                                                            ║
║   Extensiones sobre el simulador profesional base:                         ║
║                                                                            ║
║   1. MASA VARIABLE (consumo de batería/combustible)                        ║
║      • Tasa de consumo proporcional al thrust                              ║
║      • El peso m(t) decrece → el integral debe adaptarse                  ║
║      • Feedforward dinámico: FF = m(t)·g en cada paso                     ║
║                                                                            ║
║   2. MODELO 2D CON PITCH (inclinación)                                     ║
║      • Ejes Y (altitud) y X (posición horizontal)                          ║
║      • Thrust se descompone: Fy = T·cos(θ), Fx = T·sin(θ)                ║
║      • PID de altitud controla magnitud del thrust                         ║
║      • PID de posición horizontal controla el ángulo pitch (θ)             ║
║      • PID de pitch estabiliza la orientación                              ║
║      • Acoplamiento: inclinarse reduce la componente vertical              ║
║                                                                            ║
║   3. TRAYECTORIAS DINÁMICAS                                                ║
║      • Sinusoidal, circular, escalera, lemniscata                          ║
║      • Evasión de obstáculos dinámicos                                     ║
║      • Métricas de tracking error                                          ║
║                                                                            ║
║   Uso: python pid_extendido.py                                             ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass, field
from typing import Tuple, List, Callable, Optional
from copy import deepcopy
from enum import Enum


# ════════════════════════════════════════════════════════════════════════
# 1. CONTROLADOR PID (reutilizado del módulo base, simplificado)
# ════════════════════════════════════════════════════════════════════════

class PIDController:
    """
    PID profesional con filtro derivativo, anti-windup por clamping,
    derivativo sobre medición, y feedforward dinámico.
    """

    def __init__(self, kp, ki, kd, setpoint=0.0, output_min=-100, output_max=100,
                 derivative_filter_N=10.0, ff_value=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.output_min = output_min
        self.output_max = output_max
        self.N = derivative_filter_N
        self.ff_value = ff_value  # feedforward dinámico

        # Estado interno
        self._integral = 0.0
        self._prev_measurement = None
        self._prev_error = None
        self._D_filtered = 0.0

    def update(self, measurement, dt):
        """Calcula salida PID. Retorna (output, diagnostics)."""
        if self._prev_measurement is None:
            self._prev_measurement = measurement
            self._prev_error = self.setpoint - measurement

        error = self.setpoint - measurement

        # ── Proporcional ──
        P = self.kp * error

        # ── Integral con anti-windup (clamping) ──
        increment = 0.5 * (error + self._prev_error) * dt  # trapezoidal
        test_integral = self._integral + self.ki * increment
        test_output = self.ff_value + P + test_integral + self._D_filtered

        # Solo integrar si no satura o si reduce la saturación
        if test_output > self.output_max or test_output < self.output_min:
            if error * self._integral <= 0:  # error reduce saturación
                self._integral += self.ki * increment
        else:
            self._integral += self.ki * increment

        I = self._integral

        # ── Derivativo sobre medición con filtro ──
        d_meas = -(measurement - self._prev_measurement)
        td = self.kd / self.kp if self.kp > 0 else 0
        tf = td / self.N if self.N > 0 else 0

        if tf > 0 and dt > 0:
            alpha = (2 * tf) / (2 * tf + dt)
            self._D_filtered = alpha * self._D_filtered + self.kd * (1 - alpha) * 2 * d_meas / dt
        elif dt > 0:
            self._D_filtered = self.kd * d_meas / dt

        D = self._D_filtered

        # ── Suma + feedforward ──
        output_unsat = self.ff_value + P + I + D
        output = np.clip(output_unsat, self.output_min, self.output_max)

        self._prev_error = error
        self._prev_measurement = measurement

        diag = {'P': P, 'I': I, 'D': D, 'FF': self.ff_value,
                'error': error, 'output': output,
                'integral_state': self._integral,
                'saturated': abs(output - output_unsat) > 1e-6}
        return output, diag

    def reset(self):
        self._integral = 0.0
        self._D_filtered = 0.0
        self._prev_measurement = None
        self._prev_error = None


# ════════════════════════════════════════════════════════════════════════
# 2. MODELO FÍSICO DEL DRON 2D CON MASA VARIABLE
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DroneParams2D:
    """Parámetros del dron 2D con consumo de batería."""
    # Masa
    initial_mass: float = 1.5          # Masa inicial total [kg]
    fuel_mass: float = 0.3             # Masa de combustible/batería [kg]
    burn_rate: float = 0.002           # Tasa de consumo [kg/N·s] (masa perdida por N de thrust por segundo)
    min_mass: float = 1.0              # Masa mínima (estructura sin combustible)

    # Física
    gravity: float = 9.81
    drag_coeff_y: float = 0.1          # Drag vertical [N·s²/m²]
    drag_coeff_x: float = 0.08         # Drag horizontal
    moment_of_inertia: float = 0.02    # Inercia rotacional [kg·m²]
    arm_length: float = 0.25           # Longitud del brazo [m]

    # Actuadores
    thrust_max: float = 40.0
    thrust_min: float = 0.0
    max_pitch_rate: float = 3.0        # Velocidad máxima de pitch [rad/s]
    max_pitch: float = np.radians(45)  # Pitch máximo [rad]
    actuator_lag: float = 0.02

    # Ruido y perturbaciones
    sensor_noise_y: float = 0.0
    sensor_noise_x: float = 0.0
    wind_profiles: List = field(default_factory=list)
    # Cada perfil: {'start': t, 'duration': s, 'force_x': N, 'force_y': N}


class DroneModel2D:
    """
    Dron con dinámica 2D:
    - Posición (x, y), velocidad (vx, vy)
    - Pitch angle (θ), pitch rate (ω)
    - Thrust T se descompone: Fy = T·cos(θ), Fx = T·sin(θ)
    - Masa variable por consumo de batería
    """

    def __init__(self, params: DroneParams2D, start_x=0.0, start_y=0.0):
        self.p = params
        # Estado
        self.x = start_x
        self.y = start_y
        self.vx = 0.0
        self.vy = 0.0
        self.pitch = 0.0       # θ [rad] (positivo = inclinado a la derecha)
        self.pitch_rate = 0.0  # ω [rad/s]
        self.mass = params.initial_mass
        self.fuel_remaining = params.fuel_mass
        self.actual_thrust = 0.0
        self.rng = np.random.default_rng(42)

    def step(self, commanded_thrust: float, commanded_pitch: float,
             dt: float, t: float):
        """
        Avanza un paso de simulación 2D.

        commanded_thrust: magnitud total del empuje [N]
        commanded_pitch: ángulo de pitch deseado [rad]

        Retorna: dict con estado completo
        """
        # ── Dinámica del actuador (lag) ──
        alpha = dt / (self.p.actuator_lag + dt)
        self.actual_thrust = self.actual_thrust + alpha * (commanded_thrust - self.actual_thrust)
        self.actual_thrust = np.clip(self.actual_thrust, self.p.thrust_min, self.p.thrust_max)

        # ── Control de pitch (modelo simplificado de torque) ──
        pitch_target = np.clip(commanded_pitch, -self.p.max_pitch, self.p.max_pitch)
        pitch_error = pitch_target - self.pitch
        # Controlador P rígido de orientación (simula autopilot interno)
        torque = 50.0 * pitch_error - 20.0 * self.pitch_rate
        angular_acc = torque / self.p.moment_of_inertia
        self.pitch_rate += angular_acc * dt
        self.pitch_rate = np.clip(self.pitch_rate, -self.p.max_pitch_rate * 5, self.p.max_pitch_rate * 5)
        self.pitch += self.pitch_rate * dt
        self.pitch = np.clip(self.pitch, -self.p.max_pitch, self.p.max_pitch)

        # ── Consumo de combustible ──
        if self.fuel_remaining > 0:
            fuel_consumed = self.p.burn_rate * self.actual_thrust * dt
            fuel_consumed = min(fuel_consumed, self.fuel_remaining)
            self.fuel_remaining -= fuel_consumed
            self.mass = self.p.min_mass + self.fuel_remaining
        else:
            self.mass = self.p.min_mass

        # ── Descomposición del thrust ──
        thrust_y = self.actual_thrust * np.cos(self.pitch)
        thrust_x = self.actual_thrust * np.sin(self.pitch)

        # ── Perturbaciones de viento ──
        wind_fx, wind_fy = 0.0, 0.0
        for wp in self.p.wind_profiles:
            if wp['start'] <= t < wp['start'] + wp['duration']:
                wind_fx += wp.get('force_x', 0.0)
                wind_fy += wp.get('force_y', 0.0)

        # ── Fuerzas verticales ──
        gravity = -self.mass * self.p.gravity
        drag_y = -self.p.drag_coeff_y * self.vy * abs(self.vy)
        fy_net = thrust_y + gravity + drag_y + wind_fy

        # ── Fuerzas horizontales ──
        drag_x = -self.p.drag_coeff_x * self.vx * abs(self.vx)
        fx_net = thrust_x + drag_x + wind_fx

        # ── Integración ──
        ay = fy_net / self.mass
        ax = fx_net / self.mass

        self.vy += ay * dt
        self.vx += ax * dt
        self.y += self.vy * dt
        self.x += self.vx * dt

        # Suelo
        if self.y < 0:
            self.y = 0.0
            self.vy = max(0.0, self.vy)

        # ── Mediciones con ruido ──
        meas_y = self.y + (self.rng.normal(0, self.p.sensor_noise_y)
                           if self.p.sensor_noise_y > 0 else 0)
        meas_x = self.x + (self.rng.normal(0, self.p.sensor_noise_x)
                           if self.p.sensor_noise_x > 0 else 0)

        return {
            'x': self.x, 'y': self.y,
            'vx': self.vx, 'vy': self.vy,
            'meas_x': meas_x, 'meas_y': meas_y,
            'pitch': self.pitch, 'pitch_rate': self.pitch_rate,
            'mass': self.mass, 'fuel': self.fuel_remaining,
            'thrust_actual': self.actual_thrust,
            'thrust_y': thrust_y, 'thrust_x': thrust_x,
            'wind_fx': wind_fx, 'wind_fy': wind_fy,
        }


# ════════════════════════════════════════════════════════════════════════
# 3. GENERADORES DE TRAYECTORIA
# ════════════════════════════════════════════════════════════════════════

def trajectory_sinusoidal(t, amp_y=2.0, amp_x=3.0, freq_y=0.1, freq_x=0.08,
                          base_y=10.0, base_x=0.0):
    """Trayectoria sinusoidal en 2D."""
    y = base_y + amp_y * np.sin(2 * np.pi * freq_y * t)
    x = base_x + amp_x * np.sin(2 * np.pi * freq_x * t)
    return x, y


def trajectory_circular(t, radius=3.0, center_x=0.0, center_y=10.0, freq=0.05):
    """Trayectoria circular."""
    angle = 2 * np.pi * freq * t
    x = center_x + radius * np.sin(angle)
    y = center_y + radius * np.cos(angle)
    return x, y


def trajectory_lemniscate(t, scale=3.0, center_x=0.0, center_y=10.0, freq=0.04):
    """Trayectoria en forma de ∞ (lemniscata de Bernoulli)."""
    angle = 2 * np.pi * freq * t
    denom = 1 + np.sin(angle) ** 2
    x = center_x + scale * np.cos(angle) / denom
    y = center_y + scale * np.sin(angle) * np.cos(angle) / denom
    return x, y


def trajectory_staircase(t, step_time=6.0, heights=None, positions=None):
    """Trayectoria escalera (cambios discretos)."""
    if heights is None:
        heights = [8, 12, 15, 10, 14, 8]
    if positions is None:
        positions = [0, 2, -1, 3, 0, -2]
    idx = min(int(t / step_time), len(heights) - 1)
    return float(positions[idx]), float(heights[idx])


def trajectory_obstacle_avoidance(t, base_y=10.0, base_x=0.0):
    """
    Trayectoria que simula evasión de obstáculos:
    vuelo estable → subida rápida → desvío lateral → regreso
    """
    if t < 5:
        return base_x, base_y
    elif t < 8:
        # Subida evasiva
        frac = (t - 5) / 3
        return base_x, base_y + 8 * frac
    elif t < 12:
        # Desvío lateral
        frac = (t - 8) / 4
        return base_x + 6 * np.sin(np.pi * frac), base_y + 8
    elif t < 16:
        # Descenso
        frac = (t - 12) / 4
        return base_x + 6 * (1 - frac), base_y + 8 * (1 - frac)
    elif t < 22:
        # Segundo obstáculo: zigzag
        frac = (t - 16) / 6
        return base_x + 4 * np.sin(4 * np.pi * frac), base_y + 3 * np.sin(2 * np.pi * frac)
    else:
        return base_x, base_y


# ════════════════════════════════════════════════════════════════════════
# 4. SIMULACIÓN 2D COMPLETA
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult2D:
    """Resultado de simulación 2D."""
    time: np.ndarray = None
    # Posición y referencia
    x: np.ndarray = None
    y: np.ndarray = None
    ref_x: np.ndarray = None
    ref_y: np.ndarray = None
    # Velocidad
    vx: np.ndarray = None
    vy: np.ndarray = None
    # Control
    thrust: np.ndarray = None
    pitch: np.ndarray = None
    pitch_cmd: np.ndarray = None
    # Masa
    mass: np.ndarray = None
    fuel: np.ndarray = None
    # PID internals
    P_y: np.ndarray = None
    I_y: np.ndarray = None
    D_y: np.ndarray = None
    P_x: np.ndarray = None
    I_x: np.ndarray = None
    error_y: np.ndarray = None
    error_x: np.ndarray = None
    # Métricas
    tracking_error_rms: float = 0.0
    max_tracking_error: float = 0.0
    total_fuel_used: float = 0.0
    label: str = ""


def simulate_2d(drone_params: DroneParams2D,
                trajectory_fn: Callable,
                total_time: float = 30.0,
                dt: float = 0.01,
                pid_gains_y: Tuple = (8.0, 2.0, 6.0),
                pid_gains_x: Tuple = (1.5, 0.3, 2.0),
                label: str = "",
                adaptive_ff: bool = True) -> SimResult2D:
    """
    Simula el dron 2D siguiendo una trayectoria.

    Arquitectura de control:
    - PID_altitude: error_y → thrust magnitude
    - PID_position: error_x → desired pitch angle
    - Inner loop pitch: autopilot interno (P-D rígido)
    """
    n = int(total_time / dt)
    # Start drone at trajectory initial position
    init_x, init_y = trajectory_fn(0)
    drone = DroneModel2D(drone_params, start_x=init_x, start_y=init_y)

    # Controladores
    pid_y = PIDController(
        kp=pid_gains_y[0], ki=pid_gains_y[1], kd=pid_gains_y[2],
        output_min=drone_params.thrust_min, output_max=drone_params.thrust_max,
    )
    # X control: PID on position, but with VERY high Kd to damp velocity
    # The derivative acts on measurement (velocity), providing natural damping
    pid_x = PIDController(
        kp=pid_gains_x[0], ki=pid_gains_x[1], kd=pid_gains_x[2],
        output_min=-drone_params.max_pitch, output_max=drone_params.max_pitch,
        derivative_filter_N=5.0,  # Heavy filtering to avoid noise amplification
    )

    # Preallocar resultado
    res = SimResult2D(
        time=np.zeros(n), x=np.zeros(n), y=np.zeros(n),
        ref_x=np.zeros(n), ref_y=np.zeros(n),
        vx=np.zeros(n), vy=np.zeros(n),
        thrust=np.zeros(n), pitch=np.zeros(n), pitch_cmd=np.zeros(n),
        mass=np.zeros(n), fuel=np.zeros(n),
        P_y=np.zeros(n), I_y=np.zeros(n), D_y=np.zeros(n),
        P_x=np.zeros(n), I_x=np.zeros(n),
        error_y=np.zeros(n), error_x=np.zeros(n),
        label=label,
    )

    for i in range(n):
        t = i * dt
        res.time[i] = t

        # ── Referencia ──
        ref_x, ref_y = trajectory_fn(t)
        res.ref_x[i] = ref_x
        res.ref_y[i] = ref_y

        # ── Feedforward dinámico (compensa masa actual) ──
        current_ff = drone.mass * drone.p.gravity if adaptive_ff else \
                     drone.p.initial_mass * drone.p.gravity
        pid_y.ff_value = current_ff

        # ── PID de altitud → thrust ──
        pid_y.setpoint = ref_y
        thrust_cmd, diag_y = pid_y.update(drone.y, dt)

        # ── PID de posición horizontal → pitch deseado ──
        pid_x.setpoint = ref_x
        pitch_cmd, diag_x = pid_x.update(drone.x, dt)

        # ── Step de la planta ──
        state = drone.step(thrust_cmd, pitch_cmd, dt, t)

        # ── Guardar ──
        res.x[i] = state['x']
        res.y[i] = state['y']
        res.vx[i] = state['vx']
        res.vy[i] = state['vy']
        res.thrust[i] = state['thrust_actual']
        res.pitch[i] = state['pitch']
        res.pitch_cmd[i] = pitch_cmd
        res.mass[i] = state['mass']
        res.fuel[i] = state['fuel']
        res.P_y[i] = diag_y['P']
        res.I_y[i] = diag_y['I']
        res.D_y[i] = diag_y['D']
        res.P_x[i] = diag_x['P']
        res.I_x[i] = diag_x['I']
        res.error_y[i] = diag_y['error']
        res.error_x[i] = diag_x['error']

    # ── Métricas ──
    dist = np.sqrt((res.x - res.ref_x)**2 + (res.y - res.ref_y)**2)
    res.tracking_error_rms = np.sqrt(np.mean(dist**2))
    res.max_tracking_error = np.max(dist)
    res.total_fuel_used = drone_params.fuel_mass - drone.fuel_remaining

    return res


# ════════════════════════════════════════════════════════════════════════
# 5. VISUALIZACIÓN
# ════════════════════════════════════════════════════════════════════════

COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c']


def _style_ax(ax, ylabel='', xlabel='', title=''):
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=7)
    if ylabel: ax.set_ylabel(ylabel, color='#c9d1d9', fontsize=8)
    if xlabel: ax.set_xlabel(xlabel, color='#c9d1d9', fontsize=8)
    if title: ax.set_title(title, color='#8b949e', fontsize=9, pad=6)
    ax.grid(True, alpha=0.12, color='#30363d')
    for s in ax.spines.values():
        s.set_color('#30363d')


def plot_2d_result(res: SimResult2D, save_path: str = None):
    """Gráfica completa de una simulación 2D."""
    fig = plt.figure(figsize=(16, 14), facecolor='#0d1117')
    gs = gridspec.GridSpec(4, 3, figure=fig, hspace=0.38, wspace=0.35,
                           left=0.06, right=0.96, top=0.94, bottom=0.04)

    fig.suptitle(f'Simulación 2D — {res.label}',
                 color='#f0f6fc', fontsize=14, fontweight='bold')

    t = res.time

    # ── 1. Trayectoria XY (grande) ──────────────────────────────
    ax_xy = fig.add_subplot(gs[0:2, 0:2])
    _style_ax(ax_xy, 'Y — Altitud (m)', 'X — Posición (m)', 'Trayectoria 2D')
    ax_xy.plot(res.ref_x, res.ref_y, '--', color='#ffd700', linewidth=1.5,
               alpha=0.7, label='Referencia', zorder=5)
    # Colorear la trayectoria real por tiempo
    points = np.array([res.x, res.y]).T.reshape(-1, 1, 2)
    from matplotlib.collections import LineCollection
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(t[0], t[-1])
    lc = LineCollection(segments, cmap='viridis', norm=norm, linewidth=2, alpha=0.9)
    lc.set_array(t[:-1])
    ax_xy.add_collection(lc)
    cbar = plt.colorbar(lc, ax=ax_xy, pad=0.02, aspect=30)
    cbar.set_label('Tiempo (s)', color='#8b949e', fontsize=8)
    cbar.ax.tick_params(colors='#8b949e', labelsize=7)
    # Marcar inicio y fin
    ax_xy.plot(res.x[0], res.y[0], 'o', color='#2ecc71', markersize=10,
               zorder=10, label='Inicio')
    ax_xy.plot(res.x[-1], res.y[-1], 's', color='#e74c3c', markersize=10,
               zorder=10, label='Fin')
    ax_xy.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d',
                 labelcolor='#c9d1d9', loc='lower left')
    ax_xy.set_aspect('equal', adjustable='datalim')

    # ── 2. Masa y combustible ───────────────────────────────────
    ax_mass = fig.add_subplot(gs[0, 2])
    _style_ax(ax_mass, 'Masa (kg)', title='Masa Variable (consumo batería)')
    ax_mass.plot(t, res.mass, color='#f39c12', linewidth=1.5)
    ax_mass.fill_between(t, res.mass, alpha=0.1, color='#f39c12')
    ax_twin = ax_mass.twinx()
    ax_twin.plot(t, res.fuel / res.fuel[0] * 100 if res.fuel[0] > 0 else res.fuel,
                 color='#e74c3c', linewidth=1, linestyle='--', alpha=0.7)
    ax_twin.set_ylabel('Batería (%)', color='#e74c3c', fontsize=7)
    ax_twin.tick_params(colors='#e74c3c', labelsize=6)

    # ── 3. Pitch ────────────────────────────────────────────────
    ax_pitch = fig.add_subplot(gs[1, 2])
    _style_ax(ax_pitch, 'Pitch (°)', title='Ángulo de Inclinación')
    ax_pitch.plot(t, np.degrees(res.pitch), color='#9b59b6', linewidth=1.0,
                  label='Real', alpha=0.8)
    ax_pitch.plot(t, np.degrees(res.pitch_cmd), color='#3498db', linewidth=0.8,
                  linestyle='--', label='Comandado', alpha=0.6)
    ax_pitch.axhline(0, color='#484f58', linewidth=0.5, linestyle=':')
    ax_pitch.legend(fontsize=6, facecolor='#161b22', edgecolor='#30363d',
                    labelcolor='#c9d1d9')

    # ── 4. Altitud Y vs tiempo ──────────────────────────────────
    ax_y = fig.add_subplot(gs[2, 0:2])
    _style_ax(ax_y, 'Altitud Y (m)', title='Seguimiento Vertical')
    ax_y.plot(t, res.ref_y, '--', color='#ffd700', linewidth=1.2, alpha=0.7, label='Ref Y')
    ax_y.plot(t, res.y, color='#2ecc71', linewidth=1.5, label='Y real')
    ax_y.fill_between(t, res.ref_y, res.y, alpha=0.08, color='#e74c3c')
    ax_y.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # ── 5. Posición X vs tiempo ─────────────────────────────────
    ax_x = fig.add_subplot(gs[2, 2])
    _style_ax(ax_x, 'Posición X (m)', title='Seguimiento Horizontal')
    ax_x.plot(t, res.ref_x, '--', color='#ffd700', linewidth=1.2, alpha=0.7, label='Ref X')
    ax_x.plot(t, res.x, color='#3498db', linewidth=1.5, label='X real')
    ax_x.legend(fontsize=6, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # ── 6. Thrust y componentes PID ─────────────────────────────
    ax_thr = fig.add_subplot(gs[3, 0])
    _style_ax(ax_thr, 'N', 'Tiempo (s)', 'Thrust Total')
    ax_thr.plot(t, res.thrust, color='#3498db', linewidth=1.0)
    ax_thr.axhline(res.mass[0] * 9.81, color='#484f58', linewidth=0.5,
                   linestyle=':', label=f'm₀·g={res.mass[0]*9.81:.1f}N')
    ax_thr.axhline(res.mass[-1] * 9.81, color='#e74c3c', linewidth=0.5,
                   linestyle=':', label=f'm_f·g={res.mass[-1]*9.81:.1f}N')
    ax_thr.legend(fontsize=6, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # ── 7. Componentes PID altitud ──────────────────────────────
    ax_pid = fig.add_subplot(gs[3, 1])
    _style_ax(ax_pid, 'N', 'Tiempo (s)', 'PID Altitud (P,I,D)')
    ax_pid.plot(t, res.P_y, color='#2ecc71', linewidth=0.8, label='P')
    ax_pid.plot(t, res.I_y, color='#3498db', linewidth=0.8, label='I')
    ax_pid.plot(t, res.D_y, color='#e74c3c', linewidth=0.8, label='D')
    ax_pid.axhline(0, color='#484f58', linewidth=0.5)
    ax_pid.legend(fontsize=6, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # ── 8. Error de tracking ────────────────────────────────────
    ax_err = fig.add_subplot(gs[3, 2])
    _style_ax(ax_err, 'm', 'Tiempo (s)', 'Error de Tracking')
    dist = np.sqrt(res.error_x**2 + res.error_y**2)
    ax_err.plot(t, dist, color='#e74c3c', linewidth=1.0, alpha=0.8)
    ax_err.fill_between(t, dist, alpha=0.1, color='#e74c3c')
    ax_err.text(0.95, 0.95,
                f'RMS: {res.tracking_error_rms:.3f}m\n'
                f'Max: {res.max_tracking_error:.3f}m\n'
                f'Fuel: {res.total_fuel_used:.3f}kg',
                transform=ax_err.transAxes, fontsize=8, color='#c9d1d9',
                verticalalignment='top', horizontalalignment='right',
                fontfamily='monospace',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#0d1117',
                          edgecolor='#30363d', alpha=0.9))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()


def plot_comparison_2d(results: List[SimResult2D], title: str,
                       save_path: str = None):
    """Compara múltiples simulaciones 2D."""
    fig = plt.figure(figsize=(16, 12), facecolor='#0d1117')
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.35,
                           left=0.06, right=0.96, top=0.93, bottom=0.06)
    fig.suptitle(title, color='#f0f6fc', fontsize=13, fontweight='bold')

    # ── 1. Trayectoria XY comparativa ───────────────────────────
    ax_xy = fig.add_subplot(gs[0:2, 0:2])
    _style_ax(ax_xy, 'Y (m)', 'X (m)', 'Trayectorias 2D')
    # Referencia (del primer resultado)
    ax_xy.plot(results[0].ref_x, results[0].ref_y, '--', color='#ffd700',
               linewidth=2, alpha=0.6, label='Referencia', zorder=5)
    for i, res in enumerate(results):
        c = COLORS[i % len(COLORS)]
        ax_xy.plot(res.x, res.y, color=c, linewidth=1.5, alpha=0.8,
                   label=res.label)
    ax_xy.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d',
                 labelcolor='#c9d1d9')
    ax_xy.set_aspect('equal', adjustable='datalim')

    # ── 2. Masa vs tiempo ───────────────────────────────────────
    ax_mass = fig.add_subplot(gs[0, 2])
    _style_ax(ax_mass, 'kg', title='Masa (consumo batería)')
    for i, res in enumerate(results):
        ax_mass.plot(res.time, res.mass, color=COLORS[i % len(COLORS)],
                     linewidth=1.2, alpha=0.8)

    # ── 3. Pitch comparativo ────────────────────────────────────
    ax_pitch = fig.add_subplot(gs[1, 2])
    _style_ax(ax_pitch, 'deg', title='Pitch')
    for i, res in enumerate(results):
        ax_pitch.plot(res.time, np.degrees(res.pitch),
                      color=COLORS[i % len(COLORS)], linewidth=0.8, alpha=0.7)
    ax_pitch.axhline(0, color='#484f58', linewidth=0.5)

    # ── 4. Error Y ──────────────────────────────────────────────
    ax_ey = fig.add_subplot(gs[2, 0])
    _style_ax(ax_ey, 'm', 'Tiempo (s)', 'Error Vertical')
    for i, res in enumerate(results):
        ax_ey.plot(res.time, res.error_y, color=COLORS[i % len(COLORS)],
                   linewidth=0.8, alpha=0.8)
    ax_ey.axhline(0, color='#484f58', linewidth=0.5)

    # ── 5. Error X ──────────────────────────────────────────────
    ax_ex = fig.add_subplot(gs[2, 1])
    _style_ax(ax_ex, 'm', 'Tiempo (s)', 'Error Horizontal')
    for i, res in enumerate(results):
        ax_ex.plot(res.time, res.error_x, color=COLORS[i % len(COLORS)],
                   linewidth=0.8, alpha=0.8)
    ax_ex.axhline(0, color='#484f58', linewidth=0.5)

    # ── 6. Tabla de métricas ────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[2, 2])
    ax_tbl.set_facecolor('#161b22')
    ax_tbl.axis('off')

    text = "  MÉTRICAS\n  " + "─" * 32 + "\n"
    for i, res in enumerate(results):
        text += (f"  {COLORS[i % len(COLORS)][1:3].upper()} {res.label[:18]:<18s}\n"
                 f"     RMS:  {res.tracking_error_rms:>7.3f} m\n"
                 f"     Max:  {res.max_tracking_error:>7.3f} m\n"
                 f"     Fuel: {res.total_fuel_used:>7.3f} kg\n")
    ax_tbl.text(0.05, 0.95, text, transform=ax_tbl.transAxes,
                fontfamily='monospace', fontsize=8, color='#c9d1d9',
                verticalalignment='top')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()


# ════════════════════════════════════════════════════════════════════════
# 6. ESCENARIOS
# ════════════════════════════════════════════════════════════════════════

def run_all():
    """Batería completa de escenarios extendidos."""

    DT = 0.01
    T = 40.0

    # ── Parámetros base del dron 2D ──
    base_drone = DroneParams2D(
        initial_mass=1.5, fuel_mass=0.3, burn_rate=0.002, min_mass=1.0,
        gravity=9.81, drag_coeff_y=0.1, drag_coeff_x=0.08,
        thrust_max=40.0, actuator_lag=0.02,
        sensor_noise_y=0.02, sensor_noise_x=0.02,
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 8: Masa Variable — FF Adaptativo vs Estático
    # ═══════════════════════════════════════════════════════════════
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 8: Masa Variable — Feedforward Adaptativo       ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Dron con alto consumo para hacer visible el efecto
    heavy_drone = DroneParams2D(
        initial_mass=2.0, fuel_mass=0.8, burn_rate=0.005, min_mass=1.2,
        gravity=9.81, drag_coeff_y=0.1, drag_coeff_x=0.08,
        thrust_max=50.0, actuator_lag=0.02,
    )

    def hover_then_step(t):
        return 0.0, 10.0 if t < 5 else 15.0

    res_adaptive = simulate_2d(
        heavy_drone, hover_then_step, T, DT,
        pid_gains_y=(8.0, 2.0, 6.0), pid_gains_x=(1.0, 0.2, 1.5),
        label="FF Adaptativo m(t)·g", adaptive_ff=True
    )
    res_static = simulate_2d(
        heavy_drone, hover_then_step, T, DT,
        pid_gains_y=(8.0, 2.0, 6.0), pid_gains_x=(1.0, 0.2, 1.5),
        label="FF Estático m₀·g", adaptive_ff=False
    )

    print(f"  Adaptativo → RMS={res_adaptive.tracking_error_rms:.3f}m  "
          f"Fuel={res_adaptive.total_fuel_used:.3f}kg  "
          f"Masa final={res_adaptive.mass[-1]:.3f}kg")
    print(f"  Estático   → RMS={res_static.tracking_error_rms:.3f}m  "
          f"Fuel={res_static.total_fuel_used:.3f}kg  "
          f"Masa final={res_static.mass[-1]:.3f}kg")

    plot_comparison_2d(
        [res_adaptive, res_static],
        'Escenario 8: Masa Variable — Feedforward Adaptativo vs Estático\n'
        f'Masa: {heavy_drone.initial_mass}→{heavy_drone.min_mass}kg '
        f'(burn_rate={heavy_drone.burn_rate} kg/N·s)',
        save_path='escenario8_masa_variable.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 9: Modelo 2D — Trayectoria Circular
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 9: Modelo 2D — Trayectoria Circular             ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    res_circle = simulate_2d(
        base_drone, trajectory_circular, T, DT,
        pid_gains_y=(10.0, 2.5, 7.0), pid_gains_x=(0.3, 0.05, 1.5),
        label="Circular (R=3m)"
    )
    print(f"  RMS={res_circle.tracking_error_rms:.3f}m  "
          f"Max error={res_circle.max_tracking_error:.3f}m  "
          f"Fuel={res_circle.total_fuel_used:.3f}kg")
    plot_2d_result(res_circle, save_path='escenario9_circular.png')

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 10: Comparación de Trayectorias
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 10: Comparación de Trayectorias                  ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    gains_y = (10.0, 2.5, 7.0)
    gains_x = (0.3, 0.05, 1.5)  # Bajo Kp, alto Kd — doble integrador vía pitch

    trajectories = [
        ("Sinusoidal", trajectory_sinusoidal),
        ("Circular", trajectory_circular),
        ("Lemniscata (∞)", trajectory_lemniscate),
        ("Escalera", trajectory_staircase),
    ]

    results_traj = []
    for name, fn in trajectories:
        res = simulate_2d(base_drone, fn, T, DT,
                          pid_gains_y=gains_y, pid_gains_x=gains_x, label=name)
        results_traj.append(res)
        print(f"  {name:20s} → RMS={res.tracking_error_rms:.3f}m  "
              f"Max={res.max_tracking_error:.3f}m  Fuel={res.total_fuel_used:.3f}kg")

    plot_comparison_2d(
        results_traj,
        'Escenario 10: Comparación de Trayectorias Dinámicas\n'
        'Mismas ganancias PID, diferentes trayectorias de referencia',
        save_path='escenario10_trayectorias.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 11: Evasión de Obstáculos con Viento
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 11: Evasión de Obstáculos con Ráfagas de Viento ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    windy_drone = DroneParams2D(
        **{**base_drone.__dict__,
           'wind_profiles': [
               {'start': 7.0, 'duration': 2.0, 'force_x': 3.0, 'force_y': -2.0},
               {'start': 18.0, 'duration': 3.0, 'force_x': -4.0, 'force_y': 1.5},
           ]}
    )

    res_obst_calm = simulate_2d(
        base_drone, trajectory_obstacle_avoidance, 30.0, DT,
        pid_gains_y=(10.0, 2.5, 7.0), pid_gains_x=(0.3, 0.05, 1.5),
        label="Sin viento"
    )
    res_obst_wind = simulate_2d(
        windy_drone, trajectory_obstacle_avoidance, 30.0, DT,
        pid_gains_y=(10.0, 2.5, 7.0), pid_gains_x=(0.3, 0.05, 1.5),
        label="Con ráfagas"
    )

    print(f"  Sin viento  → RMS={res_obst_calm.tracking_error_rms:.3f}m  "
          f"Max={res_obst_calm.max_tracking_error:.3f}m")
    print(f"  Con ráfagas → RMS={res_obst_wind.tracking_error_rms:.3f}m  "
          f"Max={res_obst_wind.max_tracking_error:.3f}m")

    plot_comparison_2d(
        [res_obst_calm, res_obst_wind],
        'Escenario 11: Evasión de Obstáculos con Perturbaciones de Viento\n'
        'Ráfagas: 3N@t=7s (→,↓) y -4N@t=18s (←,↑)',
        save_path='escenario11_obstaculos_viento.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 12: Efecto de las Ganancias del PID Horizontal
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 12: Tuning del PID Horizontal (Posición X)      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    gain_sets = [
        ("Conservador",  (10, 2.5, 7), (0.15, 0.02, 0.8)),
        ("Balanceado",   (10, 2.5, 7), (0.3, 0.05, 1.5)),
        ("Agresivo",     (10, 2.5, 7), (0.6, 0.1, 2.5)),
    ]

    results_gains = []
    for name, gy, gx in gain_sets:
        res = simulate_2d(base_drone, trajectory_sinusoidal, T, DT,
                          pid_gains_y=gy, pid_gains_x=gx, label=name)
        results_gains.append(res)
        print(f"  {name:15s} Kp_x={gx[0]:<4} → RMS={res.tracking_error_rms:.3f}m  "
              f"Max pitch={np.degrees(np.max(np.abs(res.pitch))):.1f}°")

    plot_comparison_2d(
        results_gains,
        'Escenario 12: Tuning del PID Horizontal\n'
        'Misma trayectoria sinusoidal, diferentes ganancias del controlador X',
        save_path='escenario12_tuning_horizontal.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 13: Análisis Detallado — Lemniscata con Todo
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 13: Análisis Detallado — Lemniscata Completa    ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    full_drone = DroneParams2D(
        initial_mass=1.8, fuel_mass=0.5, burn_rate=0.003, min_mass=1.0,
        gravity=9.81, drag_coeff_y=0.1, drag_coeff_x=0.08,
        thrust_max=45.0, actuator_lag=0.02,
        sensor_noise_y=0.03, sensor_noise_x=0.03,
        wind_profiles=[
            {'start': 12.0, 'duration': 4.0, 'force_x': 2.0, 'force_y': -1.5},
            {'start': 28.0, 'duration': 3.0, 'force_x': -3.0, 'force_y': 2.0},
        ]
    )

    res_full = simulate_2d(
        full_drone, trajectory_lemniscate, T, DT,
        pid_gains_y=(12.0, 3.0, 8.0), pid_gains_x=(0.3, 0.05, 1.5),
        label="Lemniscata + viento + batería + ruido"
    )

    print(f"  RMS={res_full.tracking_error_rms:.3f}m  "
          f"Max={res_full.max_tracking_error:.3f}m  "
          f"Fuel={res_full.total_fuel_used:.3f}kg  "
          f"Masa final={res_full.mass[-1]:.2f}kg")

    plot_2d_result(res_full, save_path='escenario13_lemniscata_completa.png')

    # ═══════════════════════════════════════════════════════════════
    print("\n" + "═" * 64)
    print("  Simulaciones 2D extendidas completadas.")
    print("  Gráficas guardadas:")
    print("    • escenario8_masa_variable.png")
    print("    • escenario9_circular.png")
    print("    • escenario10_trayectorias.png")
    print("    • escenario11_obstaculos_viento.png")
    print("    • escenario12_tuning_horizontal.png")
    print("    • escenario13_lemniscata_completa.png")
    print("═" * 64)


# ════════════════════════════════════════════════════════════════════════
# 7. PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_all()
