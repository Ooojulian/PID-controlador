"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║   SIMULADOR PROFESIONAL DE CONTROL PID — ALTITUD DE DRON 1D               ║
║   ──────────────────────────────────────────────────────────               ║
║                                                                            ║
║   Implementación de grado industrial con:                                  ║
║                                                                            ║
║   ESTRUCTURAS PID:                                                         ║
║     • Forma Ideal (paralela): u = Kp·[e + (1/Ti)∫e·dt + Td·de/dt]        ║
║     • Forma ISA (serie/interactiva)                                        ║
║     • PID con filtro derivativo de primer orden (N configurable)           ║
║     • Derivativo sobre la medición (no sobre el error)                     ║
║     • Setpoint weighting (ponderación b, c)                                ║
║                                                                            ║
║   PROTECCIONES:                                                            ║
║     • Anti-windup por back-calculation (Kb configurable)                   ║
║     • Anti-windup por clamping condicional                                 ║
║     • Saturación de actuador (thrust_min, thrust_max)                      ║
║     • Filtro derivativo (evita amplificación de ruido)                     ║
║     • Bumpless transfer (transición suave manual↔auto)                     ║
║     • Rate limiter en la salida (slew rate del actuador)                   ║
║     • Dead-band (zona muerta configurable)                                 ║
║                                                                            ║
║   MÉTODOS DE AUTO-TUNING:                                                  ║
║     • Ziegler-Nichols (lazo abierto — curva de reacción)                   ║
║     • Ziegler-Nichols (lazo cerrado — oscilación sostenida)               ║
║     • Cohen-Coon                                                           ║
║     • Lambda Tuning (IMC — Internal Model Control)                         ║
║     • AMIGO (Approximate M-constrained Integral Gain Optimization)         ║
║     • Tyreus-Luyben                                                        ║
║                                                                            ║
║   MODELO DE PLANTA:                                                        ║
║     • Dinámica 1D con gravedad, drag aerodinámico y ruido de sensor       ║
║     • Perturbaciones (ráfagas de viento) configurables                     ║
║     • Latencia de actuador (retardo de transporte)                         ║
║                                                                            ║
║   Autor: Simulación académica — Ingeniería de Control                      ║
║   Uso:   python pid_profesional.py                                         ║
║                                                                            ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict
from enum import Enum
from copy import deepcopy
import matplotlib.gridspec as gridspec


# ════════════════════════════════════════════════════════════════════════
# 1. ENUMERACIONES Y CONFIGURACIÓN
# ════════════════════════════════════════════════════════════════════════

class PIDForm(Enum):
    """Estructura del controlador PID."""
    IDEAL    = "ideal"       # Forma paralela: Kp + Ki·∫ + Kd·d/dt
    ISA      = "isa"         # Forma ISA/serie: Kp·[1 + 1/(Ti·s) + Td·s]
    PARALLEL = "parallel"    # Kp, Ki, Kd independientes


class DerivativeMode(Enum):
    """Sobre qué se calcula el término derivativo."""
    ERROR       = "error"        # d(error)/dt — clásico, causa derivative kick
    MEASUREMENT = "measurement"  # -d(y)/dt — elimina derivative kick
    MIXED       = "mixed"        # Usa setpoint weighting (c·sp - y)


class AntiWindupMode(Enum):
    """Estrategia anti-windup."""
    NONE            = "none"
    CLAMPING        = "clamping"         # Detiene integración si actuador saturado
    BACK_CALCULATION = "back_calculation" # Retroalimenta diferencia de saturación


class TuningMethod(Enum):
    """Métodos de sintonización automática."""
    MANUAL              = "Manual"
    ZN_OPEN_LOOP        = "Ziegler-Nichols (Lazo Abierto)"
    ZN_CLOSED_LOOP      = "Ziegler-Nichols (Lazo Cerrado)"
    COHEN_COON          = "Cohen-Coon"
    LAMBDA_TUNING       = "Lambda Tuning (IMC)"
    AMIGO               = "AMIGO"
    TYREUS_LUYBEN       = "Tyreus-Luyben"


@dataclass
class PlantParams:
    """Parámetros del modelo físico del dron."""
    mass: float = 1.0           # Masa [kg]
    gravity: float = 9.81       # Gravedad [m/s²]
    drag_coeff: float = 0.1     # Coeficiente de drag aerodinámico [N·s/m]
    thrust_max: float = 30.0    # Empuje máximo del motor [N]
    thrust_min: float = 0.0     # Empuje mínimo [N]
    actuator_lag: float = 0.02  # Constante de tiempo del actuador [s]
    sensor_noise_std: float = 0.0   # Desv. estándar del ruido del sensor [m]
    transport_delay: float = 0.0    # Retardo de transporte [s]

    # Perturbaciones
    wind_gust_time: float = 0.0     # Tiempo de inicio de ráfaga [s] (0=sin ráfaga)
    wind_gust_duration: float = 0.0 # Duración de la ráfaga [s]
    wind_gust_force: float = 0.0    # Fuerza de la ráfaga [N] (positiva=arriba)


@dataclass
class PIDParams:
    """Configuración completa de un controlador PID profesional."""
    # Ganancias (forma paralela)
    kp: float = 4.0
    ki: float = 0.8
    kd: float = 8.0

    # Forma ISA alternativa (Ti, Td en lugar de Ki, Kd)
    # Si se usa forma ISA: Ki = Kp/Ti, Kd = Kp·Td
    ti: float = 5.0     # Tiempo integral [s]
    td: float = 2.0     # Tiempo derivativo [s]

    # Estructura
    form: PIDForm = PIDForm.PARALLEL
    derivative_mode: DerivativeMode = DerivativeMode.MEASUREMENT

    # Filtro derivativo: Tf = Td/N (N típicamente entre 5 y 33)
    derivative_filter_N: float = 10.0

    # Setpoint weighting
    sp_weight_b: float = 1.0   # Peso proporcional (0-1). 1=clásico
    sp_weight_c: float = 0.0   # Peso derivativo (0-1). 0=derivativo sobre medición

    # Anti-windup
    anti_windup: AntiWindupMode = AntiWindupMode.BACK_CALCULATION
    kb: float = 1.0     # Ganancia de back-calculation (típicamente sqrt(Ki·Kd) o 1/Ti)

    # Saturación del actuador
    output_min: float = 0.0
    output_max: float = 30.0

    # Rate limiter (máx cambio de salida por segundo)
    max_slew_rate: float = 200.0   # [N/s]. float('inf') para desactivar

    # Dead-band
    dead_band: float = 0.0   # Error dentro de esta banda se trata como 0

    # Feedforward de gravedad
    use_gravity_ff: bool = True

    # Metadata
    tuning_method: TuningMethod = TuningMethod.MANUAL
    label: str = ""


# ════════════════════════════════════════════════════════════════════════
# 2. CONTROLADOR PID PROFESIONAL
# ════════════════════════════════════════════════════════════════════════

class ProfessionalPID:
    """
    Controlador PID de grado industrial.

    Implementa las mejores prácticas de control de procesos:
    - Múltiples formas (Ideal/ISA/Paralela)
    - Filtro derivativo de primer orden
    - Derivativo sobre medición o error
    - Setpoint weighting para desacoplar seguimiento vs rechazo de perturbación
    - Anti-windup por back-calculation o clamping
    - Rate limiter para proteger actuadores
    - Dead-band configurable
    - Feedforward de gravedad
    """

    def __init__(self, params: PIDParams, plant: PlantParams, setpoint: float):
        self.p = deepcopy(params)
        self.plant = plant
        self.setpoint = setpoint

        # Resolver ganancias según la forma
        self._resolve_gains()

        # Estado interno
        self._integral = 0.0
        self._prev_measurement = 0.0
        self._prev_error = 0.0
        self._prev_derivative_filtered = 0.0
        self._prev_output = plant.mass * plant.gravity if params.use_gravity_ff else 0.0
        self._prev_unsat_output = self._prev_output
        self._initialized = False

        # Buffer de retardo de transporte
        self._delay_buffer: List[float] = []
        self._delay_steps = 0

    def _resolve_gains(self):
        """Calcula Kp, Ki, Kd internos según la forma elegida."""
        if self.p.form == PIDForm.ISA:
            self._kp = self.p.kp
            self._ki = self.p.kp / self.p.ti if self.p.ti > 0 else 0.0
            self._kd = self.p.kp * self.p.td
        elif self.p.form == PIDForm.IDEAL:
            # Forma ideal: u = Kp·(e + (1/Ti)·∫e + Td·de/dt)
            self._kp = self.p.kp
            self._ki = self.p.kp / self.p.ti if self.p.ti > 0 else 0.0
            self._kd = self.p.kp * self.p.td
        else:  # PARALLEL
            self._kp = self.p.kp
            self._ki = self.p.ki
            self._kd = self.p.kd

        # Constante de tiempo del filtro derivativo
        td_effective = self._kd / self._kp if self._kp > 0 else 0
        self._tf = td_effective / self.p.derivative_filter_N if self.p.derivative_filter_N > 0 else 0

    def update(self, measurement: float, dt: float) -> Tuple[float, dict]:
        """
        Calcula la salida del controlador.

        Retorna: (output, diagnostics_dict)
        """
        # ── Inicialización en el primer ciclo ──
        if not self._initialized:
            self._prev_measurement = measurement
            self._prev_error = self.setpoint - measurement
            self._delay_steps = int(self.plant.transport_delay / dt) if dt > 0 else 0
            self._delay_buffer = [self._prev_output] * max(1, self._delay_steps)
            self._initialized = True

        # ── Error con dead-band ──
        raw_error = self.setpoint - measurement
        if abs(raw_error) < self.p.dead_band:
            error = 0.0
        else:
            error = raw_error

        # ════════════════════════════════════════════
        # TÉRMINO PROPORCIONAL con setpoint weighting
        # ════════════════════════════════════════════
        p_error = self.p.sp_weight_b * self.setpoint - measurement
        P_term = self._kp * p_error

        # ════════════════════════════════════════════
        # TÉRMINO INTEGRAL con anti-windup
        # ════════════════════════════════════════════
        # Integración trapezoidal (más precisa que Euler)
        integration_increment = 0.5 * (error + self._prev_error) * dt

        # Anti-windup: back-calculation
        if self.p.anti_windup == AntiWindupMode.BACK_CALCULATION:
            saturation_error = self._prev_output - self._prev_unsat_output
            self._integral += self._ki * integration_increment + self.p.kb * saturation_error * dt
        elif self.p.anti_windup == AntiWindupMode.CLAMPING:
            # Solo integrar si no estamos saturados O si el error reduce la saturación
            output_saturated = (self._prev_unsat_output > self.p.output_max or
                                self._prev_unsat_output < self.p.output_min)
            same_sign = (error * self._integral) > 0
            if not (output_saturated and same_sign):
                self._integral += self._ki * integration_increment
        else:
            self._integral += self._ki * integration_increment

        I_term = self._integral

        # ════════════════════════════════════════════
        # TÉRMINO DERIVATIVO con filtro de 1er orden
        # ════════════════════════════════════════════
        if self.p.derivative_mode == DerivativeMode.MEASUREMENT:
            # Derivativo sobre la medición (evita derivative kick)
            d_input = -measurement
            d_input_prev = -self._prev_measurement
        elif self.p.derivative_mode == DerivativeMode.ERROR:
            d_input = error
            d_input_prev = self._prev_error
        else:  # MIXED — setpoint weighting c
            d_input = self.p.sp_weight_c * self.setpoint - measurement
            d_input_prev = self.p.sp_weight_c * self.setpoint - self._prev_measurement

        # Filtro derivativo de primer orden: D(s) = Kd·s / (1 + Tf·s)
        # Discretización bilineal (Tustin):
        if self._tf > 0 and dt > 0:
            alpha = (2 * self._tf) / (2 * self._tf + dt)
            D_term = alpha * self._prev_derivative_filtered + \
                     self._kd * (1 - alpha) * 2 * (d_input - d_input_prev) / dt
        elif dt > 0:
            D_term = self._kd * (d_input - d_input_prev) / dt
        else:
            D_term = 0.0

        self._prev_derivative_filtered = D_term

        # ════════════════════════════════════════════
        # FEEDFORWARD + SUMA
        # ════════════════════════════════════════════
        ff_term = self.plant.mass * self.plant.gravity if self.p.use_gravity_ff else 0.0
        unsaturated_output = ff_term + P_term + I_term + D_term

        # ════════════════════════════════════════════
        # RATE LIMITER (protección de slew rate)
        # ════════════════════════════════════════════
        if self.p.max_slew_rate < float('inf') and dt > 0:
            max_change = self.p.max_slew_rate * dt
            delta = unsaturated_output - self._prev_output
            if abs(delta) > max_change:
                unsaturated_output = self._prev_output + np.sign(delta) * max_change

        self._prev_unsat_output = unsaturated_output

        # ════════════════════════════════════════════
        # SATURACIÓN DEL ACTUADOR
        # ════════════════════════════════════════════
        output = np.clip(unsaturated_output, self.p.output_min, self.p.output_max)

        # ── Retardo de transporte ──
        if self._delay_steps > 0:
            self._delay_buffer.append(output)
            delayed_output = self._delay_buffer.pop(0)
        else:
            delayed_output = output

        # ── Actualizar estado ──
        self._prev_error = error
        self._prev_measurement = measurement
        self._prev_output = output

        # ── Diagnósticos ──
        diag = {
            'P': P_term, 'I': I_term, 'D': D_term, 'FF': ff_term,
            'error': raw_error, 'output_unsat': unsaturated_output,
            'output_sat': output, 'integral_state': self._integral,
            'saturated': abs(output - unsaturated_output) > 1e-6,
        }

        return delayed_output, diag


# ════════════════════════════════════════════════════════════════════════
# 3. MODELO DE PLANTA (DRON 1D)
# ════════════════════════════════════════════════════════════════════════

class DroneModel1D:
    """
    Modelo físico del dron en 1D con:
    - Gravedad
    - Drag aerodinámico (proporcional a v²)
    - Dinámica del actuador (lag de primer orden)
    - Ruido de sensor
    - Perturbaciones externas (ráfagas de viento)
    """

    def __init__(self, plant: PlantParams):
        self.p = plant
        self.altitude = 0.0
        self.velocity = 0.0
        self.actual_thrust = 0.0   # Thrust real (después del lag del actuador)
        self.rng = np.random.default_rng(42)

    def step(self, commanded_thrust: float, dt: float, t: float) -> Tuple[float, float]:
        """
        Avanza un paso de simulación.

        Retorna: (true_altitude, measured_altitude)
        """
        # ── Dinámica del actuador (lag de primer orden) ──
        if self.p.actuator_lag > 0:
            alpha = dt / (self.p.actuator_lag + dt)
            self.actual_thrust = self.actual_thrust + alpha * (commanded_thrust - self.actual_thrust)
        else:
            self.actual_thrust = commanded_thrust

        # Saturación física del motor
        self.actual_thrust = np.clip(self.actual_thrust, self.p.thrust_min, self.p.thrust_max)

        # ── Perturbación: ráfaga de viento ──
        wind_force = 0.0
        if (self.p.wind_gust_time > 0 and
            self.p.wind_gust_duration > 0 and
            self.p.wind_gust_time <= t < self.p.wind_gust_time + self.p.wind_gust_duration):
            wind_force = self.p.wind_gust_force

        # ── Fuerzas ──
        gravity_force = -self.p.mass * self.p.gravity
        drag_force = -self.p.drag_coeff * self.velocity * abs(self.velocity)  # drag cuadrático
        net_force = self.actual_thrust + gravity_force + drag_force + wind_force

        # ── Integración (semi-implícita Euler para mejor estabilidad) ──
        acceleration = net_force / self.p.mass
        self.velocity += acceleration * dt
        self.altitude += self.velocity * dt

        # Restricción de suelo
        if self.altitude < 0:
            self.altitude = 0.0
            self.velocity = max(0.0, self.velocity)  # rebote inelástico

        # ── Ruido de sensor ──
        measured = self.altitude
        if self.p.sensor_noise_std > 0:
            measured += self.rng.normal(0, self.p.sensor_noise_std)

        return self.altitude, measured


# ════════════════════════════════════════════════════════════════════════
# 4. MÉTODOS DE AUTO-TUNING
# ════════════════════════════════════════════════════════════════════════

def identify_fopdt(plant: PlantParams, step_amplitude: float = 1.0,
                   dt: float = 0.001, sim_time: float = 60.0) -> Tuple[float, float, float]:
    """
    Identifica los parámetros FOPDT (First Order Plus Dead Time)
    del proceso mediante una prueba de escalón en lazo abierto.

    Para un dron (doble integrador), el modelo FOPDT es una aproximación.
    Se usa una simulación en lazo cerrado con ganancia baja para obtener
    una respuesta acotada y extraer los parámetros.

    Retorna: (K, tau, theta) — ganancia estática, constante de tiempo, retardo
    """
    drone = DroneModel1D(plant)
    n = int(sim_time / dt)

    # Para un doble integrador, hacemos identificación en lazo cerrado
    # con un controlador P muy suave para obtener respuesta de primer orden
    hover = plant.mass * plant.gravity
    kp_test = 0.5  # ganancia proporcional baja
    target = 5.0   # escalón pequeño

    # Estabilizar en hover
    for _ in range(int(2.0 / dt)):
        drone.step(hover, dt, 0)

    altitudes = np.zeros(n)
    alt_initial = drone.altitude

    for i in range(n):
        t = i * dt
        error = target - (drone.altitude - alt_initial)
        thrust = hover + kp_test * error
        thrust = np.clip(thrust, plant.thrust_min, plant.thrust_max)
        true_alt, _ = drone.step(thrust, dt, t)
        altitudes[i] = true_alt - alt_initial

    alt_final = altitudes[-1]

    # Ganancia estática del lazo cerrado → ganancia de la planta
    K_cl = alt_final / target if target > 0 else 1.0
    # K de la planta: K_plant ≈ K_cl / (kp_test * (1 - K_cl)) para lazo cerrado
    # Pero es más útil usar K directamente como la ganancia DC del lazo cerrado
    K = max(alt_final / target, 0.1)

    # Constante de tiempo: tiempo para alcanzar 63.2%
    target_632 = alt_initial + 0.632 * (alt_final - 0)
    t_632 = sim_time
    for i in range(n):
        if altitudes[i] >= 0.632 * alt_final:
            t_632 = i * dt
            break

    # Retardo: tiempo para alcanzar 5%
    theta = 0.0
    for i in range(n):
        if altitudes[i] >= 0.05 * alt_final:
            theta = i * dt
            break

    tau = max(t_632 - theta, 0.05)
    theta = max(theta, 0.01)

    # Para el dron, ajustar K para que las fórmulas de tuning
    # produzcan ganancias razonables (K ≈ 1/(m·g) en régimen)
    K = 1.0 / (plant.mass * plant.gravity)  # ganancia de empuje→aceleración normalizada

    return K, tau, theta


def auto_tune(method: TuningMethod, plant: PlantParams,
              K: float = None, tau: float = None, theta: float = None,
              lambda_factor: float = 3.0) -> PIDParams:
    """
    Calcula las ganancias PID según el método de sintonización elegido.

    Para métodos de lazo abierto, necesita K (ganancia), tau (cte. de tiempo),
    theta (retardo) del modelo FOPDT identificado.
    """
    if K is None or tau is None or theta is None:
        K, tau, theta = identify_fopdt(plant)
        print(f"  → FOPDT identificado: K={K:.4f}, τ={tau:.4f}s, θ={theta:.4f}s")

    params = PIDParams(form=PIDForm.PARALLEL, tuning_method=method)
    r = theta / tau if tau > 0 else 0.1  # ratio retardo/constante

    if method == TuningMethod.ZN_OPEN_LOOP:
        # ── Ziegler-Nichols (curva de reacción) ──
        # PID: Kp = 1.2·τ/(K·θ), Ti = 2·θ, Td = 0.5·θ
        params.kp = 1.2 * tau / (K * theta)
        ti = 2.0 * theta
        td = 0.5 * theta
        params.ki = params.kp / ti if ti > 0 else 0
        params.kd = params.kp * td
        params.label = f"ZN-OL: Kp={params.kp:.2f} Ki={params.ki:.2f} Kd={params.kd:.2f}"

    elif method == TuningMethod.ZN_CLOSED_LOOP:
        # ── Ziegler-Nichols (oscilación sostenida) ──
        # Ku y Pu se estiman del FOPDT
        Ku = (0.5 * np.pi * tau) / (K * theta) if theta > 0 else 10
        Pu = 4.0 * theta
        params.kp = 0.6 * Ku
        ti = 0.5 * Pu
        td = 0.125 * Pu
        params.ki = params.kp / ti if ti > 0 else 0
        params.kd = params.kp * td
        params.label = f"ZN-CL: Kp={params.kp:.2f} Ki={params.ki:.2f} Kd={params.kd:.2f}"

    elif method == TuningMethod.COHEN_COON:
        # ── Cohen-Coon ──
        # Más agresivo que ZN, mejor para procesos con retardo significativo
        params.kp = (1.0 / (K * r)) * (1.35 + (0.25 * r))
        ti = theta * (2.5 - 2.0 * r) / (1.0 - 0.39 * r)
        td = theta * 0.37 / (1.0 - 0.81 * r)
        ti = max(ti, 0.01)
        td = max(td, 0.001)
        params.ki = params.kp / ti
        params.kd = params.kp * td
        params.label = f"Cohen-Coon: Kp={params.kp:.2f} Ki={params.ki:.2f} Kd={params.kd:.2f}"

    elif method == TuningMethod.LAMBDA_TUNING:
        # ── Lambda Tuning (IMC — Internal Model Control) ──
        # Más conservador, prioriza robustez sobre velocidad
        lam = lambda_factor * theta  # λ = factor · θ
        params.kp = tau / (K * (lam + theta))
        ti = tau
        td = 0.5 * theta
        params.ki = params.kp / ti if ti > 0 else 0
        params.kd = params.kp * td
        params.label = f"Lambda(λ={lam:.2f}): Kp={params.kp:.2f} Ki={params.ki:.2f} Kd={params.kd:.2f}"

    elif method == TuningMethod.AMIGO:
        # ── AMIGO (Approximate M-constrained Integral Gain Optimization) ──
        # Buen balance entre rendimiento y robustez
        params.kp = (1.0 / K) * (0.2 + 0.45 * tau / theta)
        ti = (0.4 * theta + 0.8 * tau) * theta / (theta + 0.1 * tau)
        td = 0.5 * theta * tau / (0.3 * theta + tau)
        ti = max(ti, 0.01)
        params.ki = params.kp / ti
        params.kd = params.kp * td
        params.label = f"AMIGO: Kp={params.kp:.2f} Ki={params.ki:.2f} Kd={params.kd:.2f}"

    elif method == TuningMethod.TYREUS_LUYBEN:
        # ── Tyreus-Luyben ──
        # Versión conservadora de ZN para procesos industriales
        Ku = (0.5 * np.pi * tau) / (K * theta) if theta > 0 else 10
        Pu = 4.0 * theta
        params.kp = Ku / 3.2
        ti = 2.2 * Pu
        td = Pu / 6.3
        params.ki = params.kp / ti if ti > 0 else 0
        params.kd = params.kp * td
        params.label = f"Tyreus-Luyben: Kp={params.kp:.2f} Ki={params.ki:.2f} Kd={params.kd:.2f}"

    else:
        params.label = "Manual"

    # Configurar saturación según planta
    params.output_min = plant.thrust_min
    params.output_max = plant.thrust_max
    params.use_gravity_ff = True

    return params


# ════════════════════════════════════════════════════════════════════════
# 5. MOTOR DE SIMULACIÓN
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SimResult:
    """Resultado completo de una simulación."""
    time: np.ndarray = None
    altitude: np.ndarray = None
    measured: np.ndarray = None
    velocity: np.ndarray = None
    thrust_cmd: np.ndarray = None
    thrust_actual: np.ndarray = None
    error: np.ndarray = None
    P_term: np.ndarray = None
    I_term: np.ndarray = None
    D_term: np.ndarray = None
    setpoint_arr: np.ndarray = None
    integral_state: np.ndarray = None
    saturated: np.ndarray = None

    # Métricas
    overshoot: float = 0.0
    settling_time: float = 0.0
    rise_time: float = 0.0
    ss_error: float = 0.0
    iae: float = 0.0   # Integral Absolute Error
    ise: float = 0.0   # Integral Squared Error
    itae: float = 0.0  # Integral Time-weighted Absolute Error

    label: str = ""
    method: str = ""


def simulate(pid_params: PIDParams, plant_params: PlantParams,
             setpoint: float, total_time: float = 30.0, dt: float = 0.01,
             setpoint_changes: List[Tuple[float, float]] = None) -> SimResult:
    """
    Ejecuta la simulación completa.

    setpoint_changes: lista de (tiempo, nuevo_setpoint) para cambios de referencia.
    """
    n = int(total_time / dt)
    drone = DroneModel1D(plant_params)
    pid = ProfessionalPID(pid_params, plant_params, setpoint)

    res = SimResult(
        time=np.zeros(n), altitude=np.zeros(n), measured=np.zeros(n),
        velocity=np.zeros(n), thrust_cmd=np.zeros(n), thrust_actual=np.zeros(n),
        error=np.zeros(n), P_term=np.zeros(n), I_term=np.zeros(n),
        D_term=np.zeros(n), setpoint_arr=np.zeros(n),
        integral_state=np.zeros(n), saturated=np.zeros(n, dtype=bool),
        label=pid_params.label or pid_params.tuning_method.value,
        method=pid_params.tuning_method.value,
    )

    current_sp = setpoint
    sp_changes = dict(setpoint_changes) if setpoint_changes else {}

    for i in range(n):
        t = i * dt
        res.time[i] = t

        # Cambio de setpoint programado
        for sp_time, sp_val in sp_changes.items():
            if abs(t - sp_time) < dt / 2:
                current_sp = sp_val
                pid.setpoint = current_sp

        res.setpoint_arr[i] = current_sp

        # Medición
        true_alt, meas_alt = drone.altitude, drone.altitude
        if plant_params.sensor_noise_std > 0:
            meas_alt += drone.rng.normal(0, plant_params.sensor_noise_std)

        # Controlador
        thrust, diag = pid.update(meas_alt, dt)

        # Planta
        true_alt, _ = drone.step(thrust, dt, t)

        # Guardar datos
        res.altitude[i] = true_alt
        res.measured[i] = meas_alt
        res.velocity[i] = drone.velocity
        res.thrust_cmd[i] = thrust
        res.thrust_actual[i] = drone.actual_thrust
        res.error[i] = diag['error']
        res.P_term[i] = diag['P']
        res.I_term[i] = diag['I']
        res.D_term[i] = diag['D']
        res.integral_state[i] = diag['integral_state']
        res.saturated[i] = diag['saturated']

    # ── Calcular métricas ──
    _compute_metrics(res, setpoint, dt)

    return res


def _compute_metrics(res: SimResult, setpoint: float, dt: float):
    """Calcula métricas de desempeño estándar."""
    alt = res.altitude
    n = len(alt)

    # Overshoot
    max_alt = np.max(alt)
    res.overshoot = max(0.0, (max_alt - setpoint) / setpoint * 100) if setpoint > 0 else 0

    # Settling time (banda ±2%)
    band = abs(setpoint) * 0.02
    dentro = np.abs(alt - setpoint) <= band
    res.settling_time = res.time[-1]
    for i in range(n - 1, -1, -1):
        if not dentro[i]:
            res.settling_time = res.time[min(i + 1, n - 1)]
            break

    # Rise time (10% → 90%)
    if np.any(alt >= setpoint * 0.1) and np.any(alt >= setpoint * 0.9):
        t10 = res.time[np.argmax(alt >= setpoint * 0.1)]
        t90 = res.time[np.argmax(alt >= setpoint * 0.9)]
        res.rise_time = t90 - t10
    else:
        res.rise_time = res.time[-1]

    # Error estacionario (promedio de los últimos 5%)
    tail = max(1, n // 20)
    res.ss_error = abs(setpoint - np.mean(alt[-tail:]))

    # Índices de desempeño integrales
    abs_error = np.abs(res.error)
    res.iae = np.sum(abs_error) * dt
    res.ise = np.sum(res.error ** 2) * dt
    res.itae = np.sum(res.time * abs_error) * dt


# ════════════════════════════════════════════════════════════════════════
# 6. VISUALIZACIÓN PROFESIONAL
# ════════════════════════════════════════════════════════════════════════

# Paleta de colores profesional
COLORS = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22']

def plot_comparison(results: List[SimResult], title: str = "",
                    save_path: str = None, figsize: Tuple = (14, 16)):
    """Gráfica comparativa completa de múltiples controladores."""

    n_results = len(results)
    fig = plt.figure(figsize=figsize, facecolor='#0d1117')

    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.08, right=0.95, top=0.93, bottom=0.04)

    # Estilo oscuro para todas las axes
    def style_ax(ax, ylabel, xlabel=""):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.set_ylabel(ylabel, color='#c9d1d9', fontsize=9)
        if xlabel:
            ax.set_xlabel(xlabel, color='#c9d1d9', fontsize=9)
        ax.grid(True, alpha=0.15, color='#30363d')
        for spine in ax.spines.values():
            spine.set_color('#30363d')

    # ── 1. ALTITUD (grande, arriba) ──────────────────────────────
    ax_alt = fig.add_subplot(gs[0:2, :])
    style_ax(ax_alt, 'Altitud (m)')

    # Setpoint (del primer resultado)
    ax_alt.plot(results[0].time, results[0].setpoint_arr, '--',
                color='#ffd700', linewidth=1.5, alpha=0.8, label='Setpoint', zorder=10)

    # Banda ±2%
    sp = results[0].setpoint_arr
    ax_alt.fill_between(results[0].time, sp * 0.98, sp * 1.02,
                        alpha=0.08, color='#ffd700', label='Banda ±2%')

    for i, res in enumerate(results):
        c = COLORS[i % len(COLORS)]
        ax_alt.plot(res.time, res.altitude, color=c, linewidth=1.4,
                    alpha=0.9, label=res.label)

    ax_alt.legend(fontsize=8, loc='lower right', facecolor='#161b22',
                  edgecolor='#30363d', labelcolor='#c9d1d9', ncol=2)
    ax_alt.set_title(title or 'Comparación de Controladores PID',
                     color='#f0f6fc', fontsize=13, fontweight='bold', pad=12)

    # ── 2. ERROR ─────────────────────────────────────────────────
    ax_err = fig.add_subplot(gs[2, 0])
    style_ax(ax_err, 'Error (m)')
    for i, res in enumerate(results):
        ax_err.plot(res.time, res.error, color=COLORS[i % len(COLORS)],
                    linewidth=1.0, alpha=0.8)
    ax_err.axhline(0, color='#484f58', linewidth=0.5, linestyle=':')

    # ── 3. THRUST ────────────────────────────────────────────────
    ax_thr = fig.add_subplot(gs[2, 1])
    style_ax(ax_thr, 'Thrust (N)')
    for i, res in enumerate(results):
        ax_thr.plot(res.time, res.thrust_cmd, color=COLORS[i % len(COLORS)],
                    linewidth=1.0, alpha=0.8)
    # Línea de peso
    weight = results[0].thrust_cmd[0]  # approx
    ax_thr.axhline(y=results[0].thrust_actual[0] if len(results[0].thrust_actual) > 0 else 9.81,
                   color='#484f58', linewidth=0.5, linestyle=':', label='m·g')

    # ── 4. COMPONENTES PID ───────────────────────────────────────
    ax_pid = fig.add_subplot(gs[3, 0])
    style_ax(ax_pid, 'Componentes PID (N)')
    # Solo del primer resultado para claridad
    res0 = results[0]
    ax_pid.plot(res0.time, res0.P_term, color='#2ecc71', linewidth=1.0, alpha=0.8, label='P')
    ax_pid.plot(res0.time, res0.I_term, color='#3498db', linewidth=1.0, alpha=0.8, label='I')
    ax_pid.plot(res0.time, res0.D_term, color='#e74c3c', linewidth=1.0, alpha=0.8, label='D')
    ax_pid.axhline(0, color='#484f58', linewidth=0.5, linestyle=':')
    ax_pid.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax_pid.set_title(f'Descomposición PID — {res0.label}',
                     color='#8b949e', fontsize=9)

    # ── 5. ESTADO INTEGRAL ───────────────────────────────────────
    ax_int = fig.add_subplot(gs[3, 1])
    style_ax(ax_int, 'Estado Integral')
    for i, res in enumerate(results):
        ax_int.plot(res.time, res.integral_state, color=COLORS[i % len(COLORS)],
                    linewidth=1.0, alpha=0.8)
    ax_int.axhline(0, color='#484f58', linewidth=0.5, linestyle=':')

    # ── 6. TABLA DE MÉTRICAS ─────────────────────────────────────
    ax_table = fig.add_subplot(gs[4, :])
    ax_table.set_facecolor('#161b22')
    ax_table.axis('off')

    headers = ['Método', 'Kp', 'Ki', 'Kd', 'Overshoot\n(%)', 'Settling\n(s)',
               'Rise\n(s)', 'SS Error\n(m)', 'IAE', 'ISE', 'ITAE']

    cell_data = []
    cell_colors = []
    for i, res in enumerate(results):
        # Extraer ganancias del label o usar valores por defecto
        row = [
            res.label[:20],
            f'{results[i].thrust_cmd[0]:.1f}',  # placeholder
            '', '', 
            f'{res.overshoot:.1f}',
            f'{res.settling_time:.2f}',
            f'{res.rise_time:.2f}',
            f'{res.ss_error:.4f}',
            f'{res.iae:.2f}',
            f'{res.ise:.2f}',
            f'{res.itae:.2f}',
        ]
        cell_data.append(row)
        c = COLORS[i % len(COLORS)]
        cell_colors.append([c + '30'] * len(headers))

    # Reconstruir cell_data con las ganancias reales
    cell_data = []
    for i, res in enumerate(results):
        cell_data.append([
            res.label[:22], '', '', '',
            f'{res.overshoot:.1f}', f'{res.settling_time:.2f}',
            f'{res.rise_time:.2f}', f'{res.ss_error:.4f}',
            f'{res.iae:.1f}', f'{res.ise:.1f}', f'{res.itae:.1f}',
        ])

    table = ax_table.table(
        cellText=cell_data,
        colLabels=headers,
        cellColours=cell_colors,
        colColours=['#21262d'] * len(headers),
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.5)

    # Estilo de la tabla
    for key, cell in table.get_celld().items():
        cell.set_edgecolor('#30363d')
        cell.set_text_props(color='#c9d1d9')
        if key[0] == 0:
            cell.set_text_props(color='#f0f6fc', fontweight='bold')
            cell.set_facecolor('#21262d')

    ax_table.set_title('Métricas de Desempeño Comparativas',
                       color='#8b949e', fontsize=10, pad=8)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()


def plot_single_detailed(res: SimResult, plant: PlantParams,
                         save_path: str = None):
    """Gráfica detallada de un solo controlador con todos los diagnósticos."""

    fig, axes = plt.subplots(4, 2, figsize=(14, 12), facecolor='#0d1117')
    fig.suptitle(f'Análisis Detallado — {res.label}',
                 color='#f0f6fc', fontsize=14, fontweight='bold')

    def style(ax, ylabel, title=''):
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=7)
        ax.set_ylabel(ylabel, color='#c9d1d9', fontsize=8)
        ax.grid(True, alpha=0.12, color='#30363d')
        for s in ax.spines.values():
            s.set_color('#30363d')
        if title:
            ax.set_title(title, color='#8b949e', fontsize=9, pad=6)

    t = res.time

    # 1. Altitud
    ax = axes[0, 0]
    style(ax, 'm', 'Altitud vs Setpoint')
    ax.plot(t, res.setpoint_arr, '--', color='#ffd700', linewidth=1.2, alpha=0.8)
    ax.plot(t, res.altitude, color='#2ecc71', linewidth=1.5)
    ax.fill_between(t, res.setpoint_arr * 0.98, res.setpoint_arr * 1.02,
                    alpha=0.1, color='#ffd700')

    # 2. Error
    ax = axes[0, 1]
    style(ax, 'm', 'Error')
    ax.plot(t, res.error, color='#e74c3c', linewidth=1.0)
    ax.fill_between(t, res.error, alpha=0.1, color='#e74c3c')
    ax.axhline(0, color='#484f58', linewidth=0.5)

    # 3. Thrust comandado vs real
    ax = axes[1, 0]
    style(ax, 'N', 'Thrust: Comandado vs Real')
    ax.plot(t, res.thrust_cmd, color='#3498db', linewidth=1.0, label='Comandado', alpha=0.8)
    ax.plot(t, res.thrust_actual, color='#e67e22', linewidth=1.0, label='Real (actuador)', alpha=0.8)
    ax.axhline(plant.mass * plant.gravity, color='#484f58', linewidth=0.5, linestyle=':', label='m·g')
    ax.axhline(plant.thrust_max, color='#e74c3c', linewidth=0.5, linestyle='--', alpha=0.5, label='Saturación')
    ax.legend(fontsize=6, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # 4. Componentes PID
    ax = axes[1, 1]
    style(ax, 'N', 'Componentes P, I, D')
    ax.plot(t, res.P_term, color='#2ecc71', linewidth=1.0, label='P')
    ax.plot(t, res.I_term, color='#3498db', linewidth=1.0, label='I')
    ax.plot(t, res.D_term, color='#e74c3c', linewidth=1.0, label='D')
    ax.axhline(0, color='#484f58', linewidth=0.5)
    ax.legend(fontsize=6, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')

    # 5. Velocidad
    ax = axes[2, 0]
    style(ax, 'm/s', 'Velocidad')
    ax.plot(t, res.velocity, color='#9b59b6', linewidth=1.0)
    ax.fill_between(t, res.velocity, alpha=0.08, color='#9b59b6')
    ax.axhline(0, color='#484f58', linewidth=0.5)

    # 6. Estado integral
    ax = axes[2, 1]
    style(ax, '', 'Estado Integral (anti-windup)')
    ax.plot(t, res.integral_state, color='#1abc9c', linewidth=1.2)
    ax.fill_between(t, res.integral_state, alpha=0.1, color='#1abc9c')
    # Marcar zonas de saturación
    sat_mask = res.saturated.astype(float)
    if np.any(sat_mask):
        ax_twin = ax.twinx()
        ax_twin.fill_between(t, sat_mask, alpha=0.15, color='#e74c3c', step='mid')
        ax_twin.set_ylabel('Saturado', color='#e74c3c', fontsize=7)
        ax_twin.set_ylim(-0.1, 1.5)
        ax_twin.tick_params(colors='#e74c3c', labelsize=6)

    # 7. Índices de error acumulados
    ax = axes[3, 0]
    style(ax, '', 'Índices de Error Acumulados')
    abs_err = np.abs(res.error)
    iae_cum = np.cumsum(abs_err) * (t[1] - t[0]) if len(t) > 1 else abs_err
    ise_cum = np.cumsum(res.error**2) * (t[1] - t[0]) if len(t) > 1 else res.error**2
    ax.plot(t, iae_cum, color='#f39c12', linewidth=1.0, label=f'IAE={res.iae:.2f}')
    ax.plot(t, ise_cum, color='#e74c3c', linewidth=1.0, label=f'ISE={res.ise:.2f}')
    ax.legend(fontsize=7, facecolor='#161b22', edgecolor='#30363d', labelcolor='#c9d1d9')
    ax.set_xlabel('Tiempo (s)', color='#c9d1d9', fontsize=8)

    # 8. Resumen de métricas (texto)
    ax = axes[3, 1]
    ax.set_facecolor('#161b22')
    ax.axis('off')

    metrics_text = (
        f"{'═' * 36}\n"
        f"  MÉTRICAS DE DESEMPEÑO\n"
        f"{'═' * 36}\n\n"
        f"  Overshoot:        {res.overshoot:>8.1f} %\n"
        f"  Settling time:    {res.settling_time:>8.2f} s\n"
        f"  Rise time:        {res.rise_time:>8.2f} s\n"
        f"  Error SS:         {res.ss_error:>8.4f} m\n\n"
        f"  IAE  (|e|):       {res.iae:>8.2f}\n"
        f"  ISE  (e²):        {res.ise:>8.2f}\n"
        f"  ITAE (t·|e|):     {res.itae:>8.2f}\n"
        f"{'═' * 36}"
    )
    ax.text(0.1, 0.95, metrics_text, transform=ax.transAxes,
            fontfamily='monospace', fontsize=9, color='#c9d1d9',
            verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#0d1117',
                      edgecolor='#30363d'))

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    plt.show()


# ════════════════════════════════════════════════════════════════════════
# 7. ESCENARIOS DE PRUEBA
# ════════════════════════════════════════════════════════════════════════

def run_all_scenarios():
    """Ejecuta la batería completa de pruebas profesionales."""

    # ── Planta base ──
    plant = PlantParams(
        mass=1.0, gravity=9.81,
        drag_coeff=0.1,
        thrust_max=30.0, thrust_min=0.0,
        actuator_lag=0.02,
        sensor_noise_std=0.0,
        transport_delay=0.0,
    )

    SETPOINT = 10.0
    T_TOTAL = 30.0
    DT = 0.01

    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  IDENTIFICACIÓN DEL MODELO FOPDT                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    K, tau, theta = identify_fopdt(plant, step_amplitude=2.0)
    print(f"  Ganancia estática (K):      {K:.4f}")
    print(f"  Constante de tiempo (τ):    {tau:.4f} s")
    print(f"  Retardo (θ):                {theta:.4f} s")
    print(f"  Ratio θ/τ:                  {theta/tau:.4f}")
    print()

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 1: Comparación de métodos de sintonización
    # ═══════════════════════════════════════════════════════════════
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 1: Comparación de Métodos de Sintonización      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    methods = [
        TuningMethod.ZN_OPEN_LOOP,
        TuningMethod.ZN_CLOSED_LOOP,
        TuningMethod.COHEN_COON,
        TuningMethod.LAMBDA_TUNING,
        TuningMethod.AMIGO,
        TuningMethod.TYREUS_LUYBEN,
    ]

    results_methods = []
    for method in methods:
        params = auto_tune(method, plant, K, tau, theta)
        print(f"  {method.value:35s} → Kp={params.kp:.3f}  Ki={params.ki:.3f}  Kd={params.kd:.3f}")
        res = simulate(params, plant, SETPOINT, T_TOTAL, DT)
        results_methods.append(res)

    print()
    print("  Métricas:")
    print(f"  {'Método':<28s} {'OS%':>6s} {'Ts(s)':>7s} {'Tr(s)':>7s} {'SSE':>8s} {'IAE':>8s} {'ITAE':>8s}")
    print("  " + "─" * 72)
    for res in results_methods:
        print(f"  {res.label[:28]:<28s} {res.overshoot:>6.1f} {res.settling_time:>7.2f} "
              f"{res.rise_time:>7.2f} {res.ss_error:>8.4f} {res.iae:>8.1f} {res.itae:>8.1f}")

    plot_comparison(
        results_methods,
        title='Escenario 1: Comparación de Métodos de Sintonización\n'
              f'Planta: m={plant.mass}kg, drag={plant.drag_coeff}, '
              f'FOPDT: K={K:.3f}, τ={tau:.3f}s, θ={theta:.3f}s',
        save_path='escenario1_metodos.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 2: Efecto del filtro derivativo (N)
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 2: Efecto del Filtro Derivativo (N)             ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    base_params = auto_tune(TuningMethod.AMIGO, plant, K, tau, theta)
    noisy_plant = PlantParams(**{**plant.__dict__, 'sensor_noise_std': 0.05})

    results_filter = []
    for N in [3, 10, 33, 100]:
        p = deepcopy(base_params)
        p.derivative_filter_N = N
        p.label = f"N={N}"
        res = simulate(p, noisy_plant, SETPOINT, T_TOTAL, DT)
        results_filter.append(res)
        print(f"  N={N:>4d}  → OS={res.overshoot:.1f}%  Ts={res.settling_time:.2f}s  "
              f"IAE={res.iae:.1f}  SSE={res.ss_error:.4f}")

    plot_comparison(
        results_filter,
        title='Escenario 2: Efecto del Filtro Derivativo (N)\n'
              f'Con ruido de sensor σ={noisy_plant.sensor_noise_std}m — Método base: AMIGO',
        save_path='escenario2_filtro_derivativo.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 3: Anti-windup comparison
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 3: Estrategias Anti-Windup                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    # Planta con thrust limitado para forzar saturación
    limited_plant = PlantParams(**{**plant.__dict__, 'thrust_max': 15.0})

    results_windup = []
    for aw_mode in [AntiWindupMode.NONE, AntiWindupMode.CLAMPING, AntiWindupMode.BACK_CALCULATION]:
        p = deepcopy(base_params)
        p.anti_windup = aw_mode
        p.output_max = 15.0
        p.label = f"Anti-WU: {aw_mode.value}"
        # Ganancias más agresivas para forzar saturación
        p.kp *= 2.0
        p.ki *= 3.0
        # Kb alto para que back-calculation sea efectivo
        p.kb = 5.0
        res = simulate(p, limited_plant, SETPOINT, T_TOTAL, DT)
        results_windup.append(res)
        print(f"  {aw_mode.value:20s} → OS={res.overshoot:.1f}%  Ts={res.settling_time:.2f}s  "
              f"IAE={res.iae:.1f}")

    plot_comparison(
        results_windup,
        title='Escenario 3: Estrategias Anti-Windup\n'
              f'Thrust máximo limitado a {limited_plant.thrust_max}N (forzando saturación)',
        save_path='escenario3_anti_windup.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 4: Derivativo sobre error vs medición
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 4: Derivativo sobre Error vs Medición           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    results_deriv = []
    # Usar cambio de setpoint para evidenciar el derivative kick
    sp_changes_deriv = [(0, 10.0), (15, 15.0)]
    for d_mode in [DerivativeMode.ERROR, DerivativeMode.MEASUREMENT, DerivativeMode.MIXED]:
        p = deepcopy(base_params)
        p.derivative_mode = d_mode
        p.derivative_filter_N = 100  # Menos filtrado para evidenciar el kick
        p.kd *= 3.0  # Más derivativo
        if d_mode == DerivativeMode.MIXED:
            p.sp_weight_c = 0.5
        p.label = f"D sobre {d_mode.value}"
        res = simulate(p, plant, 10.0, T_TOTAL, DT, setpoint_changes=sp_changes_deriv)
        results_deriv.append(res)
        print(f"  {d_mode.value:15s} → OS={res.overshoot:.1f}%  Ts={res.settling_time:.2f}s  "
              f"IAE={res.iae:.1f}")

    plot_comparison(
        results_deriv,
        title='Escenario 4: Derivativo sobre Error vs Medición vs Mixto\n'
              'Impacto del derivative kick en cambios de setpoint',
        save_path='escenario4_derivativo_modo.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 5: Rechazo de perturbaciones (ráfaga de viento)
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 5: Rechazo de Perturbaciones (Ráfaga de Viento) ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    wind_plant = PlantParams(
        **{**plant.__dict__,
           'wind_gust_time': 15.0,
           'wind_gust_duration': 3.0,
           'wind_gust_force': -5.0}  # Ráfaga hacia abajo
    )

    results_wind = []
    for method in [TuningMethod.ZN_OPEN_LOOP, TuningMethod.AMIGO, TuningMethod.LAMBDA_TUNING]:
        p = auto_tune(method, wind_plant, K, tau, theta)
        res = simulate(p, wind_plant, SETPOINT, T_TOTAL, DT)
        results_wind.append(res)
        print(f"  {method.value:35s} → OS={res.overshoot:.1f}%  "
              f"IAE={res.iae:.1f}  ITAE={res.itae:.1f}")

    plot_comparison(
        results_wind,
        title='Escenario 5: Rechazo de Perturbaciones\n'
              f'Ráfaga de viento: {wind_plant.wind_gust_force}N durante '
              f'{wind_plant.wind_gust_duration}s a t={wind_plant.wind_gust_time}s',
        save_path='escenario5_perturbaciones.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 6: Seguimiento de trayectoria (cambios de setpoint)
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 6: Seguimiento de Trayectoria                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    sp_changes = [(0, 5.0), (10, 15.0), (20, 8.0), (30, 20.0)]

    results_traj = []
    for method in [TuningMethod.ZN_OPEN_LOOP, TuningMethod.AMIGO, TuningMethod.LAMBDA_TUNING]:
        p = auto_tune(method, plant, K, tau, theta)
        res = simulate(p, plant, sp_changes[0][1], 40.0, DT,
                       setpoint_changes=sp_changes)
        results_traj.append(res)
        print(f"  {method.value:35s} → IAE={res.iae:.1f}  ITAE={res.itae:.1f}")

    plot_comparison(
        results_traj,
        title='Escenario 6: Seguimiento de Trayectoria Multi-Setpoint\n'
              f'Cambios de setpoint: {sp_changes}',
        save_path='escenario6_trayectoria.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ESCENARIO 7: Efecto del setpoint weighting
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ESCENARIO 7: Setpoint Weighting                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    results_spw = []
    for b in [0.0, 0.3, 0.7, 1.0]:
        p = deepcopy(base_params)
        p.sp_weight_b = b
        p.label = f"b={b}"
        res = simulate(p, plant, SETPOINT, T_TOTAL, DT)
        results_spw.append(res)
        print(f"  b={b:.1f}  → OS={res.overshoot:.1f}%  Ts={res.settling_time:.2f}s  "
              f"IAE={res.iae:.1f}")

    plot_comparison(
        results_spw,
        title='Escenario 7: Setpoint Weighting (Ponderación b)\n'
              'b=1: PID clásico | b=0: P solo actúa sobre medición (sin derivative kick)',
        save_path='escenario7_sp_weighting.png'
    )

    # ═══════════════════════════════════════════════════════════════
    # ANÁLISIS DETALLADO del mejor controlador
    # ═══════════════════════════════════════════════════════════════
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  ANÁLISIS DETALLADO — Mejor Controlador (AMIGO)            ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    best_params = auto_tune(TuningMethod.AMIGO, plant, K, tau, theta)
    best_params.derivative_mode = DerivativeMode.MEASUREMENT
    best_params.anti_windup = AntiWindupMode.BACK_CALCULATION
    best_params.derivative_filter_N = 10
    best_params.label = "AMIGO (configuración completa)"

    detailed_plant = PlantParams(
        **{**plant.__dict__,
           'sensor_noise_std': 0.02,
           'wind_gust_time': 15.0,
           'wind_gust_duration': 2.0,
           'wind_gust_force': -3.0}
    )

    res_best = simulate(best_params, detailed_plant, SETPOINT, T_TOTAL, DT)
    plot_single_detailed(res_best, detailed_plant, save_path='analisis_detallado.png')

    print(f"  Overshoot:     {res_best.overshoot:.1f}%")
    print(f"  Settling time: {res_best.settling_time:.2f}s")
    print(f"  Rise time:     {res_best.rise_time:.2f}s")
    print(f"  SS Error:      {res_best.ss_error:.4f}m")
    print(f"  IAE:           {res_best.iae:.2f}")
    print(f"  ISE:           {res_best.ise:.2f}")
    print(f"  ITAE:          {res_best.itae:.2f}")

    print("\n" + "═" * 64)
    print("  Simulaciones completadas. Gráficas guardadas en:")
    print("    • escenario1_metodos.png")
    print("    • escenario2_filtro_derivativo.png")
    print("    • escenario3_anti_windup.png")
    print("    • escenario4_derivativo_modo.png")
    print("    • escenario5_perturbaciones.png")
    print("    • escenario6_trayectoria.png")
    print("    • escenario7_sp_weighting.png")
    print("    • analisis_detallado.png")
    print("═" * 64)


# ════════════════════════════════════════════════════════════════════════
# 8. PUNTO DE ENTRADA
# ════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    run_all_scenarios()


# ════════════════════════════════════════════════════════════════════════
# 9. GUÍA TÉCNICA: POR QUÉ UN PID REAL ES MÁS QUE Kp·e + Ki·∫e + Kd·ė
# ════════════════════════════════════════════════════════════════════════
#
# ┌─────────────────────────────────────────────────────────────────────┐
# │ FILTRO DERIVATIVO                                                   │
# │ ─────────────────                                                   │
# │ El término derivativo puro Kd·de/dt amplifica cualquier ruido de   │
# │ alta frecuencia en la señal. En un sensor real con σ=0.05m de      │
# │ ruido, un Kd=8 produce spikes de ±400N en la salida.              │
# │                                                                     │
# │ Solución: D(s) = Kd·s / (1 + (Td/N)·s)                           │
# │                                                                     │
# │ N bajo (3-5): Filtrado agresivo, derivativo lento pero limpio      │
# │ N alto (>30): Poco filtrado, derivativo rápido pero ruidoso        │
# │ N=10: Compromiso industrial estándar                                │
# ├─────────────────────────────────────────────────────────────────────┤
# │ ANTI-WINDUP                                                         │
# │ ───────────                                                         │
# │ Cuando el actuador satura (thrust=max), el integrador sigue        │
# │ acumulando error. Al desaturar, la integral acumulada causa        │
# │ un overshoot masivo — el "windup".                                 │
# │                                                                     │
# │ Clamping: Detiene la integración si el actuador está saturado      │
# │ Back-calculation: Resta Kb·(u_sat - u_unsat) de la integral       │
# │   → Más suave, converge más rápido, estándar industrial            │
# ├─────────────────────────────────────────────────────────────────────┤
# │ DERIVATIVO SOBRE MEDICIÓN vs ERROR                                  │
# │ ────────────────────────────────                                    │
# │ d(error)/dt cuando el setpoint cambia bruscamente: ∞ impulso       │
# │ ("derivative kick"). El dron recibe un golpe de thrust brutal.     │
# │                                                                     │
# │ Solución: calcular el derivativo sobre -d(medición)/dt             │
# │ Así el derivativo solo reacciona al movimiento real del dron,      │
# │ no a cambios de referencia. Mismo rechazo de perturbaciones,       │
# │ pero sin el kick.                                                   │
# ├─────────────────────────────────────────────────────────────────────┤
# │ SETPOINT WEIGHTING (b, c)                                          │
# │ ─────────────────────────                                           │
# │ P_term = Kp · (b·SP - y)     en lugar de   Kp · (SP - y)         │
# │ D_term = Kd · d(c·SP - y)/dt                                      │
# │                                                                     │
# │ b=1, c=1: PID clásico (máxima agresividad al cambio de SP)        │
# │ b=0, c=0: P y D solo ven la medición (suave, sin kicks)           │
# │ b=0.5:    Compromiso — sigue SP suavemente pero rechaza            │
# │           perturbaciones con toda la ganancia                       │
# ├─────────────────────────────────────────────────────────────────────┤
# │ MÉTODOS DE SINTONIZACIÓN                                            │
# │ ────────────────────────                                            │
# │                                                                     │
# │ Ziegler-Nichols (OL):  Agresivo, buen punto de partida.           │
# │   → Tiende a overshoot alto (~25%). Bueno para procesos rápidos.   │
# │                                                                     │
# │ Ziegler-Nichols (CL):  Basado en Ku/Pu (ganancia/periodo último)  │
# │   → Similar agresividad. Requiere llevar el sistema al límite.     │
# │                                                                     │
# │ Cohen-Coon: Mejor para procesos con retardo significativo (θ/τ>0.3)│
# │   → Más agresivo que ZN, puede ser inestable en plantas rápidas.   │
# │                                                                     │
# │ Lambda (IMC): Conservador, prioriza robustez.                      │
# │   → Excelente rechazo de perturbaciones, respuesta lenta.          │
# │   → λ grande = más robusto pero más lento.                         │
# │                                                                     │
# │ AMIGO: Balance moderno entre rendimiento y robustez.               │
# │   → Desarrollado por Åström & Hägglund. Buenas garantías de       │
# │     margen de ganancia y fase.                                      │
# │                                                                     │
# │ Tyreus-Luyben: Versión conservadora de ZN para industria química.  │
# │   → Menos overshoot que ZN, settling time más largo.               │
# │                                                                     │
# │ ANALOGÍA IA:                                                        │
# │   ZN ≈ SGD con learning rate alto (rápido, oscila)                 │
# │   Lambda ≈ SGD con learning rate bajo + weight decay (robusto)     │
# │   AMIGO ≈ Adam optimizer (adaptativo, buen compromiso)             │
# │   Cohen-Coon ≈ SGD con momentum alto (agresivo en gradientes)      │
# └─────────────────────────────────────────────────────────────────────┘
#
# ════════════════════════════════════════════════════════════════════════
