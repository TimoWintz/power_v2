from functools import partial
from power_v2.track import Segment, Track
from power_v2.model import ResistanceModel, RiderModel
from power_v2.optim import power_to_speed
from dataclasses import dataclass
import numpy as np
from typing import Optional, Any, Union
from scipy.optimize import (
    minimize,
    NonlinearConstraint,
    Bounds,
    fmin,
    approx_fprime,
    BFGS,
)
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from diffrax import diffeqsolve, ODETerm, Dopri5, Tsit5, Kvaerno3, ImplicitEuler, Euler
from jax import numpy as jnp
from jax import grad, jit, vmap
import diffrax
from jax import jacfwd, jacobian, hessian
import optimistix as optx
import jax

jax.config.update("jax_enable_x64", True)

# from jax.scipy.optimize import minimize
GRAVITY_ACCELERATION = 9.81


def angle_from_coordinates(lat1, lng1, lat2, lng2):
    dlng = lng2 - lng1
    y = np.sin(dlng) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlng)
    return np.arctan2(y, x)


def kmh_to_ms(x: float) -> float:
    return x / 3.6


def air_density(altitude_m: float, temperature_C: float) -> float:
    pressure_pa = 100 * 1013.25 * pow(1 - 0.0065 * altitude_m / 288.15, 5.255)
    R_air = 287
    temperature_K = temperature_C + 273
    return pressure_pa / temperature_K / R_air


grad_interp = grad(jnp.interp)


def dz_dx(x: jnp.ndarray, distance: np.ndarray, elevation: np.ndarray):
    dz = np.diff(elevation) / np.diff(distance)
    x = jax.lax.stop_gradient(x)
    index = jnp.searchsorted(distance, x, side="right")
    return jnp.take(dz, index - 1, mode="clip")


def f_force(x: jnp.ndarray, distance: np.ndarray, force: np.ndarray):
    x = jax.lax.stop_gradient(x)
    index = jnp.searchsorted(distance, x, side="right")
    return jnp.take(force, index - 1, mode="clip")


def f_wind_speed(x: jnp.ndarray, distance: np.ndarray, wind_speed: np.ndarray):
    x = jax.lax.stop_gradient(x)
    index = jnp.searchsorted(distance, x, side="right")
    return jnp.take(wind_speed, index - 1, mode="clip")


def f_rolling_resistance(
    x: jnp.ndarray, distance: np.ndarray, rolling_resistance: np.ndarray
):
    x = jax.lax.stop_gradient(x)
    index = jnp.searchsorted(distance, x, side="right")
    return jnp.take(rolling_resistance, index - 1, mode="clip")


def f_air_resistance_coef(
    x: jnp.ndarray, distance: np.ndarray, air_resistance_coef: np.ndarray
):
    x = jax.lax.stop_gradient(x)
    index = jnp.searchsorted(distance, x, side="right")
    return jnp.take(air_resistance_coef, index - 1, mode="clip")


def drag_force(
    x: jnp.ndarray,
    v: jnp.ndarray,

    distance: np.ndarray,
    wind_speed: np.ndarray,
    rolling_resistance: np.ndarray,
    air_resistance_coef: np.ndarray,
):
    wind_speed = f_wind_speed(x, distance=distance, wind_speed=wind_speed)
    rolling_resistance_force = f_rolling_resistance(
        x,
        distance=distance,
        rolling_resistance=rolling_resistance,
    )
    air_resistance_coef = f_air_resistance_coef(
        x, distance=distance, air_resistance_coef=air_resistance_coef
    )

    air_resistance_force = air_resistance_coef * abs(v + wind_speed) * (v + wind_speed)
    total_resistance = air_resistance_force + rolling_resistance_force
    return total_resistance


def dk_dx(
    x: jnp.ndarray,
    k: jnp.ndarray,
    force: jnp.ndarray,
    distance: np.ndarray,
    elevation: np.ndarray,
    wind_speed: np.ndarray,
    rolling_resistance: np.ndarray,
    air_resistance_coef: np.ndarray,
    total_mass: np.ndarray,
):
    v = jnp.sqrt(2 * k / total_mass)
    val_drag_force = drag_force(
        x, v, distance, wind_speed, rolling_resistance, air_resistance_coef
    )
    in_force = f_force(x, distance, force)
    val_dz_dx = dz_dx(x, distance, elevation)
    return (
        in_force
        - val_drag_force - jnp.sin(val_dz_dx) * GRAVITY_ACCELERATION * total_mass
    )

def update_anaerobic_capacity(
    power: float,
    duration: float,
    current_capacity: float,
    anaerobic_work_capacity: np.ndarray,
    critical_power: np.ndarray,
):
    power = jnp.clip(power, 0, None)
    return jnp.where(
        power > critical_power,
        current_capacity - duration * (power - critical_power),
        current_capacity
        + (anaerobic_work_capacity - current_capacity)
        * (1 - jnp.exp((power - critical_power) * duration / anaerobic_work_capacity)),
    )


def f_anaerobic_capacity_with_duration(
    power: jnp.ndarray,
    duration: jnp.ndarray,
    start_capacity: np.ndarray,
    anaerobic_work_capacity: np.ndarray,
    critical_power: np.ndarray,
):
    assert len(duration) == len(power)
    capacity = jnp.zeros(len(power) + 1)
    capacity = capacity.at[0].set(start_capacity[0])
    for i in range(len(power)):
        capacity = capacity.at[i + 1].set(
            update_anaerobic_capacity(
                power[i],
                duration[i],
                capacity[i],
                anaerobic_work_capacity[0],
                critical_power[0],
            )
        )
    return capacity


def f_max_power(
    duration: jnp.ndarray,
    anaerobic_capacity: jnp.ndarray,
    max_power: np.ndarray,
    anaerobic_work_capacity: np.ndarray,
    critical_power: np.ndarray,
):
    assert len(anaerobic_capacity) == len(duration) + 1
    max_power_ = (
        critical_power
        + (max_power - critical_power)
        / ((max_power - critical_power) * duration + anaerobic_work_capacity)
        * anaerobic_capacity[:-1]
    )
    return jnp.clip(max_power_, 0, None)


def f_speed(
    force: jnp.ndarray,
    start_speed: np.ndarray,
    distance: np.ndarray,
    elevation: np.ndarray,
    wind_speed: np.ndarray,
    rolling_resistance: np.ndarray,
    air_resistance_coef: np.ndarray,
    total_mass: np.ndarray,
    solver,
    max_steps: int = 10000,
):
    dt0 = distance[-1] / (max_steps - 1)
    saveat = 0.5*(distance[1:] + distance[:-1])
    term = lambda x, k, _: dk_dx(x, k, force=force, distance=distance, elevation=elevation, wind_speed=wind_speed, rolling_resistance=rolling_resistance, air_resistance_coef=air_resistance_coef, total_mass=total_mass)
    term = ODETerm(term)
    y0 = 0.5 * total_mass  * start_speed ** 2
    solution = diffeqsolve(
        term,
        solver,
        t0=0,
        t1=distance[-1],
        dt0=dt0,
        y0=y0,
        max_steps=max_steps,
        saveat=diffrax.SaveAt(ts=saveat),
    )
    return jnp.sqrt(2 * solution.ys.flatten() / total_mass)


@dataclass
class PowerOptimizer:
    track: Track
    segments: list[Segment]
    resistance_model: ResistanceModel
    rider_model: RiderModel
    standing_start: bool

    def get_wind_speed(
        self,
        start_point,
        end_point,
        wind_speed,
        wind_angle_rad,
    ) -> np.ndarray:
        lat1, lng1, _ = start_point
        lat2, lng2, _ = end_point
        angle_rad = angle_from_coordinates(lat1, lng1, lat2, lng2)
        return wind_speed * np.cos(angle_rad - wind_angle_rad)

    def get_segment_length(self) -> np.ndarray:
        return np.asarray([seg.length_m for seg in self.segments])

    def __post_init__(self):
        self.num_points = len(self.track)
        self.distance_m = self.track.distance_m.copy()
        self.elevation_m = self.track.elevation_m.copy()
        self.total_mass = np.asarray([self.resistance_model.total_mass_kg])
        self.length_m = np.diff(self.distance_m)
        self.segment_length_m = self.get_segment_length()
        self.segment_distance_m = np.cumsum(self.segment_length_m)
        idx_segment = np.searchsorted(self.segment_distance_m, self.track.distance_m)

        self.wind_speed = np.zeros(self.num_points)
        self.rolling_resistance = np.zeros(self.num_points)
        self.air_resistance_coef = np.zeros(self.num_points)

        self.anaerobic_work_capacity = np.asarray(
            [self.rider_model.anaerobic_reserve_j]
        )
        self.critical_power = np.asarray([self.rider_model.critical_power_w])
        self.max_power = np.asarray([self.rider_model.max_power_w])

        for i, seg_i in enumerate(idx_segment):
            seg = self.segments[seg_i]
            if i < len(idx_segment) - 1:
                self.wind_speed[i] = self.get_wind_speed(
                    self.track.points[i],
                    self.track.points[i + 1],
                    kmh_to_ms(seg.weather_model.wind_speed_kmh),
                    seg.weather_model.wind_angle_rad,
                )
            else:
                self.wind_speed[i] = self.wind_speed[i - 1]
            self.rolling_resistance[i] = (
                self.resistance_model.crr
                * self.resistance_model.total_mass_kg
                * GRAVITY_ACCELERATION
            )
            self.air_resistance_coef[i] = (
                0.5
                * air_density(
                    self.track.elevation_m[i], seg.weather_model.temperature_c
                )
                * self.resistance_model.cda_m2
            )
        self.step_size = 10
        self.solver = diffrax.Heun()

    def f_constraint(self, force):
        speed = self.f_speed(force)
        power = speed * force
        duration = self.length_m / speed
        anaerobic_capacity = self.f_anaerobic_capacity_with_duration(
            power,
            duration,
        )
        max_power = self.f_max_power(duration, anaerobic_capacity)
        return jnp.concatenate(
            (anaerobic_capacity / self.rider_model.critical_power_w,
            (max_power - power)/self.max_power)).min()

    
    def f_anaerobic_capacity_with_duration(self, power, duration):
        return f_anaerobic_capacity_with_duration(power, duration, self.anaerobic_work_capacity, self.anaerobic_work_capacity, self.critical_power)
    
    def f_max_power(self, duration, wp_bal):
        return f_max_power(duration, wp_bal, self.max_power, self.anaerobic_work_capacity, self.critical_power)

    def f_speed(self, force):
        return f_speed(
                force,
                np.array([0]),
                self.distance_m,
                self.elevation_m,
                self.wind_speed,
                self.rolling_resistance,
                self.air_resistance_coef,
                self.total_mass,
                max_steps=int(self.distance_m[-1] / self.step_size),
                solver=self.solver,
            )

    def f_total_time(
        self,
        power: jnp.ndarray,
    ):
        speed = self.f_speed(power)
        duration = self.length_m / speed
        return duration.sum()

    def get_initial_force(self):
        initial_power = self.rider_model.critical_power_w * np.ones(self.num_points - 1)
        initial_speed = power_to_speed(initial_power, self.air_resistance_coef, self.wind_speed, GRAVITY_ACCELERATION * np.diff(self.elevation_m) / np.diff(self.distance_m), self.resistance_model.drivetrain_efficiency, self.rolling_resistance)
        initial_force = initial_power / initial_speed
        return initial_force

    def test_speed_model(self):
        print("Test speed model...")
        force = self.get_initial_force()        
        speed = self.f_speed(force)
        print(f"Done. Average speed={speed.mean()},({speed.min(), speed.max()})")

    def compute(self):
        self.test_speed_model()
        print(f"JIT compiling functions")
        initial_force = self.get_initial_force()

        f_total_time = jit(self.f_total_time)
        g_total_time = grad(f_total_time)
        f_constraint = jit(self.f_constraint)
        g_constraint = grad(f_constraint)
        x = initial_force

        g_total_time(x)
        g_constraint(x)

        nl_constraint = NonlinearConstraint(
            f_constraint,
            lb=0,
            ub=self.rider_model.anaerobic_reserve_j / self.rider_model.critical_power_w,
            jac=g_constraint,
            hess=BFGS(),
        )

        print("Minimizing time...")
        minimizer = minimize(
            f_total_time,
            x0=x,
            jac=g_total_time,
            constraints=[nl_constraint],
        )

       
        force = minimizer.x
        speed = self.f_speed(force)
        power = speed * force
        duration = self.length_m / speed
        wp_bal = self.f_anaerobic_capacity_with_duration(power, duration)

        print("Done")
        self.track.data = {
            "power/optimal": power,
            "wp_bal/optimal": wp_bal,
            "speed/optimal": speed,
        }
        return self.track
