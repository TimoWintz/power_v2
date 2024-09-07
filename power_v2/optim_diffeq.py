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
import equinox as eqx
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from diffrax import diffeqsolve, ODETerm, Dopri5, Tsit5, Kvaerno3, ImplicitEuler, Euler
from jax import numpy as jnp
from jax import grad, jit, vmap
import diffrax
from jax import jacfwd, jacobian, hessian
import optimistix as optx
import jax
from optax import lbfgs
import optax

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


def f_search_array(x: jnp.ndarray, distance: np.ndarray, array: np.ndarray):
    x = jax.lax.stop_gradient(x)
    index = jnp.searchsorted(distance, x)
    return jnp.take(array, index - 1, mode="clip")


def f_drag_force(
    x: jnp.ndarray,
    v: jnp.ndarray,

    distance: np.ndarray,
    wind_speed: np.ndarray,
    rolling_resistance: np.ndarray,
    air_resistance_coef: np.ndarray,
):
    wind_speed = f_search_array(x, distance=distance, array=wind_speed)
    rolling_resistance_force = f_search_array(
        x,
        distance=distance,
        array=rolling_resistance,
    )
    air_resistance_coef = f_search_array(
        x, distance=distance, array=air_resistance_coef
    )
    air_resistance_force = air_resistance_coef * abs(v + wind_speed) * (v + wind_speed)
    total_resistance = air_resistance_force + rolling_resistance_force
    return total_resistance

@jit
def dk_dx(
    x: jnp.ndarray,
    k: jnp.ndarray,
    rider_force: jnp.ndarray,
    distance: np.ndarray,
    dz_dx: np.ndarray,
    wind_speed: np.ndarray,
    rolling_resistance: np.ndarray,
    air_resistance_coef: np.ndarray,
    total_mass: np.ndarray,
):
    v = jnp.sqrt(2 * jnp.clip(k, 0, None) / total_mass)
    
    val_drag_force = f_drag_force(
        x, v, distance, wind_speed, rolling_resistance, air_resistance_coef
    )
    val_rider_force = f_search_array(x, distance, array=rider_force)
    val_dz_dx = f_search_array(x, distance, array=dz_dx)
    val_gravity_force = GRAVITY_ACCELERATION * val_dz_dx * total_mass


    total_force = (
        val_rider_force
        - val_drag_force - val_gravity_force
    )
    return total_force

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
    anaerobic_capacity = jnp.clip(anaerobic_capacity, 0, None)
    max_power_ = (
        critical_power
        + (max_power - critical_power)
        / ((max_power - critical_power) * duration + anaerobic_work_capacity)
        * anaerobic_capacity[:-1]
    )
    return max_power_

#@partial(jit, static_argnums=(8,))
def f_speed(
    force: jnp.ndarray,
    start_speed: np.ndarray,
    distance: np.ndarray,
    dz_dx: np.ndarray,
    wind_speed: np.ndarray,
    rolling_resistance: np.ndarray,
    air_resistance_coef: np.ndarray,
    total_mass: np.ndarray,
    step_factor: int,
):
    x_vals = jnp.linspace(0, distance[-1], (len(distance) - 1) * step_factor + 1)
    
    def dk(x, k):
        return dk_dx(x, k, rider_force=force,
                                 distance=distance,
                                 dz_dx=dz_dx,
                                 wind_speed=wind_speed,
                                 rolling_resistance=rolling_resistance, air_resistance_coef=air_resistance_coef, total_mass=total_mass)
    k_vals = [0.5 * total_mass  * start_speed ** 2]
    for i in range(1, len(x_vals)):
        k_vals.append(k_vals[i-1] + (x_vals[i] - x_vals[i-1]) * dk(x_vals[i-1], k_vals[i-1]))
    k_vals = jnp.asarray(k_vals)
    speed = jnp.sqrt(2 * k_vals.flatten() / total_mass)
    duration = jnp.diff(x_vals)/ (0.5 * (speed[:-1] + speed[1:]))
    time = jnp.pad(jnp.cumsum(duration), (1,0))
    return jnp.diff(x_vals[::step_factor]) / jnp.diff(time[::step_factor])

    


@dataclass
class PowerOptimizer:
    track: Track
    segments: list[Segment]
    resistance_model: ResistanceModel
    rider_model: RiderModel
    standing_start: bool

    step_size: int = 10

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

        idx_segment = np.searchsorted(self.segment_distance_m, self.track.distance_m[1:])
        self.wind_speed = np.zeros(len(idx_segment))
        self.rolling_resistance = np.zeros(len(idx_segment))
        self.air_resistance_coef = np.zeros(len(idx_segment))
        self.dz_dx = np.diff(self.elevation_m) / np.diff(self.distance_m)

        self.anaerobic_work_capacity = np.asarray(
            [self.rider_model.anaerobic_reserve_j]
        )
        self.critical_power = np.asarray([self.rider_model.critical_power_w])
        self.max_power = np.asarray([self.rider_model.max_power_w])

        for i, seg_i in enumerate(idx_segment):
            seg = self.segments[seg_i]
           
            self.wind_speed[i] = self.get_wind_speed(
                self.track.points[i],
                self.track.points[i + 1],
                kmh_to_ms(seg.weather_model.wind_speed_kmh),
                seg.weather_model.wind_angle_rad,
            )
    
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
        
        self.step_factor = self.track.precision_m // self.step_size
        self.solver = diffrax.Euler()
        

    def f_obj(self, force, args):
        alpha, beta = args
        speed = self.f_speed(force)
        power = speed * force
        duration = self.length_m / speed
        anaerobic_capacity = self.f_anaerobic_capacity_with_duration(
            power,
            duration,
        )
        max_power = self.f_max_power(duration, anaerobic_capacity)
        total_time = jnp.reshape(duration.sum(), (1,))
        min_anaerobic_capacity = jnp.reshape((anaerobic_capacity / self.rider_model.critical_power_w).min(), (1,))
        max_extra_power = jnp.reshape(jnp.clip((max_power - power)/self.max_power, None, 0).min(),(1,))

        return jnp.concatenate((
            total_time,
            alpha * min_anaerobic_capacity,
            beta * max_extra_power))

    
    def f_anaerobic_capacity_with_duration(self, power, duration):
        return f_anaerobic_capacity_with_duration(power, duration, self.anaerobic_work_capacity, self.anaerobic_work_capacity, self.critical_power)
    
    def f_max_power(self, duration, wp_bal):
        return f_max_power(duration, wp_bal, self.max_power, self.anaerobic_work_capacity, self.critical_power)

    def f_speed(self, force):
        return f_speed(
                force,
                np.array([0]),
                self.distance_m,
                self.dz_dx,
                self.wind_speed,
                self.rolling_resistance,
                self.air_resistance_coef,
                self.total_mass,
                step_factor=self.step_factor
            )

    def f_total_time(
        self,
        force: jnp.ndarray,
    ):
        speed = self.f_speed(force)
        duration = self.length_m / speed
        return duration.sum()
    
    def get_min_force(self):
        rider_force = np.zeros_like(self.length_m)
        k = np.sqrt(2 * np.ones_like(self.length_m) / self.total_mass)
        pt = 0.5 * (self.distance_m[:-1] + self.distance_m[1:])
        dk_dx_ = dk_dx(pt, k, rider_force, self.distance_m, self.dz_dx, self.wind_speed, self.rolling_resistance, self.air_resistance_coef, self.total_mass)
        return np.clip(-dk_dx_, 0, None)
    

    def setup_solvers(self, y):
        print("Setting up solvers...")
        options={"jac":"bwd"}
        self.args=(500, 500)
        solver = optx.LevenbergMarquardt(rtol=1e-2, atol=1e-2)
        f_struct = jax.ShapeDtypeStruct((3,), jnp.float32)
        aux_struct = None
        # Any Lineax tags describing the structure of the Jacobian matrix d(fn)/dy.
        # (In this case it's just a 1x1 matrix, so these don't matter.)
        tags = frozenset()

        self.fn_obj = eqx.filter_jit(lambda x, args: (self.f_obj(x, args), None))
      
        # These arguments are always fixed throughout interactive solves.
        self.step = eqx.filter_jit(
            eqx.Partial(solver.step, fn=self.fn_obj, args=self.args, options=options, tags=tags)
        )
        self.state = solver.init(self.fn_obj, y, self.args, options, f_struct, aux_struct, tags)
        print("Done. Initial state:")
        self.print_state(y)
        

    def print_state(self, y):
        val = self.fn_obj(y, self.args)
        print(f"Force: {y} with total time {val[0][0]}.")
        print(f"Constraint: {val[0][1:]}")

    def compute(self):
        initial_force = self.get_min_force()
        speed = self.f_speed(initial_force)
        self.setup_solvers(initial_force)
                
        y = initial_force
        for _ in range(10):
            print("STEP: Minimizing time")
            y, self.state, aux = self.step(y=y, state=self.state)
            self.print_state(y)
            

       
        force = y
        speed = self.f_speed(force)
        power = speed * force / self.resistance_model.drivetrain_efficiency
        duration = self.length_m / speed
        wp_bal = self.f_anaerobic_capacity_with_duration(power, duration)

        print("Done")
        self.track.data = {
            "power/optimal": power,
            "wp_bal/optimal": wp_bal,
            "speed/optimal": speed,
        }
        return self.track
