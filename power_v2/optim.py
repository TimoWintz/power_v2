from functools import partial
from power_v2.track import Segment, Track
from power_v2.model import ResistanceModel, RiderModel
from dataclasses import dataclass
import numpy as np
from typing import Optional, Any, Union
from scipy.optimize import minimize, NonlinearConstraint, Bounds, fmin
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d

from jax import numpy as np
GRAVITY_ACCELERATION = 9.81



def air_density(altitude_m: float, temperature_C: float) -> float:
    pressure_pa = 100 * 1013.25 * pow(1 - 0.0065 * altitude_m / 288.15, 5.255)
    R_air = 287
    temperature_K = temperature_C + 273
    return pressure_pa / temperature_K / R_air


def angle_from_coordinates(lat1, lng1, lat2, lng2):
    dlng = lng2 - lng1
    y = np.sin(dlng) * np.cos(lat2)
    x = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlng)
    return np.arctan2(y, x)


def kmh_to_ms(x: float) -> float:
    return x / 3.6


def speed_to_power(
    speed: np.ndarray,
    air_resistance_coef: np.ndarray,
    wind_speed: np.ndarray,
    gravity_force: np.ndarray,
    efficiency: float,
    rolling_resistance: float,
    mass_times_acceleration: Union[np.ndarray, float] = 0,
) -> np.ndarray:
    air_resistance_force = (
        air_resistance_coef * abs(speed + wind_speed) * (speed + wind_speed)
    )
    return (
        1
        / efficiency
        * speed
        * (
            air_resistance_force
            + rolling_resistance
            + gravity_force
            + mass_times_acceleration
        )
    )


def power_to_speed_float(
    power: float,
    air_resistance_coef: float,
    wind_speed: float,
    gravity_force: float,
    efficiency: float,
    rolling_resistance: float,
):
    a3 = air_resistance_coef
    a2 = -2 * air_resistance_coef * wind_speed
    a1 = air_resistance_coef * wind_speed**2 + rolling_resistance + gravity_force
    a0 = -efficiency * power
    rts = np.roots(np.asarray([a3, a2, a1, a0]))
    rts = np.where(rts.imag == 0, rts, 0)
    rts = np.sort_complex(rts)
    return rts[-1].real


def power_to_speed(
    power: np.ndarray,
    air_resistance_coef: np.ndarray,
    wind_speed: np.ndarray,
    gravity_force: np.ndarray,
    efficiency: float,
    rolling_resistance: float,
):
    return np.asarray(
        [
            power_to_speed_float(
                power[i],
                air_resistance_coef[i],
                wind_speed[i],
                gravity_force[i],
                efficiency,
                rolling_resistance,
            )
            for i in range(len(power))
        ]
    )


def speed_to_wp_bal(
    speed: np.ndarray,
    air_resistance_coef: np.ndarray,
    wind_speed: np.ndarray,
    gravity_force: np.ndarray,
    length: np.ndarray,
    efficiency: float,
    rolling_resistance: float,
    anaerobic_reserve: float,
    critical_power: float,
    mass: float,
) -> np.ndarray:

    duration = length / speed
    speed_padded = np.pad(speed, (1, 0))
    acceleration = (speed_padded[1:] - speed_padded[:-1]) / duration
    power = 0.5 * (
        speed_to_power(
            speed=speed_padded[1:],
            mass_times_acceleration=acceleration * mass,
            air_resistance_coef=air_resistance_coef,
            wind_speed=wind_speed,
            gravity_force=gravity_force,
            efficiency=efficiency,
            rolling_resistance=rolling_resistance,
        )
        + speed_to_power(
            speed=speed_padded[:-1],
            mass_times_acceleration=acceleration * mass,
            air_resistance_coef=air_resistance_coef,
            wind_speed=wind_speed,
            gravity_force=gravity_force,
            efficiency=efficiency,
            rolling_resistance=rolling_resistance,
        )
    )

    current_reserve_j = anaerobic_reserve
    result = np.zeros_like(power)
    for i in range(len(power)):
        if power[i] > critical_power:
            current_reserve_j -= duration[i] * (power[i] - critical_power)
        else:
            current_reserve_j += (anaerobic_reserve - current_reserve_j) * (
                1
                - np.exp((power[i] - critical_power) * duration[i] / anaerobic_reserve)
            )
        result[i] = current_reserve_j
    return result


def speed_and_theta_to_duration(
    speed: np.ndarray, thetas: np.ndarray, length: np.ndarray
):
    length_dynamic = length * thetas
    length_static = length * (1 - thetas)

    speed_padded = np.pad(speed, (1, 0))
    duration_static = length_static / speed
    duration_dynamic = length_dynamic / 0.5 / (speed_padded[1:] + speed_padded[:-1])
    return duration_dynamic, duration_static


def get_acceleration(speed: np.ndarray, duration: np.ndarray):
    return (speed[1:] - speed[:-1]) / duration


def power_affine(
    steady_power: np.ndarray,
    transition_length: np.ndarray,
    total_length: np.ndarray,
    air_resistance_coef: np.ndarray,
    wind_speed: np.ndarray,
    gravity_force: np.ndarray,
    efficiency: float,
    rolling_resistance: float,
    mass: float,
):
    transition_length = np.clip(transition_length, 0, total_length)

    speed_static_ = np.pad(
        power_to_speed(
            power=steady_power,
            air_resistance_coef=air_resistance_coef,
            wind_speed=wind_speed,
            gravity_force=gravity_force,
            efficiency=efficiency,
            rolling_resistance=rolling_resistance,
        ),
        (1, 0),
    )

    steady_power_ = np.pad(steady_power, (1, 0))
    transition_avg_speed = 0.5 * (speed_static_[1:] + speed_static_[:-1])
    transition_duration = transition_length / transition_avg_speed
    static_duration = (total_length - transition_length) / speed_static_[1:]

    acceleration = (speed_static_[1:] - speed_static_[:-1]) / transition_duration
    transition_power = (
        0.5 * (steady_power_[1:] + steady_power_[:-1])
        + acceleration * mass * transition_avg_speed
    )

    duration = np.dstack([transition_duration, static_duration]).flatten()
    power = np.dstack([transition_power, steady_power]).flatten()
    return acceleration, duration, power


def power_to_wp_bal(
    duration: np.ndarray,
    power: np.ndarray,
    anaerobic_reserve: float,
    critical_power: float,
    wp_bal_start: float = 0,
):
    if wp_bal_start == 0:
        wp_bal_start = anaerobic_reserve
    def update_reserve(current_reserve, power, duration):
        power = max(power, 0)
        if power > critical_power:
            current_reserve -= duration * (power - critical_power)
        else:
            current_reserve += (anaerobic_reserve - current_reserve) * (
                1 - np.exp((power - critical_power) * duration / anaerobic_reserve)
            )
        return current_reserve

    wp_bal = np.zeros(1+len(duration))
    current_reserve = wp_bal_start
    wp_bal[0] = wp_bal_start
    for i in range(len(power)):
        current_reserve = update_reserve(current_reserve, power[i], duration[i])
        wp_bal[i+1] = current_reserve
    return wp_bal


def energy_to_max_power_and_time(
        energy: np.ndarray,
        wp_bal: np.ndarray,
        max_power: float,
        anaerobic_reserve: float,
        critical_power: float):
    delta_p = max_power - critical_power
    a2 = critical_power * delta_p
    a1 = - energy * delta_p + anaerobic_reserve * delta_p + wp_bal * critical_power
    a0 = -energy * wp_bal
    rts = np.roots(np.asarray([a2, a1, a0]))
    rts = np.sort(rts)
    delta_t = rts[-1]
    if delta_t < 0:
        raise RuntimeError("Wrong parameters")
    power = critical_power + (delta_p * anaerobic_reserve) / (
        delta_p * delta_t + wp_bal
    )
    return power, delta_t


def wp_bal_to_max_power(
    duration: np.ndarray,
    wp_bal: np.ndarray,
    max_power: float,
    anaerobic_reserve: float,
    critical_power: float,
    wp_bal_start: float=0
):
    if wp_bal_start == 0:
        wp_bal_start = anaerobic_reserve
    max_power_ = (
        critical_power
        + (max_power - critical_power)
        / ((max_power - critical_power) * duration + anaerobic_reserve)
        * wp_bal[:-1]
    )
    return np.clip(max_power_, 0, np.inf)


@dataclass
class PowerOptimizer:
    track: Track
    segments: list[Segment]
    resistance_model: ResistanceModel
    rider_model: RiderModel
    standing_start: bool

    def _get_wind_speed(self) -> np.ndarray:
        res = []
        for seg in self.segments:
            lat1, lng1, _ = seg.start_point
            lat2, lng2, _ = seg.end_point
            angle_rad = angle_from_coordinates(lat1, lng1, lat2, lng2)
            res.append(
                kmh_to_ms(
                    seg.weather_model.wind_speed_kmh
                    * np.cos(angle_rad - seg.weather_model.wind_angle_rad)
                )
            )
        return np.asarray(res)

    def _get_rolling_resistance(self) -> float:
        return (
            self.resistance_model.crr
            * self.resistance_model.total_mass_kg
            * GRAVITY_ACCELERATION
        )

    def _get_gravity_force(self) -> np.ndarray:
        res = []
        for seg in self.segments:
            res.append(
                self.resistance_model.total_mass_kg
                * GRAVITY_ACCELERATION
                * np.sin(np.arctan(seg.grade_pct / 100))
            )
        return np.asarray(res)

    def _get_air_resistance_coef(self) -> np.ndarray:
        res = []
        for seg in self.segments:
            res.append(
                0.5
                * air_density(seg.start_point[2], seg.weather_model.temperature_c)
                * self.resistance_model.cda_m2
            )
        return np.asarray(res)

    def _get_length(self) -> np.ndarray:
        return np.asarray([seg.length_m for seg in self.segments])

    def __post_init__(self):
        self.wind_speed_ms = self._get_wind_speed()
        self.rolling_resistance_force_N = self._get_rolling_resistance()
        self.gravity_force_N = self._get_gravity_force()
        self.air_resistance_coef = self._get_air_resistance_coef()
        self.length_m = self._get_length()

        if self.standing_start:
            self.length_m = np.concatenate(([self.track.precision_m], self.length_m))
            self.length_m[1] -= self.track.precision_m

            self.wind_speed_ms = np.pad(self.wind_speed_ms,(1,0), mode="edge" )
            self.rolling_resistance_force_N = np.pad(self.rolling_resistance_force_N,(1,0), mode="edge" )
            self.gravity_force_N = np.pad(self.gravity_force_N,(1,0), mode="edge" )
            self.air_resistance_coef = np.pad(self.air_resistance_coef,(1,0), mode="edge" )

    def speed_to_power(self, speed):
        return speed_to_power(
                speed=speed,
                air_resistance_coef=self.air_resistance_coef,
                wind_speed=self.wind_speed_ms,
                gravity_force=self.gravity_force_N,
                efficiency=self.resistance_model.drivetrain_efficiency,
                rolling_resistance=self.rolling_resistance_force_N,
            )

    def power_to_speed(self, power):
        return power_to_speed(
                power=power,
                air_resistance_coef=self.air_resistance_coef,
                wind_speed=self.wind_speed_ms,
                gravity_force=self.gravity_force_N,
                efficiency=self.resistance_model.drivetrain_efficiency,
                rolling_resistance=self.rolling_resistance_force_N,
            )

    def power_to_wp_bal(self, duration, power):
        return power_to_wp_bal(
                duration=duration,
                power=power,
                anaerobic_reserve=self.rider_model.anaerobic_reserve_j,
                critical_power=self.rider_model.critical_power_w,
            )

    def wp_bal_to_max_power(self, duration, wp_bal):
        return wp_bal_to_max_power(
                duration=duration,
                wp_bal=wp_bal,
                max_power=self.rider_model.max_power_w,
                anaerobic_reserve=self.rider_model.anaerobic_reserve_j,
                critical_power=self.rider_model.critical_power_w,
            )

    def get_optimal_static(self):
        n_segments = len(self.segments) + 1 if self.standing_start else len(self.segments)
        initial_power = self.rider_model.critical_power_w * np.ones(n_segments)
        initial_speed = self.power_to_speed(initial_power)

        absolute_max_power =  self.rider_model.max_power_w * np.ones(n_segments)
        absolute_max_speed = self.power_to_speed(absolute_max_power)

        def obj(speed):
            duration = self.length_m / speed
            return duration.sum()

        def constraint_wp_bal(speed):
            power = self.speed_to_power(speed)
            duration = self.length_m / speed
            if self.standing_start:
                power[0] += 0.5 * self.resistance_model.total_mass_kg * speed[0]**2 / duration[0]

            wp_bal = self.power_to_wp_bal(duration, power)
            return wp_bal.min() / self.rider_model.anaerobic_reserve_j,

        def constraint_power(speed):
            power = self.speed_to_power(speed)
            duration = self.length_m / speed
            wp_bal = self.power_to_wp_bal(duration, power)
            max_power = self.wp_bal_to_max_power(duration, wp_bal)

            return np.asarray(
                [
                    power.min() / self.rider_model.max_power_w,
                    1 - (power / max_power).max(),
                ]
            )

        lb_speed = self.rider_model.min_speed_ms * np.ones_like(initial_speed)
        ub_speed = absolute_max_speed

        bounds_speed = Bounds(lb=lb_speed, ub=ub_speed)
        nl_constraint_wp_bal = NonlinearConstraint(
            constraint_wp_bal,
            lb=0,
            ub=1,
        )
        nl_constraint_power = NonlinearConstraint(
            constraint_power,
            lb=0,
            ub=1,
        )

        options = {
            #"catol": 0.01
        }
        minimizer = minimize(
            obj,
            initial_speed,
            bounds=bounds_speed,
            constraints=[nl_constraint_wp_bal, nl_constraint_power],
            method="trust-constr",
            options=options,
        )

        speed = minimizer.x
        if not minimizer.success:
            print("Warning: minimization failed.")
        return speed

    def compute_final_speed(self, distance_m: np.ndarray, power: np.ndarray, start_speed: float):
        distance_m_segments = np.cumsum(self.length_m)
        dz_dx = interp1d(self.track.distance_m, self.track.grade_pct / 100)
        def de_dx(x, e):
            v = np.sqrt(2 * e / self.resistance_model.total_mass_kg)
            idx_power = np.clip(np.searchsorted(distance_m, x), 0, len(distance_m) - 1)
            idx_segment = np.clip(
                np.searchsorted(distance_m_segments, x) - 1,
                0,
                len(distance_m_segments) - 1,
            )
            wind_speed = self.wind_speed_ms[idx_segment]
            power_ = power[idx_power]
            rolling_resistance = self.rolling_resistance_force_N

            air_resistance_coef = self.air_resistance_coef[idx_segment]
            air_resistance_force = (
                air_resistance_coef * abs(v + wind_speed) * (v + wind_speed)
            )
            total_resistance = air_resistance_force
            + rolling_resistance
            return power_ / (v + 0.01) - total_resistance - self.resistance_model.total_mass_kg * GRAVITY_ACCELERATION * dz_dx(x)
        sol = solve_ivp(
            de_dx,
            (0, self.track.total_distance_m),
            [start_speed],
            t_eval=self.track.distance_m,
        )
        kinetic_energy = sol.y
        speed = np.sqrt(2 * kinetic_energy / self.resistance_model.total_mass_kg).flatten()
        return speed

    def compute(self):
        speed_target = self.get_optimal_static()
        steady_power = self.speed_to_power(speed_target)

        duration = self.length_m / speed_target
        wp_bal_steady = self.power_to_wp_bal(duration, steady_power)

        energy_diff = 0.5 * self.resistance_model.total_mass_kg * np.diff(np.pad(speed_target**2, (1,0)))
        force_diff = np.diff(np.pad(steady_power / speed_target, (1, 0)))

        if self.standing_start:
            steady_power[0] = steady_power[0] + energy_diff[0] / duration[0]

        transition_length = energy_diff / force_diff
        transition_length = transition_length[2:]

        distance_m = np.cumsum(self.length_m)
        new_distance_m = distance_m.copy()
        new_distance_m[1:-1] -= np.ceil(transition_length).astype(int)

        start_speed = 0 if self.standing_start else speed_target[0]
        min_scaling = 0.9
        max_scaling = 1.1
        for i in range(5):
            scaling = 0.5 * (max_scaling + min_scaling)
            steady_power_scaled = steady_power * scaling

            track_speed = self.compute_final_speed(
                distance_m, steady_power_scaled, start_speed
            )
            track_segment_speed = 0.5 * (track_speed[1:] + track_speed[:-1])
            track_duration = self.track.length_m / track_segment_speed
            track_power = steady_power_scaled[
                np.clip(
                    np.searchsorted(new_distance_m, self.track.distance_m),
                    0,
                    len(new_distance_m) - 1,
                )
            ]
            track_wp_bal = self.power_to_wp_bal(track_duration, track_power[:-1])
            if track_wp_bal.min() < 0:
                max_scaling = scaling
            else:
                min_scaling = scaling
            print(f"{scaling=}")
            print(f"{track_wp_bal.min()=}")
        if track_wp_bal[-1] > 0:
            print(f"adding power to final segment {track_wp_bal[-1] / duration[-1]}")
            steady_power_scaled[-1] = steady_power_scaled[-1] + track_wp_bal[-1] / duration[-1]
            track_duration = self.track.length_m / track_segment_speed
            track_power = steady_power_scaled[
                np.clip(
                    np.searchsorted(
                        new_distance_m, self.track.distance_m
                    ),
                    0,
                    len(new_distance_m) - 1,
                )
            ]
            track_wp_bal = self.power_to_wp_bal(track_duration, track_power[:-1])

        self.track.optimized_data = {
            "power_w": track_power,
            "speed_kmh": 3.6 * track_speed,
            "wp_bal_j": track_wp_bal,
        }
        return self.track
