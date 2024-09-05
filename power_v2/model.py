from dataclasses import dataclass
import numpy as np
from enum import Enum


@dataclass
class ResistanceModel:
    total_mass_kg: float
    cda_m2: float
    crr: float
    drivetrain_efficiency: float


@dataclass
class RiderModel:
    critical_power_w: float
    anaerobic_reserve_j: float
    max_power_w: float
    min_power_w: float = 10


class WindDirection(Enum):
    N = "⬇️ North"
    NE = "↙️ North-East"
    E = "⬅️ East"
    SE = "↖️ South-East"
    S = "⬆️ South"
    SW =  "↗️ South-West"
    W = "➡️ West"
    NW = "↘️ North-West"


@dataclass
class WeatherModel:
    temperature_c: float
    wind_speed_kmh: float
    wind_angle_rad: float # 0 = N, 90 = E

    def set_angle(self, dir: WindDirection):
        if dir == WindDirection.N:
            self.wind_angle_rad = 0
        elif dir == WindDirection.NE:
            self.wind_angle_rad = np.pi / 4
        elif dir == WindDirection.E:
            self.wind_angle_rad = np.pi /2
        elif dir == WindDirection.SE:
            self.wind_angle_rad = 3 * np.pi / 4
        elif dir == WindDirection.S:
            self.wind_angle_rad = np.pi
        elif dir == WindDirection.SW:
            self.wind_angle_rad = 5 * np.pi / 4
        elif dir == WindDirection.W:
            self.wind_angle_rad = 3 * np.pi / 2
        elif dir == WindDirection.NW:
            self.wind_angle_rad = 7 * np.pi / 4


def get_default_resistance_model() -> ResistanceModel:
    return ResistanceModel(
        total_mass_kg=85, cda_m2=0.3, crr=0.004, drivetrain_efficiency=0.98
    )


def get_default_rider_model() -> RiderModel:
    return RiderModel(critical_power_w=300, anaerobic_reserve_j=18000, max_power_w=800, min_power_w=10)


def get_default_weather_model() -> WeatherModel:
    return WeatherModel(temperature_c=20, wind_speed_kmh=0, wind_angle_rad=0)
