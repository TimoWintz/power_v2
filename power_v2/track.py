from dataclasses import dataclass, field
import numpy as np
import geopy.distance
from scipy.interpolate import interp1d
import bisect
from power_v2.model import WeatherModel
from typing import Optional

@dataclass
class Segment:
    start_m: int
    end_m: int
    deniv_m: int

    start_point: tuple[float, float, float]
    end_point: tuple[float, float, float]

    weather_model: WeatherModel
    

    @property
    def length_m(self) -> int:
        return self.end_m - self.start_m

    @property
    def grade_pct(self) -> float:
        return 100 * self.deniv_m / self.length_m


@dataclass
class Track:
    name: str
    points: np.ndarray  # lat, lon, elev
    precision_m: int = 50

    data: dict[str, np.ndarray] = field(default_factory=dict)
    optimized_data: dict[str, np.ndarray] = field(default_factory=dict)

    @property
    def latitude(self):
        return self.points[:, 0]

    @property
    def longitude(self):
        return self.points[:, 1]

    @property
    def length_m(self):
        return np.diff(self.distance_m)

    @property
    def elevation_m(self):
        return self.points[:, 2]

    @property
    def grade_pct(self):
        return np.pad(100 * np.diff(self.elevation_m) / np.diff(self.distance_m), (1,0))

    def resample(self, step_m: int):
        idx_valid = np.roll(self.distance_m, -1) > self.distance_m
        self.points = self.points[idx_valid]
        self.distance_m = self.distance_m[idx_valid]
        new_distance = np.arange(0, self.total_distance_m, step_m)
        points_interpolator = interp1d(self.distance_m, self.points, axis=0)
        self.points = points_interpolator(new_distance)
        self.distance_m = new_distance.astype(int)
        self.points[:, 2] = np.round(self.points[:, 2])
        self.precision_m = step_m

    def get_segments(self, max_error_m: int, weather_model: WeatherModel) -> list[Segment]:
        all_indices = list(range(len(self)))
        indices = [0, len(self) - 1]
        true_elevation = self.points[:, 2]
        while True:
            linear_interpolator = interp1d(indices, true_elevation[indices])
            error = linear_interpolator(all_indices) - true_elevation
            idx_max = np.argmax(error)
            idx_min = np.argmin(error)
            if error[idx_max] - error[idx_min] <= max_error_m:
                break
            if np.abs(error[idx_max]) > np.abs(error[idx_min]):
                bisect.insort(indices, idx_max)
            else:
                bisect.insort(indices, idx_min)

        segments = [
            Segment(
                start_m=self.distance_m[indices[i]],
                end_m=self.distance_m[indices[i + 1]],
                deniv_m=self.elevation_m[indices[i + 1]] - self.elevation_m[indices[i]],
                start_point=tuple(self.points[indices[i]]),
                end_point=tuple(self.points[indices[i + 1]]),
                weather_model=weather_model,
            )
            for i in range(len(indices) - 1)
        ]
        return segments

    def __post_init__(self):
        self.distance_m = np.zeros((len(self.points)))
        for i in range(1, len(self)):
            distance_m = geopy.distance.geodesic(
                self.points[i - 1, :2],
                self.points[i, :2],
            ).meters
            distance_m = np.sqrt(
                distance_m**2 + (self.points[i, 2] - self.points[i - 1, 2]) ** 2
            )
            self.distance_m[i] = self.distance_m[i - 1] + distance_m
        self.resample(self.precision_m)

    def __len__(self):
        return len(self.points)

    @property
    def total_distance_m(self):
        return self.distance_m[-1]
