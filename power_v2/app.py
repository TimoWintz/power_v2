from nicegui import ui, run, events
from gpxpy import gpx
import gpxpy
from power_v2.model import (
    ResistanceModel,
    RiderModel,
    WeatherModel,
    WindDirection,
    get_default_resistance_model,
    get_default_rider_model,
    get_default_weather_model,
)
from power_v2.track import Track
from power_v2.optim_diffeq import PowerOptimizer

import numpy as np
from typing import Optional
import os
from dataclasses import field, dataclass
from pathlib import Path
from matplotlib import colormaps


MAX_GRADE = 15


def get_color_from_grade(grade_pct: float) -> tuple[int, int, int]:
    cmap = colormaps.get_cmap("turbo")
    norm_grade = np.clip(0.5 + 0.5 * grade_pct / 15, 0, 1)
    return cmap(norm_grade)


@dataclass
class App:
    resistance_model: ResistanceModel = field(
        default_factory=get_default_resistance_model
    )
    rider_model: RiderModel = field(default_factory=get_default_rider_model)
    weather_model: WeatherModel = field(default_factory=get_default_weather_model)
    loaded_tracks: list[Track] = field(default_factory=list)
    selected_track: Optional[Track] = None

    data_plots: list = field(default_factory=list)

    max_error_m: int = 25

    def __post_init__(self):

        columns = [
            {
                "name": "name",
                "label": "Name",
                "field": "name",
                "sortable": True,
                "align": "left",
            },
            {
                "name": "distance",
                "label": "Distance",
                "field": "distance",
                "sortable": True,
                "align": "left",
            },
        ]

        with ui.grid(columns=2).classes("w-full"):

            ui.upload(
                label="Upload GPX", on_upload=lambda e: self.add_track_from_upload(e)
            ).props("accept=.gpx").classes("w-full")
            with ui.card():
                self.track_table = ui.table(
                    columns=columns,
                    rows=[],
                    row_key="name",
                    selection="single",
                    on_select=self.select_table_callback,
                )
                self.track_table.classes("w-full")
            with ui.card():
                self.leaflet_map = ui.leaflet()
                self.leaflet_map.clear_layers()
                self.leaflet_map.style("height:100%")
            with ui.card():

                ui.label("Weather")
                with ui.grid(columns=5).classes("w-full"):
                    ui.number(label="Temperature (°C)").bind_value(
                        self.weather_model, "temperature_c"
                    )
                    ui.number(label="Wind (km/h)").bind_value(
                        self.weather_model, "wind_speed_kmh"
                    )
                    ui.select(
                        label="Wind Direction",
                        value=WindDirection.N.value,
                        options=WindDirection._value2member_map_,
                        on_change=lambda e: self.weather_model.set_angle(e.value),
                    )

                    ui.button("Apply to all")
                    with ui.dropdown_button('Apply to segment:', auto_close=True):
                        ui.item('Item 1', on_click=lambda: ui.notify('You clicked item 1'))
                        ui.item('Item 2', on_click=lambda: ui.notify('You clicked item 2'))

                ui.label("Resistance model")
                with ui.grid(columns=4).classes("w-full"):
                    ui.number(label="Total weight (kg)", step=0.1).bind_value(
                        self.resistance_model, "total_weight_kg"
                    )
                    ui.number(label="CdA (m²)", step=0.01).bind_value(
                        self.resistance_model, "cda_m2"
                    )
                    ui.number(label="Crr", step=0.0005).bind_value(
                        self.resistance_model, "crr"
                    )
                    ui.number(label="Drivetrain efficiency", step=0.01).bind_value(
                        self.resistance_model, "drivetrain_efficiency"
                    )

                ui.label("Rider model")
                with ui.grid(columns=3).classes("w-full"):
                    ui.number(label="Critical Power (W)").bind_value(
                        self.rider_model, "critical_power_w"
                    )
                    ui.number(label="Anaerobic Reserve (kJ)").bind_value(
                        self.rider_model, "anaerobic_reserve_j"
                    )
                    ui.button(
                        "Optimize power",
                        on_click=self.optimize
                    )

        with ui.grid(columns=1).classes("w-full"):
            self.plot_card = ui.card().tight()
            self.plot_card.classes("center")

            self.plot = ui.echart(
                {
                    "xAxis": {"type": "value"},
                    "yAxis": {"type": "value"},
                    "legend": {"textStyle": {"color": "gray"}},
                    "series": [],
                }
            ).classes("w-full h-64")
            self.plot.set_visibility(False)
            self.plot.move(self.plot_card)

            self.spinner = ui.spinner(type="hourglass", size="5em")
            self.spinner.set_visibility(False)
            self.spinner.move(self.plot_card)

    def select_table_callback(self, e) -> None:
        if len(e.selection) == 0:
            self.selected_track = None
            self.clear_map()
            return
        self.selected_track = self.select_track(e.selection[0]["name"])
        self.show_track_map(self.selected_track)

    def select_track(self, name: str) -> Track:
        for t in self.loaded_tracks:
            if t.name == name:
                return t
        raise ValueError(name)

    def clear_map(self):
        self.leaflet_map.clear_layers()
        self.leaflet_map.tile_layer(
            url_template=r"https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png"
        )
        self.plot.set_visibility(False)
        for p in self.data_plots:
            p.delete()
        self.data_plots = []

    def show_track_map(self, track: Track):
        self.clear_map()
        self.leaflet_map.set_center(tuple(track.points[:, :2].mean(axis=0)))
        self.leaflet_map.generic_layer(
            name="polyline", args=[track.points[:, :2].tolist()]
        )

        distance_km = track.distance_m / 1000
        series = {
            "type": "line",
            "data": [(distance_km[i], track.elevation_m[i]) for i in range(len(track))],
            "showSymbol": False,
            "areaStyle": {},
        }
        self.plot.options["series"] = [series]
        self.plot.options["xAxis"]["min"] = 0
        self.plot.options["xAxis"]["max"] = np.round(distance_km[-1], 1)
        self.plot.options["yAxis"]["min"] = track.elevation_m.min()
        self.plot.options["yAxis"]["max"] = track.elevation_m.max()
        self.plot.update()
        self.plot.set_visibility(True)

        self.add_segments(track)

    async def optimize(self) -> None:
        self.spinner.set_visibility(True)
        opt = PowerOptimizer(self.selected_track, self.segments, self.resistance_model, self.rider_model, True)
        track = await run.cpu_bound(opt.compute)
        self.spinner.set_visibility(False)
        self.plot_data(track)

    def add_segments(self, track: Track) -> None:
        segments = track.get_segments(self.max_error_m, self.weather_model)

        self.segments = segments

        pieces = []
        for segment in segments:
            color = get_color_from_grade(segment.grade_pct)
            print(segment.start_m)
            pieces.append(
                {
                    "gt": segment.start_m / 1000,
                    "lt": segment.end_m / 1000,
                    "color": f"rgba({int(255*color[0])}, {int(255*color[1])}, {int(255*color[2])}, 0.4)",
                }
            )

        visualMap = (
            {
                "type": "piecewise",
                "show": False,
                "dimension": 0,
                "seriesIndex": 0,
                "pieces": pieces,
            },
        )
        self.plot.options["visualMap"] = visualMap
        self.plot.update()

    def plot_data(self, track: Track) -> None:
        plots: dict[str, list[np.ndarray]] = {}
        distance_km = (track.distance_m / 1000).tolist()
        for k, v in track.data.items():
            s = k.split("/")
            series_name = s[0]
            if len(s) > 1:
                variant = s[1]
            if series_name in plots:
                plots[series_name].append((v, variant))
            else:
                plots[k] = [(v,"")]
        for k, plot in plots.items():

            new_plot = ui.echart(
                    {
                        "xAxis": {"type": "value"},
                        "yAxis": {"type": "value"},
                        "legend": {"textStyle": {"color": "gray"}},
                        "series": [],
                    }
                ).classes("w-full h-64")
            def plot_data(v):
                v = v.tolist()
                if len(v) == len(distance_km):
                    return [(distance_km[i], v[i]) for i in range(len(v))]
                elif len(v) == len(distance_km) - 1:
                    res = []
                    for i in range(len(v)):
                        res.append((distance_km[i], v[i]))
                        res.append((distance_km[i+1], v[i]))
                    return res
            series = [
                {
                    "type": "line",
                    "data": plot_data(data),
                    "showSymbol": False,
                    "areaStyle": {},
                }
                for data, series_name in plot
            ]
            new_plot.options["series"] = series
            new_plot.options["xAxis"]["min"] = 0
            new_plot.options["xAxis"]["max"] = np.round(distance_km[-1], 1)
            new_plot.options["yAxis"]["min"] = 0
            new_plot.options["yAxis"]["max"] = np.ceil(np.asarray([data.max() for data, _ in plot]).max())
            new_plot.update()
            new_plot.move(self.plot_card)
            self.data_plots.append(new_plot)

    def add_new_track(self, track: Track):
        self.track_table.add_rows(
            {"name": track.name, "distance": int(track.total_distance_m / 100) / 10}
        )
        self.track_table.update()

    def add_track_from_upload(self, e: events.UploadEventArguments) -> None:
        gpx_file = gpxpy.parse(e.content)
        points = [
            p
            for track in gpx_file.tracks
            for segment in track.segments
            for p in segment.points
        ]
        track = Track(
            name=Path(e.name).stem,
            points=np.stack([[p.latitude, p.longitude, p.elevation] for p in points]),
        )
        ui.notify(
            f"Uploaded {track.name}. {len(track)} GPS points. Total distance: {int(track.total_distance_m / 100) / 10} km"
        )
        self.loaded_tracks.append(track)
        self.add_new_track(track)


app = App()


ui.run()
