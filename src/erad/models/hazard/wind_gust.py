from datetime import datetime

from pydantic import field_serializer, field_validator
from infrasys.quantities import Distance
from shapely.geometry import Polygon, Point
import plotly.graph_objects as go
import geopandas as gpd

from erad.models.hazard.base_models import BaseDisasterModel
from erad.quantities import Speed
from infrasys import Component


class GustModelArea(Component):
    name: str = ""
    max_wind_area: Polygon
    max_wind_speed: Speed
    wind_aff_distance: Distance
    decay_rate: float = 0.5

    @field_validator("max_wind_area", mode="before")
    def deserialize_polygon(cls, value):
        if isinstance(value, dict) and value.get("type") == "Polygon":
            points = [Point(c) for c in value["coordinates"]]
            return Polygon(points)
        return value

    @field_serializer("max_wind_area")
    def serialize_polygon(self, poly: Polygon, _info):
        return {"type": "Polygon", "coordinates": list(poly.exterior.coords)}

    @classmethod
    def example(cls) -> "GustModelArea":
        return GustModelArea(
            name="Gust Ex",
            max_wind_area=Polygon(
                [
                    (-97.286, 28.297),
                    (-97.280, 28.305),
                    (-97.267, 28.287),
                    (-97.275, 28.282),
                ]
            ),
            max_wind_speed=Speed(120, "miles/hour"),
            wind_aff_distance=Distance(5, "miles"),
            decay_rate=0.5,
        )


class WindGustModel(BaseDisasterModel):
    timestamp: datetime
    max_wind_areas: list[GustModelArea]

    @classmethod
    def example(cls) -> "WindGustModel":
        return WindGustModel(
            name="tstorm_ex",
            timestamp=datetime.now(),
            max_wind_areas=[GustModelArea.example()],
        )

    def plot(
        self,
        time_index: int = 0,
        figure: go.Figure = go.Figure(),
        map_obj: type[go.Scattergeo | go.Scattermap] = go.Scattermap,
    ) -> int:
        for area in self.max_wind_areas:
            lon_max, lat_max = area.max_wind_area.exterior.xy  # returns x and y sequences

            gdf = gpd.GeoDataFrame(geometry=[area.max_wind_area], crs="EPSG:4326")
            max_area = gdf.to_crs(gdf.estimate_utm_crs())
            d_reduced = area.wind_aff_distance.to(
                "meter"
            )  # Convert to meters for buffer operation
            max_area["geometry"] = max_area.buffer(
                d_reduced.magnitude, resolution=1, join_style="mitre"
            )
            buffer_area = max_area.to_crs("EPSG:4326")
            wind_aff_area = buffer_area.geometry.iloc[0]

            lon_aff, lat_aff = wind_aff_area.exterior.xy  # returns x and y sequences
            figure.add_trace(
                map_obj(
                    lon=lon_aff.tolist(),
                    lat=lat_aff.tolist(),
                    mode="markers+lines+text",
                    fill="toself",
                    marker={
                        "size": 0,
                        "color": [area.max_wind_speed.magnitude / 2],
                    },
                    hovertemplate=f"""
                        <br> <b>Surrounding wind-affected distance:</b> {area.wind_aff_distance}
                        {"<br> <b>Decay rate:</b> " + str(area.decay_rate) if area.decay_rate is not None else ""}
                    """,
                    visible=(time_index == 0),
                    name=f"{str(area.name) + ' - Reduced Wind' if area.name else 'Reduced Wind'}",
                )
            )
            figure.add_trace(
                map_obj(
                    lon=lon_max.tolist(),
                    lat=lat_max.tolist(),
                    mode="markers+lines+text",
                    fill="toself",
                    marker={
                        "size": 10,
                        "color": [area.max_wind_speed.magnitude],
                    },
                    hovertemplate=f"""
                        <br> <b>Max wind speed:</b> {area.max_wind_speed}
                        """,
                    visible=(time_index == 0),
                    name=f"{str(area.name) + ' - Max Wind' if area.name else 'Max Wind'}",
                )
            )

        return 2
