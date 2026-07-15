"""Load ERAD domain objects into DuckDB tables for vectorized computation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import duckdb
from loguru import logger

if TYPE_CHECKING:
    from erad.models.asset import Asset
    from erad.models.fragility_curve import HazardFragilityCurves
    from erad.systems.hazard_system import HazardSystem

import erad.models.hazard as hz


def load_assets(
    con: duckdb.DuckDBPyConnection, assets: list["Asset"], compute_elevation: bool = True
):
    """Bulk-load Asset objects into the DuckDB assets table."""
    con.execute(
        """
        CREATE OR REPLACE TABLE assets (
            asset_id VARCHAR,
            asset_name VARCHAR,
            asset_type INTEGER,
            asset_type_name VARCHAR,
            latitude DOUBLE,
            longitude DOUBLE,
            height_m DOUBLE,
            elevation_m DOUBLE
        )
    """
    )

    if not assets:
        return

    rows = []
    for asset in assets:
        elev = asset.elevation.to("meter").magnitude if compute_elevation else 0.0
        rows.append(
            (
                str(asset.distribution_asset),
                asset.name,
                int(asset.asset_type),
                asset.asset_type.name,
                asset.latitude,
                asset.longitude,
                asset.height.to("meter").magnitude,
                elev,
            )
        )

    con.executemany(
        "INSERT INTO assets VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        rows,
    )

    logger.info(f"Loaded {len(assets)} assets into DuckDB")


def load_hazards(con: duckdb.DuckDBPyConnection, hazard_system: "HazardSystem"):
    """Load hazard models into DuckDB tables (one table per hazard type)."""

    # Earthquake models
    con.execute(
        """
        CREATE OR REPLACE TABLE earthquake_models (
            hazard_id INTEGER,
            timestamp TIMESTAMP,
            origin_lat DOUBLE,
            origin_lon DOUBLE,
            depth_km DOUBLE,
            magnitude DOUBLE
        )
    """
    )

    # Wind models
    con.execute(
        """
        CREATE OR REPLACE TABLE wind_models (
            hazard_id INTEGER,
            timestamp TIMESTAMP,
            center_lat DOUBLE,
            center_lon DOUBLE,
            max_wind_speed_kt DOUBLE,
            radius_max_wind_nm DOUBLE,
            radius_closest_isobar_nm DOUBLE
        )
    """
    )

    # Flood models - areas stored separately with a model_id foreign key
    con.execute(
        """
        CREATE OR REPLACE TABLE flood_models (
            hazard_id INTEGER,
            timestamp TIMESTAMP,
            area_idx INTEGER,
            polygon_wkt VARCHAR,
            water_velocity_mps DOUBLE,
            water_elevation_m DOUBLE,
            soil_saturation DOUBLE,
            snow_water_equivalent_m DOUBLE,
            runoff_volume DOUBLE,
            groundwater_flow DOUBLE
        )
    """
    )

    # Fire models - areas stored separately
    con.execute(
        """
        CREATE OR REPLACE TABLE fire_models (
            hazard_id INTEGER,
            timestamp TIMESTAMP,
            area_idx INTEGER,
            polygon_wkt VARCHAR
        )
    """
    )

    # Load earthquake models
    eq_data = []
    for i, model in enumerate(hazard_system.get_components(hz.EarthQuakeModel)):
        eq_data.append(
            (
                i,
                model.timestamp,
                model.origin.y,
                model.origin.x,
                model.depth.to("km").magnitude,
                model.magnitude,
            )
        )
    if eq_data:
        con.executemany("INSERT INTO earthquake_models VALUES (?, ?, ?, ?, ?, ?)", eq_data)
        logger.info(f"Loaded {len(eq_data)} earthquake models")

    # Load wind models
    wind_data = []
    for i, model in enumerate(hazard_system.get_components(hz.WindModel)):
        wind_data.append(
            (
                i,
                model.timestamp,
                model.center.y,
                model.center.x,
                model.max_wind_speed.to("knot").magnitude,
                model.radius_of_max_wind.to("nautical_mile").magnitude,
                model.radius_of_closest_isobar.to("nautical_mile").magnitude,
            )
        )
    if wind_data:
        con.executemany("INSERT INTO wind_models VALUES (?, ?, ?, ?, ?, ?, ?)", wind_data)
        logger.info(f"Loaded {len(wind_data)} wind models")

    # Load flood models
    flood_data = []
    for i, model in enumerate(hazard_system.get_components(hz.FloodModel)):
        for j, area in enumerate(model.affected_areas):
            flood_data.append(
                (
                    i,
                    model.timestamp,
                    j,
                    area.affected_area.wkt,
                    area.water_velocity.to("meter/second").magnitude,
                    area.water_elevation.to("meter").magnitude,
                    area.soil_saturation.magnitude if area.soil_saturation else None,
                    area.snow_water_equivalent.to("meter").magnitude
                    if area.snow_water_equivalent
                    else None,
                    area.runoff_volume.magnitude if area.runoff_volume else None,
                    area.groundwater_flow.magnitude if area.groundwater_flow else None,
                )
            )
    if flood_data:
        con.executemany(
            "INSERT INTO flood_models VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", flood_data
        )
        logger.info(f"Loaded {len(flood_data)} flood model areas")

    # Load fire models
    fire_data = []
    for i, model in enumerate(hazard_system.get_components(hz.FireModel)):
        for j, area in enumerate(model.affected_areas):
            fire_data.append(
                (
                    i,
                    model.timestamp,
                    j,
                    area.affected_area.wkt,
                )
            )
    if fire_data:
        con.executemany("INSERT INTO fire_models VALUES (?, ?, ?, ?)", fire_data)
        logger.info(f"Loaded {len(fire_data)} fire model areas")


def load_fragility_curves(
    con: duckdb.DuckDBPyConnection, frag_curves: list["HazardFragilityCurves"]
):
    """Load fragility curve parameters into DuckDB for vectorized CDF evaluation."""
    con.execute(
        """
        CREATE OR REPLACE TABLE fragility_curves (
            asset_type INTEGER,
            hazard_parameter VARCHAR,
            distribution VARCHAR,
            param_1 DOUBLE,
            param_2 DOUBLE,
            param_3 DOUBLE,
            units VARCHAR
        )
    """
    )

    curve_data = []
    for hazard_frag in frag_curves:
        param_name = hazard_frag.asset_state_param
        for curve in hazard_frag.curves:
            prob_fn = curve.prob_function
            # Extract params as raw magnitudes (matching ProbabilityFunctionBuilder behavior)
            from infrasys import BaseQuantity

            params = [
                p.magnitude if isinstance(p, BaseQuantity) else p for p in prob_fn.parameters
            ]
            # Get units from first BaseQuantity parameter
            units = ""
            for p in prob_fn.parameters:
                if isinstance(p, BaseQuantity):
                    units = str(p.units)
                    break

            # Pad params to 3 (norm has 2: loc, scale; lognorm has 3: s, loc, scale)
            while len(params) < 3:
                params.append(0.0)

            curve_data.append(
                (
                    int(curve.asset_type),
                    param_name,
                    prob_fn.distribution,
                    params[0],
                    params[1],
                    params[2],
                    units,
                )
            )

    if curve_data:
        con.executemany("INSERT INTO fragility_curves VALUES (?, ?, ?, ?, ?, ?, ?)", curve_data)
        logger.info(f"Loaded {len(curve_data)} fragility curve definitions")
