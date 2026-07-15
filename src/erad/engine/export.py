"""Export simulation results from DuckDB to various formats."""

from __future__ import annotations

from pathlib import Path

import duckdb
from loguru import logger


def export_to_parquet(con: duckdb.DuckDBPyConnection, path: str | Path):
    """Export asset_states table directly to Parquet (zero-copy, columnar)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT
                a.asset_name,
                a.asset_type_name AS asset_type,
                s.asset_id,
                s.timestamp,
                s.wind_speed_mph,
                s.flood_depth_m,
                s.flood_velocity_mps,
                s.fire_distance_km,
                s.pgv_cms,
                s.pga_fraction_g,
                s.soil_saturation,
                s.snow_water_equivalent_m,
                s.runoff_volume,
                s.groundwater_flow,
                s.survival_probability
            FROM asset_states s
            JOIN assets a ON s.asset_id = a.asset_id
            ORDER BY a.asset_name, s.timestamp
        )
        TO '{path}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    )
    logger.info(f"Exported results to Parquet: {path}")


def export_to_sqlite(con: duckdb.DuckDBPyConnection, path: str | Path):
    """Export asset_states to SQLite database (compatible with legacy format)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Remove existing file to avoid conflicts
    if path.exists():
        path.unlink()

    con.execute(
        f"""
        ATTACH '{path}' AS export_db (TYPE SQLITE);
        CREATE TABLE export_db.assetstatetable AS
        SELECT
            ROW_NUMBER() OVER () AS id,
            a.asset_name,
            a.asset_type_name AS asset_type,
            CAST(s.asset_id AS VARCHAR) AS distribution_asset,
            CAST(s.timestamp AS VARCHAR) AS timestamp,
            s.survival_probability,
            s.wind_speed_mph AS "wind_speed__miles_per_hour",
            s.flood_depth_m * 3.28084 AS "flood_depth__feet",
            s.flood_velocity_mps * 3.28084 AS "flood_velocity__feet_per_second",
            s.fire_distance_km AS "fire_distance__kilometer",
            s.pga_fraction_g AS "peak_ground_acceleration__fraction_of_g",
            s.pgv_cms AS "peak_ground_velocity__centimeters_per_second"
        FROM asset_states s
        JOIN assets a ON s.asset_id = a.asset_id
        ORDER BY a.asset_name, s.timestamp;
        DETACH export_db;
    """
    )
    logger.info(f"Exported results to SQLite: {path}")


def export_to_csv(con: duckdb.DuckDBPyConnection, path: str | Path):
    """Export asset_states to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"""
        COPY (
            SELECT
                a.asset_name,
                a.asset_type_name AS asset_type,
                s.asset_id,
                s.timestamp,
                s.wind_speed_mph,
                s.flood_depth_m,
                s.flood_velocity_mps,
                s.fire_distance_km,
                s.pgv_cms,
                s.pga_fraction_g,
                s.survival_probability
            FROM asset_states s
            JOIN assets a ON s.asset_id = a.asset_id
            ORDER BY a.asset_name, s.timestamp
        )
        TO '{path}' (FORMAT CSV, HEADER)
    """
    )
    logger.info(f"Exported results to CSV: {path}")


def export_to_arrow(con: duckdb.DuckDBPyConnection):
    """Export results as a PyArrow Table (zero-copy from DuckDB)."""
    return con.execute(
        """
        SELECT
            a.asset_name,
            a.asset_type_name AS asset_type,
            s.asset_id,
            s.timestamp,
            s.wind_speed_mph,
            s.flood_depth_m,
            s.flood_velocity_mps,
            s.fire_distance_km,
            s.pgv_cms,
            s.pga_fraction_g,
            s.soil_saturation,
            s.snow_water_equivalent_m,
            s.runoff_volume,
            s.groundwater_flow,
            s.survival_probability
        FROM asset_states s
        JOIN assets a ON s.asset_id = a.asset_id
        ORDER BY a.asset_name, s.timestamp
    """
    ).arrow()


def export_to_dataframe(con: duckdb.DuckDBPyConnection):
    """Export results as a pandas DataFrame."""
    return con.execute(
        """
        SELECT
            a.asset_name,
            a.asset_type_name AS asset_type,
            s.asset_id,
            s.timestamp,
            s.wind_speed_mph,
            s.flood_depth_m,
            s.flood_velocity_mps,
            s.fire_distance_km,
            s.pgv_cms,
            s.pga_fraction_g,
            s.soil_saturation,
            s.snow_water_equivalent_m,
            s.runoff_volume,
            s.groundwater_flow,
            s.survival_probability
        FROM asset_states s
        JOIN assets a ON s.asset_id = a.asset_id
        ORDER BY a.asset_name, s.timestamp
    """
    ).fetchdf()
