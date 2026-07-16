"""Core DuckDB-powered simulation engine for ERAD."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import duckdb
from loguru import logger

from erad.engine.udfs import register_udfs
from erad.engine import loaders, hazard_calcs, fragility, scenario, export

if TYPE_CHECKING:
    from erad.models.asset import Asset
    from erad.models.fragility_curve import HazardFragilityCurves
    from erad.systems.asset_system import AssetSystem
    from erad.systems.hazard_system import HazardSystem


class SimulationEngine:
    """DuckDB-based vectorized simulation engine.

    Replaces the Python loop-based approach with columnar computation.
    All assets and hazards are loaded into DuckDB tables, and calculations
    are expressed as vectorized SQL operations with custom UDFs.

    Scales to millions of assets with automatic parallelism.
    """

    def __init__(self, db_path: str | None = None):
        """Initialize the engine.

        Args:
            db_path: Optional path for persistent DuckDB database.
                     If None, uses in-memory database.
        """
        if db_path:
            self._con = duckdb.connect(db_path)
        else:
            self._con = duckdb.connect(":memory:")

        # Configure for performance
        import os

        n_cores = os.cpu_count() or 4
        self._con.execute(f"SET threads TO {n_cores}")

        register_udfs(self._con)
        self._create_results_table()
        self._loaded = False

    @property
    def connection(self) -> duckdb.DuckDBPyConnection:
        """Access the underlying DuckDB connection for custom queries."""
        return self._con

    def _create_results_table(self):
        """Create the asset_states results table."""
        self._con.execute(
            """
            CREATE TABLE IF NOT EXISTS asset_states (
                asset_id VARCHAR,
                timestamp TIMESTAMP,
                wind_speed_mph DOUBLE,
                flood_depth_m DOUBLE,
                flood_velocity_mps DOUBLE,
                fire_distance_km DOUBLE,
                pgv_cms DOUBLE,
                pga_fraction_g DOUBLE,
                soil_saturation DOUBLE,
                snow_water_equivalent_m DOUBLE,
                runoff_volume DOUBLE,
                groundwater_flow DOUBLE,
                wind_speed_surv DOUBLE DEFAULT 1.0,
                flood_depth_surv DOUBLE DEFAULT 1.0,
                flood_velocity_surv DOUBLE DEFAULT 1.0,
                fire_distance_surv DOUBLE DEFAULT 1.0,
                pgv_surv DOUBLE DEFAULT 1.0,
                pga_surv DOUBLE DEFAULT 1.0,
                soil_saturation_surv DOUBLE DEFAULT 1.0,
                snow_water_equivalent_surv DOUBLE DEFAULT 1.0,
                runoff_volume_surv DOUBLE DEFAULT 1.0,
                groundwater_flow_surv DOUBLE DEFAULT 1.0,
                survival_probability DOUBLE DEFAULT 1.0,
                PRIMARY KEY (asset_id, timestamp)
            )
        """
        )

    def load_assets(self, assets: list["Asset"], compute_elevation: bool = True):
        """Load asset data into DuckDB."""
        loaders.load_assets(self._con, assets, compute_elevation=compute_elevation)

    def load_assets_from_dataframe(self, df):
        """Load assets directly from a pandas/polars DataFrame (bypasses Pydantic objects).

        Required columns: asset_id, asset_name, asset_type, asset_type_name,
                         latitude, longitude, height_m, elevation_m

        This is the recommended path for 100K+ assets.
        """
        self._con.execute(
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
        self._con.execute("INSERT INTO assets SELECT * FROM df")
        n = self._con.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        logger.info(f"Loaded {n} assets from DataFrame into DuckDB")

    def load_assets_from_arrays(
        self,
        asset_ids,
        asset_names,
        asset_types,
        asset_type_names: list[str],
        latitudes,
        longitudes,
        heights_m,
        elevations_m=None,
    ):
        """Load assets from numpy arrays (zero Pydantic overhead).

        This is the fastest path for million+ asset scale.
        """
        import numpy as np
        import pyarrow as pa

        n = len(latitudes)
        if elevations_m is None:
            elevations_m = np.zeros(n, dtype=np.float64)

        self._con.execute(
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

        arrow_table = pa.table(  # noqa: F841 — referenced by name in DuckDB SQL
            {
                "asset_id": pa.array(asset_ids, type=pa.string()),
                "asset_name": pa.array(asset_names, type=pa.string()),
                "asset_type": pa.array(asset_types, type=pa.int32()),
                "asset_type_name": pa.array(asset_type_names, type=pa.string()),
                "latitude": pa.array(latitudes, type=pa.float64()),
                "longitude": pa.array(longitudes, type=pa.float64()),
                "height_m": pa.array(heights_m, type=pa.float64()),
                "elevation_m": pa.array(elevations_m, type=pa.float64()),
            }
        )

        self._con.execute("INSERT INTO assets SELECT * FROM arrow_table")
        logger.info(f"Loaded {n} assets from arrays into DuckDB")

    def load_hazards(self, hazard_system: "HazardSystem"):
        """Load all hazard models into DuckDB tables."""
        loaders.load_hazards(self._con, hazard_system)

    def load_fragility_curves(self, frag_curves: list["HazardFragilityCurves"]):
        """Load fragility curve parameters into DuckDB."""
        loaders.load_fragility_curves(self._con, frag_curves)
        self._loaded = True

    def compute_hazard_vectors(self):
        """Run all hazard vector computations (vectorized)."""
        if not self._loaded:
            raise RuntimeError("Must load assets, hazards, and fragility curves before computing")

        logger.info("Computing hazard vectors (DuckDB engine)...")
        hazard_calcs.compute_earthquake_vectors(self._con)
        hazard_calcs.compute_wind_vectors(self._con)
        hazard_calcs.compute_flood_vectors(self._con)
        hazard_calcs.compute_fire_vectors(self._con)
        logger.info("Hazard vector computation complete")

    def compute_survival_probabilities(self):
        """Evaluate fragility curves and compute survival probabilities (vectorized)."""
        logger.info("Computing survival probabilities (DuckDB engine)...")
        fragility.compute_survival_probabilities(self._con)

    def run(self):
        """Run the full simulation: hazard vectors → survival probabilities."""
        self.compute_hazard_vectors()
        self.compute_survival_probabilities()

    def generate_scenarios(self, n_samples: int = 1, seed: int = 0) -> list[dict]:
        """Generate Monte Carlo damage scenarios.

        Args:
            n_samples: Number of Monte Carlo samples to generate.
            seed: Random seed for reproducibility.

        Returns:
            List of failure events (dicts with scenario_name, timestamp, asset_id, asset_name)
        """
        n_assets = self._con.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
        if n_assets > 50_000:
            return scenario.generate_scenarios_fast(self._con, n_samples, seed)
        return scenario.generate_scenarios(self._con, n_samples, seed)

    # --- Export methods ---

    def export_to_parquet(self, path: str | Path):
        """Export results to Parquet format (fastest, columnar)."""
        export.export_to_parquet(self._con, path)

    def export_to_sqlite(self, path: str | Path):
        """Export results to SQLite (compatible with legacy ERAD format)."""
        export.export_to_sqlite(self._con, path)

    def export_to_csv(self, path: str | Path):
        """Export results to CSV."""
        export.export_to_csv(self._con, path)

    def export_to_arrow(self):
        """Export results as PyArrow Table (zero-copy)."""
        return export.export_to_arrow(self._con)

    def export_to_dataframe(self):
        """Export results as pandas DataFrame."""
        return export.export_to_dataframe(self._con)

    # --- Query methods ---

    def get_asset_count(self) -> int:
        """Get number of loaded assets."""
        return self._con.execute("SELECT COUNT(*) FROM assets").fetchone()[0]

    def get_timestamp_count(self) -> int:
        """Get number of distinct timestamps in results."""
        return self._con.execute("SELECT COUNT(DISTINCT timestamp) FROM asset_states").fetchone()[
            0
        ]

    def get_failed_assets(self, threshold: float = 0.5):
        """Get assets with survival probability below threshold at any timestamp."""
        return self._con.execute(
            """
            SELECT a.asset_name, a.asset_type_name, s.timestamp, s.survival_probability
            FROM asset_states s
            JOIN assets a ON s.asset_id = a.asset_id
            WHERE s.survival_probability < ?
            ORDER BY s.survival_probability ASC
        """,
            [threshold],
        ).fetchdf()

    def query(self, sql: str):
        """Run an arbitrary SQL query against the simulation data."""
        return self._con.execute(sql)

    # --- Conversion utilities ---

    def load_from_asset_system(self, asset_system: "AssetSystem"):
        """Load an existing AssetSystem (with populated asset_state) into DuckDB.

        This allows converting Pydantic-based simulation results into DuckDB
        for fast querying, export to Parquet, or further analysis.
        """
        from erad.models.asset import Asset

        assets = list(asset_system.get_components(Asset))
        loaders.load_assets(self._con, assets, compute_elevation=False)

        # Create asset_states table and populate from existing Asset.asset_state lists
        self._create_results_table()

        rows = []
        for asset in assets:
            asset_id = str(asset.distribution_asset)
            for state in asset.asset_state:
                rows.append(
                    (
                        asset_id,
                        state.timestamp,
                        state.wind_speed.speed.to("miles/hour").magnitude
                        if state.wind_speed
                        else None,
                        state.flood_depth.distance.to("meter").magnitude
                        if state.flood_depth
                        else None,
                        state.flood_velocity.speed.to("meter/second").magnitude
                        if state.flood_velocity
                        else None,
                        state.fire_boundary_dist.distance.to("kilometer").magnitude
                        if state.fire_boundary_dist
                        else None,
                        state.peak_ground_velocity.speed.to("centimeter/second").magnitude
                        if state.peak_ground_velocity
                        else None,
                        state.peak_ground_acceleration.acceleration.to("meter/second**2").magnitude
                        / 9.80665
                        * 100
                        if state.peak_ground_acceleration
                        else None,
                        state.wind_speed.survival_probability if state.wind_speed else 1.0,
                        state.flood_depth.survival_probability if state.flood_depth else 1.0,
                        state.flood_velocity.survival_probability if state.flood_velocity else 1.0,
                        state.fire_boundary_dist.survival_probability
                        if state.fire_boundary_dist
                        else 1.0,
                        state.peak_ground_velocity.survival_probability
                        if state.peak_ground_velocity
                        else 1.0,
                        state.peak_ground_acceleration.survival_probability
                        if state.peak_ground_acceleration
                        else 1.0,
                        state.survival_probability,
                    )
                )

        if rows:
            self._con.executemany(
                """INSERT INTO asset_states (
                    asset_id, timestamp, wind_speed_mph, flood_depth_m,
                    flood_velocity_mps, fire_distance_km, pgv_cms, pga_fraction_g,
                    wind_speed_surv, flood_depth_surv, flood_velocity_surv,
                    fire_distance_surv, pgv_surv, pga_surv, survival_probability
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )

        logger.info(
            f"Loaded {len(assets)} assets with {len(rows)} asset-states from AssetSystem into DuckDB"
        )

    def to_asset_states(self) -> list:
        """Convert DuckDB results to a list of (asset_id, AssetState) tuples.

        Returns list of tuples: (asset_id: str, state: AssetState)
        """
        from infrasys.quantities import Distance

        from erad.models.asset import AssetState
        from erad.models.probability import (
            AccelerationProbability,
            DistanceProbability,
            SpeedProbability,
        )
        from erad.quantities import Acceleration, Speed

        results = self._con.execute(
            """
            SELECT
                s.asset_id, s.timestamp,
                s.wind_speed_mph, s.flood_depth_m, s.flood_velocity_mps,
                s.fire_distance_km, s.pgv_cms, s.pga_fraction_g,
                s.wind_speed_surv, s.flood_depth_surv, s.flood_velocity_surv,
                s.fire_distance_surv, s.pgv_surv, s.pga_surv
            FROM asset_states s
            ORDER BY s.asset_id, s.timestamp
        """
        ).fetchall()

        output = []
        for row in results:
            (
                asset_id,
                timestamp,
                wind_mph,
                flood_depth_m,
                flood_vel_mps,
                fire_dist_km,
                pgv_cms,
                pga_g,
                wind_surv,
                flood_depth_surv,
                flood_vel_surv,
                fire_surv,
                pgv_surv,
                pga_surv,
            ) = row

            state = AssetState(timestamp=timestamp)

            if wind_mph is not None:
                state.wind_speed = SpeedProbability(
                    speed=Speed(wind_mph, "miles/hour"),
                    survival_probability=wind_surv,
                )
            if flood_depth_m is not None:
                state.flood_depth = DistanceProbability(
                    distance=Distance(flood_depth_m, "meter"),
                    survival_probability=flood_depth_surv,
                )
            if flood_vel_mps is not None:
                state.flood_velocity = SpeedProbability(
                    speed=Speed(flood_vel_mps, "meter/second"),
                    survival_probability=flood_vel_surv,
                )
            if fire_dist_km is not None:
                state.fire_boundary_dist = DistanceProbability(
                    distance=Distance(fire_dist_km, "kilometer"),
                    survival_probability=fire_surv,
                )
            if pgv_cms is not None:
                state.peak_ground_velocity = SpeedProbability(
                    speed=Speed(pgv_cms, "centimeter/second"),
                    survival_probability=pgv_surv,
                )
            if pga_g is not None:
                state.peak_ground_acceleration = AccelerationProbability(
                    acceleration=Acceleration(pga_g / 100.0 * 9.80665, "meter/second**2"),
                    survival_probability=pga_surv,
                )

            output.append((asset_id, state))

        return output

    @classmethod
    def from_asset_system(cls, asset_system: "AssetSystem") -> "SimulationEngine":
        """Create a SimulationEngine pre-loaded with data from an existing AssetSystem.

        Use this to convert Pydantic-based results into DuckDB for fast export/analysis.

        Example:
            engine = SimulationEngine.from_asset_system(asset_system)
            engine.export_to_parquet("results.parquet")
            df = engine.export_to_dataframe()
        """
        engine = cls()
        engine.load_from_asset_system(asset_system)
        return engine

    @classmethod
    def from_parquet(cls, path: str | Path) -> "SimulationEngine":
        """Create a SimulationEngine by loading results from a Parquet file.

        Example:
            engine = SimulationEngine.from_parquet("results.parquet")
            df = engine.get_failed_assets(threshold=0.3)
        """
        path = Path(path)
        engine = cls()
        engine._con.execute(
            f"""
            CREATE TABLE asset_states AS SELECT * FROM read_parquet('{path}')
        """
        )
        # Try to extract unique assets from the results
        engine._con.execute(
            """
            CREATE TABLE IF NOT EXISTS assets AS
            SELECT DISTINCT
                asset_id,
                asset_name,
                0 AS asset_type,
                asset_type AS asset_type_name,
                0.0 AS latitude,
                0.0 AS longitude,
                0.0 AS height_m,
                0.0 AS elevation_m
            FROM asset_states
        """
        )
        n = engine._con.execute("SELECT COUNT(*) FROM asset_states").fetchone()[0]
        logger.info(f"Loaded {n} records from Parquet: {path}")
        return engine

    def close(self):
        """Close the DuckDB connection."""
        self._con.close()

    def __del__(self):
        try:
            self._con.close()
        except (RuntimeError, AttributeError):
            # Connection may already be closed or object partially initialized
            return
