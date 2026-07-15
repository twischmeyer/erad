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

    def close(self):
        """Close the DuckDB connection."""
        self._con.close()

    def __del__(self):
        try:
            self._con.close()
        except (RuntimeError, AttributeError):
            # Connection may already be closed or object partially initialized
            return
