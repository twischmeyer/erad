"""Vectorized DuckDB UDFs for hazard calculations and fragility curve evaluation."""


import duckdb
import numpy as np
from scipy import stats


def register_udfs(con: duckdb.DuckDBPyConnection):
    """Register all custom UDFs with the DuckDB connection."""

    def haversine_km(lat1, lon1, lat2, lon2):
        """Vectorized haversine distance in km. Operates on numpy arrays."""
        lat1_r = np.radians(lat1)
        lat2_r = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_r) * np.cos(lat2_r) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
        return 6371.0 * c

    def hypocentral_distance_km(epicenter_dist_km, depth_km):
        """Compute hypocentral distance from epicenter distance and depth."""
        return np.sqrt(epicenter_dist_km**2 + depth_km**2)

    def mmi_from_magnitude(magnitude, hypocentral_dist_km):
        """Modified Mercalli Intensity from Atkinson & Wald (2007)."""
        return 3.31 + 1.28 * magnitude - 1.42 * np.log10(hypocentral_dist_km)

    def pgv_from_mmi(mmi):
        """Peak ground velocity (cm/s) from MMI."""
        log_pgv = (mmi - 3.78) / 1.47
        return np.power(10.0, log_pgv)

    def pga_from_mmi(mmi):
        """Peak ground acceleration (fraction of g) from MMI."""
        log_pga = (mmi - 1.78) / 3.70
        return np.power(10.0, log_pga)

    def holland_wind_speed_kt(distance_nm, v_max_kt, r_max_wind_nm, r_closest_isobar_nm):
        """Holland (1980) wind model. Returns wind speed in knots."""
        distance_nm = np.asarray(distance_nm, dtype=np.float64)
        v_max_kt = np.asarray(v_max_kt, dtype=np.float64)
        r_max_wind_nm = np.asarray(r_max_wind_nm, dtype=np.float64)
        r_closest_isobar_nm = np.asarray(r_closest_isobar_nm, dtype=np.float64)

        k = 1.14
        b = 10.0

        a = np.log(b) / (r_closest_isobar_nm - r_max_wind_nm)
        m = (1.0 / r_max_wind_nm) * np.log(k / (k - 1.0))

        result = np.zeros_like(distance_nm, dtype=np.float64)

        # Region 1: r < r_max_wind
        mask1 = (distance_nm >= 0) & (distance_nm < r_max_wind_nm)
        result[mask1] = k * v_max_kt * (1.0 - np.exp(-m * distance_nm[mask1]))

        # Region 2: r_max_wind <= r < r_closest_isobar
        mask2 = (distance_nm >= r_max_wind_nm) & (distance_nm < r_closest_isobar_nm)
        result[mask2] = v_max_kt * np.exp(-a * (distance_nm[mask2] - r_max_wind_nm))

        # Region 3: r >= r_closest_isobar → 0
        return result

    def normal_cdf(x, loc, scale):
        """Vectorized normal CDF."""
        return stats.norm.cdf(x, loc, scale)

    def lognormal_cdf(x, s, loc, scale):
        """Vectorized lognormal CDF. Matches scipy.stats.lognorm.cdf(x, s, loc, scale)."""
        return stats.lognorm.cdf(x, s, loc, scale)

    def km_to_nautical_miles(km):
        """Convert kilometers to nautical miles."""
        return km / 1.852

    def knots_to_mph(knots):
        """Convert knots to miles per hour."""
        return knots * 1.15078

    # Register all UDFs using string type names (compatible across DuckDB versions)
    con.create_function(
        "haversine_km",
        haversine_km,
        ["DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE"],
        "DOUBLE",
        type="native",
    )

    con.create_function(
        "hypocentral_distance_km",
        hypocentral_distance_km,
        ["DOUBLE", "DOUBLE"],
        "DOUBLE",
        type="native",
    )

    con.create_function(
        "mmi_from_magnitude",
        mmi_from_magnitude,
        ["DOUBLE", "DOUBLE"],
        "DOUBLE",
        type="native",
    )

    con.create_function(
        "pgv_from_mmi",
        pgv_from_mmi,
        ["DOUBLE"],
        "DOUBLE",
        type="native",
    )

    con.create_function(
        "pga_from_mmi",
        pga_from_mmi,
        ["DOUBLE"],
        "DOUBLE",
        type="native",
    )

    con.create_function(
        "holland_wind_speed_kt",
        holland_wind_speed_kt,
        ["DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE"],
        "DOUBLE",
        type="native",
    )

    con.create_function(
        "normal_cdf",
        normal_cdf,
        ["DOUBLE", "DOUBLE", "DOUBLE"],
        "DOUBLE",
        type="native",
    )

    con.create_function(
        "lognormal_cdf",
        lognormal_cdf,
        ["DOUBLE", "DOUBLE", "DOUBLE", "DOUBLE"],
        "DOUBLE",
        type="native",
    )

    con.create_function(
        "km_to_nautical_miles",
        km_to_nautical_miles,
        ["DOUBLE"],
        "DOUBLE",
        type="native",
    )

    con.create_function(
        "knots_to_mph",
        knots_to_mph,
        ["DOUBLE"],
        "DOUBLE",
        type="native",
    )
