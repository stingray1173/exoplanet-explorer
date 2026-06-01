import pandas as pd
import numpy as np
import requests
import io
import os

# 1. The Master ADQL Query
# We pull exactly the parameters you requested, plus st_rad for Stefan-Boltzmann calculations.
QUERY = """
SELECT 
    pl_name, hostname, sy_snum, sy_pnum, 
    discoverymethod, disc_year, disc_facility, disc_locale, pl_controv_flag,
    pl_orbper, pl_orbsmax, pl_rade, pl_masse, pl_dens, pl_orbeccen, 
    pl_insol, pl_eqt, 
    st_spectype, st_teff, st_mass, st_met, st_metratio, st_logg, st_lum, st_rad,
    ra, dec, sy_dist, sy_gaiamag
FROM pscomppars
"""

URL = "https://exoplanetarchive.ipac.caltech.edu/TAP/sync"


def fetch_and_process_data():
    print("Initiating API request to NASA Exoplanet Archive...")
    response = requests.get(URL, params={"query": QUERY, "format": "csv"})

    if response.status_code != 200:
        print(f"FAILED. HTTP Status Code: {response.status_code}")
        print(response.text)
        return

    # Load raw data into Pandas
    df = pd.read_csv(io.StringIO(response.text))
    print(
        f"Acquired {len(df)} planetary records. Commencing rigorous cleaning & math..."
    )

    # 2. Orbital Sanity Checks
    # Fill missing eccentricity with 0 (circular orbit).
    # Clip max eccentricity at 0.99 to prevent division by zero in insolation math.
    df["pl_orbeccen"] = df["pl_orbeccen"].fillna(0).clip(upper=0.99)

    # 3. Deriving Stellar Luminosity (Linear Scale)
    # NASA's st_lum is log10(L_solar). We convert to linear L_solar.
    df["st_lum_lin"] = 10 ** df["st_lum"]

    # Stefan-Boltzmann Rescue: If st_lum is missing, calculate it using Radius & Temperature
    missing_lum = df["st_lum_lin"].isna()
    can_calc_lum = df["st_rad"].notna() & df["st_teff"].notna()

    # L = R^2 * (T/5778)^4
    df.loc[missing_lum & can_calc_lum, "st_lum_lin"] = (
        df.loc[missing_lum & can_calc_lum, "st_rad"] ** 2
    ) * ((df.loc[missing_lum & can_calc_lum, "st_teff"] / 5778) ** 4)

    # 4. The Provenance Flag & Insolation Math
    # 0 = Raw NASA Data | 1 = Our Derived Math
    df["is_derived_insol"] = 0

    missing_insol = df["pl_insol"].isna()
    can_calc_insol = (
        df["st_lum_lin"].notna() & df["pl_orbsmax"].notna() & (df["pl_orbsmax"] > 0)
    )
    target_rows = missing_insol & can_calc_insol

    # Time-Averaged Insolation: S = L / (a^2 * sqrt(1 - e^2))
    df.loc[target_rows, "pl_insol"] = df.loc[target_rows, "st_lum_lin"] / (
        (df.loc[target_rows, "pl_orbsmax"] ** 2)
        * np.sqrt(1 - (df.loc[target_rows, "pl_orbeccen"] ** 2))
    )

    # Flag the rows we just modified so journalists/users know it's derived
    df.loc[target_rows, "is_derived_insol"] = 1

    # Drop the temporary linear luminosity column to save space (st_lum remains)
    df = df.drop(columns=["st_lum_lin"])

    # 5. Spatial Engineering (3D Coordinates)
    # If sy_dist is missing, X,Y,Z become NaN automatically (Three.js will ignore them for 3D map,
    # but the Celestial Map can still use the raw ra/dec).
    ra_rad = np.radians(df["ra"])
    dec_rad = np.radians(df["dec"])
    dist = df["sy_dist"]

    df["x"] = dist * np.cos(dec_rad) * np.cos(ra_rad)
    df["y"] = dist * np.cos(dec_rad) * np.sin(ra_rad)
    df["z"] = dist * np.sin(dec_rad)

    # 6. Payload Optimization (Precision Trimming)
    # Decimals beyond 3 places offer no visual/narrative value but bloat file size.
    float_cols = [
        "pl_orbper",
        "pl_orbsmax",
        "pl_rade",
        "pl_masse",
        "pl_dens",
        "pl_orbeccen",
        "pl_insol",
        "pl_eqt",
        "st_teff",
        "st_mass",
        "st_met",
        "st_logg",
        "st_lum",
        "st_rad",
        "ra",
        "dec",
        "sy_dist",
        "sy_gaiamag",
        "x",
        "y",
        "z",
    ]

    # Round and convert NaNs to None so they become clean JSON 'nulls'
    df[float_cols] = df[float_cols].round(3).replace({np.nan: None})

    # 7. Export the Static JSON
    # Ensure it writes precisely to your 'public' folder
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.abspath(
        os.path.join(script_dir, "..", "public", "exoplanets.json")
    )

    df.to_json(output_path, orient="records")

    print(f"SUCCESS: Pipeline complete. Processed {len(df)} systems.")
    print(f"Data compressed and exported to: {output_path}")


if __name__ == "__main__":
    fetch_and_process_data()
