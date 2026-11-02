"""Build the analysis dataset from Ghana MIS 2019 and Ghana DHS 2022.

One row per child aged 6-59 months with a malaria RDT result, with child, household,
cluster-environment and pre-interview weather predictors.

Usage: python build_dataset.py <data_raw folder> <output csv>
"""
import json
import sys
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from rasterio.features import geometry_mask
from shapely import contains_xy
from shapely.geometry import shape
from shapely.ops import unary_union

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2])

SURVEYS = {
    2019: dict(pr="dhs/GHPR82FL.DTA", hr="dhs/GHHR82FL.DTA", gc="covariates/GHGC82FL.csv",
               gps="gps/2019/GHGE81FL.shp", gc_year=2015),
    2022: dict(pr="dhs/GHPR8CFL.DTA", hr="dhs/GHHR8CFL.DTA", gc="covariates/GHGC8AFL.csv",
               gps="gps/2022/GHGE8AFL.shp", gc_year=2015),
}
PR_COLS = ["hv001", "hv002", "hvidx", "hv005", "hv103", "hv021", "hv022", "hv024", "hv025", "hv006",
           "hv007", "hv040", "hv104", "hc1", "hv270", "hml12", "hc61", "hml35", "hml32",
           "hv009", "hv014"]
HR_COLS = ["hv001", "hv002", "hv201", "hv205", "hv225", "hv206", "hv213", "hv214", "hv215",
           "hml1", "hv253"]

# WHO/UNICEF JMP categories
IMPROVED_TOILET = {"flush to piped sewer system", "flush to septic tank", "flush to pit latrine",
                   "flush, don't know where", "flush, bio-digester (biofil)",
                   "ventilated improved pit latrine (vip)", "pit latrine with slab", "composting toilet"}
UNIMPROVED_TOILET = {"flush to somewhere else", "pit latrine without slab/open pit", "bucket toilet",
                     "hanging toilet/latrine", "other"}
IMPROVED_WATER = {"piped into dwelling", "piped to yard/plot", "piped to neighbor", "public tap/standpipe",
                  "tube well or borehole", "protected well", "protected spring", "rainwater",
                  "tanker truck", "cart with small tank", "bottled water", "sachet water"}
SURFACE_WATER = {"river/dam/lake/ponds/stream/canal/irrigation channel"}

# DHS finished housing materials
FINISHED_FLOOR = {"cement", "ceramic/marble/porcelain tiles/terrazo", "ceramic/marble/porcelain/tiles/terrazo",
                  "linoleum/rubber carpet", "parquet or polished wood", "vinyl or asphalt strips",
                  "woolen carpet/synthetic carpet"}
FINISHED_WALL = {"cement", "cement blocks", "bricks", "stone with lime/cement", "covered adobe",
                 "wood planks/shingles"}
FINISHED_ROOF = {"zinc/aluminum", "wood", "wood planks", "ceramic tiles", "ceramic/brick tiles", "cement",
                 "roofing shingles", "asbestos/slate roofing sheets"}

# Ghana's 16 regions mapped to the 10 used in the 2019 MIS
REGION_16_TO_10 = {
    "western": "western", "western north": "western", "central": "central",
    "greater accra": "greater accra", "eastern": "eastern", "ashanti": "ashanti",
    "bono": "brong ahafo", "bono east": "brong ahafo", "ahafo": "brong ahafo",
    "volta": "volta", "oti": "volta", "northern": "northern", "savannah": "northern",
    "north east": "northern", "upper east": "upper east", "upper west": "upper west",
}

# DHS geospatial covariates. Malaria Atlas Project prevalence, incidence and ITN coverage
# are excluded: they are partly modelled from DHS data and would leak the outcome.
GC_TIMED = ["Aridity", "Mean_Temperature", "Maximum_Temperature", "Minimum_Temperature", "Rainfall",
            "Enhanced_Vegetation_Index", "PET", "Wet_Days", "Land_Surface_Temperature",
            "Day_Land_Surface_Temp", "Night_Land_Surface_Temp", "Diurnal_Temperature_Range",
            "UN_Population_Density"]
# Time-varying covariates use the 2015 values, which both files contain. Of the static covariates, only the
# human footprint agrees between the two files at nearby locations; night-time lights, travel time, growing
# season length and drought episodes differ between the files (or are mostly missing) and are not used.
GC_STATIC = ["Global_Human_Footprint"]

# DHS displacement: up to 2 km urban, up to 5 km rural (1% of rural clusters up to 10 km)
BUFFER_KM = {"U": 2.0, "R": 5.0}


def low(s):
    return s.astype("string").str.strip().str.lower().fillna("").astype(object)


def load_survey(year, cfg):
    pr = pd.read_stata(RAW / cfg["pr"], columns=PR_COLS)
    hr = pd.read_stata(RAW / cfg["hr"], columns=HR_COLS)
    pr["rdt"] = low(pr["hml35"])
    # de facto children (slept in the household the night before), as in DHS malaria prevalence tables
    k = pr[pr["rdt"].isin(["positive", "negative"]) & (low(pr["hv103"]) == "yes")].merge(hr, on=["hv001", "hv002"], how="left",
                                                             validate="many_to_one")
    d = pd.DataFrame(index=k.index)
    d["survey_year"] = year
    d["cluster"] = k["hv001"].astype(int)
    d["cluster_uid"] = f"{year}_" + d["cluster"].astype(str)
    d["household"] = k["hv002"].astype(int)
    d["line"] = k["hvidx"].astype(int)
    d["weight"] = k["hv005"].astype(float) / 1e6
    d["psu"] = k["hv021"].astype(int)
    d["strata"] = low(k["hv022"])
    d["interview_month"] = k["hv006"].astype(int)
    d["interview_year"] = k["hv007"].astype(int)

    d["malaria_rdt"] = (k["rdt"] == "positive").astype(int)
    mic = low(k["hml32"])
    d["malaria_microscopy"] = np.where(mic == "positive", 1, np.where(mic == "negative", 0, np.nan))

    d["region_original"] = low(k["hv024"])
    d["region10"] = d["region_original"].map(lambda r: REGION_16_TO_10.get(r, r))
    d["urban"] = (low(k["hv025"]) == "urban").astype(int)
    alt = pd.to_numeric(k["hv040"], errors="coerce")
    d["altitude_m"] = alt.where(alt < 9000)

    d["age_months"] = pd.to_numeric(k["hc1"], errors="coerce")
    d["female"] = (low(k["hv104"]) == "female").astype(int)
    net = low(k["hml12"])
    d["slept_under_itn"] = net.isin(["only treated (itn) nets", "both treated (itn) and untreated nets"]).astype(int)
    d["slept_under_any_net"] = ((net != "did not sleep under a net") & (net != "")).astype(int)
    # 2019 splits secondary into middle/JSS/JHS and secondary/SSS/SHS; 2022 has one "secondary" category
    d["mother_education"] = low(k["hc61"]).replace({".a": "not in household", "": "not in household",
                                                    "middle/jss/jhs": "secondary", "secondary/sss/shs": "secondary"})

    d["wealth_quintile"] = low(k["hv270"]).map({"poorest": 1, "poorer": 2, "middle": 3, "richer": 4, "richest": 5})
    d["household_size"] = pd.to_numeric(k["hv009"], errors="coerce")
    d["children_under5"] = pd.to_numeric(k["hv014"], errors="coerce")
    d["nets_per_person"] = (pd.to_numeric(k["hml1"], errors="coerce") / d["household_size"]).clip(upper=3)
    d["electricity"] = (low(k["hv206"]) == "yes").astype(int)
    d["sprayed_irs_12m"] = (low(k["hv253"]) == "yes").astype(int)

    toilet, shared = low(k["hv205"]), low(k["hv225"]) == "yes"
    d["sanitation_jmp"] = pd.Series(np.select(
        [toilet == "no facility/bush/field", toilet.isin(UNIMPROVED_TOILET),
         toilet.isin(IMPROVED_TOILET) & shared, toilet.isin(IMPROVED_TOILET) & ~shared],
        ["open defecation", "unimproved", "limited", "basic"], default="unknown"), index=d.index)
    d["sanitation_score"] = d["sanitation_jmp"].map({"open defecation": 0, "unimproved": 1, "limited": 2, "basic": 3})
    water = low(k["hv201"])
    d["water_jmp"] = np.select([water.isin(IMPROVED_WATER), water.isin(SURFACE_WATER), water != ""],
                               ["improved", "surface water", "unimproved"], default="unknown")
    d["floor_finished"] = low(k["hv213"]).isin(FINISHED_FLOOR).astype(int)
    d["wall_finished"] = low(k["hv214"]).isin(FINISHED_WALL).astype(int)
    d["roof_finished"] = low(k["hv215"]).isin(FINISHED_ROOF).astype(int)
    d["housing_finished_count"] = d[["floor_finished", "wall_finished", "roof_finished"]].sum(axis=1)

    gc = pd.read_csv(RAW / cfg["gc"])
    keep = {"DHSCLUST": "cluster"}
    keep.update({f"{s}_{cfg['gc_year']}": "gc_" + s.lower() for s in GC_TIMED if f"{s}_{cfg['gc_year']}" in gc})
    keep.update({s: "gc_" + s.lower() for s in GC_STATIC if s in gc})
    gc = gc[list(keep)].rename(columns=keep)
    cov = [c for c in gc if c != "cluster"]
    gc[cov] = gc[cov].mask(gc[cov] < 0)  # missing-value codes (-9999, -997, ...); none of these covariates is negative
    if gc["gc_enhanced_vegetation_index"].max() > 1:  # 2019 file stores EVI x 10,000
        gc["gc_enhanced_vegetation_index"] /= 10000
    gc.loc[gc["gc_enhanced_vegetation_index"] == 0, "gc_enhanced_vegetation_index"] = np.nan  # 0 = no data
    return d.merge(gc, on="cluster", how="left")


def era5_fields():
    ds = xr.open_dataset(RAW / "era5/era5_t2m_d2m_2019_2025.nc")
    if "expver" in ds.coords:
        print("ERA5 expver values:", sorted(set(map(str, np.atleast_1d(ds.expver.values)))))
    t = ds["t2m"] - 273.15
    td = ds["d2m"] - 273.15
    rh = 100 * np.exp(17.625 * td / (243.04 + td)) / np.exp(17.625 * t / (243.04 + t))  # Magnus formula
    return ds, t, rh


def chirps_files():
    for f in sorted((RAW / "chirps").glob("chirps_ghana_*.tif")):
        _, _, y, m = f.stem.split("_")
        yield f, int(y), int(m)


def regional_weather():
    # geoBoundaries gbOpen ADM1 (16 regions), merged to the 10 pre-2019 regions
    gj = json.loads((RAW / "boundaries/geoBoundaries-GHA-ADM1.geojson").read_text())
    groups = {}
    for f in gj["features"]:
        name = f["properties"]["shapeName"].replace(" Region", "").strip().lower()
        groups.setdefault(REGION_16_TO_10[name], []).append(shape(f["geometry"]))
    polys = {r: unary_union(g) for r, g in groups.items()}

    rows, masks = [], None
    for f, y, m in chirps_files():
        with rasterio.open(f) as src:
            a = src.read(1).astype(float)
            if masks is None:
                masks = {r: ~geometry_mask([p], out_shape=a.shape, transform=src.transform, all_touched=True)
                         for r, p in polys.items()}
        a[a < 0] = np.nan  # -9999 outside land
        rows += [{"region10": r, "year": y, "month": m, "rain_mm": np.nanmean(a[mk])} for r, mk in masks.items()]
    rain = pd.DataFrame(rows)

    ds, t, rh = era5_fields()
    lon, lat = np.meshgrid(ds.longitude.values, ds.latitude.values)
    rows = []
    for r, p in polys.items():
        inside = contains_xy(p.buffer(0.15), lon, lat)
        for i, vt in enumerate(pd.to_datetime(ds.valid_time.values)):
            rows.append({"region10": r, "year": vt.year, "month": vt.month,
                         "temp_c": float(t.values[i][inside].mean()), "rh_pct": float(rh.values[i][inside].mean())})
    return rain.merge(pd.DataFrame(rows), on=["region10", "year", "month"])


def cluster_weather():
    gps = []
    for year, cfg in SURVEYS.items():
        g = gpd.read_file(RAW / cfg["gps"])
        g = pd.DataFrame({"survey_year": year, "cluster": g["DHSCLUST"].astype(int),
                          "urban_rural": g["URBAN_RURA"], "lat": g["LATNUM"], "lon": g["LONGNUM"]})
        g.loc[(g.lat == 0) & (g.lon == 0), ["lat", "lon"]] = np.nan  # no coordinates
        gps.append(g)
    gps = pd.concat(gps, ignore_index=True).dropna(subset=["lat"]).reset_index(drop=True)

    # CHIRPS: mean of pixel centres within the displacement buffer; nearest pixel if no centre lies inside
    f0 = next(chirps_files())[0]
    with rasterio.open(f0) as src:
        rr, cc = np.meshgrid(np.arange(src.height), np.arange(src.width), indexing="ij")
        px_lon, px_lat = map(np.array, rasterio.transform.xy(src.transform, rr.ravel(), cc.ravel()))
    idx = {}
    for g in gps.itertuples():
        dist = np.hypot((px_lon - g.lon) * 111.32 * np.cos(np.radians(g.lat)), (px_lat - g.lat) * 110.57)
        sel = np.where(dist <= BUFFER_KM[g.urban_rural])[0]
        idx[(g.survey_year, g.cluster)] = sel if len(sel) else np.array([dist.argmin()])
    rows = []
    for f, y, m in chirps_files():
        with rasterio.open(f) as src:
            a = src.read(1).astype(float).ravel()
        a[a < 0] = np.nan
        rows += [{"survey_year": k[0], "cluster": k[1], "year": y, "month": m, "rain_mm": np.nanmean(a[s])}
                 for k, s in idx.items()]
    rain = pd.DataFrame(rows)

    # ERA5: bilinear interpolation at the cluster point
    ds, t, rh = era5_fields()
    lat_da, lon_da = xr.DataArray(gps.lat.values, dims="pt"), xr.DataArray(gps.lon.values, dims="pt")
    ti = t.interp(latitude=lat_da, longitude=lon_da).values
    ri = rh.interp(latitude=lat_da, longitude=lon_da).values
    era = pd.concat([pd.DataFrame({"survey_year": gps.survey_year, "cluster": gps.cluster, "year": vt.year,
                                   "month": vt.month, "temp_c": ti[i], "rh_pct": ri[i]})
                     for i, vt in enumerate(pd.to_datetime(ds.valid_time.values))])
    return rain.merge(era, on=["survey_year", "cluster", "year", "month"], how="outer")


def add_lags(d, w, keys, prefix):
    """Weather 1, 2 and 3 months before the interview month, plus 3-month summaries."""
    d = d.copy()
    w = w.assign(ym=w.year * 12 + w.month - 1)
    d["ym"] = d.interview_year * 12 + d.interview_month - 1
    for lag in (1, 2, 3):
        x = w[keys + ["ym", "rain_mm", "temp_c", "rh_pct"]].assign(ym=w.ym + lag)
        x = x.rename(columns={"rain_mm": f"{prefix}_rain_lag{lag}", "temp_c": f"{prefix}_temp_lag{lag}",
                              "rh_pct": f"{prefix}_rh_lag{lag}"})
        d = d.merge(x, on=keys + ["ym"], how="left")
    d[f"{prefix}_rain_3m"] = d[[f"{prefix}_rain_lag{i}" for i in (1, 2, 3)]].sum(axis=1, min_count=3)
    d[f"{prefix}_temp_3m"] = d[[f"{prefix}_temp_lag{i}" for i in (1, 2, 3)]].mean(axis=1)
    d[f"{prefix}_rh_3m"] = d[[f"{prefix}_rh_lag{i}" for i in (1, 2, 3)]].mean(axis=1)
    return d.drop(columns="ym")


d = pd.concat([load_survey(y, c) for y, c in SURVEYS.items()], ignore_index=True)
d = add_lags(d, regional_weather(), ["region10"], "reg")
d = add_lags(d, cluster_weather(), ["survey_year", "cluster"], "cl")
d.to_csv(OUT, index=False)
print(f"{len(d)} children, {d.shape[1]} columns saved to {OUT}")
print(d.groupby("survey_year").malaria_rdt.agg(children="size", positive="sum"))
