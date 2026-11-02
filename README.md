# malaria-u5-ghana-prediction
# Machine learning models for early prediction of malaria risk in children under five in Ghana

Analysis code for the paper by Roger Labul (University of Ghana), based on his MSc thesis. Six models (Decision Tree, Logistic Regression,
Random Forest, K-Nearest Neighbours, Gradient Boosting, XGBoost) are developed on the 2019 Ghana Malaria
Indicator Survey and tested once on the 2022 Ghana Demographic and Health Survey.

## Files

| File | Purpose |
|---|---|
| `build_dataset.py` | Builds the analysis dataset from the raw survey, GPS, covariate and climate files |
| `malaria_u5_analysis.ipynb` | Full analysis in Google Colab |
| `data_dictionary.csv` | Every dataset column, with the number of missing values in each survey |
| `requirements.txt` | Python package versions |

## Data sources

| Data | Source | Files |
|---|---|---|
| Ghana MIS 2019, Ghana DHS 2022 | [The DHS Program](https://dhsprogram.com) (registration and approval required) | GHPR82FL, GHHR82FL, GHGE81FL, GHGC82FL; GHPR8CFL, GHHR8CFL, GHGE8AFL, GHGC8AFL |
| Rainfall | [CHIRPS v3.0](https://data.chc.ucsb.edu/products/CHIRPS/v3.0/monthly/global/tifs/), monthly, January 2019 to December 2025, downloaded 29 October 2026 | chirps_ghana_YYYY_MM.tif |
| Temperature, dew point | [ERA5 monthly averaged data on single levels](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means) (2 m temperature, 2 m dew point), January 2019 to December 2025, downloaded 29 October 2026, expver 0001 | era5_t2m_d2m_2019_2025.nc |
| Region boundaries | [geoBoundaries](https://www.geoboundaries.org) gbOpen ADM1, Ghana | geoBoundaries-GHA-ADM1.geojson |

DHS data cannot be redistributed, so the derived dataset is not included.

## Reproduce

1. Arrange the raw files:
   ```
   data_raw/dhs/            GHPR82FL.DTA GHHR82FL.DTA GHPR8CFL.DTA GHHR8CFL.DTA
   data_raw/gps/2019/       GHGE81FL.shp (.dbf .shx .prj)
   data_raw/gps/2022/       GHGE8AFL.shp (.dbf .shx .prj)
   data_raw/covariates/     GHGC82FL.csv GHGC8AFL.csv
   data_raw/chirps/         chirps_ghana_YYYY_MM.tif
   data_raw/era5/           era5_t2m_d2m_2019_2025.nc
   data_raw/boundaries/     geoBoundaries-GHA-ADM1.geojson
   ```
2. Build the dataset:
   ```
   pip install -r requirements.txt
   python build_dataset.py data_raw malaria_u5_dataset.csv
   ```
3. Upload `malaria_u5_dataset.csv` to `My Drive/malaria_u5/` and run `malaria_u5_analysis.ipynb` in Google Colab.
   The notebook prints the dataset checksum (MD5) and says whether it matches the thesis dataset.
   Results are written to `My Drive/malaria_u5/outputs/`.

## Licence

MIT
