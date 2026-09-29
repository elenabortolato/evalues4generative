# Data

`italy_covid_daily.csv` contains Italy's national daily COVID-19 series, one
row per day from 2020-02-24 to 2025-01-08. It merges two public sources.

**Dipartimento della Protezione Civile**, "COVID-19 Italia – Monitoraggio
situazione", file
`dati-andamento-nazionale/dpc-covid19-ita-andamento-nazionale.csv` in
https://github.com/pcm-dpc/COVID-19 (licence CC-BY-4.0). Columns, renamed from
the Italian originals:

- `new_positives` (`nuovi_positivi`)
- `icu_admissions` (`ingressi_terapia_intensiva`, available from 2020-12-03)
- `icu_occupancy` (`terapia_intensiva`)
- `hospital_occupancy` (`totale_ospedalizzati`)
- `tests_cumulative` (`tamponi`)
- `deaths_cumulative` (`deceduti`)

**Oxford COVID-19 Government Response Tracker** (Hale et al., 2021, *Nature
Human Behaviour*), file `data/timeseries/stringency_index_avg.csv` in
https://github.com/OxCGRT/covid-policy-tracker (licence CC-BY-4.0). Column:

- `stringency`, Italy's national stringency index, from 0 to 100.

Downloaded on 2026-09-29 through raw.githubusercontent.com.

## Regional file

`italy_regions_daily.csv` has one row per region per day, for the 21 Italian
regions and autonomous provinces, from 2020-10-01 to 2022-05-15.

- **`new_positives`, `icu_admissions`:** from the Protezione Civile regional
  file `dati-regioni/dpc-covid19-ita-regioni.csv` (same repository and licence
  as above).
- **`colour`:** the region's COVID-19 restriction tier (white, yellow, orange,
  red), translated from `colore` (bianco, giallo, arancione, rosso). Tiers
  were set by government ordinance under the DPCM of 3 November 2020 and
  later decrees, from monitoring indicators such as incidence and Rt. It is
  missing before 2020-11-06. Source: https://github.com/imcatta/restrizioni_regionali_covid
  (`dataset.csv`), a compilation of the ordinances; see also
  https://github.com/jumpjack/ZoneCovidItalia.
