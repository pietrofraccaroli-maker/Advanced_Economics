# Hurricane Economic Propagation in Florida — Project README

**Course:** Advanced Techniques in Applied Economics (Spring 2026)  
**Professor:** Piotr Zwiernik — Universitat Pompeu Fabra  
**Authors:** Pietro Fraccaroli, Gimelgo Xirinda

---

## Research Question

Does the economic impact of hurricanes in Florida propagate through channels beyond geographic proximity? Evidence from a Gaussian Graphical Model approach.

---

## Motivation

Three major hurricanes have struck Florida in recent years:

- **Hurricane Irma** (2017)
- **Hurricane Michael** (2018)
- **Hurricane Ian** (2022)

Each caused severe damage across multiple counties. The key question is whether counties *not directly hit* are nonetheless affected through indirect economic linkages — for example through tourism flows, supply chains, or labor market connections — rather than through physical destruction alone.

**Illustrative example:** Lee County (Ian's landfall) and Osceola County (Orlando/Disney area, ~300 km away). If a hurricane devastating Lee depresses visitor flows statewide, Osceola may suffer economically even without any physical damage. This channel is invisible to standard geographic proximity matrices.

---

## Methodology

The project proceeds in two stages.

### Stage 1 — Panel Regression with Spatial Lags

For each county `i`, regress monthly nighttime light intensity on local hurricane indicators *and* the hurricane indicators of geographically contiguous counties (first-order contiguity — counties sharing a physical border):

```
NTL_i(t) = α_i + γ_t + β₁·FEMA_i(t) + β₂·Σ_{j ∈ N(i)} FEMA_j(t) + β₃·Precip_i(t) + ε_i(t)
```

Where:
- `NTL_i(t)` = nighttime light intensity in county `i` at month `t` (proxy for economic activity)
- `α_i` = county fixed effects (absorb time-invariant structural differences between counties)
- `γ_t` = month fixed effects (absorb common seasonal patterns across all counties)
- `FEMA_i(t)` = hurricane impact indicator for county `i` (from FEMA disaster declarations)
- `Σ_{j ∈ N(i)} FEMA_j(t)` = sum or average of FEMA indicators for contiguous counties (spatial lag — **key addition suggested by Prof. Zwiernik**)
- `Precip_i(t)` = monthly precipitation as a control for weather effects unrelated to hurricanes
- `ε_i(t)` = residual

**What the fixed effects control for:**
- County FE: structural differences in brightness/economic size that are stable over time (e.g., Miami-Dade is always brighter than Gilchrist County)
- Month FE: seasonal patterns common to all counties (e.g., December lights are higher everywhere due to holidays)

**Why the spatial lag matters (Prof. Zwiernik's suggestion):** Including neighboring counties' FEMA indicators in Stage 1 removes the geographically proximate propagation channel already at the regression stage. The residuals `ε_i(t)` are therefore "cleaner" — they capture variation not explained by direct damage *nor* by damage in neighboring counties. Any structure found in Stage 2 is thus more credibly attributable to non-geographic economic channels.

The coefficients on the spatial lag term are themselves informative: they measure how much a hurricane hitting a neighboring county depresses local economic activity, controlling for local damage.

---

### Stage 2 — Graphical Lasso on Residuals

Stack the residuals `ε_i(t)` into a matrix of dimension `T × 67` (months × counties). Apply the Graphical Lasso to estimate a sparse precision matrix Θ over Florida's 67 counties.

The precision matrix encodes **conditional dependence**: a non-zero entry (i,j) means counties `i` and `j` are directly linked after controlling for all other counties. This is the key distinction from simple correlation — the graphical lasso separates direct from indirect linkages.

**Comparison:** The estimated dependency structure is then compared against the geographic contiguity matrix (binary matrix where entry (i,j) = 1 if counties share a border, 0 otherwise).

| Pair | Contiguous? | Connected by glasso? | Interpretation |
|---|---|---|---|
| Lee — Collier | Yes | Probably No | Geographic channel already removed in Stage 1 |
| Lee — Osceola | No | Yes (if found) | Pure economic channel (e.g. tourism) |
| Lee — Miami-Dade | No | No | No direct linkage |

**Main finding of interest:** Connections in the graphical lasso that do not correspond to geographic contiguity suggest economic propagation channels — tourism, supply chains, labor markets — that go beyond physical proximity. Since Stage 1 already controls for geographic spillovers, this evidence is robust to the most obvious critique.

---

## Data Sources

| Dataset | Variable | Source | Frequency | Coverage |
|---|---|---|---|---|
| NOAA VIIRS | Nighttime light intensity | Google Earth Engine | Monthly | 2012–present |
| FEMA Disaster Declarations | Hurricane impact indicator | OpenFEMA API (CSV) | Event-level → monthly | 2012–present |
| NOAA precipitation | Monthly precipitation control | NOAA Climate Data | Monthly | 2012–present |
| US Census TIGER/Line | County boundaries (shapefile) | Census Bureau | Static | Current |

---

## Geographic Network Construction

The contiguity matrix is constructed from the US Census shapefile of Florida counties using the `spdep` package in R:

```r
library(spdep)
library(sf)

# Load Florida county shapefile
florida <- st_read("florida_counties.shp")

# Build contiguity matrix (first-order, shared border)
nb <- poly2nb(florida, queen = FALSE)  # rook contiguity = shared border
W <- nb2mat(nb, style = "B")           # binary matrix 67x67
```

This produces the 67×67 binary matrix used both as the spatial lag structure in Stage 1 and as the benchmark in Stage 2.

---

## R Implementation Sketch

### Stage 1 — Panel regression

```r
library(plm)

# Panel data structure: county x month
pdata <- pdata.frame(df, index = c("county_fips", "year_month"))

# Regression with county and time fixed effects
model <- plm(
  NTL ~ FEMA_local + FEMA_neighbors + precipitation,
  data = pdata,
  effect = "twoways",   # county FE + month FE
  model = "within"
)

# Extract residuals
residuals_matrix <- matrix(residuals(model), nrow = T, ncol = 67)
```

### Stage 2 — Graphical Lasso

```r
library(glasso)

# Estimate covariance from residuals
S <- cov(residuals_matrix)

# Apply graphical lasso (tune rho via cross-validation or BIC)
fit <- glasso(S, rho = 0.1)

# Precision matrix
Theta <- fit$wi

# Adjacency matrix (non-zero off-diagonal entries)
adj_glasso <- (abs(Theta) > 1e-6) * 1
diag(adj_glasso) <- 0
```

### Comparison

```r
# Compare glasso adjacency with geographic contiguity
agreement <- sum(adj_glasso == W) / (67 * 67)

# Edges in glasso but not in contiguity matrix = non-geographic channels
non_geographic <- adj_glasso - (adj_glasso * W)
```

---

## Project Timeline (approx. 2 weeks)

| Week | Tasks |
|---|---|
| Week 1 | Data download (VIIRS via GEE, FEMA via API), contiguity matrix construction, panel regression (Stage 1) |
| Week 2 | Graphical lasso (Stage 2), comparison with contiguity matrix, write-up |

---

## Key References

- Felbermayr & Gröschl — economic impacts of weather anomalies (conceptual inspiration)
- Friedman, Hastie & Tibshirani (2008) — Graphical Lasso
- Course Lecture 2 (Zwiernik, UPF 2026) — Graphical models and sparse precision matrices
