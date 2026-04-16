# USRDB Tariff Archetype Library

A data-driven library of representative U.S. commercial electricity tariff archetypes derived from the [U.S. Utility Rate Database (USRDB)](https://openei.org/wiki/Utility_Rate_Database).

This repository accompanies the paper:

> **A Data-Driven Archetype Library of U.S. Commercial Electricity Tariffs from the Utility Rate Database**
> Khazaei, A., Feng, F., Li, Z., Han, X., & Pang, Z.

## Repository Structure

```
USRDB-Tariff-Archetype-Library/
├── README.md
├── requirements.txt
├── LICENSE
├── archetypes/                  # Machine-readable archetype JSON files
│   ├── demand/                  # TOU demand-charge archetypes (4 clusters)
│   ├── energy_1period/          # One-period energy-rate archetypes (6 clusters)
│   ├── energy_2period/          # Two-period energy-rate archetypes (6 clusters)
│   ├── energy_3period/          # Three-period energy-rate archetypes (5 clusters)
│   ├── energy_4plus_period/     # Four+-period energy-rate archetypes (5 clusters)
│   └── flat_demand/             # Flat demand month-partition patterns (10 patterns)
├── pipeline/                    # USRDB-to-archetype processing code
│   ├── __init__.py
│   ├── usrdb_parser.py          # Step 1: Parse USRDB CSV into tidy long tables
│   ├── load_and_merge.py        # Step 2: Load data, merge with EIA utility metadata
│   ├── feature_extraction.py    # Step 3: Compute per-period pricing features
│   ├── segmentation_clustering.py # Step 4a: Rule-based + DBSCAN period segmentation
│   └── pricing_clustering.py    # Step 4b: k-means pricing archetype derivation
├── billing_engine/              # Reference billing implementation (TODO)
│   └── billing.py
└── examples/
    └── quickstart.ipynb         # Walkthrough notebook (TODO)
```

## Quick Start

### Installation
**Note:** The USRDB dataset is not included in this repository and must be obtained separately from the OpenEI Utility Rate Database.

```bash
git clone https://github.com/Armin-Khazaei/USRDB-Tariff-Archetype-Library.git
cd USRDB-Tariff-Archetype-Library
pip install -r requirements.txt
```

### Parse USRDB Data

```python
import pandas as pd
from pipeline import (
    load_usrdb, parse_usrdb_all, run_counts,
    generate_rate_features,
    classify_tou_patterns, dbscan_segmentation,
    cluster_k1_plans, cluster_k2_plans, cluster_k3_plans
)

# Step 1: Load raw USRDB CSV
# Replace with your local path to the USRDB dataset
df = load_usrdb("path/to/usrdb.csv")

# Step 2: Parse into tidy long tables
tiers_long, flat_months, summary = parse_usrdb_all(df)
periods_summary, tiers_detail = run_counts(df)

# Step 3: Extract pricing features
df_energy_features = generate_rate_features(
    df, rate_col="energyratestructure_detail",
    q_step=50, q_max=5000
)

# Step 4a: Segment period maps (rule-based + DBSCAN)
df_seg, seg_summary = classify_tou_patterns(df, "energyweekdayschedule")
df_seg, dbscan_meta = dbscan_segmentation(df_seg, schedule_col="energyweekdayschedule")

# Step 4b: Cluster pricing archetypes
df_k1, meta_k1 = cluster_k1_plans(df_energy_features, best_k=6)
df_k2, meta_k2 = cluster_k2_plans(df_energy_features, best_k=6)
```

### Load an Archetype

```python
import json

with open("archetypes/demand/archetype_01_moderate_tou.json") as f:
    archetype = json.load(f)

# archetype contains:
#   - period_map: 12x24 matrix of period indices
#   - periods: list of {tiers: [{rate, max, adj}], ...}
#   - metadata: cluster_id, share, representative_plan, ...
```

## Data Sources

- **USRDB**: Zimny-Schmitt, J. & Huggins, J. (2010). Utility Rate Database (URDB). NREL.
  DOI: [10.25984/1788456](https://doi.org/10.25984/1788456)
- **EIA Utility Data**: U.S. Energy Information Administration, Form EIA-861.

## Citation

If you use this library in your research, please cite:

```bibtex
@article{khazaei2026tariff,
  title={A Data-Driven Archetype Library of {U.S.} Commercial Electricity Tariffs
         from the Utility Rate Database},
  author={Khazaei, Armin and Feng, Fan and Li, Zhuorui and Han, Xu and Pang, Zhihong},
  journal={[Journal Name]},
  year={2026},
  note={Under review}
}
```

## License

[MIT License](LICENSE)
