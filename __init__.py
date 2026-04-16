"""
pipeline
========
USRDB tariff parsing and archetype derivation pipeline.

Modules
-------
usrdb_parser              : Parse raw USRDB CSV into tidy long tables
load_and_merge            : Load data and merge with EIA utility metadata
feature_extraction        : Compute per-period pricing features
segmentation_clustering   : DBSCAN + rule-based period segmentation
pricing_clustering        : k-means pricing archetype derivation
"""

from .usrdb_parser import (
    tidy_urdb_tiers,
    extract_flat_demand_months,
    summarize_tier_brackets,
    count_periods_and_tiers,
    run_counts,
    build_rate_detail,
    parse_usrdb_all,
    SECTIONS,
)
from .load_and_merge import (
    load_usrdb,
    build_eia_utility_table,
    merge_eia_utilities,
)
from .feature_extraction import (
    generate_rate_features,
)
from .segmentation_clustering import (
    classify_tou_patterns,
    dbscan_segmentation,
)
from .pricing_clustering import (
    cluster_k1_plans,
    cluster_k2_plans,
    cluster_k3_plans,
    cluster_k4plus_plans,
    compute_elbow,
)
