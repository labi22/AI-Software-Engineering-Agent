# RAG Retrieval Benchmark Results

- **Target Repository**: `Yield_Curve_Construction_and_Bond_Valuation_&_Risk_Lab` (14 files, 83 chunks)
- **Test Suite Size**: 25 technical evaluation queries
- **Embedding / LLM Cost**: $0.00 (Offline IR Evaluation)

## Comparative Metrics Table

| Retrieval Strategy | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR | Latency / Query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 8.0% | 20.0% | 24.0% | 24.0% | 7.2% | 0.143 | 3.7 ms |
| Bm25 | 60.0% | 72.0% | 88.0% | 88.0% | 36.8% | 0.687 | 0.6 ms |
| Hybrid (RRF k=60) | 28.0% | 68.0% | 68.0% | 68.0% | 17.6% | 0.473 | 4.8 ms |
| Hybrid (RRF k=60) + Symbol Boost | 56.0% | 68.0% | 68.0% | 68.0% | 17.6% | 0.613 | 5.6 ms |

## Failure Case Analysis

### Failure Case 1: Query q01
- **Question**: How is a bond priced using discount factors from the yield curve in the bond engine?
- **Expected File**: `bond_engine/bond_pricer.py`
- **Retrieved Files (Top 5)**: `yield_curve/curve_builder.py, app/hedge_bond.py, bond_engine/bond_pricer.py, bond_engine/bond_pricer.py, yield_curve_explore.ipynb`
- **MRR Score**: `0.333` (Rank: 3)
- **Root Cause Analysis**: Common file imports or overlapping class definitions between dashboard callers and core calculation engines.
### Failure Case 2: Query q02
- **Question**: Where is the yield to maturity (YTM) of a bond calculated using root finding?
- **Expected File**: `bond_engine/bond_pricer.py`
- **Retrieved Files (Top 5)**: `yield_curve/curve_builder.py, bond_engine/bond_pricer.py, yield_curve_explore.ipynb, yield_curve_explore.ipynb, app/hedge_bond.py`
- **MRR Score**: `0.500` (Rank: 2)
- **Root Cause Analysis**: Common file imports or overlapping class definitions between dashboard callers and core calculation engines.
### Failure Case 3: Query q05
- **Question**: How is bond convexity computed from coupon cash flows and discount rates?
- **Expected File**: `bond_engine/bond_pricer.py`
- **Retrieved Files (Top 5)**: `app/bond_portfolio.py, yield_curve_explore.ipynb, yield_curve_explore.ipynb, yield_curve_explore.ipynb, yield_curve/data_loader.py`
- **MRR Score**: `0.000` (Rank: Not in top 5)
- **Root Cause Analysis**: Common file imports or overlapping class definitions between dashboard callers and core calculation engines.

