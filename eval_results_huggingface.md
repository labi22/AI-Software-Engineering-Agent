# RAG Retrieval & Generation Benchmark Results

- **Target Repository**: `Yield_Curve_Construction_and_Bond_Valuation_&_Risk_Lab` (14 files, 83 chunks)
- **Test Suite Size**: 25 technical evaluation queries
- **Embedding / LLM Cost**: $0.00 (Offline IR Evaluation)

## Comparative Retrieval Metrics Table

| Retrieval Strategy | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR | Latency / Query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 92.0% | 100.0% | 100.0% | 100.0% | 52.0% | 0.960 | 42.0 ms |
| Bm25 | 60.0% | 72.0% | 88.0% | 88.0% | 36.8% | 0.687 | 0.4 ms |
| Hybrid (RRF k=60) | 84.0% | 100.0% | 100.0% | 100.0% | 44.8% | 0.913 | 43.5 ms |
| Hybrid (RRF k=60) + Symbol Boost | 60.0% | 100.0% | 100.0% | 100.0% | 44.8% | 0.800 | 42.8 ms |

## Failure Case Analysis

### Failure Case 1: Query q03
- **Question**: How is Macaulay duration computed for coupon-bearing bonds?
- **Expected File**: `bond_engine/bond_pricer.py`
- **Retrieved Files (Top 5)**: `bond_engine/portfolio.py, bond_engine/bond_pricer.py, bond_engine/bond_pricer.py, yield_curve_explore.ipynb, yield_curve_explore.ipynb`
- **MRR Score**: `0.500` (Rank: 2)
- **Root Cause Analysis**: Common file imports or overlapping class definitions between dashboard callers and core calculation engines.
### Failure Case 2: Query q04
- **Question**: Where is modified duration calculated from Macaulay duration and yield?
- **Expected File**: `bond_engine/bond_pricer.py`
- **Retrieved Files (Top 5)**: `bond_engine/portfolio.py, bond_engine/bond_pricer.py, bond_engine/bond_pricer.py, yield_curve_explore.ipynb, yield_curve_explore.ipynb`
- **MRR Score**: `0.500` (Rank: 2)
- **Root Cause Analysis**: Common file imports or overlapping class definitions between dashboard callers and core calculation engines.
### Failure Case 3: Query q09
- **Question**: How is a parallel rate shock curve constructed to simulate shifts across all maturities?
- **Expected File**: `bond_engine/rate_shock.py`
- **Retrieved Files (Top 5)**: `yield_curve/rates.py, bond_engine/rate_shock.py, bond_engine/rate_shock.py, yield_curve/rates.py, yield_curve_explore.ipynb`
- **MRR Score**: `0.500` (Rank: 2)
- **Root Cause Analysis**: Common file imports or overlapping class definitions between dashboard callers and core calculation engines.

