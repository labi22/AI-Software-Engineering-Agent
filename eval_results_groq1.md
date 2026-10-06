# RAG Retrieval & Generation Benchmark Results

- **Target Repository**: `Yield_Curve_Construction_and_Bond_Valuation_&_Risk_Lab` (14 files, 83 chunks)
- **Test Suite Size**: 25 technical evaluation queries
- **Embedding / LLM Cost**: $0.00 (Offline IR Evaluation)

## Comparative Retrieval Metrics Table

| Retrieval Strategy | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR | Latency / Query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 92.0% | 100.0% | 100.0% | 100.0% | 52.0% | 0.960 | 59.3 ms |
| Bm25 | 60.0% | 72.0% | 88.0% | 88.0% | 36.8% | 0.687 | 1.2 ms |
| Hybrid (RRF k=60) | 84.0% | 100.0% | 100.0% | 100.0% | 44.8% | 0.913 | 71.9 ms |
| Hybrid (RRF k=60) + Symbol Boost | 60.0% | 100.0% | 100.0% | 100.0% | 44.8% | 0.800 | 85.5 ms |

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


## Generation Evaluation Results

- **LLM Provider**: `groq`
- **Evaluated Queries**: 25

| Total Queries | Mean Faithfulness | Mean Symbol Recall | Verified Citations | Total Citations | Avg Latency / Query |
| --- | --- | --- | --- | --- | --- |
| 25 | 92.0% | 90.0% | 67 | 71 | 12232.0 ms |

### Sample Generated Answer (q01)
- **Question**: How is a bond priced using discount factors from the yield curve in the bond engine?
- **Answer**: In the **bond_engine** the price of a bond is calculated by discounting each of its cash‑flows back to the present using the discount factors supplied by a yield‑curve object.  
The routine that implements this logic is `price_bond` in `bond_engine/bond_pricer.py`:

```python
def price_bond(bond: "Bond", yield_curve):
    times, cashflows = bond.cashflows()          # 1

    price = 0.0
    for t, cf in zip(times, cashflows):          # 2
        price += cf * yield_curve.discount_factor(t)   # 3

    return price
```

**What happens at each step**

| Step | What the code does | Key line(s) |
|------|-------------------|-------------|
| 1 | The `Bond` instance is asked for its schedule of cash‑flows via `bond.cashflows()`. This call returns two parallel sequences: `times` (in years to each payment) and `cashflows` (the amounts). | `[1]` |
| 2 | The function iterates over every `(t, cf)` pair. | `[1]` |
| 3 | For each payment, the corresponding discount factor is fetched from the `yield_curve` object (`yield_curve.discount_factor(t)`). The payment is then discounted and added to the running `price` sum. | `[1]` |
| 4 | After all cash‑flows have been processed, the accumulated `price` is returned as the present value of the bond. | `[1]` |

**How the discount factor is obtained**

The `yield_curve` argument is expected to expose a `discount_factor(maturity)` method. In the sample `LocalShockCurve` implementation found in `app/hedge_bond.py`, this method is:

```python
def discount_factor(self, maturity):
    r = self.rate(maturity)
    return np.exp(-r * maturity)
```

[2]  

So the bond engine’s pricing routine is essentially a weighted sum of the bond’s future cash‑flows, where the weights are the present‑value discount factors supplied by the yield‑curve object.
- **Verified Citations**: bond_engine/bond_pricer.py (L29-36), app/hedge_bond.py (L29-31), bond_engine/bond_pricer.py (L75-87)
- **Faithfulness Score**: 100.0%
- **Symbol Recall**: 100.0%

