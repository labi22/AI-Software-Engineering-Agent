# RAG Retrieval & Generation Benchmark Results

- **Target Repository**: `Yield_Curve_Construction_and_Bond_Valuation_&_Risk_Lab` (14 files, 83 chunks)
- **Test Suite Size**: 25 technical evaluation queries
- **Embedding / LLM Cost**: $0.00 (Offline IR Evaluation)

## Comparative Retrieval Metrics Table

| Retrieval Strategy | Hit@1 | Hit@3 | Hit@5 | Recall@5 | Precision@5 | MRR | Latency / Query |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dense | 8.0% | 20.0% | 24.0% | 24.0% | 7.2% | 0.143 | 2.6 ms |
| Bm25 | 60.0% | 72.0% | 88.0% | 88.0% | 36.8% | 0.687 | 0.3 ms |
| Hybrid (RRF k=60) | 28.0% | 68.0% | 68.0% | 68.0% | 17.6% | 0.473 | 3.2 ms |
| Hybrid (RRF k=60) + Symbol Boost | 56.0% | 68.0% | 68.0% | 68.0% | 17.6% | 0.613 | 3.1 ms |

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


## Generation Evaluation Results

- **LLM Provider**: `groq`
- **Evaluated Queries**: 5

| Total Queries | Mean Faithfulness | Mean Symbol Recall | Verified Citations | Total Citations | Avg Latency / Query |
| --- | --- | --- | --- | --- | --- |
| 5 | 100.0% | 100.0% | 15 | 15 | 9048.8 ms |

### Sample Generated Answer (q01)
- **Question**: How is a bond priced using discount factors from the yield curve in the bond engine?
- **Answer**: ### How the engine prices a bond

In the *bond engine* a bond’s market value is obtained by **discounting each future cash‑flow back to the present** using the discount factor supplied by the yield curve.

1. **Cash‑flow extraction** –  
   Every `Bond` instance exposes a `cashflows()` method that returns two sequences:
   ```python
   times, cashflows = bond.cashflows()      # times in years, cashflows in currency units
   ```
   (This method is not shown in the snippets, but it is used in the duration and convexity helpers, e.g. `[4]`.)

2. **Discounting each payment** –  
   For every time `t` the engine calls the yield curve’s `discount_factor(t)`:
   ```python
   df = yield_curve.discount_factor(t)      #  df = exp(-rate(t) * t)
   ```
   The `discount_factor` itself is defined in `yield_curve/curve_builder.py`:
   ```python
   def discount_factor(self, maturity):
       r = self.rate(maturity)
       return np.exp(-r * maturity)          # [1]
   ```
   where `self.rate(maturity)` interpolates the continuous‑compounding yield from the cubic‑spline representation of the curve.

3. **Present‑value sum** –  
   The bond’s price is the sum of all discounted cash‑flows:
   ```python
   price = sum(cf * df for cf, df in zip(cashflows, dfs))      # implicit in price_bond
   ```
   Although the explicit definition of `price_bond` isn’t in the retrieved snippets, it is invoked in several places:
   - In the duration helper (`macaulay_duration` – `[4]`).
   - In the convexity helper (`convexity` – `[3]`).
   - In the hedge optimisation routine (`optimize_hedge` – `[2]`).

   Those calls show that `price_bond(bond, yield_curve)` must return the discounted‑cash‑flow sum described above.

4. **Putting it together** –  
   A minimal implementation that matches the surrounding code would look like:

   ```python
   def price_bond(bond, yield_curve):
       times, cashflows = bond.cashflows()
       df = [yield_curve.discount_factor(t) for t in times]
       return float(sum(cf * d for cf, d in zip(cashflows, df)))
   ```

   This function uses the yield‑curve’s continuous‑compounding discount factors, which are themselves derived from a cubic‑spline interpolation of the raw yield data (`[1]`).

---

**Summary**  
A bond is priced by summing its future cash‑flows, each multiplied by the corresponding discount factor obtained from the yield curve. The yield curve supplies a continuous‑compounding rate via a cubic‑spline, which is exponentiated to get the discount factor (`[1]`). This present‑value sum is the bond’s price used throughout the engine (`[3]`, `[4]`, and `[2]`).
- **Verified Citations**: yield_curve/curve_builder.py (L18-51), app/hedge_bond.py (L45-80), bond_engine/bond_pricer.py (L75-87)
- **Faithfulness Score**: 100.0%
- **Symbol Recall**: 100.0%

