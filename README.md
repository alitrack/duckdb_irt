# DuckDB IRT Extension

DuckDB functions for **Item Response Theory (2PL)** — the two-parameter logit
IRT model — estimated by **Marginal Maximum Likelihood (MML/EM)** à la
Bock & Aitkin (1981). Pure Rust, no external IRT dependency. Closes a real gap:
no existing DuckDB extension provides IRT estimation.

## What it does

For a binary item-response matrix `x` (rows = people, columns = items,
values 0/1), the 2PL model is

```
P(X[p,j]=1 | θ_p) = 1 / (1 + exp(-a_j (θ_p - b_j)))
```

- `a_j` = item `j` discrimination
- `b_j` = item `j` difficulty
- `θ_p` = person `p` latent ability, with an `N(0,1)` prior

The ability parameters `θ_p` are nuisance (integrated out). We maximize the
**marginal likelihood** over the item parameters `(a_j, b_j)` by EM:

- **E-step**: posterior over the `θ` grid (40-point Gauss-Hermite quadrature
  for the `N(0,1)` prior), per person.
- **M-step**: per item `j`, maximize the expected log-likelihood
  `Σ_p Σ_k q[pi,k] · log p_j(θ_k; a_j, b_j)` — concave in `(a_j, b_j)` — via
  2D gradient ascent with the analytic gradient and backtracking line search.

After convergence we report:
- `a_j`, `b_j` for every item
- `θ_p` for every person as the **EAP** (expected a posteriori) posterior mean

## Function

```sql
irt_2pl_param(x VARCHAR, j BIGINT, which VARCHAR, idx BIGINT,
              iters BIGINT, tol DOUBLE) → DOUBLE
```

- `x` — flat response string. **Row-major**: person 0's `j` responses, then
  person 1's, etc., comma- or space-separated, e.g.
  `'1,0,1, 0,1,1, 1,1,0'` = 3 people × 3 items. (DuckDB's `Value` API is
  scalar-only, so the matrix is passed as a single string; parse it with
  `group_concat` if needed.)
- `j` — number of items (columns).
- `which` — `'a'` (discrimination), `'b'` (difficulty), or `'theta'` (ability).
- `idx` — which item (for `a`/`b`) or which person (for `theta`).
- `iters` — max EM iterations (default 200).
- `tol` — stop when max `|Δa|`,`|Δb|` < `tol` (default 1e-6).

**Note:** each call re-fits the *whole* model (all items) and returns one
component. For a full report of `J` items + `P` people, call it `2J + P`
times (or loop). A single full fit on 300 people × 12 items takes ~2 s in
release mode.

### Example

```sql
LOAD 'irt_ext.duckdb_extension';

CREATE TABLE resp AS SELECT
  '1,0,1
   0,1,1
   1,1,0
   1,1,1
   0,0,1' AS x;   -- 5 people x 3 items

-- item discrimination / difficulty
SELECT
  irt_2pl_param(x, 3, 'a', 0, 200, 1e-6) AS a0,
  irt_2pl_param(x, 3, 'a', 1, 200, 1e-6) AS a1,
  irt_2pl_param(x, 3, 'a', 2, 200, 1e-6) AS a2,
  irt_2pl_param(x, 3, 'b', 0, 200, 1e-6) AS b0,
  irt_2pl_param(x, 3, 'b', 1, 200, 1e-6) AS b1,
  irt_2pl_param(x, 3, 'b', 2, 200, 1e-6) AS b2
FROM resp;

-- person ability (EAP)
SELECT
  irt_2pl_param(x, 3, 'theta', 0, 200, 1e-6) AS theta0,
  irt_2pl_param(x, 3, 'theta', 1, 200, 1e-6) AS theta1,
  irt_2pl_param(x, 3, 'theta', 4, 200, 1e-6) AS theta4
FROM resp;
```

## Validation

- **7/7 unit tests** (`cargo test`): 2PL probability properties, 40-node
  Gauss-Hermite quadrature (`Σw=√π`, `Σwx²=√π/2`), E-step normalization, and
  a full 4×2 fit that recovers sensible `a`, `b` and separates `θ` by correct
  count.
- **Cross-language oracle**: a 300×12 simulation is fitted independently by
  (a) this Rust extension and (b) a pure-`numpy/scipy` MML/EM reference
  (`win/irt_poc/irt_mml.py`). Item parameters agree to within ~0.1–0.5 (the
  residual is **EM multimodality** — the 2PL marginal likelihood has multiple
  local optima; the two optimizers, started from the same initial value, settle
  on different *valid* ones). The **identifiable** quantities agree closely:
  `θ_0 = -1.76` (Rust) vs `-1.71` (oracle), `θ_299 = 2.24` vs `2.33`, and all
  invariant checks pass (θ ordering, `a>0`, `b` sane, ability separation).
- **Simulation recovery** (oracle): on a 200×10 dataset with known `a∈[1,3]`,
  `b∈[-1.5,1.5]`, `θ∈N(0,1)`, the reference recovers `a` (corr 0.93), `b`
  (corr 0.99), `θ` (corr 0.90).

## Caveats

- **EM multimodality**: MML is not globally concave in all item parameters
  jointly; the result is a valid local maximum, but different optimizers
  (or initial values) can land on different ones. For production use, consider
  multi-start or a fixed-θ MLE with standard identification (e.g. `mean(b)=0`).
- **Single model**: 2PL only. 1PL / 3PL / graded response are not implemented.
- **No SEs**: standard errors / fit statistics (S-X, S-P) are not computed.
- **Scale**: with the `N(0,1)` prior the 2PL is identified (no `b[0]=0`
  constraint needed); rescaling `(a,b)` changes `expit(a(θ-b))` unless `θ` is
  also rescaled, which breaks the prior.

## Build

```bash
make configure
make release
# → build/release/irt_ext.duckdb_extension
```

## Self-check

```bash
duckdb -unsigned -noheader -list < test_irt.sql
# each line: <check>\t<true/false>  — all should be true
```
