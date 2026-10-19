# Mechanism Selection for Private Matrix Completion

Code for the paper *Mechanism Selection for Private Matrix Completion*

Differentially private matrix completion by input perturbation privatizes every observed
rating once with an entrywise mechanism and then reconstructs the matrix from the release.
This repository implements the data-free rule that picks that mechanism. Each candidate
channel is summarized by two public quantities, its attenuation `alpha` and its
conditional noise `V`; the rule returns the candidate with the smallest effective noise
`sigma2_eff = V / alpha^2`, computed from the rating grid and the budget alone, so the
choice spends no privacy budget. The release is then corrected publicly,
`Xtilde = (Y - c) / alpha`, and completed by a bias-aware factorization fitted with
alternating least squares (ALS).

The repository contains every script used to compute the numbers, tables and figure of
the paper, and nothing else.

## Contents

```
dpmc/                      library
  channels.py              the twelve entrywise eps-DP mechanisms, their channel statistics
                           (alpha, V, sigma2_eff, c), samplers and the selection rule
  staircase.py             staircase mechanism retuned for squared error
  closed_forms.py          closed-form sigma2_eff, the two crossover budgets, the
                           reference-law tilt sweep and the law-free dominance condition
  optimal.py               LP lower bound over the affine-mean class, RR-on-bins, and the
                           loader of the tabulated channels used in the runs
  optimal_tables.txt       transition matrices of the LP-optimal channel and RR-on-bins on
                           both grids at the seven budgets, exactly as used end to end
  mf.py                    bias-aware matrix factorization by ALS, one-user half-step
  nuclear.py               nuclear-norm reconstruction by ADMM (ablation)
  dpsgd.py                 DP-SGD and pure-eps DP-GD baselines, Renyi-DP accountant
  data.py                  MovieLens-25M, Amazon and Goodreads loaders, cached blocks,
                           splits, RMSE, Recall@10 and NDCG@10
  selection.py             deterministic seeding and per-seed K-fold CV on privatized values
  results.py               reading result files
  stats.py                 paired t-test, confidence intervals, Spearman correlation
experiments/               one driver per experiment; each writes one JSON file
  run_dpmc.py              DPMC for one mechanism (or the rule), given budgets and seeds
  run_pythia.py            budget-spending, data-dependent selection (Pythia style)
  run_loo_bias.py          leave-one-out measurement of the ALS design correlation
  run_leverage.py          mean of t(x)^2 under the leverage law of the ALS half-steps
  run_timing.py            wall-clock cost of every stage
  run_reference.py         non-private ceiling, global mean, bias-only model, on every block
  run_dpsgd.py             DP-SGD (Gaussian) and DP-GD (pure eps, Laplace)
  run_nuclear.py           nuclear-norm reconstruction of the same release
  run_synthetic.py         synthetic sweep over the number of rating levels k
  make_optimal_tables.py   solves the LP and RR-on-bins programs
analysis/                  turns result files into the paper's tables and figure
scripts/
  fetch_data.sh            downloads both datasets
  build_blocks.py          builds the evaluation blocks
  run_all.sh               runs every experiment and every analysis script
```

## Requirements

Python 3.9 or later with NumPy, SciPy, pandas and matplotlib (`requirements.txt`).
SciPy is needed only for the linear program in `dpmc/optimal.py`; everything else uses
NumPy. No GPU is used.

```
python3 -m pip install -r requirements.txt
```

## Data

All three datasets are public and are downloaded by `scripts/fetch_data.sh`:

| dataset | source | levels | range | block used |
|---|---|---|---|---|
| MovieLens-25M | https://files.grouplens.org/datasets/movielens/ml-25m.zip | 10 half stars | [0.5, 5] | 2000 most active users x 2000 most rated items |
| Amazon Video Games (5-core) | https://mcauleylab.ucsd.edu/public_datasets/data/amazon_2023/benchmark/5core/rating_only/Video_Games.csv.gz | 5 stars | [1, 5] | 2000 x 2000 (1969 users remain), 48,196 ratings |
| Goodreads comics and graphic novels | https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads/byGenre/goodreads_interactions_comics_graphic.json.gz | 5 stars | [1, 5] | 2000 x 2000, 393,613 explicit ratings |
| MovieLens-25M, random users | as MovieLens-25M | 10 half stars | [0.5, 5] | 2000 users drawn uniformly at random, their 2000 most rated items, 224,266 ratings |

`scripts/build_blocks.py` keeps the `m` most active users and then the `n` most rated items
among them (for the random-user block, `m` users drawn with seed 0) and caches each block
as `cache_<dataset>_<m>x<n>_s0.npz`. Goodreads ratings of 0 mean "shelved, not rated" and
are dropped. Every split is a
seeded permutation of the observed entries: 10% test, the rest training.

## Quick checks (seconds, no data)

```
python3 -m dpmc.optimal              # every tabulated channel is stochastic and within budget
python3 -m dpmc.stats                # t-test p-values against t-table critical values
cd analysis
python3 channel_tables.py            # effective-noise table, crossovers vs k, tilt sweep, law-free bounds
python3 optgap_table.py              # rule vs the exact LP optimum, RR-on-bins statistics
python3 family_check.py              # certified full-family frontier, k = 3..50 (minutes)
```

These reproduce the data-free results of the paper exactly: the effective noise of the
mechanisms in the effective-noise table on both grids, the crossover budgets (1.0237 and 4.2784 on the 10-level
grid, 1.2220 and 2.4853 on the 5-level grid, and the table over k), their invariance to the
width of the rating scale, the range of the second crossover over the tilt family, the
law-free boundaries (0.617, 2.40, 7.15), and the LP lower bound with its excess.

## Reproducing everything

```
bash scripts/run_all.sh
```

downloads the data, builds the blocks, runs every experiment and writes every table under
`results/`. `JOBS` sets the number of parallel processes and `THREADS` the BLAS threads per
process (defaults: one process per five cores, five threads each). `SCALING=1` also runs
the 4000 x 4000 and 8000 x 8000 MovieLens blocks. Finished result files are skipped, so
the script can be stopped and restarted.

| paper | driver(s) | analysis |
|---|---|---|
| effective-noise table, crossovers vs k, tilt range, law-free bounds | none (closed form) | `channel_tables.py` |
| cost of the short list against the LP optimum, RR-on-bins statistics | none (LP) | `optgap_table.py` |
| test RMSE of DPMC and every fixed mechanism | `run_dpmc.py` for each mechanism and budget; `run_dpsgd.py`; `run_reference.py` | `main_table.py` |
| per-budget agreement, rank correlation, pairwise agreement, exchange brackets, rule vs fixed defaults | `run_dpmc.py` | `agreement.py` |
| LP-optimal channel and RR-on-bins end to end | `run_dpmc.py --mech lpopt / rrbins` | `optimal_endtoend.py` |
| staircase mechanism | `run_dpmc.py --mech stair` | `staircase_check.py` |
| budget-spending selection | `run_pythia.py` | `pythia_table.py` |
| ALS design correlation (leave-one-out) | `run_loo_bias.py` | `loo_table.py` |
| mean of t^2 under the leverage law of the ALS half-steps | `run_leverage.py` | printed by `run_all.sh` |
| Goodreads at eps = 1, one-bit release refitted at ridge 300 | `run_dpmc.py --lams 300` | `main_table.py --runs results/diag` |
| wall-clock cost | `run_timing.py` | `timing_table.py` |
| correction, reconstruction and mechanism ablations | `run_dpmc.py --debias 0`, `run_nuclear.py` | `ablation_table.py` |
| synthetic sweep over k | `run_synthetic.py` | `synthetic_table.py` |
| figure (prediction beside measurement) | `run_dpmc.py` | `make_figure.py` |

A single configuration, for example the one-bit mechanism on MovieLens at three budgets
and ten seeds:

```
python3 experiments/run_dpmc.py --dataset ml-25m --data_dir data/ml --mech shin \
    --eps_list 0.1,1,8 --seeds 0-9 --out results/shin_ml.json
```

`--mech rule` applies the selection rule at every budget instead of fixing a mechanism.

## What each driver does

**`run_dpmc.py`.** For every budget and seed: split the ratings (10% test), privatize the
training ratings with the mechanism, apply the public correction, choose `(rank, lam)`
from ranks {8, 16} and lam in {1, 3, 10, ..., 100000} by 5-fold cross-validation on that
seed's corrected privatized values, refit 15 ALS iterations on all of them, and score test
RMSE on the held-out true ratings together with Recall@10 and NDCG@10 (all items outside the
user's training ratings ranked, held-out ratings of at least 4 relevant). Selection is
independent at each budget and seed and never reads a true rating, so every budget is one
eps-DP release followed by post-processing. Below its switching budget the hybrid is the
one-bit randomizer, and it is seeded and run as such, so the two releases are identical.

**`run_pythia.py`.** Spends `f * eps` on a Laplace histogram of the training ratings
(l1 sensitivity 2 under the rating-level relation, the observation pattern being public),
selects by effective noise under the estimated law at the remaining budget `(1 - f) eps`,
releases with the selected mechanism at that budget and reconstructs and scores exactly as
`run_dpmc.py`.
The two releases compose to `eps`.

**`run_loo_bias.py`.** For one mechanism and budget and 50 sampled users, fits ALS on the
full release and on the release with the user's row removed (same initialization), and
compares the user's half-step under the two designs. It records the gap, the deterministic
bound of the leave-one-out reduction, the residual of the exact identity behind it (at
machine precision), and the privacy-noise standard deviation that the risk identity
assigns to the leave-one-out half-step.

**`run_leverage.py`.** Fits the biased factorization to the true seed-0 training ratings
at several ridge weights and, for every rating, computes its weight in the risk identity of
the user and of the item half-step, `w = a^T G^2 a` with `G = (A^T A + lam I)^{-1}`. It
reports the mean of `t(x)^2` under those weights next to its marginal and uniform values;
the one-bit and piecewise effective noises are affine in that mean, so it decides which of
the two the risk identity favors on the block.

**`run_timing.py`.** Times the selection rule, the Pythia-style selector, the release, the
correction and one ALS fit on each block, and the LP and RR-on-bins programs on both grids.

**`run_dpsgd.py`.** Full-batch gradient descent with per-example clipping. With
`--noise gauss` the Gaussian mechanism is accounted by Renyi DP and reported as
`(eps, 1e-6)`-DP; with `--noise laplace_pure` the noise is Laplace calibrated to the l1
sensitivity of the clipped per-rating gradient, `C sqrt(2(d + 1))`, giving pure eps-DP.
Learning rate and epoch count are selected on true validation ratings, a non-private
advantage given to these baselines.

## Running time

Measured by `experiments/run_timing.py` on one NVIDIA Grace CPU node (aarch64, Python 3.11,
NumPy 2.2, 8 BLAS threads), at eps = 1 with the one-bit mechanism, rank 16, ridge 300 and
15 ALS iterations:

| block | training ratings | rule | Pythia-style histogram | release | correction | one ALS fit |
|---|---|---|---|---|---|---|
| MovieLens 2000 x 2000 | 1,632,685 | 0.06 ms | 50 ms | 14 ms | 1.1 ms | 5.3 s |
| MovieLens 4000 x 4000 | 3,693,114 | 0.06 ms | 115 ms | 35 ms | 2.7 ms | 12.9 s |
| MovieLens 8000 x 8000 | 6,692,336 | 0.06 ms | 214 ms | 75 ms | 7.5 ms | 25.3 s |
| Amazon 1969 x 2000 | 43,376 | 0.05 ms | 0.8 ms | 0.4 ms | 0.03 ms | 0.85 s |

Cross-validation performs 110 fits per budget (5 folds, 2 ranks, 11 ridge weights) on four
fifths of the training data, so selection on the MovieLens block costs roughly
110 x 5.3 s, about ten minutes per budget and mechanism, and about a minute and a half on
Amazon; the five scored fits come on top. The linear program of `dpmc/optimal.py` takes 12
to 20 s per budget on the 10-level grid and 3 to 6 s on the 5-level grid, and RR-on-bins
under 10 ms; neither reads data.

## Determinism

Every random draw comes from a NumPy generator seeded by integers: the split by the seed,
the privatization by `(seed, round(1000 eps), sha256(mechanism) mod 1000)`, the
cross-validation folds by `9000 + seed`, and the factor initialization by 0. Python's
built-in string hash is never used, so reruns are bit-identical on the same NumPy version.
The LP-optimal and RR-on-bins channels are read from `dpmc/optimal_tables.txt`, the exact
tables used for the reported runs (12 significant digits); `make_optimal_tables.py`
re-solves them and reproduces their effective noise, but a different LP solver version may
return a different optimal vertex of the same value, which is why the tables are shipped.

Every number in the paper was produced by these drivers and analysis scripts, run as in
`scripts/run_all.sh`.

## Output format

Each `run_dpmc.py` result is a JSON object with `meta` (arguments and data shape),
`channel` (alpha, V, sigma2_eff and the intercept at every budget), `select` (the full
cross-validation table and the chosen rank and ridge weight, per budget and seed) and
`records` (one per budget and seed: mechanism, eps, seed, test RMSE, Recall@10, NDCG@10,
rank, ridge weight, fit time). The other drivers follow the same layout.

## License

MIT, see `LICENSE`.
