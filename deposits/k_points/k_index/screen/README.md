# Goldilocks dense-mesh screen

Ranks candidate crystal structures by how likely they are to need a dense
k-point mesh, so that machine time spent extending a k-mesh dataset goes where
the labels are scarce.

This model does not advise a calculation. It answers a question about a
dataset: given thousands of structures nobody has run yet, which ones are worth
running. Its output is an ordering.

## What it predicts

Whether a structure's converged mesh sits at **rung 11 or above** on Goldilocks
Core's ordered ladder of k-point meshes, where rung 0 is the Gamma-only
`(1, 1, 1)` mesh. In the training record 9.5% of structures sit at or above it.

The rung is part of what the answer means, so it appears in the target
contract, `goldilocks.k_index_dense.ladder_0based.ge11.v1`. A screen cutting at
a different rung answers a different question and is not a drop-in replacement.

## How to read it

The model returns a score between 0 and 1. **Sort by it and take as many as the
budget allows.** The class label attached to each prediction comes from a plain
0.5 cut and is the lesser half of the output; no single operating point was
tuned, because what decides the cut is how much machine time there is.

Measured on the validation split, taking the top fraction of a ranked pool:

```text
  take        n     of them dense    precision    recall    vs random
  top  1%     18         17            0.944       0.099       9.8x
  top  2%     36         31            0.861       0.181       8.9x
  top  5%     89         70            0.787       0.409       8.2x
  top 10%    178        116            0.652       0.678       6.8x
  top 15%    266        144            0.541       0.842       5.6x
  top 25%    444        158            0.356       0.924       3.7x
```

Fractions, not counts, because a campaign ranks a pool of its own size: taking
2000 of 13175 candidates is taking the top 15%, and that row is what applies.

## How good it is

On 1775 structures held out of training and of every choice made while
building it:

```text
  ROC-AUC                   0.960
  PR-AUC                    0.752      baseline 0.050
  Matthews correlation      0.683      baseline 0.000
  balanced accuracy         0.888
  recall                    0.831
  precision                 0.619
```

The baseline is the majority class, which is right 90.3% of the time and finds
no dense structure at all. PR-AUC against a 0.050 base rate is the number that
speaks to screening: precision-recall, not accuracy, is what a ranking is
judged on when the positive class is one structure in ten.

At the nominal 0.5 cut the model calls 231 of 1775 test structures dense and is
right about 143 of them, missing 29 of the 172 that are. Read as a ranking
rather than a cut, those errors matter less than the table above suggests: a
missed structure that still scores highly is still acquired.

## When not to use it

- **It is not a k-mesh recommendation.** It cannot tell you what mesh to run,
  only that a structure probably needs a dense one. For the mesh itself, use the
  k-index forest.
- **It is not calibrated as a probability.** The scores order structures well;
  nothing here shows that a score of 0.7 means a 70% chance.
- **Outside MC3D it is untested.** It learned from MC3D bulk crystals with
  Quantum ESPRESSO SCF settings. Surfaces, molecules and other codes are not
  covered.
- **The rung is 0-based and indexes this particular ladder.** The same integer
  means something else on a ladder built differently.

## Training data

PSDI record `d5ds2-64f16`, CC BY 4.0 — the same 17757 MC3D structures with
converged Quantum ESPRESSO k-mesh studies that the k-index forest learned from.
The snapshot records the rung that was measured; this model is trained on the
coarser question derived from it, so the two models cannot drift apart.

Split 70/10/10/10 and grouped by reduced composition, so polymorphs of one
composition cannot straddle the split, and stratified so each split carries the
same dense fraction:

```text
  split          structures    dense     share
  train              12429      1166      9.4%
  validation          1776       171      9.6%
  calibration         1777       174      9.8%
  test                1775       172      9.7%
```

## Inputs

The 174-column `cslr.v1` contract: 146 composition descriptors (Magpie,
stoichiometry, valence orbital), 7 structure descriptors, 7 direct-lattice
columns and 14 reciprocal-lattice columns. The same extractor the k-index
forest uses. No learned representation, no other model's output, no artifact
dependencies — every column is a deterministic function of the crystal.

## What is in the record

`k_index_screen.pkl` is a scikit-learn random forest of 500 trees, fitted with
balanced class weights because the positive class is under a tenth of the data
and an unbalanced fit learns to answer "sparse" and rank nothing.

`model.json` records the feature contract and all 174 column names, the target
contract, the rung the classes were derived at, the measured ranking quality
above, the hyperparameters, and the artifact digest. `ranking.json` carries the
ranking record on its own.

Loading unpickles a file, which executes code, so `model.json` pins the
estimator's SHA-256 and the runtime refuses to load an artifact that does not
match it. It also refuses one that does not record the rung it screens at, or
that carries no measured ranking quality: a screen that cannot say what a
budget buys is not usable as one.
