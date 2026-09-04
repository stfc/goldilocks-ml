# Goldilocks CSLR quantile forest for k-point mesh index

A quantile random forest that answers one question: how far up Goldilocks
Core's ordered ladder of k-point meshes does this crystal have to go.

It returns a **rung on that ladder**, not a spacing in reciprocal space. Rung 0
is the Gamma-only (1, 1, 1) mesh, rungs are **0-based**, and the ladder the
labels come from was enumerated to 50 k-points per reciprocal-lattice axis.

This is a different quantity from the k-distance predicted by record
`q3bye-wep37`, which is a spacing in inverse angstroms that a consumer converts
into a mesh through the reciprocal lattice. A k-index needs no conversion: the
answer *is* the position in a table the consumer already has.

## Files

- `k_index_qrf.pkl`: the fitted quantile forest.
- `calibration.json`: the separately recorded interval calibration.
- `model.json`: feature contract, all 174 column names in order, target
  contract, hyperparameter search, calibration, and the rule that decides which
  number the model publishes. Written by the run that fitted the model.

Everything needed to run the model is here: download this record and nothing
else. This model requires no support files from any other record.

## What it predicts, and what it publishes

**These are not the same thing, and the difference is the point of this
release.**

The forest estimates a whole distribution over rungs. A model has to return one
number, and on this ladder the two directions of being wrong cost very
different things:

```text
a rung too low    the mesh is too coarse; the calculation is under-converged
                  and the answer is wrong without looking wrong

a rung too high   a denser mesh than was needed; it costs machine time
```

Mean absolute error prices those the same. A model selected on it publishes the
middle of its distribution — and read at its median, this forest comes in below
the true rung 29.7% of the time.

So it does not publish its median. It publishes:

```text
rung = round_half_up(q0.90 estimate)    then  + 2  if that rung is 11 or above
```

The quantile level and the band offsets were chosen on the validation split as
the cheapest rule keeping under-prediction at or below 6%, and are recorded in
`model.json` under `decision`. A consumer applies nothing further: the number in
the prediction is the rung to use.

## Training data

PSDI record `d5ds2-64f16`, CC BY 4.0. The record holds 18220 MC3D structures
with Quantum ESPRESSO k-mesh convergence studies; 17757 of them converged and
carry a label, and only those were used. The other 463 have a structure and no
answer.

Labels run from rung 0 to rung 41 and are heavily skewed:

```text
rungs 0-3     50.8% of the dataset
rungs 4-10    39.8%
rungs 11-20    8.9%
rungs 21-41    0.6%
```

Five of the rungs below 41 have no example at all.

The split is 70/10/10/10 train/validation/calibration/test, seed 42, grouped by
reduced composition — 15712 groups over 17757 structures, so two polymorphs of
one composition cannot land on opposite sides of it. No group spans more than
one split.

The local snapshot the run consumed is sealed at SHA-256
`66eb62879cb65aef18b3e74d73349831eaa2ebd1ec10de88ab58a77d368e82bd`, and the
training protocol pins that digest.

## Input features

The feature contract is `cslr.v1`: 174 columns, in the order recorded in
`model.json`, matching the extractor Goldilocks Core already uses.

```text
block          columns   what it is
composition        146   132 Magpie element statistics, 6 stoichiometry norms,
                         8 valence-orbital occupations
structure            7   space group, crystal system as an integer,
                         centrosymmetry, symmetry-operation count, density,
                         volume per atom, packing fraction
lattice              7   a, b, c, alpha, beta, gamma, and cell volume
reciprocal          14   reciprocal lengths, angles and volume, metric-tensor
                         invariants, anisotropy ratios
                   ---
total              174
```

No SOAP descriptors and no learned metallicity representation. The column order
is part of the contract; a loader that reorders the columns produces wrong
answers silently, so read the names out of `model.json` rather than assuming
them.

Matminer cannot compute a packing fraction for an element with no tabulated
atomic radius. 45 of the 17757 training structures contain He, Ne, Kr or Xe, and
their whole 7-column structure block was written as zeros. A consumer must apply
the same fallback, deterministically, or those crystals will be described
differently at inference than they were at training.

## Measured performance

On the 1775-structure test split, which was scored once after the quantile
level, the band offsets and the interval calibration were all settled on other
splits:

```text
                                too coarse   mean excess   exact rung     mae      r2
this model, as published             0.044         +2.42        0.163   2.634   0.042
the same forest read at its median   0.297         -0.30        0.438   1.118   0.729
baseline: always rung 3              0.492         -1.77        0.184   2.707  -0.208
```

Read the first two columns. The floor was set at 6% on validation, and the test
split, which chose nothing, comes in at 4.4% at a mean of 2.42 rungs more mesh
than was needed.

MAE and r2 are not measuring accuracy here. They price a deliberate bias as if
it were error. The correlation with the truth is 0.828 for the published value
against 0.858 for the median, so the model's ability to rank structures by the
mesh they need is intact, and what changed is where the number sits relative to
the truth. Of the published mean squared error of 14.54, some 5.88 is the
deliberate lift: a perfect rule sitting exactly 2.42 rungs above every true rung
would itself score r2 = 0.612 against a target variance of 15.17.

The median row is a diagnostic about the estimator, recomputed from this
artifact. No run scores it, because the model does not publish a median.

Banded on the rung this model publishes — the conditional the rule controls:

```text
published rung   count   too coarse   mean excess
below 6            954        0.050         +1.02
6 to 10            444        0.036         +2.40
11 and above       377        0.037         +6.01
```

## Scope and limitations

**The top of the ladder is not reliable.** Banded on the *true* rung rather than
the published one, structures that genuinely need rung 11 or above are
under-converged 14.6% of the time, well outside the 6% this model otherwise
honours. The band offsets brought that down from 27.4% and cannot go further:
they lift the structures the model *places* in the top band, and these are the
ones it does not. Treat a prediction for a structure you expect to be demanding
as a lower bound, and check convergence directly.

This is a data limit before it is a modelling one. A quantile forest returns a
quantile of labels it saw in a leaf, so it cannot reach rungs the training set
barely contains, and 0.6% of the labels sit above rung 20.

**The contract is 0-based.** A consumer that feeds this number into a 1-based
ladder gets a mesh one step too coarse, every time, silently. `model.json`
declares `goldilocks.k_index.ladder_0based.max50.v1` for exactly this reason.

**The ladder must be the same one.** These rungs index the mesh table used by
record `d5ds2-64f16`, enumerated to 50 k-points per axis. A differently
constructed ladder gives the same integers a different meaning.

**No metallicity information.** k-point density is a question about the Fermi
surface, and none of the 174 columns knows whether the crystal is a metal. That
is a deliberate choice to match Core's existing extractor, not evidence that it
does not matter.

**Applicability.** Trained on MC3D bulk crystals with Quantum ESPRESSO
self-consistent-field settings. Nothing here has been checked on surfaces,
molecules, low-dimensional systems, or other codes and pseudopotential families.

## Runtime and safe loading

`k_index_qrf.pkl` is a Python pickle. **Unpickling executes code.** Verify the
digest in `manifest.json` before loading it, and load it only from this record:

```bash
shasum -a 256 k_index_qrf.pkl
# bbabd7ed9be6a229251f145984b055232af68cfc9cf37e83b0f6c2c4ca5bc5e4
```

`model.json` pins the same digest, and the `goldilocks-ml` loader refuses to
unpickle a file that does not match it.

The estimator was fitted with these versions, and pickles are not portable
across incompatible ones:

```text
python             3.13
scikit-learn       1.7.2
sklearn-quantile   0.1.1
numpy              2.5.2
matminer           0.10.1
pymatgen           2026.5.4
```

matminer and pymatgen are load-bearing for the *features*, not the estimator: a
different Magpie table or a different symmetry finder changes the 174 columns,
and therefore the answer.

The estimator returns 11 quantile levels, in the order recorded in `model.json`
under `levels`. The published rung comes from the `decision` block; the 5th and
95th percentiles are the interval.

## The interval is a diagnostic, not the answer

`model.json` records a 90% interval and a split-conformal calibration whose
correction came out at exactly 0.0 — the raw interval already covered 94.8% of
the calibration split.

That zero is a property of the target. A quantile forest returns quantiles of
integer labels, so 22.5% of the calibration set has a nonconformity score of
exactly zero, and the correction lands inside that atom. Conformal calibration
cannot make a fine adjustment to an interval whose endpoints are whole rungs.

Test coverage is 0.953 overall but 0.805 for structures whose true rung is 11 or
above, at a mean width of 10.5 rungs. Do not read the interval as a uniform
guarantee. The published rung, not the interval, is what carries the stated 6%
floor.

## Reproducibility

Fitted by a versioned training protocol in stfc/goldilocks-ml, from a sealed
dataset snapshot, with a fixed seed on CPU.

**The run reproduces bit for bit.** A second run over the same snapshot, reusing
the recorded split assignment, produced an identical estimator pickle, model
record, metrics and predictions — the same SHA-256 as the file in this record.
The decision rule reproduces with it: same quantile level, same band offsets.

`model.json` carries the hyperparameter search that chose `min_samples_leaf`,
the validation scores of every candidate, and the trials behind the decision
rule, so the choices can be inspected rather than taken on trust.
