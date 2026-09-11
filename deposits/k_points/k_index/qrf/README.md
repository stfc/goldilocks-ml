# Goldilocks CSLR quantile forest for k-point mesh index

A quantile random forest that answers one question: how far up Goldilocks
Core's ordered ladder of k-point meshes does this crystal have to go.

It returns a **rung on that ladder**, not a spacing in reciprocal space. Rung 1
is the Gamma-only `(1, 1, 1)` mesh, rungs are **1-based**, and the ladder was
enumerated down to a minimum k-distance of **0.03 inverse angstrom** rather
than a fixed count per axis.

This is a different quantity from the k-distance predicted by record
`q3bye-wep37`, which is a spacing in inverse angstroms that a consumer converts
into a mesh through the reciprocal lattice. A k-index needs no conversion: the
answer *is* the position in a table the consumer already has.

## This is a new version of the same record

The superseded version of this record was trained on `d5ds2-64f16` and
published a 0-based rung, with the ladder enumerated to 50 k-points per axis.
That enumeration bound was never real: the ladder was always built to a
minimum k-distance, and `50` was a name for something else entirely. This
version fixes both things at once, retraining on `52713-55d86` -- the record
that already carries the corrected ladder -- rather than shifting the old
labels by one.

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
middle of its distribution -- and read at its median, this forest comes in
below the true rung 29.9% of the time on the held-out test split.

So it does not publish its median. It publishes:

```text
rung = round_half_up(q0.95 estimate)
```

with no band lift needed on top: at this level every band already honours the
floor on its own (see below). The quantile level was chosen on the validation
split as the cheapest one keeping under-prediction at or below 6%, and is
recorded in `model.json` under `decision`. A consumer applies nothing further:
the number in the prediction is the rung to use.

⚠️ The publishing *policy* itself -- whether a 6% floor is the right one to buy
with this much extra mesh -- is being reconsidered separately (issue #71). This
release only moves the ladder and corrects the contract; it keeps the
superseded record's policy so the two effects are not measured at once.

## Training data

PSDI record `52713-55d86`, CC BY 4.0. The same 17757 MC3D structures that
`d5ds2-64f16` held, relabelled under the corrected ladder -- **not a uniform
+1 shift**: the 1-based enumeration also drops a small number of repeated
meshes the old per-axis count did not, so a handful of structures move by a
different amount or not at all.

Convergence is judged on **total energy alone**: the first of three
consecutive rungs whose energies agree within **1 meV/atom**. No force
criterion is applied.

Labels run from rung 1 to rung 42 and are heavily skewed, the same shape the
superseded record had:

```text
rungs 1-4     50.8% of the dataset
rungs 5-11    39.8%
rungs 12-21    8.9%
rungs 22-42    0.6%
```

The split is 70/10/10/10 train/validation/calibration/test, seed 42, grouped by
reduced composition -- 15712 groups over 17757 structures, so two polymorphs of
one composition cannot land on opposite sides of it. No group spans more than
one split. Because the underlying structures and the split are unchanged from
the superseded record, this is the same partition, on the same crystals, with
corrected labels.

The local snapshot the run consumed is sealed at SHA-256
`5a6cffc002123f9d6f2f1f34be8622f32a6446b64f84c7e4d5601043392b4bea`, and the
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
atomic radius. Structures containing He, Ne, Kr, Ar or Xe have their whole
7-column structure block written as zeros. A consumer must apply the same
fallback, deterministically, or those crystals will be described differently
at inference than they were at training.

## Measured performance

On the 1775-structure test split, which was scored once after the quantile
level and the interval calibration were both settled on other splits:

```text
                                too coarse   mean excess   exact rung     mae      r2
this model, as published             0.035         +2.96        0.097   3.124  -0.153
the same forest read at its median   0.299         -0.29        0.437   1.124   0.729
```

Read the first two columns. The floor was set at 6% on validation, and the test
split, which chose nothing, comes in at 3.5% at a mean of 2.96 rungs more mesh
than was needed -- more than the superseded record's 2.42 rungs, because the
same 6% floor lands on a higher quantile level (0.95, against 0.90) once the
ladder and its labels change.

MAE and r2 are not measuring accuracy here. They price a deliberate bias as if
it were error, and read at its median this forest is essentially as accurate as
the superseded one was (test MAE 1.124 against 1.118, exact rung 43.7% against
43.8%): what changed is not the estimator's skill, it is how the safety-floor
policy responds to the corrected labels.

Banded on the rung this model publishes -- the conditional the rule controls:

```text
published rung   count   too coarse   mean excess
below 7             796        0.026         +1.63
7 to 11              512        0.031         +3.15
12 and above         469        0.058         +4.85
```

## Scope and limitations

**The top of the ladder is not reliable, and is measurably harder here than on
the superseded record.** Banded on the *true* rung rather than the published
one, structures that genuinely need rung 12 or above -- the same physical cut
the superseded record's rung 11 named -- are under-converged 17.8% of the
time, against 14.6% at that cut on the superseded record. Treat a prediction
for a structure you expect to be demanding as a lower bound, and check
convergence directly.

This is a data limit before it is a modelling one. A quantile forest returns a
quantile of labels it saw in a leaf, so it cannot reach rungs the training set
barely contains, and 0.6% of the labels sit above rung 21.

**The contract is 1-based.** A consumer that feeds this number into a 0-based
ladder gets a mesh one step too dense, every time, silently. `model.json`
declares `goldilocks.k_index.ladder_1based.v2` for exactly this reason, and
records the ladder's floor (`min_k_distance = 0.03`) as a structured field
rather than encoding it into the contract name.

**The ladder must be the same one.** These rungs index the mesh table used by
record `52713-55d86`, enumerated down to a minimum k-distance of 0.03 inverse
angstrom. A differently constructed ladder gives the same integers a different
meaning.

**No metallicity information.** k-point density is a question about the Fermi
surface, and none of the 174 columns knows whether the crystal is a metal. That
is a deliberate choice to match Core's existing extractor, not evidence that it
does not matter.

**Applicability.** Trained on MC3D bulk crystals with Quantum ESPRESSO
self-consistent-field settings. Nothing here has been checked on surfaces,
molecules, low-dimensional systems, or other codes and pseudopotential
families.

**Small cells, made worse by this release.** The known over-prediction on
structures with few training analogues (see the model index) is not merely
unaffected here -- it is larger, because the decision level moved from 0.90 to
0.95 with no band lift on top. Primitive diamond silicon (2 atoms) now
predicts rung 34, a (34, 34, 34) mesh of 39304 k-points, against rung 26 and
19683 k-points on the superseded record and roughly 512 in common practice.
This is a known, tracked limitation (issue #68 in stfc/goldilocks-ml), not
something this release set out to change -- but it is a real cost of porting
the safety-floor policy forward unchanged, and worth weighing when #71
reconsiders that policy.

## Runtime and safe loading

`k_index_qrf.pkl` is a Python pickle. **Unpickling executes code.** Verify the
digest in `manifest.json` before loading it, and load it only from this record:

```bash
shasum -a 256 k_index_qrf.pkl
# 42dc3ee1d7973ca2f299ef0155d0e39f4fd034d45efb4595182ff6696600e0f7
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
correction came out at exactly 0.0 -- the raw interval already covered 95.3% of
the test split.

That zero is a property of the target. A quantile forest returns quantiles of
integer labels, so a meaningful fraction of the calibration set has a
nonconformity score of exactly zero, and the correction lands inside that atom.
Conformal calibration cannot make a fine adjustment to an interval whose
endpoints are whole rungs.

Do not read the interval as a uniform guarantee. The published rung, not the
interval, is what carries the stated 6% floor.

## Reproducibility

Fitted by a versioned training protocol in stfc/goldilocks-ml, from a sealed
dataset snapshot, with a fixed seed on CPU.

**The run reproduces bit for bit.** A second run over the same snapshot, reusing
the recorded split assignment, produced an identical estimator pickle -- the
same SHA-256 as the file in this record, and the same one the superseded
version's own retrain would have produced had it used this snapshot, since
nothing about the forest's fit depends on the band edges a later evaluation
pass chose to report.

`model.json` carries the hyperparameter search that chose `min_samples_leaf`,
the validation scores of every candidate, and the trials behind the decision
rule, so the choices can be inspected rather than taken on trust.
