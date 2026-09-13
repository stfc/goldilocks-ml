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

Three changes landed here in sequence, on the same fitted forest.

The first retrained on `52713-55d86` -- the record that already carries the
corrected, floor-based 1-based ladder -- rather than shifting the previous
version's `d5ds2-64f16` labels by one, and reissued the target contract as
`ladder_1based.v2`: the superseded `ladder_1based.max50.v1` declared a
50-per-axis enumeration cap that was never real, since the ladder was always
built to a minimum k-distance.

The second changed what quantile is published (issue #71). Ported forward
unchanged, the previous publishing policy -- hold under-prediction at 6% --
landed on q0.95 with no band lift left to give on this record, at a mean cost
of almost 3 rungs of extra mesh per recommendation. That version published
q0.6 instead, with no band lift at all.

The third replaced the single shared quantile with a level chosen per band.
A single q0.6 for every structure means the easy majority and the hard tail
share one number: measured on q0.6, structures that truly need rung 12 or
above under-predicted 76.7% of the time while the bulk of the dataset (rung
< 7) already sat at 12.1%. This version cuts on the model's own median and
picks a separate published level per band, so the safety margin the hard tail
needs is not paid by the easy majority. See "What it predicts, and what it
publishes" below for the trade this makes.

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
middle of its distribution -- and read at its exact median (q0.5), this forest
comes in below the true rung 29.9% of the time on the held-out test split.

A single shared quantile cannot serve both halves of that distribution well.
The previous policy published q0.6 for every structure -- close to the median,
cheap on average, but measured at 76.7% under-prediction for structures that
truly need rung 12 or above, against 12.1% for the easy majority (rung < 7)
at the same level. Raising that one shared level to fix the hard tail would
have taxed the easy majority with mesh it does not need, which is exactly the
trade the q0.6 policy was adopted to avoid in the first place (issue #71).

This record instead publishes a different level per band of its own median:

```text
band = the model's own round_half_up(q0.5 estimate)
rung = round_half_up(q<level for that band> estimate)
```

```text
band (on the model's own median)   published level
below rung 7                       q0.90
rung 7 and above                   q0.975
```

Each level is the cheapest one that keeps *that band's own* under-prediction
at or below 5% on the validation split, not the dataset average -- a rare
band's failures no longer hide behind a common band's successes. The cut and
the levels are recorded in `model.json` under `decision`. A consumer applies
nothing further: the number in the prediction is the rung to use.

This is a deliberate trade, not a free improvement: under-prediction on the
test split fell to 4.2% overall, and to 14.7% for the hardest structures --
those genuinely needing rung 12 or above -- down from 76.7% under the
previous policy. In exchange, the model recommends denser meshes on average
than a policy tuned purely for average accuracy would. See "Measured
performance" below for what that costs, and "Scope and limitations" for what
it does not fix.

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

On the 1775-structure test split, scored once after the quantile level and
the interval calibration were both settled on other splits:

```text
                                              this model   q0.6 (previous policy)
mesh dense enough                                  95.8%                    77.1%
dense enough for the hardest structures            85.3%                    23.3%
(rung 12 and above)
```

Banded on the rung this model actually publishes -- the conditional a
consumer has in hand:

```text
published rung   share of predictions   mesh dense enough
below 7                          54.5%                  95.4%
7 to 11                          19.7%                  95.7%
12 and above                     25.8%                  96.5%
```

Under-prediction is now roughly flat across the published range, rather than
concentrated in the hardest structures, which is the point of choosing a
level per band instead of one shared level. The trade is a higher average
mesh size than a policy tuned purely for average accuracy would give: this
version recommends meshes several rungs denser on average than the previous
one, most noticeably for structures already predicted to need rung 12 or
more. `model.json` records the validation MAE achieved by each band; the full
metric suite behind this trade is recorded in the training run in
`stfc/goldilocks-ml` for anyone who wants to audit it in detail.

## Scope and limitations

**Treat a prediction for a demanding structure as a lower bound, not a
guarantee.** Bands are cut on the model's own estimate, not the true
difficulty, so a structure that turns out harder than expected can still be
served a level meant for an easier case. On the test split this still affects
14.7% of the structures that genuinely need rung 12 or above -- well down
from earlier policies, but not zero. If you expect a structure to be
demanding, check convergence directly rather than treating the prediction as
final.

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

**Very small, simple cells may get a denser recommendation than common
practice.** A small unit cell has a large reciprocal lattice, which this
model reads as evidence of the hard, high-rung tail regardless of how simple
the chemistry actually is -- primitive silicon, for example, is recommended a
considerably denser mesh than typical practice uses. Goldilocks Core applies
an additional ceiling on mesh density based on the structure's metallicity to
keep this bounded; a consumer using this model directly, outside Core, should
apply a similar sanity check for small, simple cells.

**Confirm a published rung is within your structure's own ladder before
indexing into it.** For a very small number of simple, high-symmetry cells,
the ladder enumerated down to the 0.03 Å⁻¹ floor is short enough that a
published rung can exceed it, so indexing directly into a per-structure
ladder table without this check can raise an out-of-range error for those
cells. Goldilocks Core guards against this by clamping to the densest
available mesh for that structure; a consumer using this artifact directly
should add the same check.

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
interval, is what carries the stated under-prediction floor -- 5% per band on
validation under this policy, not a single dataset-wide 25% or 6%.

## Reproducibility

Fitted by a versioned training protocol in stfc/goldilocks-ml, from a sealed
dataset snapshot, with a fixed seed on CPU.

**The run reproduces bit for bit.** This version and the one it replaces share
the exact same estimator pickle SHA-256, because the two are the same forest:
switching from one shared quantile to a level chosen per band changed
`model.json` alone. Refitting from scratch over the same snapshot and split
assignment reproduces that same pickle again.

`model.json` carries the hyperparameter search that chose `min_samples_leaf`
and its validation score for every candidate, plus the decision rule's own
validation MAE for each band, so the choices can be inspected rather than
taken on trust.
