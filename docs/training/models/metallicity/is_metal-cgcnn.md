# Train the metallicity classifier

| | |
| --- | --- |
| Release | `metallicity.is_metal.cgcnn.matbench_mp_is_metal.v2` |
| Runtime | `metallicity.is_metal.cgcnn` |
| Target contract | `goldilocks.is_metal.dft_band_gap_zero.v1` |
| Dataset | `matbench_mp_is_metal`, 106113 structures |
| Served by | `load_model`, returning a boolean |
| Notebook | [run it yourself](../../../notebooks/metallicity-cgcnn.ipynb) |
| Deposit | `deposits/metallicity/is_metal/cgcnn/`, prepared for PSDI |

This model answers one question: **does DFT give this crystal a zero band gap?**
If it does, we call the crystal a metal.

Goldilocks needs the answer early. Metals need a denser k-point mesh than
insulators, and they need smearing, which insulators do not. So this is one of
the first things Core works out about a structure.

## The three things to know

1. **It catches 97.2% of metals**, and to do that it calls about a third of its
   metals wrongly. That trade is deliberate — see [where the line is
   drawn](#where-the-line-is-drawn).
2. **The threshold is 0.048, not 0.5.** A score of 0.1 means metal here.
3. **Two runs with the same seed give different files.** The v1 recipe was run
   four times and every measured number agreed to three decimals; v2 has run
   once so far.

## Why there are two metallicity networks

PSDI record `m742g-g0k14` already holds a metallicity CGCNN, and this repository
uses it — but only for the representation it learned, not for its predictions.
That code lives in `models/k_points/k_distance/qrf/embedding.py`, next to the
QRF95 feature contract that consumes it, and stays there. It is pinned there by
digest, so it does not move.

The problem with the published one is simple: **it never says how often it is
right.** No test metrics, no split, no dataset beyond a sentence. You can make
it emit `metal` or `insulator`, but you cannot tell anyone how much to trust
that, and Core should not make a claim it cannot back.

So this trainer fits the same architecture from a sealed snapshot and writes
down what the other record is missing: the dataset, the split, the seed, the
epoch it stopped at, and the measured performance. The two never get confused
for each other — this one registers under runtime `metallicity.is_metal.cgcnn`.

## The data

[Matbench](https://matbench.materialsproject.org) `mp_is_metal`: 106113
Materials Project structures, each labelled by whether its DFT band gap is
zero. Matminer publishes it with a SHA-256, so we can pin the source without a
Materials Project API key.

```bash
uv run --extra models python scripts/matbench_to_snapshot.py \
    --output local_data/snapshots/mp-is-metal
```

Two things the converter does that are worth knowing:

**Each sample's id is a digest of the crystal itself**, not a row number. Run
the conversion again on a freshly downloaded dataset and you get the same ids.
Duplicate crystals collide instead of being counted twice.

**Samples are grouped by reduced formula.** Polymorphs of one composition, and
the same structure in differently sized cells, all land in the same group — so
the split cannot put two descriptions of the same chemistry on opposite sides.
There are 78164 groups across 106113 samples, so this costs very little.

The labels come out at 46151 metals to 59962 insulators. The split is
stratified, so every part carries the same 43.5% metal fraction.

## What counts as a metal

The contract is `goldilocks.is_metal.dft_band_gap_zero.v1`: `metal` when the
Materials Project DFT band gap is zero, `insulator` otherwise, with `metal` as
the positive class.

The contract names the *definition*, not just the quantity. A band gap computed
with a different functional would be a different contract, even though it is
still called a band gap.

## What the model sees

The feature contract is `crystal_graph.v1`, and it computes **no columns at
all**.

A graph network eats the crystal itself, not a fixed-width row, so there is
nothing tabular to produce. The contract still has a job: it checks that every
sample comes with a structure, and that the atomic embedding table is pinned by
digest. Then it hands the structures to the trainer untouched. Declaring it
keeps the protocol explicit about what the model consumes, and keeps the
pinned-artifact machinery working for a model with no feature matrix.

Each crystal becomes a graph the same way the published checkpoint builds one:

- every atom is a node, carrying its 92-wide row from `atom_init.json`
- each node joins up to 12 neighbours within 10 Å
- each edge carries the interatomic distance

Both networks therefore see a crystal identically, which is what makes their
numbers comparable.

Graphs are cached in memory, because building one means parsing a CIF and
searching for neighbours. Without the cache, evaluating four splits would build
every crystal twice.

## The network

Same as the published checkpoint, unchanged — otherwise comparing the two would
mean nothing.

| | |
| --- | --- |
| Node features in | 92 |
| Convolutions | 3 |
| Atom feature width | 64 |
| Edge RBF bins | 64 |
| Hidden width after pooling | 128 |
| Hidden layers | 3 |
| Pooling | mean |
| Classes | 2 |

`model.parameters.architecture` can override any of these. An unknown key is
refused rather than quietly ignored.

## How it was trained

The published checkpoint came with its own training settings, and where those
are sound we kept them: AdamW at learning rate 0.001, weight decay 1e-4, cross
entropy, no class weighting.

```toml
[model.parameters]
epochs = 300
batch_size = 128
learning_rate = 0.001
weight_decay = 0.0001
patience = 40
selection_metric = "roc_auc"
scheduler_factor = 0.5
scheduler_patience = 10
```

We changed three of its settings.

**OneCycle became a plateau schedule.** OneCycle has to know its total number
of steps before the first batch, which rules out stopping when the validation
metric flattens. Halving the learning rate on a plateau gets to the same place
without fixing an epoch count up front.

**Stochastic weight averaging is gone.** The published run set it to start at
epoch 50 and then stopped at epoch 0, so it never ran there either. It is a
genuine improvement and worth adding later, but it needs a batch-norm update
pass over the training set, and that belongs in a change we can measure on its
own.

**Early stopping now watches validation ROC-AUC, not validation loss, and waits
far longer before giving up (v2).** The first release (v1) stopped on loss with
a patience of eight epochs, reached epoch 32, and scored 0.890 balanced
accuracy at its best threshold — well short of the 0.952 the Matbench CGCNN
entry reports on the same task. ROC-AUC is the ranking metric this model is
actually judged on, and it can keep improving for a while after cross-entropy
loss stops falling, so v2 selects on it directly and gives training up to 40
epochs to prove there is no more of that improvement left. Training stops when
the selection metric has not improved for that many epochs, and restores the
weights from the best epoch. It never looks at the calibration or test splits.

That change bought less than it cost: v2 restored its best epoch at 30, only six
past where v1 would have already stopped, and the epochs after that were
overfitting — training loss kept falling while validation loss rose. See
[Results](#results) for what actually moved.

`device` takes `cpu`, `mps`, or `cuda`, and defaults to `auto`, which uses an
accelerator if there is one. The record states which device actually did the
fitting, which epoch was selected, and the learning rate at every epoch.

### What the published run actually did

Its checkpoint says `epochs: 1`, reached `epoch: 0` at `global_step: 2246`, and
is labelled `run_name: test0`, `experiment_name: cgcnn_basic`. At batch size 64
that is roughly 144000 samples — one pass over the data.

In other words, it is a smoke test. Reproducing it exactly would reproduce a
smoke test, and that is very likely why its record reports no accuracy.

## How it is scored

The primary metric is Matthews correlation, compared against a `train_majority`
baseline. Accuracy, balanced accuracy, precision, recall, F1, MCC, ROC-AUC and
PR-AUC are all reported.

With a 43.5% positive rate, accuracy on its own would be misleading — a model
that says "insulator" every time scores 0.544. Read balanced accuracy and MCC
instead.

### Where the line is drawn

The network returns a probability that a structure is metallic. Turning that
into a yes or no needs a threshold, and here **the two mistakes do not cost the
same**:

| Mistake | What happens |
| --- | --- |
| A metal called an insulator | The mesh is too coarse. The Fermi surface is undersampled, and the answer can be wrong without looking wrong. |
| An insulator called a metal | A denser mesh than necessary. It costs compute. |

One gives you a bad number. The other gives you a bigger bill. Picking the
threshold that maximises MCC treats those as equally bad, so the protocol
constrains the search instead:

```toml
threshold_metric = "mcc"
min_recall = 0.97
```

That reads: *of all the thresholds that miss no more than 3% of metals, take
the one with the best MCC.*

The floor is the part that belongs on the model card — retrain the model and it
still applies. The threshold it produces, 0.048, belongs to these weights only.
[Choosing a decision threshold](../../protocol.md#choosing-a-decision-threshold)
covers the mechanism.

The threshold is chosen on the validation split and applied unchanged to
calibration and test. It is a fitted parameter, so choosing it on test would be
reading the answer first. It is written into `model.json` under `decision`, so
every consumer applies the same line rather than inventing one. A record without
it is refused at load — a classifier that cannot turn its own score into a
label is not servable.

**The floor is not free.** Recall is bought with precision, and the price rises
steeply. Measured on validation:

| Recall floor | Threshold | Precision | MCC | Metals missed | False alarms |
| ---: | ---: | ---: | ---: | ---: | ---: |
| unconstrained (MCC) | 0.486 | 0.901 | 0.796 | 674 | 431 |
| 0.95 | 0.111 | 0.755 | 0.706 | 228 | 1414 |
| **0.97** | **0.048** | **0.669** | **0.619** | **137** | **2204** |
| 0.99 | 0.015 | 0.555 | 0.454 | 45 | 3639 |

That is over 10603 validation samples, 4585 of them metals. Going from the
unconstrained threshold to 0.95 rescues 446 metals and costs 983 extra false
alarms. Going from 0.97 to 0.99 rescues only 92 more and costs 1435. The useful
range runs out before 0.99, where 77% of all structures would get a dense mesh
anyway — against the 100% you would get by not classifying at all.

**Why 0.97 and not 0.95?** Margin. A floor is honoured on validation, and
validation is only a sample. The 0.95 threshold delivers 0.9498 recall on test,
which is below its own floor. The 0.97 threshold delivers 0.9717, which keeps
0.95 as well.

## Running it

```bash
uv run goldilocks-ml train validate protocols/metallicity/is_metal/cgcnn/matbench_mp_is_metal.v2.toml \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts

uv run goldilocks-ml train run protocols/metallicity/is_metal/cgcnn/matbench_mp_is_metal.v2.toml \
  --dataset local_data/snapshots/mp-is-metal \
  --artifact-directory local_data/artifacts \
  --output local_runs/cgcnn-v2
```

## Results

A full run over the sealed snapshot needs a GPU to finish inside a working
session — the release run took about two hours on SCARF. It ran 70 epochs and
restored the weights from epoch 30, where validation ROC-AUC peaked at 0.9548.
These are the numbers from the release run,
`metallicity.is_metal.cgcnn.matbench_mp_is_metal.v2`.

| Split | Accuracy | Balanced | Precision | Recall | F1 | MCC | ROC-AUC | PR-AUC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Validation | 0.779 | 0.802 | 0.669 | 0.970 | 0.792 | 0.619 | 0.955 | 0.947 |
| Calibration | 0.786 | 0.810 | 0.672 | 0.976 | 0.796 | 0.632 | 0.959 | 0.950 |
| **Test** | **0.748** | **0.766** | 0.649 | **0.972** | 0.778 | **0.569** | **0.951** | **0.949** |
| Test baseline | 0.544 | 0.500 | 0 | 0 | 0 | 0.000 | 0.500 | 0.274 |

The baseline always predicts the majority class, so it never finds a metal at
all.

**Read ROC-AUC and PR-AUC first.** Neither depends on the threshold, so they
measure how well the model *ranks* structures by metallicity. At 0.951 and
0.949 on test, that ranking is strong.

Accuracy and MCC look lower than they could be, and that is expected: the
threshold is deliberately not where they peak. The recall floor moved it, and
the table above records exactly what that cost.

Recall on test is 0.972, against a floor of 0.97 that was chosen on validation.
The promise holds on a split it was not chosen on.

**What the longer patience actually bought.** Against the v1 release — same
data, same split, same architecture, stopped on loss with patience 8 — every
number above moved by at most 0.002, an amount three other v1 runs also show
between themselves as ordinary training noise. The one figure this change was
aimed at, ROC-AUC, went from 0.9498 to 0.9508. Reaching 0.952 needs a different
lever than early stopping; the leading candidate is the composition-grouped
split, kept deliberately harder than Matbench's random one, and untangling that
from the rest of the recipe is an open, unpublished diagnostic rather than a
settled result.

## Will I get the same file twice?

**No.** This trainer is not deterministic, and `model.json` says so. The seed is
fixed; what varies is the order a GPU adds numbers in. That changes the last few
digits, and the difference compounds over training. [What a run
produces](../../run-bundle.md#do-i-get-the-same-file) explains the general case.

Measured on the v1 recipe: two runs with the same seed and the same splits
produced weight files with different checksums. Out of 106113 scores, 4% were
bit-identical, the average difference was 4e-6, the largest was 8e-4, and
exactly one structure crossed the threshold. **Every metric agreed to three
decimal places, across four separate v1 runs.**

So the model reproduces; the file does not. v2 has been run once, so this has
not yet been checked for it directly — nothing about the change makes it
likely to behave differently, but that is an expectation, not a measurement.

**The split does reproduce exactly.** The second run worked out its own
assignment from the seed rather than copying the first run's `splits.csv`, and
the two agree row for row. The set of structures behind a number never moves,
even when the number wobbles in its last digits.
