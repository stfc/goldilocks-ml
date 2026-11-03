# Backbone parity: v1.0 (mace fork) against v2.0 (mace-torch 0.3.17)

PSDI record `1g8rw-q8128` v2.0 replaces the backbone file. This compares the
two versions on the same structures, each in the environment it was published
for, through the public goldilocks-ml API: `embed_structures`, the
`is_magnetic` predictor's scores, and `relax()`.

```text
v1.0       torch 2.10.0, mace-torch 0.3.12, sphericart-torch 1.0.9, e3nn 0.4.4
           backbone sha256 bdf6b8ccad5585eefd49ef5bcc4d8fab260abc04626ec4940d49a3dea310acb6
v2.0       torch 2.10.0, mace-torch 0.3.17, sphericart-torch 2.0.4, e3nn 0.4.4
           backbone sha256 fb43d7aadc63cbfbd00550a17e109553b6c051d6d6f4ab6ebd741849d92dc8bf
```

v1.0 ran the `CheukHinHoJerry/mace` fork at
`19cdf6692c48e068a24e06cfe1ffc670e8aea3dd` with goldilocks-ml 0.2.3. v2.0 ran
this branch. Both used the same `is_magnetic.pt`.

## Result

```text
embeddings      pass
scores          pass
decisions       pass     0 flips
relax moments   pass     19 structures both runs accept
relax energies  pass     19 structures both runs accept
relax status    differs  20 of 21 agree; differs: mp-13
collinear       pass
```

## Structures

- 13156 structures embedded and scored: every MatPES PBE test-split
  structure with a label (max |site moment| <= 0.05 or >= 0.5 uB), plus
  hand-built bcc Fe, diamond Si and fcc Pt and eight well-known MatPES
  entries (mp-13 bcc Fe, mp-150 fcc Fe, mp-23 Ni, mp-54 Co, mp-149 Si,
  mp-126 Pt, mp-19009 NiO, mp-19770 Fe2O3).
- 21 structures relaxed with `relax()`: bcc Fe from a [0, 0, 2.2]
  seed, six of the named entries, and 14 magnetic test-split structures of at
  most 24 atoms drawn with seed 117. All but the first start from
  `seed_moments_fm_fim`.

The MatPES records come from `materialyze/matpes/pbe-2025.2` (CC-BY-4.0), with
the train/valid/test split the classifier was trained on:
`matpes_pbe_material_records.json.gz`, sha256 `f58328fe21bf7b82342bf09b867658e34855eb487dfcafa7310c31f3d2c405f9`.

## Embeddings and scores

```text
embedding max |diff|          0   (tolerance 0.0001; largest |value| 8.05)
embedding worst structure     -
structures bit-identical      13156 of 13156
score max |diff|              0   (tolerance 1e-05)
decision flips at 0.3239   0
closest score to threshold    0.325192 (matpes-20240214_684736_1), diff there 0
```

Decisions on the 13145 labelled test-split structures:

```text
         accuracy  precision  recall  confusion [[TN, FP], [FN, TP]]
v1.0     0.9351    0.9127     0.9546  [[6335, 570], [283, 5957]]
v2.0     0.9351    0.9127     0.9546  [[6335, 570], [283, 5957]]
```

These are not the model card's metrics. The card reports the research
checkpoint's own test evaluation; this split is the one in the MatPES records
file used here.

## relax()

Tolerances: 0.001 uB on any moment component, 0.0001 eV on
the energy, checked where both runs return a result relax() accepts. `max|m|`
is the largest |m_z| v2.0 returned; `transverse` is the largest |m_x| or |m_y|.

```text
structure                   formula       atoms  status v1.0           status v2.0             max|m|   max|dm|   |dE| eV transverse
hand:Fe-bcc                 Fe-bcc            2  ok                    ok                       2.426         0         0          0
mp-13                       Fe                2  invalid               converged_after_retry    4.213         -         -          0
mp-150                      Fe                1  ok                    ok                       4.189         0         0          0
mp-23                       Ni                1  ok                    ok                       1.800         0         0          0
mp-54                       Co                2  ok                    ok                       3.367     7e-08   4.5e-10          0
mp-19009                    NiO               2  ok                    ok                       1.744         0         0          0
mp-19770                    Fe2O3            10  converged_after_retry converged_after_retry    3.625         0         0          0
matpes-20240214_1184221_0   FeSnRh2           4  ok                    ok                       2.673         0         0          0
matpes-20240214_1190448_5   Na2SO5           16  warning_not_converged warning_not_converged    0.443         0         0          0
matpes-20240214_1206312_1   SmCr2Si2C         6  converged_after_retry converged_after_retry    5.500         0         0          0
matpes-20240214_1210171_3   Pu4Ni            10  invalid               invalid                      -         -         -          0
matpes-20240214_1219777_14  Pu4ZrC5          10  converged_after_retry converged_after_retry    6.438         0         0          0
matpes-20240214_1226013_9   CoNiSe2           4  ok                    ok                       3.248         0         0          0
matpes-20240214_3155_2      Sc3FeC4           8  ok                    ok                       3.851         0         0          0
matpes-20240214_567807_0    TmZrSb            6  converged_after_retry converged_after_retry    1.642         0         0          0
matpes-20240214_866189_0    CeCd2             3  ok                    ok                       0.986         0         0          0
matpes-20240214_972367_10   UAlO3             5  ok                    ok                       2.081         0         0          0
mp-1097310                  MnBeIr2           4  ok                    ok                       3.890         0         0          0
mp-1225200                  FeCoSn2          12  converged_after_retry converged_after_retry    3.690         0         0          0
mp-1518245                  BaSrPrWO6        10  converged_after_retry converged_after_retry    1.500         0         0          0
mp-1640067                  Li2Fe(CoO3)2     22  converged_after_retry converged_after_retry    3.614         0         0          0
```

### Where the status differs

mace-torch 0.3.17 clamps |m| / m_max to at most 1 before the Chebyshev
moment basis (`MagneticScaleShiftMACE.forward` in
`mace/modules/extensions.py`); the fork does not. The two backbones are
therefore identical for moments up to the element's `m_max` and differ
above it. relax() rejects a result above 1.05 m_max, but LBFGS's line search
can try such moments on the way, and then the two runs follow different
paths.

Energy of mp-13 with every site at the same moment along z, no SCF
(m_max: Fe 5.1528 uB):

```text
m (uB)               E v1.0 (eV)               E v2.0 (eV)
   3.6            -17.1556786908            -17.1556786908  same
   3.7            -18.7389893955            -18.7389893955  same
   3.8            -20.8986211789            -20.8986211789  same
   3.9            -21.7682792126            -21.7682792126  same
   4.0            -18.8767228049            -18.8767228049  same
   4.1             -19.250235893             -19.250235893  same
   4.2            -42.2024223643            -42.2024223643  same
   4.3            -54.2039442395            -54.2039442395  same
   4.4             1490.26327944             1490.26327944  same
   4.5             48340.7831537             48340.7831537  same
   4.6             6396995.23381             6396995.23381  same
   4.7              130843627.91              130843627.91  same
   4.8            -1740902422.76            -1740902422.76  same
   4.9            -595233058.112            -595233058.112  same
   5.0             104193600.746             104193600.746  same
   5.1              37820662.733              37820662.733  same
   5.2         1.13644164953e+25         5.73216198637e+13  differs
   5.3         1.31370575754e+41         5.85381543618e+13  differs
   5.4          1.4872721031e+52         5.97778609456e+13  differs
```

Both versions agree that this energy surface is unphysical above about
4.2 uB, well inside m_max. Neither relaxed result for this structure is
physical: MatPES's own DFT gives it 2.25 uB. v2.0's retry stops at a point
relax() accepts; v1.0's retry did not.

## Reproduce

`scripts/mmace_parity/` holds the scripts that produced this file:
`select_structures.py` picks the structures from the MatPES records,
`run_pipeline.py` and `scan_moments.py` run once per environment, and
`compare.py` writes this report.
