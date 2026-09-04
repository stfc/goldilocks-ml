# Models

Models are filed under the calculation setting they advise, not under the
algorithm that produces them.

## How a model is named

```text
k_points . k_distance . qrf . goldilocks_kdist_ultra . v1
└ setting   └ quantity   └ family └ dataset            └ version
```

| Part | Answers |
| --- | --- |
| **setting** | Which calculation input this advises. Core's vocabulary. |
| **quantity** | What the number is. Decides how Core converts it. |
| **family** | The kind of model fitted. |
| **dataset** | The snapshot's own record id. |
| **version** | Bumped when the same combination is trained again. |

Setting and quantity are separate because a k-point mesh can be reached from a
k-distance *or* a k-index, and those need different conversions. Keeping them
apart lets a second model join a setting without disturbing the first.

### On `metallicity.is_metal`

Some names read as if they say the same thing twice, and this one will not be
the last: a magnetism classifier would be `magnetism.is_magnetic`.

The two parts are still doing different jobs. The setting is what Core routes
on, and it stays `metallicity` whether the answer arrives as a boolean or as a
band gap in eV. The quantity is what Core has to interpret, and a `false` is
not a `0.03`. A band-gap regressor for the same setting would be
`metallicity.band_gap`, sitting beside this one and reaching the same decision
by a different route — exactly as `k_index` now sits beside `k_distance`.

Magnetism shows this more plainly than metallicity does, because it needs three
quantities rather than two: `is_magnetic` decides whether to switch spin
polarisation on, `ordering` decides which arrangement to converge towards, and
`magnetic_moments` is what goes into the input file. Only the first repeats its
setting's name.

The repetition is inherited rather than invented. The published target contract
is `goldilocks.is_metal.dft_band_gap_zero.v1`, whose quantity segment is
`is_metal`; names follow the contract word for word instead of coining a
tidier synonym, because a name that drifts from its contract is worse than a
name that repeats itself.

## What exists

| Setting | Kind | Quantity | Family | Status |
| --- | --- | --- | --- | --- |
| [k-point mesh](k_points/index.md) | input | `k_distance` | QRF | published, **historical** |
| [k-point mesh](k_points/k_index-qrf.md) | input | `k_index` | QRF | trained, not deposited |
| [Metallicity](metallicity/index.md) | property | `is_metal` | CGCNN | trained, not deposited |
| [Metallicity](metallicity/representation-cgcnn.md) | property | representation | CGCNN | published, **historical** |
| [Magnetism](magnetism/index.md) | input | — | — | planned |
| [Hubbard U](hubbard_u/index.md) | input | — | — | planned |

**Historical** means the record's latest version is its last. Both published
records were fitted before this repository existed, from workflows that were
not versioned protocols, so neither carries a training run anyone can repeat.
They stay loadable and citable; a successor is a new record, not a new version.

Two of those rows share a setting, because the same architecture trained on the
same labels can give you two different things. The second level of the name
says which:

```text
deposits/metallicity/is_metal/cgcnn/         a decision
deposits/metallicity/representation/cgcnn/   64 numbers another model consumes
```

The representation record says `role: feature_extractor`, and `load_model`
declines to serve it, naming the reason.

**Input** settings are written into a DFT input file. **Property** settings are
facts about the material that inform several inputs at once.

Which model Core reaches for when the caller names none is Core's decision, and
Core's registry records it — not this table. Core does not read these PSDI
records at all today; it downloads its own copies from Hugging Face.
