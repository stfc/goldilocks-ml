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
by a different route — exactly as `k_index` will sit beside `k_distance`.

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
| [k-point mesh](k_points/index.md) | input | `k_distance` | QRF | **Core default** |
| [Metallicity](metallicity/index.md) | property | `is_metal` | CGCNN | trained |
| [Magnetism](magnetism/index.md) | input | — | — | planned |
| [Hubbard U](hubbard_u/index.md) | input | — | — | planned |

**Input** settings are written into a DFT input file. **Property** settings are
facts about the material that inform several inputs at once.

**Core default** is the model Core reaches for when the caller names none. It is
a separate decision from whether a model is good — Core's registry, not this
one, records it.
