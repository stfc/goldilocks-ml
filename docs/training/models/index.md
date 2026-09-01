# Models

Models are filed under the calculation setting they advise, not under the
algorithm that produces them. A setting can have several models, and swapping
between them should never change what Goldilocks Core does with the answer.

## How a model is named

Every model has a five-part name, read left to right:

```text
k_points . k_distance . qrf . goldilocks_kdist_ultra . v1
└ setting   └ quantity   └ family └ dataset            └ version
```

| Part | Answers |
| --- | --- |
| **setting** | Which calculation input this advises. Core's vocabulary, not ours. |
| **quantity** | What the number actually is. Decides how Core converts it. |
| **family** | The kind of model fitted. |
| **dataset** | Which data it learned from — the snapshot's own record id. |
| **version** | Bumped when the same combination is trained again. |

The middle two are the ones people usually collapse. They are different
questions: a k-point mesh can be reached from a k-distance *or* a k-index, and
those are different numbers needing different conversions, even from the same
family of model. Keeping them apart is what lets a second model join a setting
without disturbing the first.

## What exists

| Setting | Kind | Quantity | Family | Dataset | Status |
| --- | --- | --- | --- | --- | --- |
| [k-point mesh](k_points/index.md) | input | `k_distance` | QRF | `goldilocks-kdist-ultra` | **Core default** |
| | | `k_index` | — | — | no model |
| [Metallicity](metallicity/index.md) | property | `is_metal` | CGCNN | `matbench_mp_is_metal` | trained |
| [Magnetism](magnetism/index.md) | input | — | — | — | planned |
| [Hubbard U](hubbard_u/index.md) | input | — | — | — | planned |

**Input** settings are written straight into a DFT input file. **Property**
settings are things about the material that inform several inputs — metallicity
changes both how dense a mesh needs to be and which smearing is appropriate, so
it is predicted once and used in more than one place.

## What "Core default" means

Core carries a registry naming which model it reaches for when the caller does
not choose one. A model can be trained, published, and perfectly good without
being the default; the default is a separate decision about what should happen
when nobody is looking.

Today Core's only default is the k-distance model. The metallicity classifier
is used, but as an *input to that model's features* rather than as advice in
its own right — Core does not yet have anywhere to put a metallicity answer.
