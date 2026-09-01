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
