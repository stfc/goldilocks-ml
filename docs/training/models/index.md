# Models

Models are filed under the calculation setting they advise, not the algorithm
that produced them.

| Setting | Predicts | Model | Status |
| --- | --- | --- | --- |
| [k-point mesh](k_points/index.md) | `k_distance` | [QRF95](k_points/k_distance-qrf.md) | published, historical |
| [k-point mesh](k_points/index.md) | `k_index` | [k-index forest](k_points/k_index-qrf.md) | trained here |
| [Metallicity](metallicity/index.md) | `is_metal` | [CGCNN](metallicity/is_metal-cgcnn.md) | published |
| [Metallicity](metallicity/index.md) | a representation | [CGCNN](metallicity/representation-cgcnn.md) | published, historical |
| [Magnetism](magnetism/index.md) | — | — | planned |
| [Hubbard U](hubbard_u/index.md) | — | — | planned |

**Historical** means the record's latest version is its last. Both were fitted
before this repository existed, so neither carries a training run anyone can
repeat. They stay loadable and citable.

## How a model is named

```text
k_points . k_distance . qrf . goldilocks_kdist_ultra . v1
└ setting   └ quantity   └ family └ dataset            └ version
```

| Part | Answers |
| --- | --- |
| setting | which calculation input this advises |
| quantity | what the number is, which decides how it is converted |
| family | the kind of model fitted |
| dataset | the snapshot's own record id |
| version | bumped when the same combination is trained again |

Setting and quantity are separate because one setting can be reached from more
than one quantity — a k-point mesh from either a k-distance or a k-index — and
those need different conversions.

Every deposit definition lives in `deposits/`, and is the example to copy when
you [publish your own](../../publishing.md).
