# proteopt
Common interface to protein design tools and structure predictors

This package provides three things:
1. A [huge docker image](https://hub.docker.com/r/timodonnell/proteopt-complete)
with a number of protein design and protein structure prediction tools installed
(with model weights) in the same python environment
2. A python API to call (most of) these tools using a reasonably convenient interface
3. A flask server with an API to call these tools over HTTP 

This is all very much WIP. Contributions are welcome.

## Tools
| Tool                                                                 | Task                  | Installed in the proteopt docker image? | Part of proteopt python API?              | Unit tests for API                    |
|----------------------------------------------------------------------|-----------------------|-----------------------------------------|-------------------------------------------|---------------------------------------|
| [AlphaFold](https://github.com/deepmind/alphafold)                   | structure prediction  | yes                                     | [yes](proteopt/alphafold.py)              | [yes](test/test_alphafold.py)         |
| [OmegaFold](https://github.com/HeliXonProtein/OmegaFold)             | structure prediction  | yes                                     | [yes](proteopt/omegafold.py)              | [yes](test/test_omegafold.py)         |
| [OpenFold](https://github.com/aqlaboratory/openfold)                 | structure prediction  | yes                                     | no                                        | no                                    |
| [ESMFold](https://github.com/facebookresearch/esm)                   | structure prediction  | no (TODO)                               | no                                        | no                                    |
| [RFDesign](https://github.com/RosettaCommons/RFDesign) hallucination | design                | yes                                     | [yes](proteopt/rfdesign_hallucination.py) | [yes](test/test_hallucination.py)     |
| [RFDesign](https://github.com/RosettaCommons/RFDesign) inpainting    | design                | yes                                     | [yes](proteopt/rfdesign_inpainting.py)    | [yes](test/test_inpainting.py)        |
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)               | fixed-backbone design | yes                                     | [yes](proteopt/proteinmpnn.py)            | [yes](test/test_proteinmpnn.py)       |
| [ColabDesign](https://github.com/sokrypton/ColabDesign) AfDesign     | design                | yes                                     | no                                        | no                                    |
| [RFDiffusion](https://github.com/RosettaCommons/RFDiffusion)         | design                | yes                                     | yes (motif scaffolding only)              | [yes](test/test_rfdiffusion_motif.py) |

