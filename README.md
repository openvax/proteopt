# proteopt
Common interface to protein design tools and structure predictors

The goal is provide a Python API that wraps recent protein design and structure prediction tools. We want to
be able to experiment with multiple approaches for same problem without rewriting our task for each tool's API.

This package also provides a huge docker image with the tools installed, including
model weights.

This is all very much WIP. Contributions are welcome.

## Tools
| Tool                                                                 | Task | Installed in the proteopt docker image? | Part of proteopt python API?              | Unit tests for API                |
|----------------------------------------------------------------------|------|-----------------------------------------|-------------------------------------------|-----------------------------------|
| [AlphaFold](https://github.com/deepmind/alphafold)                   | structure prediction | yes                                     | [yes](proteopt/alphafold.py)              | [yes](test/test_alphafold.py)     |
| [OmegaFold](https://github.com/HeliXonProtein/OmegaFold)             | structure prediction | yes                                     | [yes](proteopt/omegafold.py)              | [yes](test/test_omegafold.py)     |
| [OpenFold](https://github.com/aqlaboratory/openfold)                 | structure prediction | yes                                     | no                                        | no                                |
| [RFDesign](https://github.com/RosettaCommons/RFDesign) hallucination | design | yes                                     | [yes](proteopt/rfdesign_hallucination.py) | [yes](test/test_hallucination.py) |
| [RFDesign](https://github.com/RosettaCommons/RFDesign) inpainting    | design | yes                                     | [yes](proteopt/rfdesign_inpainting.py)    | [yes](test/test_inpainting.py)    |
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN)               | fixed-backbone design | yes                                     | [yes](proteopt/proteinmpnn.py)            | [yes](test/test_proteinmpnn.py)   |
| [ColabDesign](https://github.com/sokrypton/ColabDesign) AfDesign     | design | yes                                     | no                                        | no                                |
| [RFDiffusion](https://github.com/RosettaCommons/RFDiffusion)         | design | yes                                     | no                                        | no                                |

