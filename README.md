# proteopt
Common interface to protein design tools and structure predictors

The goal is provide a Python API that wraps recent protein design and structure prediction tools. We want to
be able to experiment with multiple approaches for same problem without rewriting our task for each tool's API.

This package also provides a large docker image with the tools installed (see [docker/base](docker/base)) as well as an 
image that extends it to include a jupyter environment for interactive work (see [docker/full](docker/full)).

For now this is all very much WIP. Contributions are welcome.

## Tools
| Tool       | Task | Installed in the docker? | Python API? | Unit tests for API |
|------------|------|--------------------------------|-------------|--------------------|
| [AlphaFold](https://github.com/deepmind/alphafold) | structure prediction | yes | [yes](proteopt/alphafold.py) | [yes](test/test_alphafold.py) |
| [RFDesign](https://github.com/RosettaCommons/RFDesign) hallucination | design | yes | [yes](proteopt/rfdesign_hallucination.py) | [yes](test/test_hallucination.py) |
| [RFDesign](https://github.com/RosettaCommons/RFDesign) inpainting | design | yes | [yes](proteopt/rfdesign_inpainting.py) | [yes](test/test_inpainting.py) |
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | fixed-backbone design | yes | [yes](proteopt/proteinmpnn.py) | [yes](test/test_proteinmpnn.py) |
| [RFDiffusion](https://github.com/RosettaCommons/RFDiffusion) | design | yes | no | no |
| [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) | structure prediction | yes | no | no |

