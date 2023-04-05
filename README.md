# proteopt
Common interface to protein design tools and structure predictors

The goal is provide a Python API that wraps multiple recent protein design and structure prediction tools. We want to
be able to experiment with multiple approaches for same problem without rewriting our task for each tool's API.

This package also provides a large docker image with the tools installed (see [docker/base](docker/base)) as well as an 
image that extends it to include a jupyter environment for interactive work (see [docker/full](docker/full)).

For now this is all very much WIP. Contributions are welcome.

## Tools

| Tool       | Task | Installed in the docker image? | Python API? | Unit tests for API                |
|------------|------------------------------------------------|
| [AlphaFold](https://github.com/deepmind/alphafold) | structure prediction | yes | yes | yes |
| [OmegaFold](https://github.com/HeliXonProtein/OmegaFold) | structure prediction | yes | no | no |
| [RFDesign](https://github.com/RosettaCommons/RFDesign) | design | yes | yes | yes |
| [ProteinMPNN](https://github.com/dauparas/ProteinMPNN) | fixed-backbone design | yes | yes | yes |
| [RFDiffusion](https://github.com/RosettaCommons/RFDiffusion) | design | yes | no | no |

