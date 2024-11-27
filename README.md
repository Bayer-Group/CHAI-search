# CHAI Search
CHAI-Search can be used for retrosynthetic searches with the DFPN* algorithm. The search algorithm needs a reaction prediction and reaction filter model to work as intended. Which type of model is used is completely customizable and only has to follow template-based approach, see retrosynthetic search tools like:
- Genheden S, Thakkar A, Chadimova V, et al (2020) AiZynthFinder: a fast, robust and flexible open-source software for retrosynthetic planning. ChemRxiv. Preprint. https://doi.org/10.26434/chemrxiv.12465371.v1.
- Segler MH, Preuss M, and Waller MP (2018) Planning Chemical Syntheses with deep neural networks and symbolic AI, Nature, 555(7698), doi:10.1038/nature25978.

This is just an implementation detail, in principle the DFPN* is agnostic to the type of reaction prediction model used (template-based, template-free).

This repository provides a basic implementation for a retrosynthetic route search planning tool, but is not a complete tool suite like:
[ASKCOS](https://gitlab.com/mlpds_mit/askcosv2) or [AiZynthFinder](https://github.com/MolecularAI/aizynthfinder)

## Prerequisites
The following requirements have to be met:

- Linux
- Anaconda (Miniconda)
- reaction prediction/expansion policy model
- filter policy model
- a building block (stock) file

## Installation & Usage

conda env create --file environment.yml

python main.py -t 60 -i input.smi

## Contributors
- [Florian Mrugalla](https://github.com/FloMru) 
- [Yannic Alber](https://github.com/YannicAlber)

## License
See license file

## References
Mrugalla F, Franz C, Alber Y, Mogk G, Villalba M, Mrziglod T, Schewior K (2024) Generating Diversity and Securing Completeness in Algorithmic Retrosynthesis. Submitted.

Franz C, Mogk G, Mrziglod T, Schewior K (2022) Completeness and Diversity in Depth-First Proof-Number Search with Applications to Retrosynthesis. IJCAI. https://doi.org/10.24963/ijcai.2022/658
