<div align="left">
  <img src="https://github.com/mir-group/CiderPress/blob/main/docs/logos/cider_logo_and_name.png" height="80px"/>
</div>

CiderPress: Machine Learned Exchange-Correlation Functionals
------------------------------------------------------------
![Build Status](https://github.com/mir-group/CiderPress/actions/workflows/basic_tests.yml/badge.svg)

CiderPress provides tools for training and evaluating CIDER functionals for use in Density Functional Theory calculations. Interfaces to the GPAW and PySCF codes are included.

Please see the [CiderPress website](https://mir-group.github.io/CiderPress/) for installation instructions and documentation.

## Questions and Comments

Find a bug? Areas of code unclearly documented? Other questions? Feel free to contact
Kyle Bystrom at kylebystrom@gmail.com AND/OR create an issue on the [Github page](https://github.com/mir-group/CiderPress/).

## Citing

If you find CiderPress or CIDER functionals useful in your research, please cite the following article
```
@article{PhysRevB.110.075130,
  title = {Nonlocal machine-learned exchange functional for molecules and solids},
  author = {Bystrom, Kyle and Kozinsky, Boris},
  journal = {Phys. Rev. B},
  volume = {110},
  issue = {7},
  pages = {075130},
  numpages = {30},
  year = {2024},
  month = {Aug},
  publisher = {American Physical Society},
  doi = {10.1103/PhysRevB.110.075130},
  url = {https://link.aps.org/doi/10.1103/PhysRevB.110.075130}
}
```
The above article introduces the CIDER23X functionals and much of the algorithms in CiderPress. If you use the CIDER24X functionals, please also cite
```
@article{doi:10.1021/acs.jctc.4c00999,
  author = {Bystrom, Kyle and Falletta, Stefano and Kozinsky, Boris},
  title = {Training Machine-Learned Density Functionals on Band Gaps},
  journal = {Journal of Chemical Theory and Computation},
  volume = {20},
  number = {17},
  pages = {7516-7532},
  year = {2024},
  doi = {10.1021/acs.jctc.4c00999},
  note ={PMID: 39178337},
  URL = {https://doi.org/10.1021/acs.jctc.4c00999},
  eprint = {https://doi.org/10.1021/acs.jctc.4c00999}
}
```
