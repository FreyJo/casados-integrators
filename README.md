# casados integrators

A `CasADi` Python wrapper for the `acados` integrators.

## Publication
Fast integrators with sensitivity propagation for use in CasADi - Jonathan Frey, Jochem De Schutter, Moritz Diehl
https://arxiv.org/abs/2211.01982

Please cite as
```
@inproceedings{Frey2023,
	year = {2023},
	booktitle = {Proceedings of the European Control Conference (ECC)},
	author = {Frey, Jonathan and De Schutter, Jochem and Diehl, Moritz},
	title = {Fast integrators with sensitivity propagation for use in {C}as{AD}i},
}
```

## Installation
1. Install `acados` and its Python interface following https://docs.acados.org/

2. Clone this repository

3. Enter the local repository and install `casados` using:
```
pip install -e .
```

## Examples and Supported versions
The pendulum example has been tested with Python 3.11 and `CasADi` 3.6.5.
Other examples are not tested with `CasADi` 3.6.x.

## CasADi 3.5.
Please check out [`casados-integrators v1.0.0`](https://github.com/FreyJo/casados-integrators/releases/tag/v1.0.0)
