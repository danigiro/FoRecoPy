# FoRecoPy: Forecast Reconciliation in Python <img src="sphinx/source/forecopy.svg" alt="logo" align="right" width="150" style="border: none; float: right;"/>

[![pytest](https://github.com/danigiro/FoRecoPy/actions/workflows/test.yml/badge.svg)](https://github.com/danigiro/FoRecoPy/actions/workflows/test.yml) [![codecov](https://codecov.io/github/danigiro/FoRecoPy/graph/badge.svg?token=S7CMY2OT3U)](https://codecov.io/github/danigiro/FoRecoPy)

Forecast reconciliation is a post-forecasting process aimed at improving the
accuracy and coherence of forecasts for a system of **linearly constrained
time series** (e.g., hierarchical, grouped, or temporal structures).

The **FoRecoPy** package is inspired by the
[R package FoReco](https://danigiro.github.io/FoReco) and brings similar
functionality to **Python**. It is designed for researchers, practitioners,
and data scientists who use Python for time series forecasting and want
access to **state-of-the-art reconciliation methods**.

Future versions will expand the scope to include the cross-temporal framework,
non-negative constraints and probabilistic reconciliation.

## Installation

Make sure to have a working `JAX` installation (please, follow [these instructions](https://github.com/google/jax#installation)).

To install the package from PyPI, call:

```bash
pip install forecopy
```

To install the latest GitHub <RELEASE>, just call the following on the
command line:

```bash
pip install git+https://github.com/danigiro/FoRecoPy@<RELEASE>
```

## Features

- Optimal combination reconciliation via projection and structural approaches  
- Tools for both **cross-sectional** (`csrec`) and **temporal** (`terec`) reconciliation  
- Different covariance matrix approximation
- Support for custom aggregation or constraints matrices  
- Option to enforce **non-negativity** on reconciled forecasts  
- Efficient solvers suitable for high-dimensional problems  

## Quick Examples
Examples of cross-sectional and temporal forecast reconciliation are available [here](https://danigiro.github.io/FoRecoPy/overview.html)


## License

MIT License. See [LICENSE](LICENSE) for details.
