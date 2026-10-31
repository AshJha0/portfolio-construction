# portopt (Python reference)

The Python reference implementation of the portfolio-construction toolkit:
sample and Ledoit-Wolf covariance estimators, closed-form and box-constrained
Markowitz optimisation, Black-Litterman, risk parity, and a backtest. Every
algorithm is pinned in the repository's API_SPEC.md so that the C++, Rust and
Java ports reproduce the same numbers.

```bash
python -m pip install -e "python[dev]"   # from the repository root
pytest -q python
python python/demo.py
```

The bundled data in `data/` is synthetic (a three-factor model with Student-t
noise, seed 42). See the repository README for the full documentation:
https://github.com/AshJha0/portfolio-construction
