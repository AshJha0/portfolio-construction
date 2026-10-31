# Security policy

## Scope

`portopt` is a numerical library. It reads return data and parameters you
supply and computes estimators, portfolio weights and backtest metrics. It
does not connect to brokers or exchanges, place orders, handle credentials
or open network sockets. The issues most likely to matter are:

- input handling: an input that causes a crash, undefined behaviour,
  unbounded memory use or a hang, instead of the invalid-input error the
  public functions are specified to raise (`API_SPEC.md`);
- file parsing: the CSV and JSON readers in each port (for example
  `cpp/src/data_io.cpp`, `rust/src/data.rs`) given malformed files;
- the build and supply chain: a dependency, fetched archive or CI workflow
  that could be tampered with.

A wrong number, or disagreement between ports, is a bug rather than a
security issue. Please open a normal issue for it.

## Supported versions

Only the latest release on `main` receives fixes.

## Reporting a vulnerability

Please report privately through GitHub's
[private vulnerability reporting](https://github.com/AshJha0/portfolio-construction/security/advisories/new)
rather than in a public issue. Include the port, the version or commit, a
minimal input that reproduces the problem, and what you observed.

We aim to acknowledge a report within a week. A fix is released with a note
in `CHANGELOG.md`, crediting the reporter unless they prefer otherwise.

## What the repository does to reduce risk

- Every public entry point validates its inputs (NaN and infinity, shapes,
  symmetry and positive definiteness, bounds, views) before computing.
- The C++ build fetches GoogleTest only for tests, from a release archive
  pinned by SHA-256 (`cpp/CMakeLists.txt`).
- `rust/Cargo.lock` is committed, so Rust dependency versions are fixed.
- The Python package depends only on numpy at run time.
- CI builds the C++ tests under AddressSanitizer and UBSan with Clang
  (`.github/workflows/ci.yml`).
- CodeQL scans Python, C++, Rust and Java weekly and on pull requests
  (`.github/workflows/codeql.yml`); its actions are pinned by commit SHA.
- Dependabot proposes weekly updates for pip, cargo, Maven and GitHub
  Actions (`.github/dependabot.yml`).
