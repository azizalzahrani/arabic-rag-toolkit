# Releasing

This repository is configured for tag-based releases.

## One-Time Setup

1. Create the project on PyPI with the name `arabic-rag-toolkit`.
2. In PyPI, configure a trusted publisher for this GitHub repository and the workflow `.github/workflows/release.yml`.
3. In GitHub, keep the `pypi` environment available for the publish job.

## Release Steps

1. Update the version in `pyproject.toml` and `arabic_rag/__init__.py`, and add an entry to `CHANGELOG.md`.
2. Run local checks:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest -v
python -m build
twine check dist/*
```

3. Commit the version bump.
4. Create and push a tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

## What Happens On Tag Push

- GitHub Actions builds the sdist and wheel.
- The package is published to PyPI through trusted publishing.
- A GitHub Release is created automatically with the built artifacts attached.
