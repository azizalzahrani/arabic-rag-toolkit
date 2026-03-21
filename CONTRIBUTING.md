# Contributing

Thanks for contributing to `arabic-rag-toolkit`.

## Development Setup

```bash
git clone https://github.com/azizalzahrani/arabic-rag-toolkit.git
cd arabic-rag-toolkit
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Workflow

1. Create a branch for your change.
2. Keep changes focused and document user-facing behavior.
3. Add or update tests when behavior changes.
4. Run the test suite before opening a pull request.

## Test Command

```bash
pytest -v
```

## Pull Requests

- Describe the problem and the chosen fix.
- Include sample input/output when the change affects retrieval or generation behavior.
- Call out any dependency or packaging changes explicitly.

## Maintainer Notes

Release instructions live in `RELEASING.md`.
