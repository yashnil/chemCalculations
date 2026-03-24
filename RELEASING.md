# Releasing to PyPI

This project is published as **`chemcalculations`** on [PyPI](https://pypi.org/project/chemcalculations/).

## One-time setup

1. **Accounts**
   - Register at [pypi.org](https://pypi.org/account/register/) and [test.pypi.org](https://test.pypi.org/account/register/) (recommended for a dry run).
   - Enable **two-factor authentication (2FA)** on both — PyPI **requires** 2FA to upload.

2. **API tokens** (recommended over passwords)
   - PyPI → Account settings → **API tokens** → Add token.
   - **First release of a new project:** choose scope **Entire account** (all projects). A token scoped only to **Project: `chemcalculations`** will often get **`403 Forbidden`** until that project already exists on PyPI—after the first successful upload you can create a project-scoped token if you want.
   - Save the token somewhere safe (password manager). For `twine`, username is always **`__token__`** and the **password** is the **full** token string (starts with `pypi-`).

3. **Install release tools** (from the repo root):

   ```bash
   pip install -e ".[release]"
   ```

## Every release

1. **Bump the version** in `pyproject.toml` (`[project]` → `version = "0.1.1"`, etc.). PyPI does not allow re-uploading the same version.

2. **Run tests**:

   ```bash
   pip install -e ".[dev]"
   pytest
   ```

3. **Clean old builds** (optional but avoids confusion):

   ```bash
   rm -rf dist/ build/ src/*.egg-info
   ```

4. **Build** source distribution + wheel:

   ```bash
   python -m build
   ```

   Artifacts appear under **`dist/`**: `chemcalculations-<version>.tar.gz` and `.whl`.

5. **Check** the artifacts:

   ```bash
   twine check dist/*
   ```

6. **Upload to TestPyPI first** (recommended):

   ```bash
   twine upload --repository testpypi dist/*
   ```

   Install in a **fresh venv** and smoke-test:

   ```bash
   pip install --index-url https://test.pypi.org/simple/ chemcalculations
   python -c "import chemcalculations; print(chemcalculations.__version__)"
   ```

7. **Upload to PyPI** (production):

   ```bash
   twine upload dist/*
   ```

   When prompted, username: **`__token__`**, password: **your PyPI API token** (including the `pypi-` prefix).

8. **Verify** the project page: `https://pypi.org/project/chemcalculations/`

## Troubleshooting

### `HTTPError: 403 Forbidden` from `upload.pypi.org`

1. **Token scope** — For the **first** upload, use an **Entire account** API token. Recreate the token at [pypi.org/manage/account/token/](https://pypi.org/manage/account/token/) if needed.
2. **Username** — Must be exactly **`__token__`** (not your PyPI email or display name).
3. **Password** — Paste the **full** token including the `pypi-` prefix; no leading/trailing spaces or quotes.
4. **Right registry** — A **TestPyPI** token only works on TestPyPI; production uploads need a token created on **pypi.org**.
5. **More detail** — Run `twine upload dist/* --verbose` and read the response body.

## Notes

- **Git tag** (optional but good practice): `git tag v0.1.0 && git push origin v0.1.0`
- **First upload**: If the name `chemcalculations` is taken by someone else, change `[project]` → `name` in `pyproject.toml` before uploading.
- Large files: training data and checkpoints stay **out** of the sdist; only package code under `src/chemcalculations/` is included (see `MANIFEST.in`).
