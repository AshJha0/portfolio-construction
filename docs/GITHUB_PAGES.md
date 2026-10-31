# Publishing the documentation site (GitHub Pages)

The site at https://ashjha0.github.io/portfolio-construction/ is the static
page `docs/index.html`, served by GitHub Pages straight from the `docs/`
folder of `main`. There is no build step and no site generator.

## One-time setup

1. Repository **Settings > Pages**.
2. Under *Build and deployment*, choose **Source: Deploy from a branch**,
   **Branch: `main`**, **Folder: `/docs`**, then **Save**.
3. After the first build (about a minute) the site is live at the address
   above. Every push to `main` that changes `docs/` republishes it.

## What is published

- `docs/index.html`: the landing page. It links to `LEARN.md`,
  `COOKBOOK.md`, `API_SPEC.md`, `CHANGELOG.md` and `docs/ARCHITECTURE.md`
  on GitHub, where Markdown and Mermaid render.
- `docs/img/*.png`: the three figures the page and the README show,
  referenced by relative path (`img/frontier.png` from the page,
  `docs/img/frontier.png` from the README).
- Every other file under `docs/` is reachable by its path but served raw,
  for example `https://ashjha0.github.io/portfolio-construction/ARCHITECTURE.md`.
  The Mermaid sources in `docs/diagrams/` are not rendered on the site.

## Regenerating the figures

The figures are computed, not drawn by hand:

```bash
python -m pip install -e "python[docs]"   # numpy and matplotlib
python docs/make_figures.py               # writes docs/img/*.png
```

Rerun it whenever the data, the generator or a function the figures use
changes, and commit the PNGs with that change.

## Keeping the page accurate

The page is static HTML, so nothing updates it automatically. When any of
the following change, update `docs/index.html` and `README.md` in the same
commit:

- the version (badge, header paragraph and footer);
- the golden case count (23, from `data/golden/golden.json`);
- the CI matrix (from `.github/workflows/ci.yml`);
- the quickstart commands;
- any number quoted from the demo or the figures (for example the
  tangency gross leverage 4.14 and the ERC spread 1.3e-12, both printed by
  `python python/demo.py`).

Release notes belong in `CHANGELOG.md`, not on the page.

Check the page at phone width (375 px) before pushing: there should be no
horizontal scrolling outside the code blocks.
