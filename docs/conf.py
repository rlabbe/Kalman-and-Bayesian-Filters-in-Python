"""Sphinx config for the web version of the book.

Build from the repo root:  sphinx-build -c docs . _build/html
(-c keeps this config out of the way of the notebooks, which stay where they are.)
"""

project = "Kalman and Bayesian Filters in Python"
author = "Roger Labbe"
copyright = "Roger R. Labbe Jr."

extensions = ["myst_nb"]

root_doc = "docs/index"
source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb"}
exclude_patterns = [
    "_build", ".venv", "Supporting_Notebooks", "experiments", "animations", "pdf",
    "kf_book", "README.md", "contributing.md", "table_of_contents.ipynb", "**/.ipynb_checkpoints",
]

# the notebooks ship with their outputs; don't re-run them while building the docs
nb_execution_mode = "off"
myst_enable_extensions = ["dollarmath", "amsmath", "html_image", "colon_fence"]
myst_heading_anchors = 3
suppress_warnings = ["myst.header", "myst.xref_missing"]

html_theme = "shibuya-page"
html_title = project
html_theme_options = {
    "look": "alabaster",
    "github_url": "https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python",
}
html_static_path = []
html_extra_path = ["_extra"]


# Every chapter opens with boilerplate cells (`%matplotlib inline`, and book_format.set_style(), which just
# injects the old notebook CSS). They mean nothing on the web, so drop whole cells whose code is only that.
_BOILERPLATE = {
    "%matplotlib inline",
    "#format the book\nimport book_format\nbook_format.set_style()",
}


def _drop_boilerplate_cells(app, doctree):
    from docutils import nodes

    for cell in list(doctree.findall(nodes.container)):
        if "cell" not in cell["classes"]:
            continue
        code = [n.astext().strip() for n in cell.findall(nodes.literal_block) if "cell_input" in
                " ".join(p for a in [n] for p in _ancestor_classes(a))]
        if code and all(c in _BOILERPLATE for c in code):
            cell.parent.remove(cell)


def _ancestor_classes(node):
    while node is not None:
        yield from node.get("classes", [])
        node = node.parent


def setup(app):
    app.connect("doctree-read", _drop_boilerplate_cells)
