# Web version of the book

```sh
uv venv --python 3.12 && uv pip install -r docs/requirements.txt
uv run sphinx-build -c docs . _build/html   # then open _build/html/index.html
```

The notebooks are rendered with their saved outputs; nothing is re-executed.
