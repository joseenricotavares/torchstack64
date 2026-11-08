# Changelog

What changed in each version, newest first. A pull request that bumps `version` in
`pyproject.toml` adds the section of that version here. A change takes a line or two: what changed
for the reader, not how it works.

## 0.2.1

- The package handles missing optional plotting dependencies and missing installed metadata without
  crashing, so the library still imports and the training history callback can keep local files safely.

## 0.2.0

- The library arrives as it stood in its first home, unchanged: the training loop and its callbacks,
  the evaluation primitives, the artifact bundle and the predictor contract, with their tests and the
  MIT license. The package is still imported as `dlkit`.
- The README opens with the logo and what TorchStack64 is for.

## 0.1.1

Shorten the package docstring lines so lint passes.
