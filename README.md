<p align="center">
  <img src="docs/logo.png" alt="A torch in an inventory slot, sixty-four of them" width="180">
</p>

# TorchStack64

TorchStack64 is a versatile collection of utilities and building blocks for deep learning workflows, primarily built around the PyTorch ecosystem. The name conveys the idea of a well-stocked toolkit, bringing together complementary resources in a single package. Focused on modularity, reusability, and practical convenience, TorchStack64 aims to streamline recurring tasks across model experimentation, training, evaluation, and serving, providing a set of useful tools that extend the everyday PyTorch development experience.

Until its first change, the package inside is still imported as `dlkit`:

# dlkit

Reusable building blocks extracted from doing epoch-based PyTorch training with
MLflow/Optuna/rich tooling more than once: a training loop plus a callback
system, and a small, framework-agnostic contract for packaging and serving
whatever a training run produces.

`dlkit` is deliberately **not** a universal ML framework. It exists to avoid
re-writing the same training loop and the same "how do I get this model into
an API" plumbing on every new project.

## What it is built for

- **`dlkit.training`** (including `dlkit.training.optim`, `dlkit.training.callbacks`)
  — PyTorch. An epoch-based `NNTrainer` (train/validate/early-stop/checkpoint
  loop) that any `torch.nn.Module`-based model can plug into by implementing
  three methods.
- **`dlkit.evaluation`** — small, composable, side-effect-free evaluation
  primitives built on scikit-learn: a `Metric` protocol and a catalog of
  metrics generic to any classification problem (accuracy, F1), under
  `dlkit.evaluation.metrics`, plus the `Evaluator` that composes them.
- **`dlkit.artifacts` / `dlkit.inference`** — framework-agnostic. A model
  artifact bundle format and a `Predictor` protocol that work equally well
  for a PyTorch model or a plain scikit-learn pipeline, as long as whatever
  produced it can write the bundle and whatever serves it can implement
  `predict()`.
- **`dlkit.mlflow_logging`** — one function, `log_evaluation`, that logs an
  `EvaluationResult` to MLflow. It lives at the package root, not inside
  `dlkit.evaluation`, precisely so `dlkit.evaluation` can stay 100% pure:
  compose a list of metrics, run them against `(y_true, y_pred)`, get a
  result back - no I/O, no side effects, no coupling to any tracking or
  plotting library.

## Explicit non-goals

These are deliberate scope boundaries, not oversights:

- No attempt to generalize non-epoch training loops (e.g. scikit-learn's
  one-shot `.fit()`) into `NNTrainer`. If a model doesn't train in epochs, it
  doesn't need `dlkit.training` at all — it only needs to satisfy the
  `dlkit.artifacts`/`dlkit.inference` contracts to be servable.
- No built-in regression or multi-label support. `dlkit.artifacts.LabelSchema`
  assumes discrete classes.
- No attempt to cover streaming or online learning.
- No fixed evaluation taxonomy. `dlkit.evaluation` gives you `Metric` + `Evaluator`
  and a couple of ready-made generic metrics; anything domain-specific is expected to
  be implemented by the consuming project as its own `Metric`.
- No tooling for non-text modalities. `dlkit.inference.Predictor[InputT]` is
  generic over its input type. An `ImagePredictor` could satisfy
  `Predictor[PIL.Image.Image]` without touching the protocol, but `dlkit`
  ships zero concrete tooling for anything but the contract itself (no image
  transforms, no tokenizers).

## What already works without any dlkit change

Binary and multiclass classification both work out of the box: nothing in
`dlkit.training` or `dlkit.evaluation` assumes a fixed number of classes or a
fixed set of metrics — the `Monitor`/`Metric` abstractions accept any
`Metric` implementation, including ones defined entirely outside `dlkit`.

## Installing

```bash
pip install -e src/dlkit                            # base: NNTrainer, artifacts, inference, generic evaluation
pip install -e "src/dlkit[mlflow,plotting]"          # add extras as needed
pip install -e "src/dlkit[all]"                      # every extra at once
```

Extras: `mlflow`, `callbacks-rich`, `callbacks-optuna`, `plotting`. Each
isolates its heavy/optional dependency (`mlflow`, `rich`, `optuna`,
`matplotlib`) so a lean serving environment never has to install them

## Minimal example

The full cycle - train, get logging/progress/plotting, save, and
serve - is where dlkit's value shows up:

```python
from dataclasses import dataclass
import torch
import torch.nn as nn

from dlkit.training import NNTrainer, Monitor, OPTIMIZERS, OptimizerType
from dlkit.training.callbacks import build as build_callbacks
from dlkit.evaluation import AccuracyMetric
from dlkit.artifacts import ArtifactBundle, BundleFile, LabelSchema


# 1. Your concrete config - a few lines, once per project.
@dataclass
class MyConfig:
    epochs: int = 10
    early_stopping_patience: int = 3
    monitor: Monitor = Monitor(metric=AccuracyMetric(), mode="max")
    optimizer: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2


# 2. Your trainer - only what's specific to your model.
class MyTrainer(NNTrainer[MyConfig]):
    def __init__(self, model, train_loader, val_loader, config, device, callbacks=None):
        super().__init__(model, train_loader, val_loader, config, device, callbacks)
        self.optimizer = OPTIMIZERS[config.optimizer](model.parameters(), lr=config.learning_rate)

    def _train_epoch(self) -> float: ...  # your forward/backward
    def predict(self, loader): ...  # your validation forward pass

    def save(self, output_dir) -> ArtifactBundle:
        return ArtifactBundle.write(
            output_dir,
            flavor="my-model",
            label_schema=LabelSchema(kind="categorical", classes=(0, 1)),
            files={"weights": BundleFile("weights.pt", lambda p: torch.save(self.best.state, p))},
        )


# 3. Train, with MLflow logging + a progress bar + a training-curve plot for free.
trainer = MyTrainer(model, train_loader, val_loader, MyConfig(), device, callbacks=build_callbacks())
trainer.fit()
bundle = trainer.save("artifacts/my-model")


# 4. Serve - any process, same artifact format.
class MyPredictor:
    label_schema = LabelSchema(kind="categorical", classes=(0, 1))

    @classmethod
    def load(cls, bundle: ArtifactBundle, device="cpu") -> "MyPredictor": ...
    def predict(self, inputs): ...


predictor = MyPredictor.load(ArtifactBundle.read("artifacts/my-model"))
predictor.predict([...])
```
