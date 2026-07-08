# Add Toto 2.0 (Toto-2.0-2.5B-FT) as a foundation model, run GIFT-Eval, prepare PR

> This file is the approved implementation plan, committed to the repo so a
> fresh Claude Code session (e.g. on a remote HPC cluster) can pick up the
> work with full context. See the "Continuing this work in a new session"
> section at the bottom before starting.

## Context

TSOrchestra (a fork of TimeCopilot) currently only supports Toto **1.0** via
[src/models/foundation/toto.py](src/models/foundation/toto.py), which wraps the
`timecopilot-toto==0.1.3` package (Datadog's `toto-ts`, pinned in
[environment.yml:369](environment.yml#L369)). The user wants to give their
forecasting pipeline access to the newer, larger **Toto 2.0** models —
specifically to load the `Datadog/Toto-2.0-2.5B-FT` checkpoint, run it through
the existing GIFT-Eval evaluation harness
([src/data/evaluator.py](src/data/evaluator.py)), and submit the results as a
PR to the official leaderboard at
https://huggingface.co/spaces/Salesforce/GIFT-Eval (submissions actually land
via PR to https://github.com/SalesforceAIResearch/gift-eval, which powers that
Space).

Toto 2.0 is a separate codebase/package from Toto 1.0 (confirmed by reading
Datadog's `toto` GitHub repo directly): different pip package (`toto-2`,
`from toto2 import ...`), different Python/torch minimums (Python **3.12+**,
torch **2.4.0+**, vs. this repo's current Python 3.11.11 pin), and a
completely different inference API (dict-based `{target, target_mask,
series_ids}` inputs, quantile-tensor output — no `.mean`/`.quantile()`
distribution object like Toto 1.0's `TotoForecaster`).

Crucially, Datadog ships an official GluonTS adapter, `Toto2GluonTSModel`,
whose `create_predictor(batch_size, device)` returns a stock
`gluonts.torch.model.predictor.PyTorchPredictor` — the exact same predictor
class that [Moirai](src/models/foundation/moirai.py) produces via its
`get_predictor()` method. This repo already has a generic bridge for that
shape of model: [GluonTSForecaster](src/models/common/gluonts_forecaster.py),
which implements `Forecaster.forecast(df, h, freq, level, quantiles)` once,
generically, on top of any subclass that supplies a `get_predictor()` context
manager. Moirai uses this bridge instead of hand-rolling DataFrame/tensor
plumbing like Toto 1.0 does. Toto 2.0 should do the same — this means the new
adapter can be small (no manual padding/masking/dict-construction code needed;
GluonTS's own transform + instance-splitter pipeline inside
`Toto2GluonTSModel` already handles that).

## Environment note: this must run on a GPU HPC cluster, not locally

This repo's `environment.yml` targets a remote Linux GPU cluster (`prefix`
points at a `/u/.../conda/envs/tso` style path, includes `nvidia-cu12`
packages, and [pipeline/eval.py](pipeline/eval.py) reads
`SLURM_ARRAY_TASK_ID` to select GIFT-Eval dataset configs from a SLURM job
array). It is not runnable on a local laptop.

The user runs this on **Delta** (NCSA's GPU cluster) via SSH / VS Code
Remote-SSH. Delta has several storage tiers, but note that **`$SCRATCH` is
not set on this account** — do not rely on it. This work uses the user's
**`bdem` allocation** instead, which has plenty of free space (unlike the
`bcqc` allocation, whose `/projects` and `/work/hdd` quotas are nearly full):

- **codebase + final results → `/projects/bdem/tliao2/`** — the `projects`
  tier is backed up and appropriate for source code and durable outputs
  (`all_results.csv`, `config.json` for the GIFT-Eval submission). The
  `bdem` allocation here has a 500 GB soft quota and is currently near-empty.
- **large caches (model checkpoints + datasets) → `/work/hdd/bdem/tliao2/`**
  — the `work` tier is the high-throughput space meant for large job I/O.
  The `bdem` allocation here has a ~1 TB soft quota and is currently
  near-empty. Note: there is **no `/work/nvme/bdem`** (only `bcqc` has an
  NVMe allocation), so `bdem` job I/O runs on the HDD tier — fine for this
  workload (read checkpoints + datasets once), just not as fast as NVMe.

**This project's storage footprint is large and must go on `bdem`, not
home**:
- The `Datadog/Toto-2.0-2.5B-FT` checkpoint alone is **9.82 GB**
  (`model.safetensors`).
- The full **GIFT-Eval benchmark** (98 dataset/term configs across many
  domains) is a large multi-GB download in its own right, on top of the
  model checkpoints for every ensemble member (Moirai, Sundial, Toto 1.0,
  TimesFM). The user's home HF cache is already ~13 GB; expect the full set
  to grow to ~30–50 GB.
- Clone the repo into `/projects/bdem/tliao2/` and redirect the HuggingFace
  cache to the `bdem` work tier so downloads don't fill the small home quota
  (~100 GB, already ~69 GB used):
  ```bash
  # codebase (backed up)
  cd /projects/bdem/tliao2
  git clone https://github.com/tliao730/TSorchestra.git

  # large caches (work tier) — set before any download/run
  export HF_HOME=/work/hdd/bdem/tliao2/huggingface
  ```
  Do **not** clone into `$HOME` — the home quota is small and would fill up
  quickly from the model cache (HuggingFace downloads to
  `~/.cache/huggingface` by default, hence the `HF_HOME` redirect above).
- The `projects` tier is backed up, so final results are durable there. If
  anything is instead staged under `work` (not backed up, may be purged after
  inactivity), copy the final `all_results.csv` / `config.json` back to
  `/projects/bdem/tliao2/` or commit them to git once produced.

## Repo / git setup

The working repo was originally cloned from the upstream lab repo
(`DC-research/TSorchestra`). The user has forked it to
**`https://github.com/tliao730/TSorchestra`** so that this work can be
committed and eventually used to open a PR (either back upstream, or as the
basis for the separate GIFT-Eval leaderboard PR described below).

On Delta, clone the **fork**, not the upstream repo:
```bash
git clone https://github.com/tliao730/TSorchestra.git
```
If a remote is later found already pointing at the upstream repo instead of
the fork, add the fork as a second remote rather than silently overwriting
`origin` — confirm with the user before changing push targets.

## Approach

### 1. Add a new `Toto2` adapter: `src/models/foundation/toto2.py`

Subclass `GluonTSForecaster` (not `Forecaster` directly — mirror
[moirai.py](src/models/foundation/moirai.py) exactly in structure):

```python
from contextlib import contextmanager

import torch
from gluonts.torch.model.predictor import PyTorchPredictor
from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig, Toto2Model

from src.models.common.gluonts_forecaster import GluonTSForecaster


class Toto2(GluonTSForecaster):
    """Toto 2.0 ... (docstring mirrors Toto 1.0's style, links to
    https://github.com/DataDog/toto and the 2.0 HF collection)"""

    def __init__(
        self,
        repo_id: str = "Datadog/Toto-2.0-2.5B-FT",
        context_length: int = 4096,
        batch_size: int = 16,
        quantiles: list[float] | None = None,
        target_dim: int = 1,
        past_feat_dynamic_real_dim: int = 0,
        alias: str = "Toto2",
    ):
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.quantiles = quantiles or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.target_dim = target_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.alias = alias
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @contextmanager
    def get_predictor(self, prediction_length: int) -> PyTorchPredictor:
        model = Toto2Model.from_pretrained(self.repo_id, map_location=self.device)
        model = model.to(self.device).eval()
        gts_config = Toto2GluonTSModelConfig(
            prediction_length=prediction_length,
            context_length=self.context_length,
            target_dim=self.target_dim,
            past_feat_dynamic_real_dim=self.past_feat_dynamic_real_dim,
            quantiles=self.quantiles,
        )
        gts_model = Toto2GluonTSModel(model, gts_config).to(self.device).eval()
        predictor = gts_model.create_predictor(
            batch_size=self.batch_size,
            device=self.device,
        )
        try:
            yield predictor
        finally:
            del predictor, gts_model, model
            torch.cuda.empty_cache()
```

Notes:
- `GluonTSForecaster.__init__` (base class) takes `repo_id, filename, alias,
  num_samples` — but Toto2 doesn't use `filename`/`num_samples` (it's
  quantile-based, not sample-based) or HF `hf_hub_download`/`torch.load`
  (`Toto2Model.from_pretrained` handles its own download). So **override
  `__init__` fully** instead of calling `super().__init__()`, same as Moirai
  does not fully reuse the base constructor semantics either — check
  `GluonTSForecaster.forecast()` at
  [gluonts_forecaster.py:116-197](src/models/common/gluonts_forecaster.py#L116-L197)
  only calls `self.get_predictor(...)`, `self.alias`, and
  `self._maybe_infer_freq` — so as long as `self.alias` is set, a custom
  `__init__` is safe.
- `num_samples` is passed by `GluonTSForecaster.forecast()` into
  `predictor.predict(gluonts_dataset, num_samples=self.num_samples)`
  ([gluonts_forecaster.py:183](src/models/common/gluonts_forecaster.py#L183)).
  Since `Toto2`'s predictor is quantile-based (`QuantileForecastGenerator`),
  confirm during implementation whether `PyTorchPredictor.predict()` silently
  ignores `num_samples` for quantile forecasters (GluonTS convention is that
  it does) — if not, set `self.num_samples = None` explicitly or check
  GluonTS's `PyTorchPredictor.predict` signature.
- Register in [src/models/foundation/\_\_init\_\_.py](src/models/foundation/__init__.py):
  add `from .toto2 import Toto2` and add `"Toto2"` to `__all__`.

### 2. Dependency changes: `environment.yml`

- Add `toto-2` (the actual PyPI/package name confirmed from
  `toto2/pyproject.toml`, NOT `toto-models`) as a new pip dependency.
- **Python version conflict**: `toto-2` requires Python **3.12+**; this repo's
  `environment.yml` pins `python=3.11.11` ([environment.yml:21](environment.yml#L21)).
  Because `Toto2` must live in the **same process** as the other ensemble
  models (see pipeline section below — it's a peer in `SLSQPEnsemble`, not a
  standalone job), an isolated env is not viable long-term: **the shared `tso`
  env itself must be bumped to Python 3.12**.
  This is a real, moderately risky change and should be done as its own
  isolated first step, separate from writing any `Toto2` code:
  1. Bump `python=3.12.x` in `environment.yml`.
  2. Rebuild the env (`conda env update` / recreate) and let pip resolve —
     watch specifically for `numba==0.62.1` (historically the slowest package
     to support new Python versions, via its pinned `llvmlite` dependency —
     `llvmlite==0.45.1` is also pinned in this env and the two must stay in
     lockstep), plus native-wheel availability for `torch==2.9.0`,
     `jax==0.7.1`/`jaxlib==0.7.1`, and `ray==2.50.1` on 3.12.
  3. If any package has no 3.12-compatible build at its pinned version, that
     forces a version bump for that package too — surface this to the user
     for a decision rather than silently changing pins beyond what's needed
     for `toto-2`.
  4. After the env rebuilds cleanly, smoke-import every existing foundation
     model class (`Moirai`, `Sundial`, `TabPFN`, `TimesFM`, `TiRex`, `Toto`)
     and run the existing test suite to confirm nothing regressed on 3.12
     *before* adding any `Toto2` code.
  This is best validated empirically (actually resolving the env) rather than
  predicted from package changelogs alone — version-compatibility research
  without an actual resolver run is not reliable enough to act on.

  **PyPI metadata check (pre-flight, not a substitute for step 2's actual
  resolve)**: `numba==0.62.1` and its pinned `llvmlite==0.45.1` both declare
  Python 3.10–3.13 support — the historically-risky package turns out fine.
  `torch==2.9.0`, `jax==0.7.1`/`jaxlib==0.7.1`, `numpy==1.26.4`, `ray==2.50.1`,
  `statsforecast==2.0.2`, `neuralforecast==3.1.2`, `mlforecast==1.0.2`, and
  `nixtla==0.7.0` all declare 3.12 support at their pinned versions.
  `pytorch-lightning==2.4.0` turned out fine on closer inspection (3.12
  support/testing landed in this exact 2.4 milestone per GitHub PR #20078,
  with one minor known `AdvancedProfiler` issue unlikely to affect this repo's
  usage of `seed_everything`). `tabpfn==2.2.1` is also confirmed fine (3.12
  explicitly classified).

  Three real risks, confirmed from PyPI JSON + GitHub sources:
  - **`gluonts==0.16.2`**: no Python 3.12 classifier, and a GitHub discussion
    (awslabs/gluonts#3223) confirms the **MXNet backend** is not
    3.12-compatible without manual patching. This repo only uses GluonTS's
    PyTorch-based predictor path ([GluonTSForecaster](src/models/common/gluonts_forecaster.py),
    [GluonTSPredictor](src/models/common/gluonts_predictor.py)) — confirm no
    code path imports `gluonts.mxnet` before assuming this is safe.
  - **`prophet==1.2.1`**: PyPI classifiers cap at Python 3.11; the 3.12
    classifier was only added in `prophet==1.3.0`. This pin will need
    bumping — check for breaking changes in the 1.2→1.3 changelog first.
  - **`mlstm-kernels==2.0.1`** (an `xlstm` dependency): only a generic
    `Python :: 3` classifier, and it depends on Triton, which has
    historically lagged in Python 3.12 wheel availability — real risk despite
    permissive metadata; must be verified empirically.

  Unknown/needs manual testing (permissive metadata, not explicitly
  confirmed): `tabpfn-time-series==1.0.3`, `xlstm==2.0.5`, `cmdstanpy==1.3.0`.
- `torch>=2.4.0` is already satisfied by the existing `torch==2.9.0` pin.

### 3. Wire into the eval pipeline: `pipeline/eval.py`

`Toto2` becomes a **permanent member of the ensemble**, alongside Moirai,
Sundial, Toto (1.0), and TimesFM — not a one-off standalone script. Update:

- [pipeline/eval.py:11](pipeline/eval.py#L11): add `Toto2` to the import from
  `src.models.foundation`.
- [pipeline/eval.py:21-26](pipeline/eval.py#L21-L26): add `Toto2(batch_size=cfg.batch_size)`
  to the `models` list passed into `SLSQPEnsemble`, e.g.:
  ```python
  models = [
      Moirai(batch_size=cfg.batch_size),
      Sundial(batch_size=cfg.batch_size),
      Toto(batch_size=cfg.batch_size),
      Toto2(batch_size=cfg.batch_size),
      TimesFM(batch_size=cfg.batch_size),
  ]
  ```

This means `Toto2` must run in the **same Python process/environment** as
every other ensemble member — ruling out an isolated conda env as a
long-term solution for the Python 3.12 requirement (see dependency section
above; an isolated env only works for a one-off standalone GIFT-Eval scoring
run, not for the ensemble/agent use case).

For the **GIFT-Eval submission specifically** (scoring Toto 2.0 alone, per
the leaderboard's per-model submission format), reuse the exact same
`pipeline/eval.py` flow but swap the `models` list to just
`[Toto2(batch_size=cfg.batch_size)]` and pass that single model straight into
`GluonTSPredictor` (skip `SLSQPEnsemble`, since ensembling isn't meaningful
for a single-model score) for that run only — e.g. via a Hydra config
override or a small variant entrypoint, without removing `Toto2` from the
main ensemble list used for the agent.

Either way, reuse [Evaluator.evaluate()](src/data/evaluator.py#L59-L136)
unchanged — it already writes results in the exact per-dataset
`eval_metrics/...` column format GIFT-Eval expects, appending to
`results.csv` per model/dataset-config via
[resolve_output_path](src/utils/path.py).

### 4. Assemble the GIFT-Eval submission

Per the official `SalesforceAIResearch/gift-eval` repo convention (confirmed
via its README/existing submissions):

```
results/Toto_2_0_2_5B_FT/
├── all_results.csv      # 98 rows × 15 columns, matches results/naive/all_results.csv schema
└── config.json
```

- `all_results.csv`: concatenate the per-dataset `results.csv` files this
  repo's `Evaluator` produces (one per `(alias, dataset_config)` pair) across
  all 98 GIFT-Eval dataset/term configs into a single file with columns
  `dataset, model, eval_metrics/MSE[mean], ..., domain, num_variates` — this
  already matches [Evaluator](src/data/evaluator.py#L107-L126)'s output
  column names, so it's a concatenation/rename step, not a reformatting step.
- `config.json`: new file, fields `model`, `model_type` (likely
  `"pretrained"` or `"fine-tuned"` given the FT checkpoint — flag to user,
  since the model card itself calls it a benchmarking-only checkpoint),
  `model_dtype`, `model_link` (the HF repo URL), `code_link` (this repo, or
  Datadog's toto repo), `org`, `testdata_leakage`, `replication_code_available`.
- Fork `SalesforceAIResearch/gift-eval`, add the `results/Toto_2_0_2_5B_FT/`
  folder, open a PR. **This is a user-facing/external action — do not open
  the PR automatically; prepare the files and let the user review before
  pushing/opening it**, per the standing guidance to confirm before
  actions visible to others.

## Verification

1. **Env sanity**: on the Delta env (Python 3.12), `pip install toto-2` and
   confirm `from toto2 import Toto2Model, Toto2GluonTSModel,
   Toto2GluonTSModelConfig` imports cleanly.
2. **Smoke test the adapter**: instantiate `Toto2()`, call
   `.forecast(df, h=...)` on a small synthetic `pd.DataFrame` (a few
   `unique_id`s, short series) and confirm it returns a DataFrame shaped like
   [Toto.forecast()](src/models/foundation/toto.py#L171-L246)'s output
   (`unique_id`, `ds`, `Toto2`, and `Toto2-q-*` columns).
2b. **Verify `num_samples` handling** noted above — confirm no crash/warning
   when `GluonTSForecaster.forecast()` passes `num_samples=100` into a
   quantile-only predictor's `.predict()`.
3. **End-to-end eval**: run the chosen `pipeline/eval.py` variant against one
   small GIFT-Eval dataset config (e.g. `m4_hourly`, short-term) and confirm
   `results.csv` is written with the expected 15 columns and sane
   (non-NaN, finite) metric values.
4. **Full GIFT-Eval sweep**: run across all 98 dataset/term configs (this is
   long-running — likely via the existing SLURM array mechanism in
   [pipeline/eval.py:40-46](pipeline/eval.py#L40-L46)), then assemble
   `all_results.csv` and diff its shape/columns against
   `results/naive/all_results.csv` in the `gift-eval` repo for schema parity.
5. **Before opening the PR**: show the user the final `results/Toto_2_0_2_5B_FT/`
   folder contents for review.

**Note per user instruction (as of this plan being written): testing steps
(1–5 above) are deliberately deferred.** Prioritize getting the environment,
dependency, and code changes (sections 1–4 of Approach) in place first; do
not block on running tests until the user asks for it.

## Continuing this work in a new session

This plan was written and approved in a Claude Code session running locally
(no GPU, no Delta access). The actual implementation is meant to happen in a
**separate Claude Code session running on Delta** (via VS Code Remote-SSH),
since that's where the GPU, large scratch storage, and SLURM scheduler are.

If you are a fresh session picking this up on Delta:
1. Read this whole file — it's the complete, approved plan with all
   technical research already done (exact Toto2 API shapes, GluonTS adapter
   classes, dependency compatibility findings). You should not need to
   re-research Toto2's source code or GIFT-Eval's submission format from
   scratch — it's all above.
2. Work under the **`bdem` allocation**, not `$SCRATCH` (which is unset on
   this account): clone/work in `/projects/bdem/tliao2/` and set
   `HF_HOME=/work/hdd/bdem/tliao2/huggingface`, per the "Environment note"
   section above.
3. Follow the "Approach" sections in order (1–4). Testing (see Verification)
   is deferred per user instruction — don't run it unprompted.
4. This file can be deleted or moved once the work is merged/no longer
   needed for reference — it's a working document, not permanent repo
   documentation.
