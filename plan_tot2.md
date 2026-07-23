# Add Toto 2.0 (Toto-2.0-2.5B-FT) as a foundation model, run GIFT-Eval, prepare PR

> This file is the approved implementation plan, committed to the repo so a
> fresh Claude Code session (e.g. on a remote HPC cluster) can pick up the
> work with full context. See the "Continuing this work in a new session"
> section at the bottom before starting.

## Status at a glance (update as you go)

- ✅ **Code written & committed** — the `Toto2` adapter
  ([src/models/foundation/toto2.py](src/models/foundation/toto2.py)), its
  registration ([src/models/foundation/\_\_init\_\_.py](src/models/foundation/__init__.py)),
  and the pipeline wiring ([pipeline/eval.py](pipeline/eval.py)) are all done.
  This code was written on a machine **without** `toto-2` installed, so its
  `toto2` API usage (class names, config fields) reflects careful reading of
  Datadog's source but has **not been import-checked** — verify on first run.
- ✅ **Environment built & import-verified** (2026-07-09) — Python 3.12.12,
  torch 2.9.0+cu128; `from toto2 import ...` and all seven foundation model
  classes (incl. `Toto2`) import cleanly, and `Toto2()` instantiates.
  Verification step 1 is therefore done; steps 2–5 (forecast smoke test,
  eval runs, submission) remain and are run by the user personally.
- ✅ `environment.yml` pins `python=3.12`,
  `prophet==1.3.0` (first release with 3.12 wheels), and `toto-2==2.0.0`
  (now on PyPI — no git install needed; `dd-unit-scaling` is also on PyPI
  and resolves automatically). The `tso` env lives at
  `/work/nvme/bcqc/tliao2/conda/envs/tso` (conda via
  `module load miniforge3-python`; envs_dirs/pkgs_dirs set in `~/.condarc`).
  **Do not put anything on `/projects/bdem`** — that allocation is over its
  file (inode) quota (822k used / 750k soft as of 2026-07-09), which fails
  writes with `Disk quota exceeded` even though block space is free. Per
  user decision, everything (env, caches, results) stays on `/work/nvme`;
  durability comes from git, not a backup tier.
- ✅ **Toto2 wired into the ensemble** — [pipeline/eval.py](pipeline/eval.py)
  builds the SLSQP ensemble from 5 hardcoded models including
  `Toto2(batch_size=cfg.batch_size)`, so the existing
  [cli/eval.sh](cli/eval.sh) sweep already scores Toto2 as an ensemble member.
  The final GIFT-Eval submission is the **5-model ensemble** (not standalone
  Toto2), so the `model` column is the ensemble alias
  (`SLSQPEnsemble_5-models_opt-<metric>_<n>-windows`) that the Evaluator writes
  automatically — no per-model rename.
  - The standalone single-model entrypoint (`pipeline/eval_toto2.py`) and its
    SLURM script (`cli/eval_toto2.sh`) were **removed** — the ensemble path
    covers Toto2, so they were redundant.
  - The submission assembler
    ([pipeline/assemble_submission.py](pipeline/assemble_submission.py)) is
    model-agnostic (scripted, hardened version of
    [notebooks/results.ipynb](notebooks/results.ipynb)); run it with
    `--alias SLSQPEnsemble_5-models_opt-mae_1-windows` to assemble the ensemble
    submission. (Its defaults still target a standalone-Toto2 submission.)
- ⏳ **Nothing run / tested yet** — no smoke test, no GIFT-Eval sweep, no
  submission assembled (Verification §1–5). Testing is deliberately deferred
  per user instruction until explicitly asked. Note the GIFT-Eval **datasets
  are not downloaded yet** (no `data/` dir, no `.env`) — that download is the
  first prerequisite of any eval run.

**Where to work:** `/work/nvme/bcqc/tliao2/` on Delta (see "Environment
note"). Not backed up → push to git often.

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
not set on this account** — do not rely on it.

**All of this work — codebase, HuggingFace cache, datasets, and results —
lives under `/work/nvme/bcqc/tliao2/`.** This is the `bcqc` allocation's
**NVMe `work` tier**: high-throughput solid-state storage, the fastest tier
available and the right place for the read-heavy checkpoint + dataset I/O
this eval does. As of this plan it holds ~396 GB of a 500 GB soft quota, so
there is headroom for this project's ~30–50 GB footprint, but keep an eye on
it. (The `bcqc` `/projects` and `/work/hdd` tiers are near-full — 890/900 GB
and 2.25/2.25 TB respectively — so do **not** use those; and there is no
`/work/nvme/bdem`, only `bcqc` has an NVMe allocation.)

**Trade-off to be aware of — the `work` tier is not backed up and may be
auto-purged after a period of inactivity.** That is fine for re-downloadable
things (checkpoints, datasets) but means the codebase and final results are
**not durable here**. Mitigations, both important:
- The codebase's durability comes from **git**: the fork at
  `https://github.com/tliao730/TSorchestra` is the backup. Commit and push
  often; treat the `/work/nvme/bcqc` working copy as disposable.
- Once the final GIFT-Eval outputs (`all_results.csv`, `config.json`) are
  produced, **commit them to git** — that is the durable copy. Do **not**
  copy to `/projects/bdem`: it is over its file (inode) quota (822k used /
  750k soft as of 2026-07-09) and writes there fail with `Disk quota
  exceeded`. Per user decision everything stays on `/work/nvme`.

**Why the footprint is large:**
- The `Datadog/Toto-2.0-2.5B-FT` checkpoint alone is **9.82 GB**
  (`model.safetensors`).
- The full **GIFT-Eval benchmark** (98 dataset/term configs across many
  domains) is a large multi-GB download in its own right, on top of the
  model checkpoints for every ensemble member (Moirai, Sundial, Toto 1.0,
  TimesFM). Expect the full set to grow to ~30–50 GB.

**Setup on a fresh Delta login — clone the fork and point the HF cache off
`$HOME` before any download/run:**
```bash
# codebase (durability = git; treat this copy as disposable)
cd /work/nvme/bcqc/tliao2
git clone https://github.com/tliao730/TSorchestra.git
cd TSorchestra

# large HF cache (bdem work-hdd tier: ~177 GB block / ~694k inode headroom
# as of 2026-07-09) — already exported in ~/.bashrc on this account
export HF_HOME=/work/hdd/bdem/tliao2/huggingface
```
Do **not** clone into `$HOME` — the home quota is small (~100 GB, already
~69 GB used) and the model cache would fill it quickly (HuggingFace downloads
to `~/.cache/huggingface` by default, hence the `HF_HOME` redirect above).

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

**Status: DONE — the file is already written and committed**
([src/models/foundation/toto2.py](src/models/foundation/toto2.py)). It
subclasses `GluonTSForecaster` (not `Forecaster` directly), mirroring
[moirai.py](src/models/foundation/moirai.py) in structure: a fully-overridden
`__init__` plus a `get_predictor(prediction_length)` context manager that
loads `Toto2Model.from_pretrained`, wraps it in `Toto2GluonTSModel` with a
`Toto2GluonTSModelConfig`, and yields the resulting `PyTorchPredictor`
(cleaning up GPU memory on exit). It imports
`from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig, Toto2Model`.

The import will fail until `toto-2` is installed (section 2) — that is
expected; the code is complete but not yet runnable. Read the committed file
for the exact implementation rather than duplicating it here.

Notes (design decisions baked into the committed file):
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
  `Toto2`'s predictor is quantile-based — its output head is a
  `QuantileKnotsOutputHead` feeding GluonTS's `QuantileForecastGenerator`
  (confirmed in toto2's `model.py`), and the GluonTS convention is that
  `QuantileForecastGenerator` ignores `num_samples`. So the value is harmless,
  but **`self.num_samples` must still be set** (any int) or
  `forecast()` raises `AttributeError` at that line. The adapter sets
  `self.num_samples = 100`. Still worth a quick runtime check (Verification
  step 2b) that no warning/error surfaces.
- The GluonTS config class is `Toto2GluonTSModelConfig` (a dataclass, verified
  in toto2's `configuration.py`). Required fields (no default):
  `prediction_length`, `context_length`, `target_dim`. Useful optional fields
  with defaults: `past_feat_dynamic_real_dim=0`, `feat_dynamic_real_dim=0`,
  `decode_block_size=None`, `has_missing_values=True`,
  `quantiles=[0.1..0.9]`, `imputation_internal="ffill"`,
  `scaler_fallback_min_obs=8`, `quantile_real_cap_k=1e4`. The adapter exposes
  `decode_block_size` as a constructor arg (default `None` = decode the whole
  horizon in one pass; set a multiple of the model's patch size to bound GPU
  memory on long horizons — likely needed for the 2.5B checkpoint on the
  longer GIFT-Eval terms).
- Register in [src/models/foundation/\_\_init\_\_.py](src/models/foundation/__init__.py):
  add `from .toto2 import Toto2` and add `"Toto2"` to `__all__`.

### 2. Dependency changes: `environment.yml`

- Add `toto-2` as a new pip dependency. The distribution name is `toto-2`
  (confirmed from `toto2/pyproject.toml`: `name = "toto-2"`, `version =
  "2.0.0"`), imported as `from toto2 import ...`. It is **not on PyPI as a
  normal wheel** — install it from Datadog's git repo, from the `toto2`
  subdirectory:
  ```
  pip install "toto-2 @ git+https://github.com/DataDog/toto.git#subdirectory=toto2"
  ```
  Its own dependencies (all satisfiable): `torch>=2.4.0`, `einops>=0.7.0`,
  `numpy>=1.26.0`, `gluonts[torch]>=0.16.0`, `huggingface-hub>=0.20.0`,
  `safetensors>=0.4.0`, `jaxtyping>=0.2.25`, `dd-unit-scaling>=0.1.0`,
  `matplotlib>=3.9.0`. Note the extra transitive dep **`dd-unit-scaling`**
  (imported as `dd_unit_scaling` / `uu` inside toto2) — it also installs from
  git and provides the u-μP unit-scaling layers the model uses.
- **Python version conflict**: `toto-2` requires Python **3.12+**; this repo's
  `environment.yml` pins `python=3.11.11` ([environment.yml:21](environment.yml#L21)).

  **Why 3.12 (verified by reading toto2's source, not just its metadata):**
  `toto2/pyproject.toml` declares `requires-python = ">=3.12"` and
  `target-version = "py312"`, so **pip will refuse to install it on 3.11**
  regardless of what the code actually uses. Reading the source, the highest
  language feature actually used is `from typing import NotRequired` (PEP 655,
  added to `typing` in Python **3.11**) in the `Toto2ModelInputs` TypedDict —
  no PEP 695 generics or other 3.12-only *syntax* were found in
  `model.py`/`configuration.py`. So the `>=3.12` floor is largely Datadog's
  **support policy**, not a hard syntactic requirement. In principle one could
  bypass `requires-python` and install on 3.11, but that fights the upstream
  declaration and risks 3.12-only code in the transitive `dd-unit-scaling`
  dep (not inspected) — **not worth it. Bump the shared env to 3.12 as
  planned.**
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

**Status: DONE — already committed.** `Toto2` is now a **permanent member of
the ensemble**, alongside Moirai, Sundial, Toto (1.0), and TimesFM — not a
one-off standalone script. Two edits, both already made:

- [src/models/foundation/\_\_init\_\_.py](src/models/foundation/__init__.py):
  `from .toto2 import Toto2` added, and `"Toto2"` added to `__all__`.
- [pipeline/eval.py:11](pipeline/eval.py#L11): `Toto2` added to the import from
  `src.models.foundation`.
- [pipeline/eval.py:21-27](pipeline/eval.py#L21-L27): `Toto2(batch_size=cfg.batch_size)`
  added to the `models` list passed into `SLSQPEnsemble`:
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
since that's where the GPU, large NVMe storage, and SLURM scheduler are.

If you are a fresh session picking this up on Delta:
1. Read this whole file — it's the complete, approved plan with all
   technical research already done (exact Toto2 API shapes, GluonTS adapter
   classes, dependency compatibility findings). You should not need to
   re-research Toto2's source code or GIFT-Eval's submission format from
   scratch — it's all above.
2. Work under **`/work/nvme/bcqc/tliao2/`** (the NVMe `work` tier), not
   `$SCRATCH` (which is unset on this account): clone the fork there and set
   note the HF cache lives at `HF_HOME=/work/hdd/bdem/tliao2/huggingface`
   (already exported in `~/.bashrc`), per the "Environment note" section
   above. Remember these tiers are not backed up — push to git often;
   final results are committed to git (do **not** copy to `/projects/bdem`,
   it is over its inode quota).
3. Approach §1 (adapter code) and §3 (pipeline wiring) are **already done and
   committed** — read them for context, but the next real work is **§2**
   (bump the env to Python 3.12 and install `toto-2`), then §4 (assemble the
   submission). Testing (see Verification) is deferred per user instruction —
   don't run it unprompted.
4. This file can be deleted or moved once the work is merged/no longer
   needed for reference — it's a working document, not permanent repo
   documentation.
