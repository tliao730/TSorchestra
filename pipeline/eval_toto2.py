import logging
import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.data.dataset import Dataset
from src.data.evaluator import Evaluator
from src.models.common.gluonts_predictor import GluonTSPredictor
from src.models.foundation import Toto2


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    """
    Standalone GIFT-Eval scoring run for Toto 2.0 (Datadog/Toto-2.0-2.5B-FT).

    Unlike pipeline/eval.py, this evaluates the single model directly (no
    SLSQPEnsemble) — the GIFT-Eval leaderboard's per-model submission format
    scores one model alone, so ensembling isn't meaningful here.
    """
    # Set seed, precision, and logging
    seed_everything(seed=cfg.seed, workers=cfg.workers, verbose=cfg.verbose)
    logging.basicConfig(**cfg.logging)

    forecaster = Toto2(batch_size=cfg.batch_size)

    predictor = GluonTSPredictor(
        forecaster=forecaster,
        batch_size=cfg.batch_size,
    )

    # Load list of dataset cfgs and use SLURM_ARRAY_TASK_ID to index the list.
    # Defaults to 38, which is the M4 Hourly dataset (short-term).
    cfg.data = cfg.data[int(os.environ.get("SLURM_ARRAY_TASK_ID", 38))]
    dataset_name, term = cfg.data.name, cfg.data.term

    logging.info(f"Loading dataset: {dataset_name} ({term}-term)")
    dataset = Dataset(name=dataset_name, term=term)

    evaluator = Evaluator(
        dataset=dataset,
        batch_size=cfg.batch_size,
        verbose=cfg.verbose,
    )

    # Evaluate the model and save the results
    evaluator.evaluate(predictor=predictor)


if __name__ == "__main__":
    main()
