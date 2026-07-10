import logging
import os

import hydra
from omegaconf import DictConfig
from pytorch_lightning import seed_everything
from src.data.dataset import Dataset
from src.data.evaluator import Evaluator
from src.models.common.gluonts_predictor import GluonTSPredictor
from src.models.ensembles.slsqp import SLSQPEnsemble
from src.models.foundation import Moirai, Sundial, Toto, Toto2, TimesFM


@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    # Set seed, precision, and logging
    seed_everything(seed=cfg.seed, workers=cfg.workers, verbose=cfg.verbose)
    logging.basicConfig(**cfg.logging)

    # Load models
    models = [
        Moirai(batch_size=cfg.batch_size),
        Sundial(batch_size=cfg.batch_size),
        Toto(batch_size=cfg.batch_size),
        Toto2(batch_size=cfg.batch_size),
        TimesFM(batch_size=cfg.batch_size),
    ]

    # Ensemble the models
    logging.info(
        f"Ensembling {len(models)} models: {', '.join([m.alias for m in models])}",
    )
    forecaster = SLSQPEnsemble(models=models, **cfg.ensemble)

    # Prepare the ensemble for evaluation
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

    # Evaluate the ensemble and save the results
    evaluator.evaluate(predictor=predictor)

    # Display the ensemble weights across all cross-validation windows
    forecaster.print_weights()


if __name__ == "__main__":
    main()
