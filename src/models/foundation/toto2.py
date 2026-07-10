from contextlib import contextmanager

import torch
from gluonts.torch.model.predictor import PyTorchPredictor
from toto2 import Toto2GluonTSModel, Toto2GluonTSModelConfig, Toto2Model

from src.models.common.gluonts_forecaster import GluonTSForecaster


class Toto2(GluonTSForecaster):
    """
    Toto 2.0 is Datadog's second-generation open-weights foundation model for
    time series forecasting. Compared to Toto 1.0 it is a larger, decoder-only
    model with a dict-based inference API (``{target, target_mask,
    series_ids}``) that produces quantile forecasts directly rather than a
    sampled distribution object.

    This adapter loads the ``Datadog/Toto-2.0-2.5B-FT`` checkpoint via
    Datadog's official GluonTS bridge (``Toto2GluonTSModel``), whose
    ``create_predictor`` returns a stock
    ``gluonts.torch.model.predictor.PyTorchPredictor`` — the same predictor
    shape Moirai produces — so all of the DataFrame/tensor plumbing is reused
    from :class:`GluonTSForecaster` rather than hand-rolled here (mirrors
    :class:`~src.models.foundation.moirai.Moirai`).

    See the [official repo](https://github.com/DataDog/toto) and the
    [Hugging Face collection](https://huggingface.co/collections/Datadog/toto)
    for more details.
    """

    def __init__(
        self,
        repo_id: str = "Datadog/Toto-2.0-2.5B-FT",
        context_length: int = 4096,
        batch_size: int = 16,
        quantiles: list[float] | None = None,
        target_dim: int = 1,
        past_feat_dynamic_real_dim: int = 0,
        decode_block_size: int | None = None,
        alias: str = "Toto2",
    ):
        """
        Args:
            repo_id (str, optional): Hugging Face Hub model ID (or local path)
                for the Toto 2.0 checkpoint. Defaults to
                "Datadog/Toto-2.0-2.5B-FT".
            context_length (int, optional): Maximum context length (input
                window size). Defaults to 4096.
            batch_size (int, optional): Batch size for inference. Defaults to
                16 (lower than Moirai's default because the 2.5B checkpoint is
                large — tune to available GPU memory).
            quantiles (list[float], optional): Quantile levels to forecast.
                Toto 2.0 is quantile-based (not sample-based). Defaults to
                the deciles [0.1, ..., 0.9].
            target_dim (int, optional): Number of target variables (for
                multivariate forecasting). Defaults to 1.
            past_feat_dynamic_real_dim (int, optional): Number of past dynamic
                real covariates. Defaults to 0.
            decode_block_size (int, optional): Block size for iterative
                (KV-cached) decoding. When None (the toto-2 default), the
                whole horizon is decoded in a single pass — simplest, but
                memory-hungry for the 2.5B checkpoint on long horizons. Set a
                multiple of the model's patch size (e.g. 64) to decode in
                blocks with median feedback and bound GPU memory if a full
                GIFT-Eval horizon OOMs. Defaults to None.
            alias (str, optional): Name used for the model in output
                DataFrames and logs. Defaults to "Toto2".

        Notes:
            **Resources:**

            - GitHub: [DataDog/toto](https://github.com/DataDog/toto)
            - HuggingFace: [Datadog/Toto-2.0-2.5B-FT](https://huggingface.co/Datadog/Toto-2.0-2.5B-FT)

            **Technical Details:**

            - The model is loaded onto the best available device (GPU if
              available, otherwise CPU).
            - ``__init__`` is overridden fully instead of calling
              ``super().__init__()``: the base ``GluonTSForecaster``
              constructor expects ``filename``/``num_samples`` and downloads
              via ``hf_hub_download``, none of which apply here
              (``Toto2Model.from_pretrained`` handles its own download and the
              model is quantile-, not sample-, based).
            - ``self.num_samples`` is still set because
              ``GluonTSForecaster.forecast()`` passes it into
              ``predictor.predict(..., num_samples=self.num_samples)``. For a
              quantile forecaster GluonTS ignores this value, but the
              attribute must exist to avoid an ``AttributeError``.
        """
        self.repo_id = repo_id
        self.context_length = context_length
        self.batch_size = batch_size
        self.quantiles = quantiles or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        self.target_dim = target_dim
        self.past_feat_dynamic_real_dim = past_feat_dynamic_real_dim
        self.decode_block_size = decode_block_size
        self.alias = alias
        # See note above: forecast() references self.num_samples even though
        # the quantile-based predictor ignores it.
        self.num_samples = 100
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
            decode_block_size=self.decode_block_size,
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
