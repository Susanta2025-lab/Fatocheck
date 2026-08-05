"""
Centralized runtime configuration for FatoCheck.

Paths, environment variables, and logging setup live here so application
modules do not recalculate directories or call logging.basicConfig() ad hoc.
"""

from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path
from typing import Literal, Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


DEFAULT_LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Artifact filenames — keep in sync with training exports and inference.
XGBOOST_ARTIFACT_NAME = "xgboost_pipeline.joblib"
LOGISTIC_REGRESSION_ARTIFACT_NAME = "logistic_regression_pipeline.joblib"
RANDOM_FOREST_ARTIFACT_NAME = "random_forest_pipeline.joblib"
BERT_ARTIFACT_DIR_NAME = "bert-base-uncased"

# Must match transformer training / inference.
BERT_MAX_LENGTH = 256

_LOGGING_CONFIGURED = False


class Settings(BaseSettings):
    """Typed runtime settings loaded from environment variables and `.env`."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    log_level: str = Field(default="INFO", validation_alias="LOG_LEVEL")
    api_host: str = Field(default="0.0.0.0", validation_alias="API_HOST")
    api_port: int = Field(default=8000, validation_alias="API_PORT")
    default_model_type: Literal["xgboost", "bert"] = Field(
        default="xgboost",
        validation_alias="MODEL_TYPE",
    )

    # Optional override for deployments/tests. When unset, uses <repo>/models/trained.
    models_dir_override: Optional[Path] = Field(
        default=None,
        validation_alias="MODELS_DIR",
    )

    @property
    def base_dir(self) -> Path:
        return Path(__file__).resolve().parents[1]

    @property
    def models_dir(self) -> Path:
        if self.models_dir_override is not None:
            return Path(self.models_dir_override)
        return self.base_dir / "models" / "trained"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / "data"

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / "logs"

    @property
    def results_dir(self) -> Path:
        return self.base_dir / "results" / "evaluation"

    @property
    def plots_dir(self) -> Path:
        return self.results_dir / "plots"

    @property
    def xgboost_model_path(self) -> Path:
        return self.models_dir / XGBOOST_ARTIFACT_NAME

    @property
    def logistic_regression_model_path(self) -> Path:
        return self.models_dir / LOGISTIC_REGRESSION_ARTIFACT_NAME

    @property
    def random_forest_model_path(self) -> Path:
        return self.models_dir / RANDOM_FOREST_ARTIFACT_NAME

    @property
    def bert_model_path(self) -> Path:
        return self.models_dir / BERT_ARTIFACT_DIR_NAME


@lru_cache
def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()


# Module-level singleton for convenient imports: `from utils.settings import settings`
settings = get_settings()


def configure_logging(
    level: Optional[str] = None,
    *,
    log_file: Optional[Path] = None,
    force: bool = False,
) -> None:
    """
    Configure process-wide logging once.

    Safe to call from the API and from CLI entry points. Subsequent calls are
    no-ops unless ``force=True``. Does not create directories unless a log file
    path is provided.
    """
    global _LOGGING_CONFIGURED

    if _LOGGING_CONFIGURED and not force:
        return

    resolved_level = (level or get_settings().log_level).upper()
    handlers: list[logging.Handler] = [logging.StreamHandler()]

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    root = logging.getLogger()
    for existing in list(root.handlers):
        root.removeHandler(existing)

    logging.basicConfig(
        level=getattr(logging, resolved_level, logging.INFO),
        format=DEFAULT_LOG_FORMAT,
        handlers=handlers,
        force=True,
    )

    _LOGGING_CONFIGURED = True
