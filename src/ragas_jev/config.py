"""Runtime settings loaded from environment variables and a `.env` file.

Field names match environment variable names (case-insensitive), so `.env`
entries such as `TYPESAFE_API_KEY` map directly onto `typesafe_api_key`.
Secrets are `SecretStr` so they never show up in reprs, logs, or result files.

Precedence: explicit arguments > environment variables > `.env` file > defaults.
The `.env` file is the first that exists of: `--env-file` / `RAGAS_JEV_ENV_FILE`,
`./.env`, and the user config file (`ragas-jev init --user` writes it).
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Generative LLM (any OpenAI-compatible endpoint; unset base URL = OpenAI API)
    openai_base_url: str | None = None
    openai_api_key: SecretStr = SecretStr("not-needed")
    openai_model: str = "gpt-5.6-luna"
    available_models: str = ""

    # JEV (TypeSafe System One API)
    typesafe_api_key: SecretStr | None = None
    typesafe_base_url: str | None = None
    typesafe_default_model: str = "jev-1.13.0"
    jev_timeout_sec: float = 30.0
    jev_max_concurrency: int = 8
    jev_max_questions_per_request: int = 40
    # JEV allows 32k tokens for state + longest question; stay well below in characters.
    jev_state_char_budget: int = 20_000

    # Routing thresholds (calibrated in Phase 5)
    ragas_jev_conf_accept: float = 0.85
    ragas_jev_conf_audit: float = 0.60
    ragas_jev_decision_threshold: float = 0.5

    # Models for each role
    ragas_jev_preprocessor_model: str | None = None
    ragas_jev_auditor_model: str = "gpt-6-luna"  # re-judges uncertain JEV decisions (Phase 5)
    ragas_jev_strong_judge_model: str = "gpt-6-sol"
    ragas_jev_baseline_judge_model: str = "gpt-6-sol"  # LLM judge that JEV is benchmarked against

    # Phase 5: escalation and calibration
    ragas_jev_audit_enabled: bool = True
    ragas_jev_calibration_file: Path | None = None  # default: packaged calibration for the pinned JEV model

    # Data protection / storage
    ragas_jev_pii_masking: bool = True
    ragas_jev_cache_dir: Path = Path(".cache/ragas_jev")

    @property
    def available_model_list(self) -> list[str]:
        return [m.strip() for m in self.available_models.split(",") if m.strip()]

    @property
    def preprocessor_model(self) -> str:
        return self.ragas_jev_preprocessor_model or self.openai_model

    @property
    def calibration_path(self) -> Path:
        if self.ragas_jev_calibration_file is not None:
            return self.ragas_jev_calibration_file
        return Path(__file__).parent / "data" / f"calibration_{self.typesafe_default_model}.json"


ENV_FILE_VAR = "RAGAS_JEV_ENV_FILE"


def user_config_dir() -> Path:
    if os.name == "nt":
        return Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming") / "ragas-jev"
    return Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "ragas-jev"


def resolve_env_file(explicit: Path | None = None) -> Path | None:
    """The `.env` file to load, or None to use environment variables only."""
    if explicit is not None:
        return explicit
    if os.environ.get(ENV_FILE_VAR):
        return Path(os.environ[ENV_FILE_VAR])
    for candidate in (Path(".env"), user_config_dir() / ".env"):
        if candidate.is_file():
            return candidate
    return None


@lru_cache
def get_settings() -> Settings:
    return Settings(_env_file=resolve_env_file())
