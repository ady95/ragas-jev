"""Runtime settings loaded from environment variables and `.env`.

Field names match environment variable names (case-insensitive), so `.env`
entries such as `TYPESAFE_API_KEY` map directly onto `typesafe_api_key`.
Secrets are `SecretStr` so they never show up in reprs, logs, or result files.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Generative LLM (openai-oauth proxy)
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
    ragas_jev_strong_judge_model: str = "gpt-6-astra"
    ragas_jev_baseline_judge_model: str = "gpt-6-sol"  # LLM judge that JEV is benchmarked against

    # Data protection / storage
    ragas_jev_pii_masking: bool = True
    ragas_jev_cache_dir: Path = Path(".cache/ragas_jev")

    @property
    def available_model_list(self) -> list[str]:
        return [m.strip() for m in self.available_models.split(",") if m.strip()]

    @property
    def preprocessor_model(self) -> str:
        return self.ragas_jev_preprocessor_model or self.openai_model


@lru_cache
def get_settings() -> Settings:
    return Settings()
