"""Settings file lookup, `ragas-jev init`, and `--env-file` (no API calls)."""

import os
from pathlib import Path

import pytest
from typer.testing import CliRunner

from ragas_jev import config
from ragas_jev.cli import app

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Empty working directory and user config dir, no settings in the environment."""
    home = tmp_path / "home"
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setenv("APPDATA", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(home))
    for var in (config.ENV_FILE_VAR, "OPENAI_MODEL"):
        monkeypatch.delenv(var, raising=False)
    config.get_settings.cache_clear()
    yield work
    config.get_settings.cache_clear()


def test_no_env_file_anywhere(isolated):
    assert config.resolve_env_file() is None


def test_lookup_order(isolated, monkeypatch):
    user_file = config.user_config_dir() / ".env"
    user_file.parent.mkdir(parents=True)
    user_file.write_text("OPENAI_MODEL=from-user\n", encoding="utf-8")
    assert config.resolve_env_file() == user_file

    Path(".env").write_text("OPENAI_MODEL=from-cwd\n", encoding="utf-8")
    assert config.resolve_env_file() == Path(".env")

    chosen = isolated / "chosen.env"
    monkeypatch.setenv(config.ENV_FILE_VAR, str(chosen))
    assert config.resolve_env_file() == chosen
    assert config.resolve_env_file(Path("explicit.env")) == Path("explicit.env")


def test_environment_variable_beats_env_file(isolated, monkeypatch):
    Path(".env").write_text("OPENAI_MODEL=from-file\n", encoding="utf-8")
    assert config.get_settings().openai_model == "from-file"
    monkeypatch.setenv("OPENAI_MODEL", "from-env")
    config.get_settings.cache_clear()
    assert config.get_settings().openai_model == "from-env"


def test_init_writes_template_once(isolated):
    runner = CliRunner()
    result = runner.invoke(app, ["init"])
    assert result.exit_code == 0, result.output
    written = Path(".env").read_text(encoding="utf-8")
    assert "TYPESAFE_API_KEY=" in written

    Path(".env").write_text("TYPESAFE_API_KEY=kept\n", encoding="utf-8")
    result = runner.invoke(app, ["init"])
    assert "already exists" in result.output
    assert Path(".env").read_text(encoding="utf-8") == "TYPESAFE_API_KEY=kept\n"

    result = runner.invoke(app, ["init", "--force", "--sample"])
    assert result.exit_code == 0, result.output
    assert Path(".env").read_text(encoding="utf-8") == written
    assert Path("sample_ko.jsonl").read_text(encoding="utf-8").startswith('{"sample_id": "smoke-001"')


def test_init_user_file_is_found_from_any_directory(isolated):
    result = CliRunner().invoke(app, ["init", "--user"])
    assert result.exit_code == 0, result.output
    assert config.resolve_env_file() == config.user_config_dir() / ".env"


def test_env_file_option(isolated):
    settings_file = isolated / "custom.env"
    settings_file.write_text("OPENAI_MODEL=from-option\n", encoding="utf-8")
    seen = {}

    @app.command("show-model-for-test", hidden=True)
    def _show() -> None:
        seen["model"] = config.get_settings().openai_model

    try:
        result = CliRunner().invoke(app, ["--env-file", str(settings_file), "show-model-for-test"])
        assert result.exit_code == 0, result.output
        assert seen["model"] == "from-option"
        missing = CliRunner().invoke(app, ["--env-file", "missing.env", "show-model-for-test"])
        assert missing.exit_code != 0
    finally:
        app.registered_commands.pop()
        os.environ.pop(config.ENV_FILE_VAR, None)  # set by the option itself, not by monkeypatch


def test_packaged_files_match_repository_copies():
    data = ROOT / "src" / "ragas_jev" / "data"
    assert (data / "env.template").read_bytes() == (ROOT / ".env.example").read_bytes()
    assert (data / "sample_ko.jsonl").read_bytes() == (ROOT / "benchmark" / "datasets" / "smoke_ko.jsonl").read_bytes()
