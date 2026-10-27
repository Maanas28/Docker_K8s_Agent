from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

from pydantic import BaseModel, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

# Feature flags for the agent's behaviour, documented in the file itself (backend/agent.toml).
AGENT_TOML = Path(__file__).resolve().parent.parent / "agent.toml"


class ContextSettings(BaseModel):
    """How the conversation is trimmed before each model call. See agent.toml for the why."""

    compact_tool_output: bool = True
    compact_log_lines: int = Field(60, ge=5)
    compact_table_rows: int = Field(40, ge=5)
    replace_outdated_outputs: bool = True
    attempt_log: bool = True
    attempt_log_size: int = Field(8, ge=1)
    relevant_tools_only: bool = True
    # Share of the context window that triggers compaction of the oldest tool outputs.
    compact_at_ratio: float = Field(0.75, gt=0, le=1)
    # project_context also states detected build facts (package manager, start command, port).
    build_facts: bool = True


# Providers the app may use. Only ones with a genuinely free tier (no card, no credits) are listed;
# see agent.toml for the catalog of models and their limits.
Provider = Literal["ollama", "groq", "gemini", "openrouter", "nvidia"]


class PoolSpec(BaseModel):
    """A provider limit that one or more models share. Unset = no limit of that kind."""

    requests_per_day: int | None = Field(None, ge=1)
    tokens_per_day: int | None = Field(None, ge=1)
    requests_per_minute: int | None = Field(None, ge=1)
    tokens_per_minute: int | None = Field(None, ge=1)
    # The provider's daily counters reset at midnight in this time zone.
    reset_timezone: str = "UTC"

    @model_validator(mode="after")
    def _known_zone(self) -> "PoolSpec":
        ZoneInfo(self.reset_timezone)  # raises for an unknown zone
        return self


class ModelSpec(BaseModel):
    """One entry of the model dropdown."""

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    label: str
    provider: Provider
    # The provider's model id. Empty for the local model = use ollama_model (AGENT_OLLAMA_MODEL).
    model: str = ""
    # The usage pool (limits + counter) this model draws from. Required for cloud models.
    pool: str | None = None


class QuotaSettings(BaseModel):
    """The usage counter for free cloud models. See agent.toml."""

    reserve_calls: int = Field(2, ge=0)
    max_wait_s: int = Field(90, ge=1)
    max_output_tokens: int = Field(4096, ge=256)
    request_timeout_s: int = Field(180, ge=10)


class ApprovalSettings(BaseModel):
    """Approval options. See agent.toml."""

    session_allow: bool = True


class RunSettings(BaseModel):
    """Running a project's image as a container (Step 4). See agent.toml."""

    health_timeout_s: int = Field(30, ge=3)
    health_interval_s: float = Field(1, gt=0)
    health_failure_threshold: int = Field(5, ge=1)
    health_paths: list[str] = ["/health", "/healthz", "/"]
    bind_address: str = "127.0.0.1"
    pass_env_file: bool = True


class KubernetesSettings(BaseModel):
    """Reading Kubernetes clusters (Phase 3). See agent.toml."""

    allowed_contexts: list[str] = ["docker-desktop"]
    kubeconfig: str | None = None  # None = ~/.kube/config (or $KUBECONFIG)
    request_timeout_s: int = Field(10, ge=1)
    log_tail_lines: int = Field(200, ge=10)


class LoopSettings(BaseModel):
    """How the agent behaves when steps fail (the build-and-fix loop). See agent.toml."""

    same_error_brake: bool = True
    failure_nudge: bool = True


class BuildSettings(BaseModel):
    """Building images from projects (Step 3). See agent.toml."""

    timeout_s: int = Field(900, ge=10)
    progress_interval_ms: int = Field(250, ge=50)
    warn_context_mb: int = Field(100, ge=1)
    max_log_lines: int = Field(5000, ge=100)


class Settings(BaseSettings):
    """Runtime settings: defaults here, overridden by agent.toml, overridden by AGENT_* env vars."""

    # env_parse_none_str lets e.g. AGENT_OLLAMA_THINK=none mean "not set".
    # env_nested_delimiter lets e.g. AGENT_CONTEXT__ATTEMPT_LOG=false reach a nested setting.
    model_config = SettingsConfigDict(
        env_prefix="AGENT_", env_file=".env", env_parse_none_str="none",
        env_nested_delimiter="__", toml_file=AGENT_TOML,
    )

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Earlier sources win: explicit arguments > env vars > .env > agent.toml > defaults.
        return (init_settings, env_settings, dotenv_settings, TomlConfigSettingsSource(settings_cls))

    ollama_host: str = "http://localhost:11434"
    # Uses native tool calls and handles multi-step work (read code, then write a file) far better
    # than qwen2.5-coder:7b, and still leaves room for Docker on 18 GB.
    ollama_model: str = "qwen3:8b"
    # Context window in tokens: how much text (prompt + history + tool output) the model sees per
    # call. Ollama's default is only a few thousand. 32k fits a project's key files plus the
    # conversation; it costs ~1-1.5 GB more RAM than 16k for an 8B model.
    ollama_num_ctx: int = 32768
    # Reasoning ("thinking") for models that support it, e.g. qwen3. false = fast replies (the
    # default: ~45 s for a Dockerfile vs ~5 min with it on); true = slower but more careful;
    # None = don't send the option (needed for models without a thinking mode, e.g. llama3.1).
    ollama_think: bool | None = False
    db_url: str = "sqlite:///agent.db"
    # Pending approvals are auto-rejected after this many seconds.
    approval_timeout_s: int = 300
    # Successful steps are unlimited (the user can press Stop). Only failures are capped: after
    # this many errors in one request (failed tools, refused pre-checks, invalid arguments...),
    # the agent stops and reports what kept failing.
    max_errors: int = 10
    # Tool output is truncated to this many characters before it is sent to the LLM.
    tool_output_llm_chars: int = 30000
    # How many recent messages are sent to the LLM.
    history_window: int = 30
    # Browser pages allowed to talk to the API. Any other website open in your browser is refused,
    # so it can't drive the agent or approve actions. Requests without an Origin header (curl,
    # scripts on this machine) are allowed.
    allowed_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]
    context: ContextSettings = ContextSettings()
    build: BuildSettings = BuildSettings()
    run: RunSettings = RunSettings()
    kubernetes: KubernetesSettings = KubernetesSettings()
    loop: LoopSettings = LoopSettings()
    approvals: ApprovalSettings = ApprovalSettings()

    # Model dropdown (agent.toml [[models]] and [pools.*]). Every chat starts on default_model.
    models: list[ModelSpec] = [ModelSpec(id="local", label="Local model", provider="ollama")]
    pools: dict[str, PoolSpec] = {}
    default_model: str = "local"
    quota: QuotaSettings = QuotaSettings()
    # API keys for the free cloud providers, from backend/.env (which you fill in yourself). SecretStr
    # keeps them out of logs and error messages; they're only ever sent to their own provider.
    groq_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    openrouter_api_key: SecretStr | None = None
    nvidia_api_key: SecretStr | None = None

    @model_validator(mode="after")
    def _check_models(self) -> "Settings":
        ids = [m.id for m in self.models]
        if len(ids) != len(set(ids)):
            raise ValueError("Model ids in agent.toml must be unique.")
        if self.default_model not in ids:
            raise ValueError(f"default_model '{self.default_model}' isn't one of the [[models]].")
        host = urlparse(self.ollama_host).hostname
        for m in self.models:
            if m.provider == "ollama":
                if host not in ("localhost", "127.0.0.1", "::1"):
                    # Keeps Ollama Cloud (paid beyond "light usage") out: local models only.
                    raise ValueError(f"ollama_host must be this machine, not {self.ollama_host}.")
                continue
            if not m.model:
                raise ValueError(f"Model '{m.id}' needs the provider's model id.")
            if m.provider == "openrouter" and not m.model.endswith(":free"):
                # Free-only rule: any other OpenRouter model costs money.
                raise ValueError(f"OpenRouter model '{m.model}' isn't a free (:free) model.")
            if m.pool is None or m.pool not in self.pools:
                raise ValueError(f"Cloud model '{m.id}' needs a pool from [pools] so its usage is counted.")
        return self


@lru_cache
def get_settings() -> Settings:
    return Settings()
