"""The model dropdown: which models exist, whether each can be used now, and a client for it.

The catalog (models, their usage pools and limits) is in agent.toml; the providers' fixed facts
(address, which key, how they report usage) are here. Only providers with a genuinely free tier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import httpx

from app.config import ModelSpec, Settings
from app.llm import LLM, OllamaLLM
from app.llm_openai import OpenAICompatLLM
from app.quota import HeaderSpec, Meter, Notify, QuotaTracker
from app.store import Store


@dataclass(frozen=True)
class ProviderInfo:
    label: str
    base_url: str = ""
    key_field: str = ""  # Settings attribute holding the API key
    key_env: str = ""  # what the user puts in backend/.env
    headers: HeaderSpec | None = None  # the provider's own daily request count, if it sends one
    thought_signatures: bool = False  # Gemini 3: tool calls must carry a signature


PROVIDERS: dict[str, ProviderInfo] = {
    "ollama": ProviderInfo("Local (Ollama)"),
    # Groq reports requests per day in x-ratelimit-*-requests (its tokens headers are per minute).
    "groq": ProviderInfo("Groq", "https://api.groq.com/openai/v1", "groq_api_key", "AGENT_GROQ_API_KEY",
                         HeaderSpec("x-ratelimit-limit-requests", "x-ratelimit-remaining-requests")),
    "gemini": ProviderInfo("Google Gemini", "https://generativelanguage.googleapis.com/v1beta/openai",
                           "gemini_api_key", "AGENT_GEMINI_API_KEY", thought_signatures=True),
    "openrouter": ProviderInfo("OpenRouter", "https://openrouter.ai/api/v1", "openrouter_api_key",
                               "AGENT_OPENROUTER_API_KEY"),
    "nvidia": ProviderInfo("NVIDIA", "https://integrate.api.nvidia.com/v1", "nvidia_api_key", "AGENT_NVIDIA_API_KEY"),
}


class ModelHub:
    """Shared by every chat: the catalog, the usage tracker and one HTTP connection pool."""

    def __init__(self, settings: Settings, store: Store, local_llm: OllamaLLM | None = None,
                 http: httpx.AsyncClient | None = None) -> None:
        self.settings = settings
        self.specs = {m.id: m for m in settings.models}
        self.tracker = QuotaTracker(store, settings.pools, settings.quota.reserve_calls)
        self.local_llm = local_llm or OllamaLLM(
            settings.ollama_host, settings.ollama_model, settings.ollama_num_ctx, settings.ollama_think
        )
        self.http = http or httpx.AsyncClient(timeout=httpx.Timeout(settings.quota.request_timeout_s, connect=10))

    def _key(self, spec: ModelSpec) -> str | None:
        info = PROVIDERS[spec.provider]
        secret = getattr(self.settings, info.key_field, None) if info.key_field else None
        return secret.get_secret_value() if secret else None

    def label(self, spec: ModelSpec) -> str:
        return f"{PROVIDERS[spec.provider].label} · {spec.label}"

    def usable(self, model_id: str) -> tuple[bool, str]:
        spec = self.specs.get(model_id)
        if spec is None:
            return False, "unknown model"
        if spec.provider == "ollama":
            return True, ""
        if not self._key(spec):
            return False, f"add {PROVIDERS[spec.provider].key_env} to backend/.env"
        return self.tracker.usable(spec.pool)  # type: ignore[arg-type]

    def client(self, model_id: str, notify: Notify, should_stop: Callable[[], bool]) -> LLM:
        """A client for one chat. Cloud clients meter every request against the model's pool."""
        spec = self.specs[model_id]
        if spec.provider == "ollama":
            return self.local_llm
        info = PROVIDERS[spec.provider]
        meter = Meter(self.tracker, spec.pool, self.label(spec), notify, should_stop, info.headers)  # type: ignore[arg-type]
        return OpenAICompatLLM(
            label=self.label(spec), base_url=info.base_url, api_key=self._key(spec) or "", model=spec.model,
            meter=meter, http=self.http, thought_signatures=info.thought_signatures,
            max_output_tokens=self.settings.quota.max_output_tokens, max_wait_s=self.settings.quota.max_wait_s,
            timeout_s=self.settings.quota.request_timeout_s,
        )

    def statuses(self, selected: str | None = None) -> list[dict[str, Any]]:
        """What the dropdown shows for every model."""
        out = []
        for spec in self.specs.values():
            usable, reason = self.usable(spec.id)
            local = spec.provider == "ollama"
            out.append({
                "id": spec.id,
                "label": spec.label,
                "provider": PROVIDERS[spec.provider].label,
                "model": spec.model or self.settings.ollama_model,
                "local": local,
                # False = its provider has no API key yet (the dropdown lists those in one line).
                "configured": local or bool(self._key(spec)),
                "pool": spec.pool,  # models with the same pool share one limit
                "usable": usable,
                "reason": reason,
                "selected": spec.id == selected,
                "usage": None if local or not spec.pool else self._usage(spec.pool),
            })
        return out

    def _usage(self, pool: str) -> dict[str, Any]:
        snap = self.tracker.snapshot(pool)
        # "exact" = the provider itself confirmed today's request count (Groq headers).
        return {**snap, "exact": snap["synced_at"] is not None, "reserve": self.settings.quota.reserve_calls}
