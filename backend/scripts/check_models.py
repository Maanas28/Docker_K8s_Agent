"""Check the model dropdown against the real providers: `uv run python -m scripts.check_models`.

Add `--find WORD …` to list every model id containing one of the words, e.g. `--find glm kimi`.

For every provider whose key is in backend/.env: does each model id in agent.toml exist there?
For OpenRouter it also lists the free (:free) models that support tool calling, since that list
rotates. Only lists models (no chat requests), so it uses none of your free daily requests.
Each key is only sent to its own provider.
"""

import asyncio
import re
import sys

import httpx

from app.config import get_settings
from app.models import PROVIDERS


async def main() -> None:
    settings = get_settings()
    words = [w.lower() for w in sys.argv[sys.argv.index("--find") + 1:]] if "--find" in sys.argv else []
    async with httpx.AsyncClient(timeout=30) as http:
        for name, info in PROVIDERS.items():
            if name == "ollama":
                continue
            wanted = [m for m in settings.models if m.provider == name]
            secret = getattr(settings, info.key_field)
            print(f"\n== {info.label} ({len(wanted)} in agent.toml)")
            if secret is None or not secret.get_secret_value().strip():  # an empty line = no key
                print(f"   no key: add {info.key_env} to backend/.env")
                continue
            try:
                r = await http.get(f"{info.base_url}/models",
                                   headers={"Authorization": f"Bearer {secret.get_secret_value()}"})
            except httpx.HTTPError as e:
                print(f"   can't reach it: {e}")
                continue
            if r.status_code != 200:
                print(f"   HTTP {r.status_code}: {r.text[:200]}")
                continue
            data = r.json().get("data", [])
            ids = {m.get("id", "").removeprefix("models/") for m in data}
            for m in wanted:
                print(f"   {'OK     ' if m.model in ids else 'MISSING'} {m.model}  ({m.id})")
            if words:
                found = sorted(i for i in ids if any(w in i.lower() for w in words))
                print(f"   ids containing {', '.join(words)} ({len(found)}):")
                for model_id in found:
                    print(f"     {model_id}")
                continue
            if name == "openrouter":
                free = sorted(
                    m["id"] for m in data
                    if m.get("id", "").endswith(":free") and "tools" in (m.get("supported_parameters") or [])
                )
                print(f"   free models with tool calling ({len(free)}):")
                for model_id in free:
                    print(f"     {model_id}")
            elif any(m.model not in ids for m in wanted):
                families = {re.match(r"[a-z]+", m.model.split("/")[-1]).group(0) for m in wanted
                            if re.match(r"[a-z]+", m.model.split("/")[-1])}
                similar = sorted(i for i in ids if any(f in i for f in families))
                print(f"   similar ids there: {', '.join(similar[:15]) or '(none)'}")


asyncio.run(main())
