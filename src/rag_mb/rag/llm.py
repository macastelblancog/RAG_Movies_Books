"""Clientes LLM intercambiables + prompts.

Condiciones del benchmark (misma pregunta, mismo modelo):
  closed_book : el LLM responde de memoria (lo que "sabe" Gemini / GPT).
  rag         : el LLM recibe los k fragmentos recuperados y debe citar [n].

Clientes:
  openai  : requiere OPENAI_API_KEY   (modelo: RMB_OPENAI_MODEL, por defecto gpt-5)
  gemini  : requiere GEMINI_API_KEY   (modelo: RMB_GEMINI_MODEL, por defecto gemini-2.5-pro)
  local   : servidor local compatible con OpenAI (llama.cpp `llama-server`). `local` = modelo por defecto;
            `local:gemma3_4b` = otro modelo de configs/corpora.yaml → rag.llm.models (cada uno en su puerto).
  manual  : exporta los prompts a JSONL/CSV para pegarlos en la interfaz web y luego importar
            las respuestas (útil sin API keys). Ver `rmb eval --client manual`.
Los nombres de modelo cambian con frecuencia: fijarlos por variable de entorno y reportarlos.
"""
from __future__ import annotations

import os
import re
from typing import Protocol

from .index import Hit

SYSTEM_RAG = (
    "You answer questions about books and their film adaptations using ONLY the numbered context passages "
    "(each labelled with its source: BOOK or FILM). Read ALL passages before answering: the answer "
    "is often in a passage that does not repeat the question's words. Before answering, check that the "
    "passages actually show the character, the work and the event the question asks about; if they do not, "
    "answer that the passages do not show it instead of guessing.\n"
    "Output format, exactly two lines:\n"
    "Answer: <at most 4 sentences, citing passage numbers like [3]; keep what the BOOK says separate from what "
    "the FILM shows; do not add names or events that are not in the passages; if the question assumes "
    "something the passages contradict, correct it; if no passage contains the answer, say that the passages "
    "do not say>\n"
    "Evidence: <ONE short quote of at most 25 words copied from the passage that supports the answer, then its "
    "number, e.g. \"...pierced with many arrows.\" [3]; or 'none'>\n"
    "Never copy whole passages. Write the Answer in the same language as the question."
)
SYSTEM_CLOSED = (
    "You answer questions about books and their film adaptations. Distinguish the BOOK from the FILM. "
    "If you are not sure, or the question rests on a false premise, say so explicitly. "
    "Be concise (max 5 sentences)."
)


def format_context(hits: list[Hit], max_chars: int = 1400) -> str:
    return "\n\n".join(f"[{h.rank}] ({h.prompt_label})\n{h.chunk.text[:max_chars]}" for h in hits)


def rag_prompt(question: str, hits: list[Hit], max_chars: int = 1400, best_last: bool = False) -> str:
    """best_last: el pasaje mejor rankeado se ubica al FINAL del contexto (junto a la pregunta), donde un
    modelo pequeño lo atiende mejor ("lost in the middle"); la numeración [n] sigue siendo el ranking."""
    ordered = sorted(hits, key=lambda h: -h.rank) if best_last else hits
    return f"Context:\n{format_context(ordered, max_chars)}\n\nQuestion: {question}"


_ANS = re.compile(r"(?is)\b(?:final\s+)?(?:answer|respuesta)\s*:\s*")
_EVI = re.compile(r"(?is)\b(?:evidence|evidencia)\s*:\s*")


def split_evidence(text: str) -> tuple[str, str]:
    """(evidencia, respuesta). Se puntúa SOLO la respuesta: la evidencia copia texto de los pasajes y
    contiene palabras clave que inflarían el acierto. Acepta ambos órdenes (Answer→Evidence o al revés).
    Si hay bloque de evidencia pero ninguna respuesta (salida truncada), la respuesta queda vacía (= fallo)."""
    t = (text or "").strip()
    a, e = _ANS.search(t), _EVI.search(t)
    if a is None:
        if e is None:
            return "", t
        return t[e.end():].strip(), ""
    if e is None:
        return "", t[a.end():].strip()
    if a.start() < e.start():                       # Answer: ... Evidence: ...
        return t[e.end():].strip(), t[a.end():e.start()].strip()
    return t[e.end():a.start()].strip(), t[a.end():].strip()   # Evidence: ... Answer: ...


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAIClient:
    def __init__(self, model: str | None = None):
        from openai import OpenAI
        self.model = model or os.environ.get("RMB_OPENAI_MODEL", "gpt-5")
        self.name = f"openai:{self.model}"
        self.client = OpenAI()

    def complete(self, system: str, user: str) -> str:
        r = self.client.chat.completions.create(
            model=self.model, messages=[{"role": "system", "content": system}, {"role": "user", "content": user}])
        return r.choices[0].message.content or ""


def local_models_config() -> dict:
    from ..config import load_config
    return (load_config().get("rag", {}) or {}).get("llm", {}) or {}


class LocalClient:
    """llama.cpp `llama-server` (u Ollama / LM Studio): endpoint /v1/chat/completions compatible con OpenAI.
    Se llama con urllib para no exigir el paquete `openai`."""

    def __init__(self, key: str | None = None, url: str | None = None, model: str | None = None):
        """key: nombre en configs/corpora.yaml → rag.llm.models (p. ej. phi4mini, gemma3_4b). Cada modelo
        tiene su propio llama-server y puerto."""
        cfg = local_models_config()
        key = key or cfg.get("default")
        m = (cfg.get("models") or {}).get(key, {}) if key else {}
        if key and not m:
            raise ValueError(f"modelo local desconocido: {key}. Opciones: {sorted(cfg.get('models') or {})}")
        self.key = key or "local"
        self.url = (url or os.environ.get("RMB_LOCAL_URL") or m.get("url") or cfg.get("url")
                    or "http://127.0.0.1:8080/v1").rstrip("/")
        self.model = model or os.environ.get("RMB_LOCAL_MODEL") or m.get("model") or cfg.get("model", "local")
        self.max_tokens = int(m.get("max_tokens", cfg.get("max_tokens", 500)))
        self.temperature = float(m.get("temperature", cfg.get("temperature", 0.0)))
        self.name = f"local:{self.key}"

    def complete(self, system: str, user: str, temperature: float | None = None,
                 max_tokens: int | None = None) -> str:
        import json
        import urllib.error
        import urllib.request
        body = json.dumps({"model": self.model, "max_tokens": max_tokens or self.max_tokens,
                           "temperature": self.temperature if temperature is None else temperature,
                           "messages": [{"role": "system", "content": system},
                                        {"role": "user", "content": user}]}).encode()
        req = urllib.request.Request(self.url + "/chat/completions", data=body,
                                     headers={"Content-Type": "application/json", "Authorization": "Bearer local"})
        try:
            with urllib.request.urlopen(req, timeout=300) as r:
                out = json.loads(r.read().decode("utf-8"))
        except urllib.error.URLError as e:
            raise ConnectionError(f"No responde el LLM local en {self.url} ({e.reason}). "
                                  "¿Está corriendo llama-server?") from e
        return out["choices"][0]["message"]["content"] or ""


class GeminiClient:
    def __init__(self, model: str | None = None):
        from google import genai
        from google.genai import types
        self.types = types
        self.model = model or os.environ.get("RMB_GEMINI_MODEL", "gemini-2.5-pro")
        self.name = f"gemini:{self.model}"
        self.client = genai.Client()

    def complete(self, system: str, user: str) -> str:
        r = self.client.models.generate_content(
            model=self.model, contents=user,
            config=self.types.GenerateContentConfig(system_instruction=system))
        return r.text or ""


class EchoClient:
    """Cliente de prueba sin red: devuelve el primer fragmento citado (baseline extractivo)."""
    name = "extractive"

    def complete(self, system: str, user: str) -> str:
        if "Context:" not in user:
            return "I don't know."
        first = user.split("Context:\n", 1)[1].split("\n\n[", 1)[0]
        return first[:600] + " [1]"


def get_client(name: str) -> LLM:
    if name == "openai":
        return OpenAIClient()
    if name == "gemini":
        return GeminiClient()
    if name == "local" or name.startswith("local:"):
        return LocalClient(name.split(":", 1)[1] if ":" in name else None)
    if name in ("extractive", "echo"):
        return EchoClient()
    raise ValueError(f"cliente desconocido: {name}")
