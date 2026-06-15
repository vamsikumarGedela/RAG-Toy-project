import logging
import os
import time

logger = logging.getLogger(__name__)

# Provider configs — all OpenAI-compatible except Anthropic
_PROVIDERS = {
    "ollama": {
        "base_url":      "http://localhost:11434/v1",
        "api_key":       "ollama",           # Ollama needs no real key
        "default_model": "llama3.2",
        "env_key":       None,
        "env_model":     "OLLAMA_MODEL",
    },
    "openai": {
        "base_url":      None,               # official OpenAI endpoint
        "api_key":       None,
        "default_model": "gpt-4o-mini",
        "env_key":       "OPENAI_API_KEY",
        "env_model":     "OPENAI_MODEL",
    },
    "openrouter": {
        "base_url":      "https://openrouter.ai/api/v1",
        "api_key":       None,
        "default_model": "openai/gpt-oss-120b:free",
        "env_key":       "OPENROUTER_API_KEY",
        "env_model":     "OPENROUTER_MODEL",
    },
    "anthropic": {
        "default_model": "claude-haiku-4-5-20251001",
        "env_key":       "ANTHROPIC_API_KEY",
        "env_model":     "ANTHROPIC_MODEL",
    },
}


class LLM:
    """
    Unified LLM interface.

    Supported providers (set via llm_provider param):
      - "ollama"      → free, runs locally, no API key needed
      - "openai"      → needs OPENAI_API_KEY
      - "openrouter"  → needs OPENROUTER_API_KEY
      - "anthropic"   → needs ANTHROPIC_API_KEY + pip install anthropic
    """

    def __init__(self, provider: str = "ollama", model: str = None, api_key: str = None, temperature: float = 0.0, timeout: float = 30.0):
        if provider not in _PROVIDERS:
            raise ValueError(f"Unknown provider '{provider}'. Choose from: {list(_PROVIDERS)}")

        self.provider = provider
        cfg = _PROVIDERS[provider]

        # resolve model
        self.model = model or os.getenv(cfg.get("env_model", ""), "") or cfg["default_model"]

        # resolve api key
        env_key = cfg.get("env_key")
        self.api_key = api_key or (os.getenv(env_key) if env_key else None) or cfg.get("api_key", "")

        if env_key and not self.api_key:
            raise EnvironmentError(
                f"Provider '{provider}' requires an API key.\n"
                f"Set the {env_key} environment variable or pass api_key='...' to RAG()."
            )

        self.base_url = cfg.get("base_url")
        self.temperature = temperature
        self.timeout = timeout

    def _openai_client(self):
        from openai import OpenAI
        kwargs = {"api_key": self.api_key, "timeout": self.timeout}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        return OpenAI(**kwargs)

    def _anthropic_client(self):
        try:
            import anthropic
        except ImportError:
            raise ImportError(
                "Anthropic provider requires the anthropic package.\n"
                "Install it: pip install anthropic"
            )
        return anthropic.Anthropic(api_key=self.api_key, timeout=self.timeout)

    def chat(self, messages: list, stream: bool = False) -> str:
        if self.provider == "anthropic":
            return self._chat_anthropic(messages, stream)
        return self._chat_openai(messages, stream)

    def stream(self, messages: list):
        """Yield tokens one at a time — no printing, caller decides what to do."""
        if self.provider == "anthropic":
            yield from self._stream_anthropic(messages)
        else:
            yield from self._stream_openai(messages)

    def _stream_openai(self, messages: list):
        client = self._openai_client()
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                )
                for chunk in response:
                    token = chunk.choices[0].delta.content or ""
                    if token:
                        yield token
                return
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 10 * (attempt + 1)
                    logger.warning("Rate limited. Retrying in %ds...", wait)
                    time.sleep(wait)
                else:
                    raise

    def _stream_anthropic(self, messages: list):
        client = self._anthropic_client()
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)
        with client.messages.stream(
            model=self.model,
            max_tokens=2048,
            temperature=self.temperature,
            system=system,
            messages=filtered,
        ) as s:
            for token in s.text_stream:
                yield token

    def _chat_openai(self, messages: list, stream: bool) -> str:
        client = self._openai_client()

        if not stream:
            response = client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
            )
            return response.choices[0].message.content or ""

        parts = []
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    stream=True,
                )
                for chunk in response:
                    token = chunk.choices[0].delta.content or ""
                    parts.append(token)
                    print(token, end="", flush=True)
                print()
                return "".join(parts)
            except Exception as e:
                if "429" in str(e) and attempt < 2:
                    wait = 10 * (attempt + 1)
                    logger.warning("Rate limited. Retrying in %ds...", wait)
                    time.sleep(wait)
                else:
                    raise
        return "".join(parts)

    def _chat_anthropic(self, messages: list, stream: bool) -> str:
        client = self._anthropic_client()

        # Anthropic separates system prompt from messages
        system = ""
        filtered = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                filtered.append(m)

        if not stream:
            response = client.messages.create(
                model=self.model,
                max_tokens=2048,
                temperature=self.temperature,
                system=system,
                messages=filtered,
            )
            return response.content[0].text

        parts = []
        with client.messages.stream(
            model=self.model,
            max_tokens=2048,
            temperature=self.temperature,
            system=system,
            messages=filtered,
        ) as s:
            for token in s.text_stream:
                parts.append(token)
                print(token, end="", flush=True)
        print()
        return "".join(parts)


def build_ask_messages(question: str, context_chunks: list, history: list = None) -> list:
    context = "\n\n".join(
        f"[{c['source']} p.{c['page']}]\n{c['text']}"
        for c in context_chunks
    )
    messages = [{
        "role": "system",
        "content": (
            "You are a precise document assistant. Answer using ONLY the provided context. "
            "If the answer is not in the context say: "
            "'I could not find relevant information in the provided documents.'\n\n"
            f"Context:\n{context}"
        ),
    }]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": question})
    return messages
