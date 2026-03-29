import base64
import time
import anthropic
from google import genai
from google.genai import types

from config import (
    ANTHROPIC_API_KEY,
    GOOGLE_API_KEY,
    LLM_PROVIDER,
    GEMINI_MODEL,
)


def call_llm(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    system: str = "",
) -> str:
    if LLM_PROVIDER == "gemini":
        return _call_gemini(messages, temperature, max_tokens, system)
    return _call_anthropic(messages, model, temperature, max_tokens, system)


def _call_anthropic(
    messages: list[dict],
    model: str,
    temperature: float,
    max_tokens: int,
    system: str,
) -> str:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("ANTHROPIC_API_KEY não configurada no .env")

    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    max_retries = 3
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            kwargs: dict = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": messages,
            }
            if system:
                kwargs["system"] = system

            response = client.messages.create(**kwargs)
            return response.content[0].text
        except (anthropic.RateLimitError, anthropic.APIStatusError) as e:
            last_error = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Anthropic: falha após {max_retries} tentativas") from last_error


def _call_gemini(
    messages: list[dict],
    temperature: float,
    max_tokens: int,
    system: str,
) -> str:
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY não configurada no .env")

    client = genai.Client(api_key=GOOGLE_API_KEY)
    contents = _to_gemini_contents(messages)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        system_instruction=system or None,
    )

    max_retries = 3
    last_error: Exception | None = None

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=contents,
                config=config,
            )
            return response.text
        except Exception as e:
            # google-genai não tem subclasses públicas estáveis — captura genérico e retenta
            last_error = e
            time.sleep(2 ** attempt)

    raise RuntimeError(f"Gemini: falha após {max_retries} tentativas — último erro: {type(last_error).__name__}: {last_error}") from last_error


def _to_gemini_contents(messages: list[dict]) -> list[types.Content]:
    contents = []
    for message in messages:
        parts: list[types.Part] = []
        content = message["content"]

        blocks = [{"type": "text", "text": content}] if isinstance(content, str) else content

        for block in blocks:
            if block["type"] == "text":
                parts.append(types.Part.from_text(text=block["text"]))
            elif block["type"] == "image":
                source = block["source"]
                image_bytes = base64.b64decode(source["data"])
                parts.append(types.Part.from_bytes(data=image_bytes, mime_type=source["media_type"]))

        contents.append(types.Content(role=message["role"], parts=parts))

    return contents
