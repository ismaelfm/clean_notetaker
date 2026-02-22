"""OpenRouter API client for vision-based page analysis.

Sends extracted text + rendered page image to a vision-capable model
via the OpenRouter chat completions endpoint.
"""

import base64
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"


def _get_config() -> tuple[str, str]:
    """Load and validate API key and model from environment."""
    api_key = os.getenv("OPENROUTER_API_KEY", "")
    model = os.getenv("OPENROUTER_MODEL", "")

    if not api_key or api_key == "your-api-key-here":
        print("[ERROR] Set your OPENROUTER_API_KEY in .env")
        sys.exit(1)

    return api_key, model


def analyze_page(
    system_prompt: str,
    user_prompt: str,
    image_bytes: bytes | None = None,
    send_image: bool = True,
    max_tokens: int = 4096,
) -> str:
    """Send a page to the model and return the response.

    Args:
        system_prompt: The system directive for the model.
        user_prompt: The per-page prompt with extracted text.
        image_bytes: PNG image bytes of the rendered page (can be None if send_image=False).
        send_image: If True, include the page image. If False, text-only.
        max_tokens: Maximum tokens in the response.

    Returns:
        The model's markdown response text.
    """
    api_key, model = _get_config()

    # Build user message content
    if send_image and image_bytes:
        b64_image = base64.b64encode(image_bytes).decode("utf-8")
        image_url = f"data:image/png;base64,{b64_image}"
        user_content = [
            {"type": "text", "text": user_prompt},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    else:
        user_content = user_prompt

    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content},
        ],
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "X-Title": "Notes Extractor",
    }

    response = httpx.post(
        OPENROUTER_API_URL,
        json=payload,
        headers=headers,
        timeout=120.0,
    )

    if response.status_code != 200:
        error_detail = response.text
        raise RuntimeError(
            f"OpenRouter API error {response.status_code}: {error_detail}"
        )

    data = response.json()

    if "error" in data:
        raise RuntimeError(f"OpenRouter error: {data['error']}")

    return data["choices"][0]["message"]["content"]
