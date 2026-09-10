"""
Shared LLM client wrapper — abstracts away the API provider.
Currently supports Groq (fast, free tier with generous limits).
"""
import json
import logging
import time

from groq import Groq

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


class LLMClient:
    """Unified LLM client wrapping Groq API."""

    def __init__(self, api_key: str | None = None):
        from src.config import LLM_API_KEY, LLM_MODEL
        self.api_key = api_key or LLM_API_KEY
        if not self.api_key:
            raise ValueError("API key required. Set GROQ_API_KEY in .env")
        self.client = Groq(api_key=self.api_key)
        self.model = LLM_MODEL

    def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        json_mode: bool = True,
        max_retries: int = 3,
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from the LLM. Returns raw text."""
        messages = []
        if system_prompt:
            sys_content = system_prompt
            if json_mode and "json" not in sys_content.lower():
                sys_content += "\nRespond with valid JSON."
            messages.append({"role": "system", "content": sys_content})
        elif json_mode:
            messages.append({"role": "system", "content": "Respond with valid JSON."})
        messages.append({"role": "user", "content": prompt})

        for attempt in range(max_retries):
            try:
                kwargs = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                }
                if json_mode:
                    kwargs["response_format"] = {"type": "json_object"}

                response = self.client.chat.completions.create(**kwargs)
                return response.choices[0].message.content
            except Exception as e:
                wait = 2 ** attempt + 1
                logger.warning("LLM attempt %d failed: %s. Retrying in %ds…",
                               attempt + 1, e, wait)
                time.sleep(wait)

        raise RuntimeError(f"All {max_retries} LLM attempts failed.")

    def generate_json(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: float = 0.1,
        max_retries: int = 3,
        max_tokens: int = 512,
    ) -> dict:
        """Generate and parse a JSON response."""
        text = self.generate(prompt, system_prompt, temperature, json_mode=True,
                             max_retries=max_retries, max_tokens=max_tokens)
        return json.loads(text)
