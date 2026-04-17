"""Target model interface.

The target model is the model being attacked. It receives adversarial
prompts from the attacker and generates responses.
"""

import logging
from openai import OpenAI

logger = logging.getLogger(__name__)


class TargetModel:
    """Interfaces with the target model via vLLM OpenAI-compatible API."""

    def __init__(self, base_url: str, model_name: str, config: dict):
        self.client = OpenAI(base_url=base_url, api_key="not-needed")
        self.model_name = model_name
        self.temperature = config.get("target_temperature", 0.0)
        self.max_tokens = config.get("target_max_tokens", 512)

    def generate(self, prompt: str) -> str:
        """Send a prompt to the target model and return its response."""
        messages = [{"role": "user", "content": prompt}]

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"Target model generation failed: {e}")
            raise
