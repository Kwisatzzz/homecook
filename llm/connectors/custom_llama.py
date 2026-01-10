import json
import logging
from typing import List, Optional
from urllib.parse import urljoin

import requests
from langchain.llms.base import LLM
import logging

logger = logging.getLogger(__name__)


class Llama(LLM):
    endpoint: str
    model: str
    max_tokens: int = 2048
    temperature: float = 0.0

    SYSTEM_PROMPT: str = (
        "Odpowiedź musi być zwięzła i w języku polskim. Nigdy nie dopisujesz nic od siebie. Wypełniasz jedynie podaną instrukcję. Nie wychodzisz poza ramy podane w instrukcji. Trzymasz się ram podanych w instrukcji."
    )

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """
        Send the prompt to the LLAMA API and get the response.
        """
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": self.SYSTEM_PROMPT,
                },
                {"role": "user", "content": prompt},
            ],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }

        headers = {"Content-Type": "application/json", "Accept-Charset": "UTF-8"}
        response = requests.post(
            self.endpoint, data=json.dumps(payload), headers=headers
        )

        if response.status_code != 200:
            raise ValueError(
                f"Error from LLAMA API: {response.status_code}, {response.text}"
            )

        choices = response.json().get("choices", [])
        if not choices:
            raise ValueError("No choices found in the response.")

        return choices[0]["message"]["content"]

    @property
    def _llm_type(self) -> str:
        return "llama_custom_chat"

    def get_model_context_window(self) -> Optional[int]:
        try:
            base_url = self.endpoint.rsplit("/v1/", 1)[0]
            models_url = urljoin(base_url, "/v1/models")

            response = requests.get(models_url)
            response.raise_for_status()
            models_data = response.json()

            for model_info in models_data.get("data", []):
                if model_info.get("id") == self.model:
                    context_keys = [
                        "max_model_len",
                        "context_length",
                        "max_position_embeddings",
                    ]
                    for key in context_keys:
                        if key in model_info:
                            logger.info(
                                f"Found context window key '{key}' with value {model_info[key]}"
                            )
                            return model_info[key]

            logger.warning(
                f"Model '{self.model}' found, but no context window key was found in its data."
            )
            return None

        except requests.RequestException as e:
            logger.error(f"Could not fetch model details from {models_url}. Error: {e}")
            return None
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while fetching model details: {e}"
            )
            return None
