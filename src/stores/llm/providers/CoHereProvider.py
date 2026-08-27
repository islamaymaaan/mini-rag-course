import logging
from typing import List, Union
import cohere

from ..LLMEnums import CoHereEnums, DocumentTypeEnum
from ..LLMInterface import LLMInterface


class CoHereProvider(LLMInterface):

    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):

        self.api_key = api_key

        self.default_input_max_characters = (
            default_input_max_characters
        )

        self.default_generation_max_output_tokens = (
            default_generation_max_output_tokens
        )

        self.default_generation_temperature = (
            default_generation_temperature
        )

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        # Cohere V2 Client
        self.client = cohere.ClientV2(
            api_key=self.api_key
        )

        self.enums = CoHereEnums
        self.logger = logging.getLogger(__name__)

    # =========================================================
    # Model Configuration
    # =========================================================

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(
        self,
        model_id: str,
        embedding_size: int
    ):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    # =========================================================
    # Helpers
    # =========================================================

    def process_text(self, text: str):
        if not text:
            return ""

        return text[
            :self.default_input_max_characters
        ].strip()

    def _clean_chat_history(
        self,
        chat_history: list
    ):

        if not chat_history:
            return []

        cleaned_history = []

        for message in chat_history:

            if not isinstance(message, dict):
                continue

            role = str(message.get("role", "")).lower().strip()
            content = message.get("content")

            if role not in {
                "system",
                "user",
                "assistant"
            }:
                continue

            if not content:
                continue

            cleaned_history.append(
                {
                    "role": role,
                    "content": str(content),
                }
            )

        return cleaned_history

    # =========================================================
    # Text Generation
    # =========================================================

    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None,
    ):

        if not self.client:
            self.logger.error("Cohere client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Cohere was not set")
            return None

        if chat_history is None:
            chat_history = []

        max_output_tokens = (
            max_output_tokens
            if max_output_tokens is not None
            else self.default_generation_max_output_tokens
        )

        temperature = (
            temperature
            if temperature is not None
            else self.default_generation_temperature
        )

        messages = self._clean_chat_history(chat_history)

        processed_prompt = prompt.strip()

        if not processed_prompt:
            self.logger.error("Prompt is empty")
            return None

        messages.append(
            {
                "role": "user",
                "content": processed_prompt,
            }
        )

        try:
            response = self.client.chat(
                model=self.generation_model_id,
                messages=messages,
                temperature=temperature,
                max_tokens=max_output_tokens,
            )

        except Exception as e:
            self.logger.exception(
                f"Error while generating text with Cohere: {e}"
            )
            return None

        if not response or not response.message or not response.message.content:
            self.logger.error("Empty or invalid response structure from Cohere")
            return None

        for content in response.message.content:
            text = getattr(content, "text", None)
            if text:
                return text

        self.logger.error("No text content found in Cohere response")
        return None

    # =========================================================
    # Embeddings
    # =========================================================

    def embed_text(
        self,
        texts: Union[str, List[str]],
        document_type: str = None
    ):

        if not self.client:
            self.logger.error("Cohere client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Cohere was not set")
            return None

        if isinstance(texts, str):
            texts = [texts]

        input_type = "search_document"

        if document_type == DocumentTypeEnum.QUERY or document_type == DocumentTypeEnum.QUERY.value:
            input_type = "search_query"

        processed_texts = [
            self.process_text(t)
            for t in texts
            if t
        ]

        if not processed_texts:
            self.logger.error("No valid text provided for embedding")
            return None

        try:
            response = self.client.embed(
                model=self.embedding_model_id,
                texts=processed_texts,
                input_type=input_type,
                embedding_types=["float"],
            )

        except Exception as e:
            self.logger.exception(
                f"Error while embedding text with Cohere: {e}"
            )
            return None

        if not response or not response.embeddings or not response.embeddings.float:
            self.logger.error("No float embeddings returned from Cohere")
            return None

        return response.embeddings.float

    # =========================================================
    # Prompt Construction
    # =========================================================

    def construct_prompt(
        self,
        prompt: str,
        role: str
    ):
        return {
            "role": str(role).lower().strip(),
            "content": prompt,
        }