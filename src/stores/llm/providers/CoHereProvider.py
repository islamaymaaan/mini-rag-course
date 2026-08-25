from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum

import cohere
import logging


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
        """
        Used for embedding input only.

        The generation prompt must NOT be limited by this
        character limit because RAG prompts can contain
        multiple retrieved documents.
        """

        if not text:
            return ""

        return text[:self.default_input_max_characters].strip()

    def _clean_chat_history(self, chat_history: list):
        """
        Keep only standard Cohere V2 messages.

        Supported roles:
        - system
        - user
        - assistant
        """

        if not chat_history:
            return []

        cleaned_history = []

        for message in chat_history:

            if not isinstance(message, dict):
                continue

            role = message.get("role")
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
                    "content": content,
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

        # -----------------------------------------------------
        # Validate client
        # -----------------------------------------------------

        if not self.client:
            self.logger.error(
                "Cohere client was not set"
            )
            return None

        # -----------------------------------------------------
        # Validate generation model
        # -----------------------------------------------------

        if not self.generation_model_id:
            self.logger.error(
                "Generation model for Cohere was not set"
            )
            return None

        # -----------------------------------------------------
        # Default values
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Clean chat history
        # -----------------------------------------------------

        messages = self._clean_chat_history(
            chat_history
        )

        # -----------------------------------------------------
        # Prepare RAG prompt
        #
        # IMPORTANT:
        # Do NOT use process_text() here.
        # The RAG prompt can be much larger than 1024 chars.
        # -----------------------------------------------------

        processed_prompt = prompt.strip()

        if not processed_prompt:
            self.logger.error(
                "Prompt is empty"
            )
            return None

        # -----------------------------------------------------
        # Add current user message
        # -----------------------------------------------------

        messages.append(
            {
                "role": "user",
                "content": processed_prompt,
            }
        )

        # -----------------------------------------------------
        # Logging
        # -----------------------------------------------------

        self.logger.info(
            f"Cohere generation model: "
            f"{self.generation_model_id}"
        )

        self.logger.info(
            f"Cohere prompt length: "
            f"{len(processed_prompt)} characters"
        )

        self.logger.info(
            f"Cohere max output tokens: "
            f"{max_output_tokens}"
        )

        self.logger.info(
            f"Cohere temperature: "
            f"{temperature}"
        )

        # -----------------------------------------------------
        # Cohere V2 Chat
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Validate response
        # -----------------------------------------------------

        if not response:
            self.logger.error(
                "Empty response from Cohere"
            )
            return None

        if not response.message:
            self.logger.error(
                "No message returned from Cohere"
            )
            return None

        if not response.message.content:
            self.logger.error(
                "No content returned from Cohere"
            )
            return None

        # -----------------------------------------------------
        # Extract text
        # -----------------------------------------------------

        for content in response.message.content:

            text = getattr(
                content,
                "text",
                None
            )

            if text:
                self.logger.info(
                    f"Cohere finish reason: "
                    f"{response.finish_reason}"
                )

                return text

        # -----------------------------------------------------
        # No text found
        # -----------------------------------------------------

        self.logger.error(
            "No text content found in Cohere response"
        )

        self.logger.error(
            f"Cohere response content: "
            f"{response.message.content}"
        )

        return None

    # =========================================================
    # Embeddings
    # =========================================================

    def embed_text(
        self,
        text: str,
        document_type: str = None
    ):

        # -----------------------------------------------------
        # Validate client
        # -----------------------------------------------------

        if not self.client:
            self.logger.error(
                "Cohere client was not set"
            )
            return None

        # -----------------------------------------------------
        # Validate embedding model
        # -----------------------------------------------------

        if not self.embedding_model_id:
            self.logger.error(
                "Embedding model for Cohere was not set"
            )
            return None

        # -----------------------------------------------------
        # Determine embedding input type
        # -----------------------------------------------------

        input_type = "search_document"

        if document_type == DocumentTypeEnum.QUERY.value:
            input_type = "search_query"

        # -----------------------------------------------------
        # Process text
        # -----------------------------------------------------

        processed_text = self.process_text(
            text
        )

        if not processed_text:
            self.logger.error(
                "Text is empty after processing"
            )
            return None

        # -----------------------------------------------------
        # Cohere V2 Embed
        # -----------------------------------------------------

        try:

            response = self.client.embed(
                model=self.embedding_model_id,
                texts=[
                    processed_text
                ],
                input_type=input_type,
                embedding_types=[
                    "float"
                ],
            )

        except Exception as e:

            self.logger.exception(
                f"Error while embedding text with Cohere: {e}"
            )

            return None

        # -----------------------------------------------------
        # Validate response
        # -----------------------------------------------------

        if not response:
            self.logger.error(
                "Empty embedding response from Cohere"
            )
            return None

        if not response.embeddings:
            self.logger.error(
                "No embeddings returned from Cohere"
            )
            return None

        if not response.embeddings.float:
            self.logger.error(
                "No float embeddings returned from Cohere"
            )
            return None

        return response.embeddings.float[0]

    # =========================================================
    # Prompt Construction
    # =========================================================

    def construct_prompt(
        self,
        prompt: str,
        role: str
    ):

        return {
            "role": role,
            "content": prompt,
        }