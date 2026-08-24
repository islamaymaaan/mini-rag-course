from ..LLMInterface import LLMInterface
from ..LLMEnums import CoHereEnums, DocumentTypeEnum
import cohere
import logging


class CoHereProvider(LLMInterface):

    def __init__(self, api_key: str,
                       default_input_max_characters: int = 1000,
                       default_generation_max_output_tokens: int = 1000,
                       default_generation_temperature: float = 0.1):

        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = default_generation_max_output_tokens
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None

        self.embedding_model_id = None
        self.embedding_size = None

        # Current Cohere SDK
        self.client = cohere.ClientV2(api_key=self.api_key)

        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[:self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = None,
        max_output_tokens: int = None,
        temperature: float = None
    ):

        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for CoHere was not set")
            return None

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

        # Cohere V2 uses one `messages` list instead of
        # `message` + `chat_history`
        messages = []

        if chat_history:
            for message in chat_history:
                # Support both the old structure:
                # {"role": "...", "text": "..."}
                #
                # and the new structure:
                # {"role": "...", "content": "..."}
                if "content" in message:
                    messages.append({
                        "role": message["role"],
                        "content": message["content"]
                    })
                elif "text" in message:
                    messages.append({
                        "role": message["role"],
                        "content": message["text"]
                    })

        messages.append({
            "role": "user",
            "content": self.process_text(prompt)
        })

        response = self.client.chat(
            model=self.generation_model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_output_tokens
        )

        if (
            not response
            or not response.message
            or not response.message.content
        ):
            self.logger.error("Error while generating text with CoHere")
            return None

        return response.message.content[0].text

    def embed_text(self, text: str, document_type: str = None):

        if not self.client:
            self.logger.error("CoHere client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for CoHere was not set")
            return None

        # Cohere V2 embedding input types
        input_type = CoHereEnums.DOCUMENT

        if document_type == DocumentTypeEnum.QUERY:
            input_type = CoHereEnums.QUERY

        response = self.client.embed(
            model=self.embedding_model_id,
            texts=[self.process_text(text)],
            input_type=input_type,
            embedding_types=["float"]
        )

        if (
            not response
            or not response.embeddings
            or not response.embeddings.float
        ):
            self.logger.error("Error while embedding text with CoHere")
            return None

        return response.embeddings.float[0]

    def construct_prompt(self, prompt: str, role: str):
        return {
            "role": role,
            "content": self.process_text(prompt)
        }

