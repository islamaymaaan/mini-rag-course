import logging
from google import genai
from google.genai import types
from ..LLMInterface import LLMInterface
from ..LLMEnums import DocumentTypeEnum


class GeminiProvider(LLMInterface):

    def __init__(
        self,
        api_key: str,
        default_input_max_characters: int = 1000,
        default_generation_max_output_tokens: int = 1000,
        default_generation_temperature: float = 0.1,
    ):
        self.api_key = api_key

        self.default_input_max_characters = default_input_max_characters
        self.default_generation_max_output_tokens = (
            default_generation_max_output_tokens
        )
        self.default_generation_temperature = default_generation_temperature

        self.generation_model_id = None
        self.embedding_model_id = None
        self.embedding_size = None

        # استخدام الـ Client الحديث من google-genai
        self.client = genai.Client(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)

    def set_generation_model(self, model_id: str):
        self.generation_model_id = model_id

    def set_embedding_model(self, model_id: str, embedding_size: int):
        self.embedding_model_id = model_id
        self.embedding_size = embedding_size

    def process_text(self, text: str):
        return text[: self.default_input_max_characters].strip()

    def generate_text(
        self,
        prompt: str,
        chat_history: list = [],
        max_output_tokens: int = None,
        temperature: float = None,
    ):
        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.generation_model_id:
            self.logger.error("Generation model for Gemini was not set")
            return None

        max_output_tokens = (
            max_output_tokens
            if max_output_tokens
            else self.default_generation_max_output_tokens
        )
        temperature = (
            temperature
            if temperature
            else self.default_generation_temperature
        )

        try:
            # إعداد الـ Configuration للـ Generation
            config = types.GenerateContentConfig(
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            )

            # دمج الـ Chat History مع الـ Prompt إذا وُجد
            contents = []
            if chat_history:
                contents.extend(chat_history)

            contents.append(
                self.construct_prompt(prompt=prompt, role="user")
            )

            response = self.client.models.generate_content(
                model=self.generation_model_id,
                contents=contents,
                config=config,
            )

            if not response or not response.text:
                self.logger.error("Error while generating text with Gemini")
                return None

            return response.text

        except Exception as e:
            self.logger.error(
                f"Exception occurred while generating text with Gemini: {e}"
            )
            return None

    def embed_text(self, text: str, document_type: str = None):
        if not self.client:
            self.logger.error("Gemini client was not set")
            return None

        if not self.embedding_model_id:
            self.logger.error("Embedding model for Gemini was not set")
            return None

        try:
            # تحديد نوع الـ Task لبناء الـ Vector بشكل دقيق في RAG
            task_type = types.TaskType.RETRIEVAL_DOCUMENT
            if document_type == DocumentTypeEnum.QUERY:
                task_type = types.TaskType.RETRIEVAL_QUERY

            response = self.client.models.embed_content(
                model=self.embedding_model_id,
                contents=self.process_text(text),
                config=types.EmbedContentConfig(
                    task_type=task_type,
                    output_dimensionality=self.embedding_size,
                ),
            )

            if (
                not response
                or not response.embedding
                or not response.embedding.values
            ):
                self.logger.error("Error while embedding text with Gemini")
                return None

            return response.embedding.values

        except Exception as e:
            self.logger.error(
                f"Exception occurred while embedding text with Gemini: {e}"
            )
            return None

    def construct_prompt(self, prompt: str, role: str):
        # Gemini يفضل الأدوار: 'user' و 'model'
        gemini_role = "model" if role == "assistant" else role
        return types.Content(
            role=gemini_role,
            parts=[types.Part.from_text(text=self.process_text(prompt))],
        )