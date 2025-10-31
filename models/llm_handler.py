import os
from groq import Groq

class LLMHandler:
    def __init__(self):
        self.client = Groq()  # Automatically reads GROQ_API_KEY from .env

    def get_completion(self, messages):
        """
        messages: list of dicts, e.g. [{"role": "user", "content": "Hello"}]
        """
        completion = self.client.chat.completions.create(
            model="openai/gpt-oss-120b",
            messages=messages
        )
        return completion.choices[0].message.content
