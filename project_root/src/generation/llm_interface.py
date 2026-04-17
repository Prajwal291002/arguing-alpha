import requests
import json


class LLMInterface:

    def __init__(self, model_name: str = "llama3"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/generate"

    def call_llm(self, prompt: str) -> str:

        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False
        }

        response = requests.post(self.api_url, json=payload)

        if response.status_code != 200:
            raise Exception(f"LLM API Error: {response.text}")

        return response.json()["response"]

    def extract_json_block(self, text: str):

        start = text.find("[")
        end = text.rfind("]")

        if start != -1 and end != -1:
            return text[start:end + 1]

        return None

    def parse_json(self, response: str):

        json_block = self.extract_json_block(response)

        if not json_block:
            return None

        try:
            return json.loads(json_block)
        except:
            return None
