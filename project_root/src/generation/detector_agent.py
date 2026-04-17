from src.generation.llm_interface import LLMInterface
import json


class DetectorAgent:

    def __init__(self, taxonomy_path: str):
        self.llm = LLMInterface()

        with open(taxonomy_path, "r") as f:
            self.taxonomy = json.load(f)

    def build_prompt(self, context: str):

        return f"""
You are a financial risk analyst.

Your task is to extract financial risks STRICTLY from the given text.

IMPORTANT RULES:
- Do NOT invent information
- Use ONLY risks explicitly mentioned in the text
- Each risk MUST include exact supporting evidence from the text
- Use ONLY the predefined categories

Categories:
{json.dumps(self.taxonomy, indent=2)}

Return ONLY valid JSON in this format:
[
  {{
    "risk_category": "...",
    "risk_description": "...",
    "evidence": "..."
  }}
]

Text:
{context}
"""

    def run(self, context: str):

        prompt = self.build_prompt(context)

        response = self.llm.call_llm(prompt)

        parsed = self.llm.parse_json(response)

        return parsed
