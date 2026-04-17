from src.generation.llm_interface import LLMInterface
import json


class OptimizedRiskAgent:

    def __init__(self):
        self.llm = LLMInterface()

    def build_prompt(self, text):

        return f"""
You are a financial risk analyst.

Your task:

1. Extract ONLY real financial risks
2. Ignore positive statements (e.g., "strong liquidity")
3. Ignore vague or unsupported risks
4. Each risk MUST have supporting evidence from text

Assign confidence:
- 1.0 → Explicit risk
- 0.7–0.9 → Strong but slightly inferred
- 0.4–0.6 → Weak

Return ONLY JSON:
[
  {{
    "risk_category": "...",
    "risk_description": "...",
    "evidence": "...",
    "confidence": 0.0
  }}
]

Text:
{text}
"""

    def run(self, text):

        if not text.strip():
            return []

        prompt = self.build_prompt(text)

        response = self.llm.call_llm(prompt)

        parsed = self.llm.parse_json(response)

        return parsed if parsed else []
