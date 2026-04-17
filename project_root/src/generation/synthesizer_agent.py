from src.generation.llm_interface import LLMInterface
import json


class SynthesizerAgent:

    def __init__(self):
        self.llm = LLMInterface()

    def build_prompt(self, validated_risks: list):

        return f"""
You are a financial analyst.

Assign a CONFIDENCE SCORE (0 to 1) to each risk.

STRICT RULES:
- 1.0 → Explicit, clearly stated risk with direct wording
- 0.7–0.9 → Strong but slightly inferred
- 0.4–0.6 → Weak or indirect risk
- <0.4 → Very weak (should rarely occur)

DO NOT assign all risks the same score.

Return ONLY JSON:
[
  {{
    "risk_category": "...",
    "risk_description": "...",
    "evidence": "...",
    "confidence": 0.0
  }}
]

Validated Risks:
{json.dumps(validated_risks, indent=2)}
"""

    def run(self, validated_risks: list):

        if not validated_risks:
            return []

        prompt = self.build_prompt(validated_risks)

        response = self.llm.call_llm(prompt)

        parsed = self.llm.parse_json(response)

        return parsed if parsed else []

