from src.generation.llm_interface import LLMInterface
import json


class SkepticAgent:

    def __init__(self):
        self.llm = LLMInterface()

    def build_prompt(self, context: str, extracted_risks: list):

        return f"""
You are a STRICT financial auditor.

Your job is to REMOVE invalid risks.

STRICT RULES:
1. A risk is VALID only if:
   - It explicitly describes a NEGATIVE or harmful condition
   - It is clearly supported by the text
2. If the sentence describes a POSITIVE condition → REMOVE
3. If the risk is vague or inferred → REMOVE
4. If evidence is not directly supporting → REMOVE

You MUST be strict. It is better to REMOVE than to keep weak risks.

Return ONLY valid JSON:
[
  {{
    "risk_category": "...",
    "risk_description": "...",
    "evidence": "..."
  }}
]

TEXT:
{context}

EXTRACTED RISKS:
{json.dumps(extracted_risks, indent=2)}
"""
    
    def run(self, context: str, extracted_risks: list):

        if not extracted_risks:
            return []

        prompt = self.build_prompt(context, extracted_risks)

        response = self.llm.call_llm(prompt)

        parsed = self.llm.parse_json(response)

        return parsed if parsed else []