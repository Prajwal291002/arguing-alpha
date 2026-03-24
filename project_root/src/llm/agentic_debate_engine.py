import os
import json
import time
import re
import dotenv
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

GEMKEY = os.getenv("GEMKEY")
class AgenticDebateEngine:

    def __init__(self, api_key: str):

        self.client = genai.Client(api_key=api_key)

        self.model_name = "gemini-2.0-flash"

        self.max_retries = 2

    # -----------------------------
    # PROMPTS
    # -----------------------------

    def detector_prompt(self, context: str) -> str:
        return f"""
You are a financial risk detection system.

TASK:
Identify financial risks from the provided text.

STRICT RULES:
- ONLY extract risks explicitly mentioned
- DO NOT infer or hallucinate
- Classify into:
  Liquidity, Credit, Operational, Supply Chain, Regulatory, Market, Legal

OUTPUT FORMAT (STRICT JSON ONLY):
{{
  "risks": [
    {{
      "risk_type": "...",
      "description": "...",
      "evidence": "...",
      "confidence": 0.0
    }}
  ]
}}

TEXT:
{context}
"""

    def skeptic_prompt(self, context: str, detected_output: str) -> str:
        return f"""
You are a financial risk auditor.

TASK:
Verify if each extracted risk is supported by the text.

RULES:
- Remove hallucinated risks
- Keep only evidence-backed claims
- Do not modify wording unnecessarily

OUTPUT FORMAT:
Return ONLY valid JSON.
Do NOT include explanations, markdown, or extra text.
{detected_output}

TEXT:
{context}
"""

    def synthesizer_prompt(self, validated_output: str) -> str:
        return f"""
You are a financial summarization system.

TASK:
Produce the final validated risk list.

RULES:
- Clean formatting
- Ensure valid JSON
- Keep only high-confidence risks

OUTPUT FORMAT (STRICT JSON ONLY):
{validated_output}
"""

    # -----------------------------
    # GEMINI CALL
    # -----------------------------

    def call_gemini(self, prompt: str) -> str:

        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                response_mime_type="application/json"
            )
        )

        return response.text

    # -----------------------------
    # JSON EXTRACTION
    # -----------------------------

    def extract_json(self, text: str):

        if not text:
            return None

        # Remove markdown wrappers if present
        text = text.strip()
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)

        # Try direct parse first
        try:
            return json.loads(text)
        except:
            pass

        # Fallback: extract first valid JSON block
        json_candidates = re.findall(r"\{[\s\S]*?\}", text)

        for candidate in json_candidates:
            try:
                return json.loads(candidate)
            except:
                continue

        return None

    # -----------------------------
    # SAFE CALL WITH RETRY
    # -----------------------------

    def safe_llm_call(self, prompt: str):

        for attempt in range(self.max_retries):
            try:
                raw_output = self.call_gemini(prompt)

                print("\n--- RAW LLM OUTPUT ---\n")
                print(raw_output[:500])  # first 500 chars

                parsed = self.extract_json(raw_output)

                if parsed:
                    return parsed

            except Exception as error:
                print(f"LLM Error: {error}")

            time.sleep(1)

        return None

    # -----------------------------
    # MAIN PIPELINE
    # -----------------------------

    def run_debate(self, context: str):

        # Detector
        detector_output = self.safe_llm_call(
            self.detector_prompt(context)
        )

        if not detector_output:
            return None

        # Skeptic
        skeptic_output = self.safe_llm_call(
            self.skeptic_prompt(context, json.dumps(detector_output))
        )

        if not skeptic_output:
            return None

        # Synthesizer
        final_output = self.safe_llm_call(
            self.synthesizer_prompt(json.dumps(skeptic_output))
        )

        return final_output

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------

    def save_output(self, output, save_path: str):

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


if __name__ == "__main__":

    from src.retrieval.hybrid_retriever import HybridRetriever

    API_KEY = f"{GEMKEY}"

    engine = AgenticDebateEngine(api_key=API_KEY)

    retriever = HybridRetriever()
    retriever.initialize()

    query = "Identify financial risks in this filing"

    results = retriever.retrieve(query, final_k=5)

    context = "\n\n".join([r["chunk_text"] for r in results])

    output = engine.run_debate(context)

    if output:
        print(json.dumps(output, indent=2))
    else:
        print("Failed to generate output.")