import os
import json
import time
import re
import requests


class AgenticDebateEngine:

    def __init__(self):

        self.model_name = "gemma3:4b"
        self.ollama_url = "http://localhost:11434/api/generate"

        self.max_retries = 2

    # -----------------------------
    # PROMPTS
    # -----------------------------

    def detector_prompt(self, context: str) -> str:
        return f"""
You are a financial risk detection system.

TASK:
Identify financial risks explicitly mentioned in the text.

RULES:
- DO NOT infer or hallucinate
- ONLY use evidence from text
- Classify risks into:
  Liquidity, Credit, Operational, Supply Chain, Regulatory, Market, Legal

OUTPUT FORMAT:
Return ONLY valid JSON.
No explanations. No markdown.

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
Validate the extracted risks.

RULES:
- Remove unsupported risks
- Keep only evidence-backed risks
- Do not hallucinate


OUTPUT FORMAT:
Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.


{detected_output}

TEXT:
{context}
"""

    def synthesizer_prompt(self, validated_output: str) -> str:
        return f"""
You are a financial summarization system.

TASK:
Produce final clean risk output.

RULES:
- Keep only high-confidence risks
- Ensure valid JSON

OUTPUT FORMAT:
Return ONLY valid JSON.
No explanations.
No markdown.
No extra text.

{validated_output}
"""

    # -----------------------------
    # OLLAMA CALL
    # -----------------------------

    def call_llm_local(self, prompt: str) -> str:

        response = requests.post(
            self.ollama_url,
            json={
                "model": self.model_name,
                "prompt": prompt,
                "stream": False
            }
        )

        print("\n--- FULL RESPONSE JSON ---\n")
        print(response.json())

        return response.json().get("response", "")

    # -----------------------------
    # JSON EXTRACTION
    # -----------------------------

    def extract_json(self, text: str):

        if not text:
            return None

        text = text.strip()

        # Remove markdown wrappers
        text = re.sub(r"```json", "", text)
        text = re.sub(r"```", "", text)

        # Direct parse
        try:
            return json.loads(text)
        except:
            pass

        # Fallback extraction
        json_candidates = re.findall(r"\{[\s\S]*?\}", text)

        for candidate in json_candidates:
            try:
                return json.loads(candidate)
            except:
                continue

        return None

    # -----------------------------
    # SAFE CALL
    # -----------------------------

    def safe_llm_call(self, prompt: str):

        for attempt in range(self.max_retries):

            try:
                raw_output = self.call_llm_local(prompt)

                print("\n--- RAW LLM OUTPUT ---\n")
                print(raw_output[:500])

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
            print("Detector failed")
            return None

        # Skeptic
        skeptic_output = self.safe_llm_call(
            self.skeptic_prompt(context, json.dumps(detector_output))
        )

        if not skeptic_output:
            print("Skeptic failed")
            return None

        # Synthesizer
        final_output = self.safe_llm_call(
            self.synthesizer_prompt(json.dumps(skeptic_output))
        )

        if not final_output:
            print("Synthesizer failed")
            return None

        return final_output

    # -----------------------------
    # SAVE OUTPUT
    # -----------------------------

    def save_output(self, output, save_path: str):

        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)


# -----------------------------
# TEST BLOCK
# -----------------------------

if __name__ == "__main__":

    from src.retrieval.hybrid_retriever import HybridRetriever

    engine = AgenticDebateEngine()

    retriever = HybridRetriever()
    retriever.initialize()

    query = "Identify financial risks in this filing"

    results = retriever.retrieve(query, final_k=1)  # reduced for local model

    context = "\n\n".join([r["chunk_text"] for r in results])

    output = engine.run_debate(context)

    if output:
        print("\n=== FINAL OUTPUT ===\n")
        print(json.dumps(output, indent=2))
    else:
        print("Failed to generate output.")