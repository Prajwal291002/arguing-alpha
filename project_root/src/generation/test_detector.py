from src.generation.detector_agent import DetectorAgent


sample_text = """
Our business is exposed to liquidity risk due to fluctuations in cash flow.
We may not be able to meet short-term obligations.
In addition, we face credit risk from customers defaulting on payments.
"""

agent = DetectorAgent("configs/risk_taxonomy.json")

results = agent.run(sample_text)

print("\n=== DETECTOR OUTPUT ===\n")
print(results)