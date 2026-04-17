from src.generation.detector_agent import DetectorAgent
from src.generation.skeptic_agent import SkepticAgent


sample_text = """
The company maintains strong liquidity and has sufficient cash reserves.

However, we face credit risk from customers defaulting on payments.

There is also significant cybersecurity risk affecting operations.
"""

detector = DetectorAgent("configs/risk_taxonomy.json")
skeptic = SkepticAgent()

detected = detector.run(sample_text)

validated = skeptic.run(sample_text, detected)

print("\n=== DETECTOR ===\n")
print(detected)

print("\n=== SKEPTIC ===\n")
print(validated)