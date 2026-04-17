from src.generation.detector_agent import DetectorAgent
from src.generation.skeptic_agent import SkepticAgent
from src.generation.synthesizer_agent import SynthesizerAgent


sample_text = """
The company maintains strong liquidity and has sufficient cash reserves.

However, we face credit risk from customers defaulting on payments.

There is potential exposure to supply chain disruptions due to reliance on overseas suppliers.

We aim to improve operational efficiency through automation.
"""

detector = DetectorAgent("configs/risk_taxonomy.json")
skeptic = SkepticAgent()
synthesizer = SynthesizerAgent()


detected = detector.run(sample_text)
validated = skeptic.run(sample_text, detected)
final_output = synthesizer.run(validated)


print("\n=== DETECTOR ===\n", detected)
print("\n=== SKEPTIC ===\n", validated)
print("\n=== FINAL OUTPUT ===\n", final_output)