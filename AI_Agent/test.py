# test_agent.py
from ai_agent import classify_item

item = "used paper plate with food residue"
location = "Chicago, IL"
result = classify_item(item, location)

print("\n--- RESULT ---")
print(result)
