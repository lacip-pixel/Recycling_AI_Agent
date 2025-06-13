# eval.py
import pandas as pd
from ai_agent import classify_item

# Load evaluation dataset
df = pd.read_csv("Evaluation_Set.csv")  # Make sure your test set is named this way

results = []

for _, row in df.iterrows():
    item = row["Item Description"]
    location = row["Location"]
    expected = row["Expected Classification"].strip().lower()

    result = classify_item(item, location)
    predicted = result["classification"]
    match = "✅" if predicted == expected else "❌"

    results.append({
        "Item Description": item,
        "Location": location,
        "Expected Classification": expected,
        "Predicted": predicted,
        "Match": match,
        "LLM Reason": result["response"]
    })

# Save output
output_df = pd.DataFrame(results)
output_df.to_csv("evaluation_results.csv", index=False)
print("Evaluation complete. Saved to evaluation_results.csv")
