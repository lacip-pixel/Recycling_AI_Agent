import pandas as pd
from ai_agent import classify_item

# Load test set
df = pd.read_csv("recycling_test_set_25_items_with_locations.csv")

# Initialize result lists
predicted_classes = []
match_flags = []
reasons = []

# Loop through each row
for index, row in df.iterrows():
    item = row["Item Description"]
    location = row["Location"]
    expected = row["Expected Classification"].strip().lower()

    print(f"Testing: {item} @ {location}")
    result = classify_item(item, location)

    # Parse result
    lines = result.lower().splitlines()
    class_line = next((l for l in lines if "classification:" in l), "")
    reason_line = next((l for l in lines if "reason:" in l), "")
    predicted = class_line.split(":")[1].strip() if ":" in class_line else "unknown"

    match = "✅" if predicted == expected else "❌"

    predicted_classes.append(predicted)
    match_flags.append(match)
    reasons.append(reason_line.strip())

# Save to new CSV
df["Predicted"] = predicted_classes
df["Match"] = match_flags
df["LLM Reason"] = reasons

df.to_csv("evaluation_results.csv", index=False)
print("✅ Done! Results saved to evaluation_results.csv.")
