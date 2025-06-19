# eval.py
import pandas as pd
from ai_agent import classify_item

def run_evaluation(input_path, output_path):
    df = pd.read_csv(input_path)
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

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    print(f"Evaluation complete. Saved to {output_path}")

# Run both evaluations
run_evaluation("evaluation_set_large.csv", "evaluation_results_set1.csv")
