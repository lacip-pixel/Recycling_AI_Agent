# eval.py
import pandas as pd
import os
from ai_agent import classify_item
from caption import caption_image

def run_combined_evaluation(input_path, output_path):
    df = pd.read_csv(input_path)
    results = []

    for _, row in df.iterrows():
        item_desc = str(row.get("Item Description", "")).strip()
        image_path = str(row.get("Image Path", "")).strip()
        location = str(row["Location"]).strip()
        expected = str(row["Expected Classification"]).strip().lower()

        # Determine item input: caption image if image path exists
        if image_path and os.path.isfile(image_path):
            try:
                item = caption_image(image_path)
                source = f"[🖼️ from {os.path.basename(image_path)}]"
            except Exception as e:
                item = "unknown item"
                source = f"[error: {e}]"
        else:
            item = item_desc
            source = "[text]"

        result = classify_item(item, location)
        predicted = result["classification"]
        match = "✅" if predicted == expected else "❌"

        results.append({
            "Item Description": item_desc,
            "Image Path": image_path,
            "Generated Caption": item if source.startswith("[🖼️") else "",
            "Location": location,
            "Expected Classification": expected,
            "Predicted": predicted,
            "Match": match,
            "LLM Reason": result["response"],
            "Input Source": source
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)
    print(f"✅ Evaluation complete. Results saved to: {output_path}")

if __name__ == "__main__":
    run_combined_evaluation("evaluation_input_130.csv", "evaluation_output.csv")
