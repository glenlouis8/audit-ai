import sys
import os

# The evals directory is not part of the installed package, so we manually add
# the src directory to sys.path to make the audit_ai package importable.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import csv
import json
import time
from audit_ai.rag.engine import process_query


def load_test_csv(file_path):
    rows = []
    if not os.path.exists(file_path):
        print(f"❌ Error: '{file_path}' not found.")
        return []
    with open(file_path, "r", encoding="utf-8") as f:
        f.readline()  # skip header row
        reader = csv.reader(f)
        for line in reader:
            if line:
                # Strip BOM markers and surrounding quotes that some CSV editors insert.
                q = line[0].replace("\ufeff", "").strip('"').strip()
                rows.append({"question": q, "ground_truth": line[1]})
    return rows


TEST_FILE = os.path.join(CURRENT_DIR, "test.csv")
RESULTS_FILE = os.path.join(CURRENT_DIR, "rag_results.json")


def collect_answers():
    test_questions = load_test_csv(TEST_FILE)
    collected_data = []

    print(f"🚀 Starting collection for {len(test_questions)} questions...")

    for i, item in enumerate(test_questions):
        print(f"[{i+1}/{len(test_questions)}] Processing: {item['question'][:50]}...")

        response = process_query(item["question"])

        collected_data.append(
            {
                "question": item["question"],
                "answer": str(response.get("answer", "")),
                "contexts": [
                    str(doc.page_content) for doc in response.get("context", [])
                ],
                "ground_truth": item["ground_truth"],
            }
        )

        # A 1-second delay between requests keeps throughput within Gemini's
        # free-tier rate limits and avoids 429 errors mid-collection.
        time.sleep(1)

    with open(RESULTS_FILE, "w") as f:
        json.dump(collected_data, f, indent=4)

    print("✅ Collection complete! Data saved to 'rag_results.json'")


if __name__ == "__main__":
    collect_answers()
