import sys
import os

# The evals directory is not part of the installed package, so we manually add
# the src directory to sys.path to make the audit_ai package importable.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import json
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)
from audit_ai.config import EVAL_JUDGE_MODEL, GOOGLE_API_KEY, EMBEDDING_MODEL
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import numpy as np
import warnings

# Suppress deprecation warnings from third-party libraries (RAGAS and the Google SDK)
# that are outside our control. We scope these tightly to avoid masking our own warnings.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="ragas")
warnings.filterwarnings("ignore", category=DeprecationWarning, module="google")

RESULTS_FILE = os.path.join(CURRENT_DIR, "rag_results.json")
REPORT_FILE = os.path.join(CURRENT_DIR, "ragas_report.md")


def generate_markdown_report(df, averages):
    """
    Writes a structured Markdown evaluation report to disk.

    The report is split into an executive summary (aggregate scores) and a
    per-question breakdown so that both high-level trends and individual failure
    cases are easy to identify.
    """
    with open(REPORT_FILE, "w") as f:
        f.write("# 📊 AuditAI: RAG Evaluation Report\n\n")
        f.write("Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")

        f.write("## 🏛️ Executive Summary\n")
        f.write("Below are the average scores across all evaluated metrics.\n\n")

        f.write("| Metric | Score | Status |\n")
        f.write("| :--- | :--- | :--- |\n")

        for metric, score in averages.items():
            if np.isnan(score):
                score = 0.0

            status = "✅ Passing" if score >= 0.7 else "⚠️ Needs Review"
            if score < 0.3: status = "❌ Failing"

            metric_name = metric.replace('_', ' ').title()
            f.write(f"| **{metric_name}** | `{score:.4f}` | {status} |\n")

        f.write("\n---\n\n")
        f.write("## 📝 Detailed Performance Breakdown\n\n")

        for i, row in df.iterrows():
            f.write(f"### Question {i+1}\n")
            f.write(f"**Question:** {row['question']}\n\n")
            f.write(f"**AI Answer:** {row['answer']}\n\n")
            f.write(f"**Ground Truth:** {row['ground_truth']}\n\n")

            f.write("**Scores:**\n")
            for m in averages.keys():
                if m in row:
                    val = row[m]
                    val_str = f"`{val:.4f}`" if not (isinstance(val, float) and np.isnan(val)) else "`N/A`"
                    f.write(f"- {m.replace('_', ' ').title()}: {val_str}\n")
            f.write("\n---\n\n")

    print(f"✅ Complete report generated: '{REPORT_FILE}'")


def run_ragas_eval():
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ Error: '{RESULTS_FILE}' not found.")
        print("   Please run 'python evals/collector.py' first to generate the dataset.")
        return

    with open(RESULTS_FILE, "r") as f:
        raw_data = json.load(f)

    print(f"📂 Loaded {len(raw_data)} records from {RESULTS_FILE}")

    # Evaluate only the first 10 records to stay within Gemini's free-tier token
    # budget. The full dataset is preserved in rag_results.json for future use.
    raw_data = raw_data[:10]
    print(f"⚠️  Running on {len(raw_data)} records...")

    dataset = Dataset.from_dict({
        "question": [e["question"] for e in raw_data],
        "answer": [e["answer"] for e in raw_data],
        "contexts": [e["contexts"] for e in raw_data],
        "ground_truth": [e["ground_truth"] for e in raw_data],
    })

    # Use the same model family for the judge as for generation to keep the
    # evaluation environment consistent with production behaviour.
    judge_llm = ChatGoogleGenerativeAI(
        model=EVAL_JUDGE_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=GOOGLE_API_KEY,
    )

    print(f"🚀 Starting RAGAS Evaluation using {EVAL_JUDGE_MODEL}...")
    # raise_exceptions=False allows partial results when a single metric fails,
    # rather than aborting the entire evaluation run.
    # max_workers=1 serialises requests to avoid rate limit bursts.
    results = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
        ],
        llm=judge_llm,
        embeddings=embeddings,
        raise_exceptions=False,
        run_config=RunConfig(max_workers=1),
    )

    df = results.to_pandas()
    df_input = dataset.to_pandas()

    # RAGAS occasionally strips input columns from its output dataframe.
    # Re-attach them from the original dataset so the report generator can
    # reference question text and ground truth without a KeyError.
    for col in ["question", "answer", "ground_truth"]:
        if col not in df.columns and col in df_input.columns:
            df[col] = df_input[col]

    numeric_df = df.select_dtypes(include=[np.number])
    averages = numeric_df.mean().to_dict()

    print("\n--- 📊 FINAL AVERAGES ---")
    for k, v in averages.items():
        print(f"{k}: {v:.4f}")

    df.to_csv(os.path.join(CURRENT_DIR, "ragas_results.csv"), index=False)
    generate_markdown_report(df, averages)


if __name__ == "__main__":
    run_ragas_eval()
