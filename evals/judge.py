import sys
import os

# The evals directory is not part of the installed package, so we manually add
# the src directory to sys.path to make the audit_ai package importable.
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(os.path.join(PROJECT_ROOT, "src"))

import json
import time
import re
import pandas as pd
import numpy as np
from langchain_google_genai import ChatGoogleGenerativeAI
from audit_ai.config import EVAL_JUDGE_MODEL, GOOGLE_API_KEY

RESULTS_FILE = os.path.join(CURRENT_DIR, "rag_results.json")
CSV_FILE = os.path.join(CURRENT_DIR, "judge_results.csv")
REPORT_FILE = os.path.join(CURRENT_DIR, "judge_report.md")

# Scored dimensions. Unlike RAGAS (which decomposes answers into atomic
# statements), this is a direct rubric-graded LLM-as-judge pass: one prompt
# scores the whole answer on each axis 1-5 with a short rationale.
DIMENSIONS = {
    "correctness": "Is the answer factually correct and aligned with the ground truth?",
    "groundedness": "Is every claim supported by the retrieved context (no hallucination)?",
    "completeness": "Does the answer cover the key points the ground truth requires?",
    "relevance": "Does the answer directly address the question without padding?",
}

JUDGE_PROMPT = """You are a strict compliance-domain evaluator grading a RAG system's answer.

Score the ANSWER on each dimension using an integer 1-5 scale
(1 = terrible, 3 = acceptable, 5 = excellent).

Dimensions:
- correctness: {correctness}
- groundedness: {groundedness}
- completeness: {completeness}
- relevance: {relevance}

QUESTION:
{question}

RETRIEVED CONTEXT:
{context}

GROUND TRUTH:
{ground_truth}

ANSWER (to grade):
{answer}

Respond with ONLY a JSON object, no markdown fences, in this exact shape:
{{"correctness": <int>, "groundedness": <int>, "completeness": <int>, "relevance": <int>, "rationale": "<one sentence>"}}
"""


def _parse_scores(raw_text):
    """
    Extract the JSON verdict from the model output. Gemini occasionally wraps
    the object in ```json fences or trailing prose, so we grab the first
    balanced {...} block rather than json.loads-ing the whole string.
    """
    match = re.search(r"\{.*\}", raw_text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None


def run_judge_eval():
    if not os.path.exists(RESULTS_FILE):
        print(f"❌ Error: '{RESULTS_FILE}' not found.")
        print("   Please run 'python evals/collector.py' first to generate the dataset.")
        return

    with open(RESULTS_FILE, "r") as f:
        raw_data = json.load(f)

    print(f"📂 Loaded {len(raw_data)} records from {RESULTS_FILE}")

    # temperature=0 for deterministic, reproducible grading across runs.
    judge_llm = ChatGoogleGenerativeAI(
        model=EVAL_JUDGE_MODEL,
        temperature=0,
        google_api_key=GOOGLE_API_KEY,
    )

    print(f"🚀 Starting LLM-as-judge evaluation using {EVAL_JUDGE_MODEL}...")

    rows = []
    for i, e in enumerate(raw_data):
        print(f"[{i+1}/{len(raw_data)}] Grading: {e['question'][:50]}...")

        prompt = JUDGE_PROMPT.format(
            question=e["question"],
            context="\n\n".join(e.get("contexts", []))[:6000],
            ground_truth=e["ground_truth"],
            answer=e["answer"],
            **{k: v for k, v in DIMENSIONS.items()},
        )

        row = {
            "question": e["question"],
            "answer": e["answer"],
            "ground_truth": e["ground_truth"],
        }
        try:
            resp = judge_llm.invoke(prompt)
            scores = _parse_scores(resp.content)
        except Exception as err:
            print(f"   ⚠️  Judge call failed: {err}")
            scores = None

        if scores is None:
            # Record NaN so a parse/API failure doesn't silently count as a
            # passing score. It drops out of the averages instead.
            for d in DIMENSIONS:
                row[d] = np.nan
            row["rationale"] = "PARSE_FAILED"
        else:
            for d in DIMENSIONS:
                val = scores.get(d)
                row[d] = float(val) if isinstance(val, (int, float)) else np.nan
            row["rationale"] = str(scores.get("rationale", ""))

        rows.append(row)

        # Match collector's pacing to stay under Gemini free-tier limits.
        time.sleep(1)

    df = pd.DataFrame(rows)
    averages = df[list(DIMENSIONS)].mean().to_dict()

    print("\n--- 📊 FINAL AVERAGES (1-5 scale) ---")
    for k, v in averages.items():
        print(f"{k}: {v:.2f}")

    df.to_csv(CSV_FILE, index=False)
    _write_report(df, averages)


def _write_report(df, averages):
    with open(REPORT_FILE, "w") as f:
        f.write("# ⚖️ AuditAI: LLM-as-Judge Evaluation Report\n\n")
        f.write("Generated on: " + pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
        f.write(f"Judge model: `{EVAL_JUDGE_MODEL}` · Scale: 1 (worst) – 5 (best)\n\n")

        f.write("## 🏛️ Executive Summary\n\n")
        f.write("| Dimension | Avg Score | Status |\n")
        f.write("| :--- | :--- | :--- |\n")
        for dim, score in averages.items():
            if np.isnan(score):
                score = 0.0
            status = "✅ Strong" if score >= 4.0 else "⚠️ Needs Review"
            if score < 3.0:
                status = "❌ Weak"
            f.write(f"| **{dim.title()}** | `{score:.2f}` | {status} |\n")

        f.write("\n---\n\n## 📝 Per-Question Verdicts\n\n")
        for i, row in df.iterrows():
            f.write(f"### Question {i+1}\n")
            f.write(f"**Question:** {row['question']}\n\n")
            scores_line = " · ".join(
                f"{d.title()}: `{row[d]:.0f}`" if not np.isnan(row[d]) else f"{d.title()}: `N/A`"
                for d in DIMENSIONS
            )
            f.write(f"**Scores:** {scores_line}\n\n")
            f.write(f"**Rationale:** {row['rationale']}\n\n")
            f.write("---\n\n")

    print(f"✅ Report generated: '{REPORT_FILE}'")


if __name__ == "__main__":
    run_judge_eval()
