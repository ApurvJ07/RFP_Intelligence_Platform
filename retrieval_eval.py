import os
import sys
import json

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

from search.engine import HybridSearchEngine

# bid=None means "search across all bids" (tests cross-bid retrieval).
# must = words that must ALL appear in a passage for it to count as correct.
QUESTIONS = [
    # ---------------- Bid1 (Dallas ISD) ----------------
    {"q": "What is the submission deadline after all addendums?", "bid": "Bid1", "must": ["july 9, 2024"]},
    {"q": "When is the pre-proposal meeting and how is it held?", "bid": "Bid1", "must": ["teams", "june 10"]},
    {"q": "How long is the contract term and what are the renewal options?", "bid": "Bid1", "must": ["three (3) year agreement"]},
    {"q": "What warranty is required for student Chromebooks and Windows laptops?", "bid": "Bid1", "must": ["one-year warranty for"]},
    {"q": "Must Form 1295 be submitted with the proposal or after award?", "bid": "Bid1", "must": ["form 1295 should be submitted"]},
    {"q": "Is laser etching required on monitors or only laptops?", "bid": "Bid1", "must": ["etching on laptops"]},
    {"q": "What software setup is required for Windows devices?", "bid": "Bid1", "must": ["autopilot is required"]},
    {"q": "How many Chromebooks are listed for evaluation pricing?", "bid": "Bid1", "must": ["50,000"]},
    {"q": "How many points does purchase price carry in the evaluation?", "bid": "Bid1", "must": ["purchase price - 40 points"]},
    {"q": "Who is the buyer and what is their email?", "bid": "Bid1", "must": ["alzate"]},
    {"q": "Which M/WBE forms must be attached to the submission?", "bid": "Bid1", "must": ["mwbe forms must be completed"]},
    {"q": "Is the EPCNT cooperative rider part of this bid?", "bid": "Bid1", "must": ["epcnt"]},
    {"q": "When is a hot swap device required?", "bid": "Bid1", "must": ["hot swap is only"]},
    {"q": "Are bid bond or insurance requirements stated?", "bid": "Bid1", "must": ["insurance and/or bond"]},
    # ---------------- Bid2 (Maryland Treasurer) ----------------
    {"q": "What is the proposal due date for the Dell laptop bid?", "bid": "Bid2", "must": ["06/10/2024"]},
    {"q": "Which affidavits are required for the Dell laptop bid?", "bid": "Bid2", "must": ["mercury affidavit"]},
    {"q": "Which master contract may bidders use?", "bid": "Bid2", "must": ["060b5400007"]},
    {"q": "What is the delivery deadline after award?", "bid": "Bid2", "must": ["45 days"]},
    {"q": "Where should invoices be emailed?", "bid": "Bid2", "must": ["stoaccountspayable"]},
    {"q": "Is a Letter of Authorization from Dell needed?", "bid": "Bid2", "must": ["letter of authorization"]},
    {"q": "What is the SKU for the Latitude 5550 base configuration?", "bid": "Bid2", "must": ["210-blyz"]},
    {"q": "What is the model number of the Thunderbolt dock?", "bid": "Bid2", "must": ["wd22tb4"]},
    {"q": "What is the procurement officer's phone number?", "bid": "Bid2", "must": ["410-260-7533"]},
    {"q": "Can the vendor charge a restocking fee?", "bid": "Bid2", "must": ["restocking fee"]},
    # ---------------- Cross-bid (no bid filter) ----------------
    {"q": "Mercury affidavit requirement", "bid": None, "must": ["mercury affidavit"]},
    {"q": "Form 1295 certificate of interested parties", "bid": None, "must": ["form 1295"]},
]

MODES = ["vector", "hybrid", "hybrid_rerank"]
KS = [1, 3, 5, 10]
TOP_K = 10


def norm(s):
    return " ".join(str(s).lower().split())


def is_relevant(text, must):
    t = norm(text)
    return all(m in t for m in must)


def oracle_count(engine, bid, must):
    """How many indexed chunks contain the answer text (sanity check)."""
    n = 0
    for c in engine.bm25_corpus:
        if bid and c["bid_id"] != bid:
            continue
        if is_relevant(c["text"], must):
            n += 1
    return n


def evaluate(engine, mode, valid_questions):
    hits = {k: 0 for k in KS}
    rr_total = 0.0
    details = []
    for item in valid_questions:
        results = engine.search(item["q"], bid_id=item["bid"], top_k=TOP_K, mode=mode)
        first_rank = None
        for rank, r in enumerate(results, start=1):
            if is_relevant(r["text"], item["must"]):
                first_rank = rank
                break
        if first_rank:
            rr_total += 1.0 / first_rank
            for k in KS:
                if first_rank <= k:
                    hits[k] += 1
        top = results[0] if results else {}
        details.append({
            "question": item["q"],
            "bid": item["bid"],
            "first_relevant_rank": first_rank,
            "top1_file": top.get("file_name"),
            "top1_page": top.get("page_number"),
        })
    n = len(valid_questions)
    metrics = {f"Recall@{k}": round(hits[k] / n, 3) for k in KS}
    metrics["MRR"] = round(rr_total / n, 3)
    return metrics, details


def main():
    engine = HybridSearchEngine()
    for folder in ["data/Bid1", "data/Bid2"]:
        n = engine.index_bid_folder(folder)
        print(f"[index] {folder}: {n} chunks (upserted)")

    valid, skipped = [], []
    for item in QUESTIONS:
        cnt = oracle_count(engine, item["bid"], item["must"])
        if cnt == 0:
            skipped.append(item)
        else:
            valid.append(item)

    print(f"\nQuestions total: {len(QUESTIONS)} | usable: {len(valid)} | skipped: {len(skipped)}")
    for s in skipped:
        print("  [SKIPPED - answer text not found in indexed chunks]", s["q"], s["must"])

    report = {"num_questions": len(valid), "skipped": [s["q"] for s in skipped], "modes": {}}
    for mode in MODES:
        print(f"\n[eval] running mode: {mode} ...")
        metrics, details = evaluate(engine, mode, valid)
        report["modes"][mode] = {"metrics": metrics, "details": details}

    # results table
    header = ["Configuration"] + [f"Recall@{k}" for k in KS] + ["MRR"]
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    for mode in MODES:
        m = report["modes"][mode]["metrics"]
        row = [mode] + [str(m[f"Recall@{k}"]) for k in KS] + [str(m["MRR"])]
        lines.append("| " + " | ".join(row) + " |")
    table = "\n".join(lines)
    print("\n" + table)

    # failures for the best-expected mode
    print("\nQuestions where hybrid_rerank missed the top 3:")
    for d in report["modes"]["hybrid_rerank"]["details"]:
        if d["first_relevant_rank"] is None or d["first_relevant_rank"] > 3:
            print("  rank:", d["first_relevant_rank"], "|", d["question"])

    os.makedirs("outputs", exist_ok=True)
    with open("outputs/retrieval_eval.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open("outputs/retrieval_eval.md", "w", encoding="utf-8") as f:
        f.write(f"Questions used: {len(valid)}\n\n{table}\n")
    print("\nSaved outputs/retrieval_eval.json and outputs/retrieval_eval.md")


if __name__ == "__main__":
    main()