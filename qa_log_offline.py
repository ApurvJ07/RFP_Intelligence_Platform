
import os
from search.engine import HybridSearchEngine

QUESTIONS = [
    ("Bid1", "What is the final proposal due date after all addenda?"),
    ("Bid1", "When is the pre-proposal meeting and how is it held?"),
    ("Bid1", "What warranty is required for student Chromebooks and Windows laptops?"),
    ("Bid1", "Must Form 1295 be submitted with the proposal or after award?"),
    ("Bid1", "Is laser etching required on monitors or only laptops?"),
    ("Bid2", "What is the proposal due date for the Dell laptop bid?"),
    ("Bid2", "Which affidavits are required for the Dell laptop bid?"),
    ("Bid2", "Which master contract may bidders use?"),
    ("Bid2", "What is the delivery deadline after award?"),
    ("Bid2", "What is the SKU for the Latitude 5550 base configuration?"),
]

os.makedirs("outputs", exist_ok=True)

print("Loading existing local search index...")
engine = HybridSearchEngine()

report = [
    "# RFP Intelligence Platform - Retrieval Evidence Report",
    "",
    "> This report uses local hybrid retrieval (ChromaDB + BM25).",
    "> Evidence excerpts are retrieved from the source documents;",
    "> they are not AI-generated or independently verified answers.",
    "",
]

successful = 0
failed = 0

for bid, question in QUESTIONS:
    print(f"\nSearching {bid}: {question}")

    try:
        results = engine.search(
            query=question,
            bid_id=bid,
            top_k=3,
            mode="hybrid"
        )

        report.extend([
            f"## Question ({bid})",
            "",
            question,
            "",
        ])

        if not results:
            report.extend([
                "**Result:** No relevant passages retrieved.",
                "",
            ])
            failed += 1
            continue

        report.extend([
            "### Retrieved evidence",
            "",
        ])

        for rank, item in enumerate(results, start=1):
            filename = item.get("file_name", "Unknown file")
            page = item.get("page_number", "Unknown")
            score = item.get("score", "N/A")
            text = item.get("text", "").strip()

            report.extend([
                f"**Passage {rank}**",
                "",
                f"- Source: `{filename}`",
                f"- Page: {page}",
                f"- Retrieval score: {score}",
                "",
                text,
                "",
            ])

        report.extend(["---", ""])
        successful += 1
        print(f"Retrieved {len(results)} passages.")

    except Exception as error:
        report.extend([
            "**Retrieval error:**",
            str(error),
            "",
            "---",
            "",
        ])
        failed += 1
        print(f"Failed: {error}")

output_path = "outputs/qa_log_retrieval.md"

with open(output_path, "w", encoding="utf-8") as file:
    file.write("\n".join(report))

print("\nRetrieval report completed.")
print(f"Questions with evidence: {successful}/10")
print(f"Questions without evidence or errors: {failed}/10")
print(f"Saved: {output_path}")
