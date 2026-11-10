
import os
import re

from search.engine import HybridSearchEngine
from agents.workflow import MultiAgentOrchestrator

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

log_path = "outputs/qa_log.md"

# Load previously completed answers.
existing_sections = {}

if os.path.exists(log_path):
    with open(log_path, "r", encoding="utf-8") as file:
        old_log = file.read()

    pattern = r"## Question \((Bid1|Bid2)\)\n(.*?)(?=\n## Question \(|\Z)"

    for match in re.finditer(pattern, old_log, re.DOTALL):
        bid = match.group(1)
        section = match.group(2).strip()

        if "**Error:**" not in section and "### Answer" in section:
            question = section.split("### Answer")[0].strip()
            existing_sections[(bid, question)] = section

print(f"Previously completed answers: {len(existing_sections)}")

engine = HybridSearchEngine()

for folder in ["data/Bid1", "data/Bid2"]:
    print(f"Indexing {folder}...")
    engine.index_bid_folder(folder)

orch = MultiAgentOrchestrator(engine)

lines = ["# RFP Intelligence Platform - Q&A Log", ""]
successful = 0
failed = 0

for bid, question in QUESTIONS:
    key = (bid, question)

    # Preserve an answer that was already successful.
    if key in existing_sections:
        print(f"\nSkipping completed question ({bid}): {question}")
        lines.extend([
            f"## Question ({bid})",
            existing_sections[key],
            "",
            "---",
            ""
        ])
        successful += 1
        continue

    print(f"\nAsking ({bid}): {question}")

    try:
        result = orch.ask_question(question, bid_id=bid)

        citations = [
            f"{source['file']} (Page {source['page']})"
            for source in result["sources"]
        ]

        section = [
            question,
            "",
            "### Answer",
            result["answer"],
            "",
            "### Sources",
            *[f"- {citation}" for citation in citations],
            "",
            f"**Confidence:** {result['confidence']}"
        ]

        lines.extend([
            f"## Question ({bid})",
            *section,
            "",
            "---",
            ""
        ])

        successful += 1
        print("Answered successfully.")

    except Exception as error:
        lines.extend([
            f"## Question ({bid})",
            question,
            "",
            f"**Error:** {error}",
            "",
            "---",
            ""
        ])

        failed += 1
        print(f"Failed: {error}")

with open(log_path, "w", encoding="utf-8") as file:
    file.write("\n".join(lines))

print("\nQ&A generation completed.")
print(f"Successful answers: {successful}/10")
print(f"Failed answers: {failed}/10")
print(f"Saved: {log_path}")
