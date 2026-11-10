import sys
import json
import argparse
from search.engine import HybridSearchEngine
from agents.workflow import MultiAgentOrchestrator

def main():
    parser = argparse.ArgumentParser(description="RFP Intelligence Platform CLI")
    parser.add_argument("target", help="Path to bid folder (e.g. data/Bid1) or 'ask'")
    parser.add_argument("--question", "-q", help="Question to ask in Q&A mode", default=None)
    parser.add_argument("--bid", "-b", help="Optional bid_id filter for Q&A", default=None)
    args = parser.parse_args()

    engine = HybridSearchEngine()
    orchestrator = MultiAgentOrchestrator(engine)

    if args.target == "ask" or args.question:
        q = args.question or input("Enter your question: ")
        ans = orchestrator.ask_question(q, bid_id=args.bid)
        print(json.dumps(ans, indent=2))
    else:
        print(f"[*] Running Multi-Agent Extraction Pipeline on: {args.target} ...")
        result = orchestrator.run_extraction(args.target)
        print(f"[+] Extraction complete! Saved to outputs/{result['bid_id']}.json")
        print(json.dumps(result["validation"], indent=2))

if __name__ == "__main__":
    main()