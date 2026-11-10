import argparse
import json
from search.engine import HybridSearchEngine


def build_engine():
    engine = HybridSearchEngine()
    for folder in ["data/Bid1", "data/Bid2"]:
        engine.index_bid_folder(folder)
    return engine


def main():
    p = argparse.ArgumentParser(description="RFP Intelligence Platform CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="search the bid documents (no LLM used)")
    s.add_argument("query")
    s.add_argument("--bid", default=None, help="Bid1 or Bid2 (default: all)")
    s.add_argument("--top-k", type=int, default=5)
    s.add_argument("--mode", default="hybrid_rerank", choices=["vector", "hybrid", "hybrid_rerank"])

    a = sub.add_parser("ask", help="ask a question (uses 1 LLM request)")
    a.add_argument("question")
    a.add_argument("--bid", default=None)

    args = p.parse_args()
    engine = build_engine()

    if args.cmd == "search":
        results = engine.search(args.query, bid_id=args.bid, top_k=args.top_k, mode=args.mode)
        for i, r in enumerate(results, 1):
            print(f"\n#{i}  {r['bid_id']} | {r['file_name']} | page {r['page_number']}")
            print("    " + str(r["text"])[:300].replace("\n", " "))
    else:
        from agents.workflow import MultiAgentOrchestrator
        print(json.dumps(MultiAgentOrchestrator(engine).ask_question(args.question, bid_id=args.bid), indent=2))


if __name__ == "__main__":
    main()