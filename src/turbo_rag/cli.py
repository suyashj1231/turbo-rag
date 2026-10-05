"""CLI entry point. After `uv sync --all-extras`:

    uv run rag ingest                     # data/raw -> chunks + embeddings
    uv run rag ask "..." [--mode dense]   # one question, cited answer
    uv run rag chat                       # multi-turn REPL (persisted sessions)
    uv run rag serve                      # HTTP API on :8000
    uv run rag make-questions             # synthetic DEV set from the corpus
    uv run rag validate-golden            # check golden labels against the corpus
    uv run rag eval [--set dev]           # retrieval ablation + LLM judge (golden by default)
    uv run rag ragas [--gate]             # Ragas metrics; --gate fails below thresholds
"""

import argparse

from .retrieval.retriever import MODES


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="ingest data/raw into chunks + embeddings")

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--mode", choices=MODES, default=None)

    sub.add_parser("chat", help="interactive multi-turn chat")

    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--port", type=int, default=8000)

    mq = sub.add_parser("make-questions", help="generate a synthetic dev set")
    mq.add_argument("-n", type=int, default=45)

    sub.add_parser("validate-golden", help="check golden-set labels against the corpus")

    ev = sub.add_parser("eval", help="retrieval ablation + LLM-as-judge scoreboard")
    ev.add_argument("--set", dest="eval_set", choices=("golden", "dev"), default="golden")
    ev.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ev.add_argument("--no-llm", action="store_true", help="retrieval metrics only (no LLM calls)")
    ev.add_argument("--gate", action="store_true", help="exit 1 if retrieval metrics are below thresholds")

    rg = sub.add_parser("ragas", help="Ragas faithfulness / answer relevancy / context precision")
    rg.add_argument("--set", dest="eval_set", choices=("golden", "dev"), default="golden")
    rg.add_argument("--limit", type=int, default=None, help="evaluate only the first N questions")
    rg.add_argument("--gate", action="store_true", help="exit 1 if any metric is below its threshold")

    args = parser.parse_args()

    if args.command == "ingest":
        from .pipeline import run_ingest
        run_ingest()
    elif args.command == "ask":
        from .pipeline import run_ask
        run_ask(args.question, args.mode)
    elif args.command == "chat":
        from .pipeline import run_chat
        run_chat()
    elif args.command == "serve":
        import uvicorn
        uvicorn.run("turbo_rag.api:app", port=args.port)
    elif args.command == "make-questions":
        from .eval.make_questions import make_questions
        make_questions(args.n)
    elif args.command == "validate-golden":
        from .eval.golden import run_validate
        raise SystemExit(run_validate())
    elif args.command == "eval":
        from .eval.harness import run_eval
        raise SystemExit(run_eval(args.modes, args.eval_set, use_llm=not args.no_llm, gate=args.gate))
    elif args.command == "ragas":
        from .eval.ragas_eval import run_ragas
        raise SystemExit(run_ragas(args.eval_set, limit=args.limit, gate=args.gate))


if __name__ == "__main__":
    main()
