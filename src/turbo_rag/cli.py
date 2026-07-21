"""CLI entry point. Plumbing — provided. After `uv sync`:

    uv run rag ingest          # PDFs in data/raw -> chunks + embeddings
    uv run rag ask "..."       # one question through the pipeline
    uv run rag eval            # Stage 2: the scoreboard
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(prog="rag")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="ingest data/raw PDFs into chunks + embeddings")

    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")

    sub.add_parser("eval", help="run the eval scoreboard (Stage 2)")

    args = parser.parse_args()

    if args.command == "ingest":
        from .pipeline import run_ingest
        run_ingest()
    elif args.command == "ask":
        from .pipeline import run_ask
        print(run_ask(args.question))
    elif args.command == "eval":
        from .eval.harness import run_eval
        run_eval()


if __name__ == "__main__":
    main()
