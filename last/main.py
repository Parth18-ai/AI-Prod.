"""Ask the research assistant.

    python main.py "What is RAG?"     # one question
    python main.py                    # chat loop (type 'quit' to stop)
    python main.py --diagram          # print the graph as a Mermaid diagram
"""
import sys

from graph import build_graph


def ask(app, question: str):
    try:
        result = app.invoke({"question": question, "trace": []})
    except Exception as e:
        print(f"\nSomething went wrong: {e}")
        print("Is Ollama running? Did you run `ollama pull llama3.2`?")
        return

    print("\n=== ANSWER ===")
    print(result.get("answer", "(no answer)"))
    if result.get("sources"):
        print("\n=== SOURCES ===")
        for s in result["sources"]:
            print(" ", s)
    print("\n=== HOW I GOT HERE ===")
    for step in result.get("trace", []):
        print(" -", step)


def main():
    app = build_graph()
    args = sys.argv[1:]

    if args and args[0] == "--diagram":
        print(app.get_graph().draw_mermaid())
    elif args:
        ask(app, " ".join(args))
    else:
        print("Research assistant ready. Type a question (or 'quit').")
        while True:
            q = input("\nYou: ").strip()
            if q.lower() in {"quit", "exit", ""}:
                break
            ask(app, q)


if __name__ == "__main__":
    main()
