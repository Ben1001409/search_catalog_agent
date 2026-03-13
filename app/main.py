from __future__ import annotations

from uuid import uuid4

from agent import build_agent


def run_cli() -> None:
    agent = build_agent()
    thread_id = str(uuid4())
    config = {"configurable": {"thread_id": thread_id}}

    print("Product Catalog Agent (type 'exit' to quit)")
    while True:
        user_input = input("\nYou: ").strip()
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break

        result = agent.invoke(
            {"messages": [{"role": "user", "content": user_input}]},
            config=config,
        )
        last = result["messages"][-1]
        print(f"Agent: {last.content}")


if __name__ == "__main__":
    run_cli()
