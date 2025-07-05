import sys
import json
from .graph import app
from .state import GraphState

def run_agent():
    """The main entry point for the research agent CLI."""
    if len(sys.argv) < 2:
        print("Usage: docker compose run --rm agent \"<question>\"", file=sys.stderr)
        sys.exit(1)

    question = sys.argv[1]
    MAX_ITER = 2

    # Initialize the state for the graph
    initial_state = GraphState(
        question=question,
        queries=[],
        documents=[],
        need_more=False,
        final_answer="",
        citations=[],
        loop_count=0,
        max_iter=MAX_ITER,
        errors=[]
    )

    config = {}

    try:
        # Invoke the LangGraph app
        final_state = app.invoke(initial_state, config=config)

        # Print the final JSON output
        output = {
            "answer": final_state.get("final_answer"),
            "citations": final_state.get("citations", [])
        }
        print(json.dumps(output, indent=2))

    except Exception as e:
        sys.exit(1)

if __name__ == "__main__":
    run_agent()
