from langgraph.graph import StateGraph, END
from .state import GraphState
from .nodes import (
    generate_queries_node,
    web_search_node,
    reflect_node,
    synthesize_node,
)

def state_has_error(state: GraphState) -> bool:
    return bool(state.get("errors", []))


def should_continue_after_generate(state: GraphState) -> str:
    if state_has_error(state):
        print("--- [generate] Error detected ---")
        return "end_step"
    return "next_step"

def should_continue_after_search(state: GraphState) -> str:
    if state_has_error(state) or not state.get("documents"):
        print("--- [search] Error or empty documents ---")
        return "end_step"
    return "next_step"

def should_continue_after_reflect(state: GraphState) -> str:
    if state_has_error(state):
        print("--- [reflect] Error detected ---")
        return "end_step"
    if state["need_more"] and state["loop_count"] < state["max_iter"]:
        print("--- [reflect] Need more, loop ---")
        return "next_step"
    return "end_step"


# Create the StateGraph
workflow = StateGraph(GraphState)

# Add the nodes to the graph
workflow.add_node("generate_queries", generate_queries_node)
workflow.add_node("web_search", web_search_node)
workflow.add_node("reflect", reflect_node)
workflow.add_node("synthesize", synthesize_node)

# Set the entry point of the graph
workflow.set_entry_point("generate_queries")

# Add the edges connecting the nodes
workflow.add_conditional_edges(
    "generate_queries",
    should_continue_after_generate,
    {
        "next_step": "web_search",
        "end_step": "synthesize",
    },
)
# Add conditional edge from web_search to either reflect or synthesize based on error detection
workflow.add_conditional_edges(
    "web_search",
    should_continue_after_search,
    {
        "next_step": "reflect",
        "end_step": "synthesize",
    },
)

# Add the conditional edge from the reflect node
workflow.add_conditional_edges(
    "reflect",
    should_continue_after_reflect,
    {
        "next_step": "web_search",
        "end_step": "synthesize",
    },
)

# The synthesize node is the final step
workflow.add_edge("synthesize", END)

# Compile the graph into a runnable application
app = workflow.compile()
