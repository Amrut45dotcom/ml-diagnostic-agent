from langgraph.graph import StateGraph, START, END

from state import AgentState
from nodes import (
    inspect_metrics,
    generate_hypotheses,
    select_discriminating_experiment,
    run_experiment,
    update_hypotheses
)

builder = StateGraph(AgentState)

builder.add_node("inspect_metrics", inspect_metrics)
builder.add_node("generate_hypotheses", generate_hypotheses)
builder.add_node("select_discriminating_experiment", select_discriminating_experiment)
builder.add_node("run_experiment", run_experiment)
builder.add_node("update_hypotheses", update_hypotheses)

builder.add_edge(START, "inspect_metrics")
builder.add_edge("inspect_metrics", "generate_hypotheses")
builder.add_edge("generate_hypotheses", "select_discriminating_experiment")
builder.add_edge("select_discriminating_experiment", "run_experiment")
builder.add_edge("run_experiment", "update_hypotheses")
builder.add_edge("update_hypotheses", END)

graph = builder.compile()