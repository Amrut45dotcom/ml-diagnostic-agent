from graph import graph


state = {
    "filepath": "data2/training_results.csv",
    "accuracy_filepath": "data2/accuracy_results.csv",
    "target_lr": 0.05,
    "metrics_summary": "",
    "raw_metrics": [],
    "hypotheses": [],
    "selected_experiment": None,
    "experiments_run": [],
    "iteration": 0,
    "budget": 5,
    "conclusion": None
}

result = graph.invoke(state)
print("Selected experiment:", result["selected_experiment"])
print("Experiments run:", result["experiments_run"])
print("Final hypotheses:", result["hypotheses"])

# print(result)