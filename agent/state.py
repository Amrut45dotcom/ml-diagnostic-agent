from typing import TypedDict, List, Dict, Optional


class Hypothesis(TypedDict):
    name: str
    confidence: float
    status: str
    evidence: List[str]
    other_description: Optional[str]


class Experiment(TypedDict):
    name: str
    result_summary: str


class AgentState(TypedDict):
    filepath: str
    accuracy_filepath: str
    run_key_column: str
    run_key: str  
    metrics_summary: str
    raw_metrics: List[Dict]

    hypotheses: List[Hypothesis]
    experiments_run: List[Experiment]

    selected_experiment: Optional[str]

    iteration: int
    budget: int

    conclusion: Optional[str]

    update_history: List[Dict]