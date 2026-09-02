from typing import Dict, Literal, Annotated, List, Optional
import pandas as pd
from pydantic import BaseModel, Field, model_validator
from state import AgentState
from langchain_groq import ChatGroq
import os


from bank import EXPERIMENT_BANK
BankKey = Literal[tuple(EXPERIMENT_BANK.keys())]

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

class HypothesisOutput(BaseModel):
    name: Literal["high_lr", "label_noise", "overfitting", "data_leak", "dist_mismatch", "other"]
    other_description: Optional[str] = None
    confidence: Annotated[float, Field(gt=0, lt=1)]
    status: Literal["active", "ruled_out", "confirmed"]
    evidence: list[str]

    @model_validator(mode="after")
    def check_other_has_description(self):
        if self.name == "other" and not self.other_description:
            raise ValueError("other_description is required when name is 'other'")
        return self

class HypothesesOutput(BaseModel):
    hypotheses: list[HypothesisOutput]

class ExperimentSelection(BaseModel):
    selected_key: BankKey
    reasoning: str

def inspect_metrics(state: AgentState) -> dict:

    filepath = state["filepath"]
    accuracy_filepath = state["accuracy_filepath"]
    target_lr = state["target_lr"]  # the "sick" run under diagnosis

    df = pd.read_csv(filepath)
    accuracy_df = pd.read_csv(accuracy_filepath)

    group = df[df["lr"] == target_lr].sort_values("epoch")
    accuracy = accuracy_df.loc[accuracy_df["lr"] == target_lr, "test_accuracy"].iloc[0]

    first = group.iloc[0]
    final = group.iloc[-1]

    summary = (
        f"LR={target_lr}: "
        f"train_loss {first['train_loss']:.3f} -> {final['train_loss']:.3f}, "
        f"val_loss {first['val_loss']:.3f} -> {final['val_loss']:.3f}, "
        f"grad_norm {first['gradient_norm']:.3f} -> {final['gradient_norm']:.3f}, "
        f"test_accuracy {accuracy:.2f}%"
    )

    return {
        "metrics_summary": summary,
        "raw_metrics": group.to_dict(orient="records")
    }

llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0
)

structured_llm = llm.with_structured_output(
    HypothesesOutput,
    method="json_schema"
)


def generate_hypotheses(state: AgentState) -> dict:
    summary = state["metrics_summary"]

    result = structured_llm.invoke(summary)

    hypotheses = []

    for h in result.hypotheses:
        hypothesis = {
            "name": h.name,
            "confidence": h.confidence,
            "status": h.status,
            "evidence": h.evidence,
            "other_description": h.other_description
        }

        hypotheses.append(hypothesis)

    print(hypotheses)

    return {
    "hypotheses": hypotheses
    }


def select_discriminating_experiment(state: AgentState) -> dict:
    active = [h for h in state["hypotheses"] if h["status"] == "active"]

    if not active:
        raise ValueError("No active hypotheses to discriminate between")

    hypotheses_text = "\n".join(
    f"- {h['name']}"
    + (f" ({h['other_description']})" if h['name'] == "other" and h.get('other_description') else "")
    + f" (confidence: {h['confidence']}): {', '.join(h['evidence'])}"
    for h in active
)

    bank_text = "\n".join(
        f"- {key}: {entry['description']}"
        for key, entry in EXPERIMENT_BANK.items()
    )

    already_run = {e["name"] for e in state["experiments_run"]}
    remaining_bank = {k: v for k, v in EXPERIMENT_BANK.items() if k not in already_run}
    bank_text = "\n".join(f"- {k}: {v['description']}" for k, v in remaining_bank.items())

    prompt = (
        f"Active hypotheses:\n{hypotheses_text}\n\n"
        f"Available experiments:\n{bank_text}\n\n"
        "Pick the ONE experiment that would best confirm, rule out, or "
        "separate between the active hypotheses above."
    )

    structured_experiment_llm = llm.with_structured_output(
    ExperimentSelection,
    method="function_calling"
    )
    result = structured_experiment_llm.invoke(prompt)

    print("Selection reasoning:", result.reasoning)

    return {"selected_experiment": result.selected_key}


def run_experiment(state: AgentState) -> dict:
    key = state["selected_experiment"]
    entry = EXPERIMENT_BANK[key]

    experiment_record = {
        "name": key,
        "result_summary": entry["summary"]
    }

    return {
        "experiments_run": state["experiments_run"] + [experiment_record]
    }


class UpdatedHypothesis(BaseModel):
    name: Literal["high_lr", "label_noise", "overfitting", "data_leak", "dist_mismatch", "other"]
    other_description: Optional[str] = None
    confidence: Annotated[float, Field(gt=0, lt=1)]
    status: Literal["active", "ruled_out", "confirmed"]
    evidence: list[str]
    evidence_relation: Literal["supports", "contradicts", "unrelated"]
    justification: str  # explicit reasoning for THIS update, forced

    @model_validator(mode="after")
    def check_other_has_description(self):
        if self.name == "other" and not self.other_description:
            raise ValueError("other_description is required when name is 'other'")
        return self


class UpdatedHypothesesOutput(BaseModel):
    updated_hypotheses: list[UpdatedHypothesis]


structured_update_llm = llm.with_structured_output(
    UpdatedHypothesesOutput,
    method="json_schema"
)


def update_hypotheses(state: AgentState) -> dict:
    hypotheses = state["hypotheses"]
    latest_experiment = state["experiments_run"][-1]

    hypotheses_text = "\n".join(
        f"- {h['name']}"
        + (f" ({h['other_description']})" if h['name'] == "other" and h.get('other_description') else "")
        + f" (current status: {h['status']}, confidence: {h['confidence']}): "
        f"{', '.join(h['evidence'])}"
        for h in hypotheses
    )

    prompt = (
        f"Current hypotheses:\n{hypotheses_text}\n\n"
        f"New experiment result — {latest_experiment['name']}:\n"
        f"{latest_experiment['result_summary']}\n\n"
        "For EVERY hypothesis above, you must explicitly decide whether this new "
        "evidence SUPPORTS it, CONTRADICTS it, or is UNRELATED to it. "
        "Do not leave a hypothesis unchanged unless you can justify in one sentence "
        "why the new evidence doesn't affect it. "
        "A hypothesis can only stay 'active' if you state a specific, unresolved "
        "question about it. If evidence contradicts a hypothesis, mark it "
        "'ruled_out' — do not keep contradicted hypotheses active out of caution. "
        "If evidence strongly and specifically supports a hypothesis with no "
        "remaining ambiguity, mark it 'confirmed'."
    )

    result = structured_update_llm.invoke(prompt)
    input_names = {h["name"] for h in hypotheses}
    output_names = {h.name for h in result.updated_hypotheses}

    if input_names != output_names:
        missing = input_names - output_names
        extra = output_names - input_names
        raise ValueError(
            f"Hypothesis name mismatch after LLM update. "
            f"Missing: {missing or 'none'}. Unexpected new names: {extra or 'none'}."
        )
    existing_by_name = {h["name"]: h for h in hypotheses}
    updated = []
    for h in result.updated_hypotheses:
        old_evidence = existing_by_name[h.name]["evidence"]
        merged_evidence = old_evidence + [e for e in h.evidence if e not in old_evidence]

        confidence = h.confidence

        if h.status == "ruled_out":
            confidence = min(confidence, 0.15)
        elif h.status == "confirmed":
            confidence = max(confidence, 0.85)

        updated.append({
            "name": h.name,
            "confidence": confidence,
            "status": h.status,
            "evidence": merged_evidence,
        })

    print("Update justifications:")
    for h in result.updated_hypotheses:
        print(f"  - {h.name}: {h.evidence_relation} -> {h.status} | {h.justification}")

    history_entry = {
    "experiment": latest_experiment["name"],
    "updates": [
        {
            "name": h.name,
            "evidence_relation": h.evidence_relation,
            "justification": h.justification,
            "status": h.status,
        }
        for h in result.updated_hypotheses
    ],
}

    return {
        "hypotheses": updated,
        "update_history": state.get("update_history", []) + [history_entry],
    }


def should_continue(state: AgentState) -> str:
    hypotheses = state["hypotheses"]
    active = [h for h in hypotheses if h["status"] == "active"]
    round_count = len(state.get("experiments_run", []))

    MAX_ROUNDS = 5

    if len(active) <= 1:
        return "end"

    if round_count >= MAX_ROUNDS:
        return "end"

    experiments_run_names = {e["name"] for e in state["experiments_run"]}
    if experiments_run_names >= set(EXPERIMENT_BANK.keys()):
        return "end"

    return "continue"