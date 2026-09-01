from typing import Dict, Literal, Annotated, List
import pandas as pd
from pydantic import BaseModel, Field
from state import AgentState
from langchain_groq import ChatGroq
import os


from bank import EXPERIMENT_BANK
BankKey = Literal[tuple(EXPERIMENT_BANK.keys())]

from dotenv import load_dotenv
load_dotenv()
api_key = os.getenv("GROQ_API_KEY")

class HypothesisOutput(BaseModel):
    name: str
    confidence: Annotated[float, Field(gt=0, lt=1)]
    status: Literal["active", "ruled_out", "confirmed"]
    evidence: list[str]

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
            "evidence": h.evidence
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
        f"- {h['name']} (confidence: {h['confidence']}): {', '.join(h['evidence'])}"
        for h in active
    )

    bank_text = "\n".join(
        f"- {key}: {entry['description']}"
        for key, entry in EXPERIMENT_BANK.items()
    )

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
    name: str
    confidence: Annotated[float, Field(gt=0, lt=1)]
    status: Literal["active", "ruled_out", "confirmed"]
    evidence: list[str]
    evidence_relation: Literal["supports", "contradicts", "unrelated"]
    justification: str  # explicit reasoning for THIS update, forced

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
        f"- {h['name']} (current status: {h['status']}, confidence: {h['confidence']}): "
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

    updated = []
    for h in result.updated_hypotheses:
        updated.append({
            "name": h.name,
            "confidence": h.confidence,
            "status": h.status,
            "evidence": h.evidence,
        })

    print("Update justifications:")
    for h in result.updated_hypotheses:
        print(f"  - {h.name}: {h.evidence_relation} -> {h.status} | {h.justification}")

    return {"hypotheses": updated}