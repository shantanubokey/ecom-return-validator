"""
Evaluation Metrics for Return Validation
- Per-field F1, Precision, Recall
- Overall accept_return accuracy
- Hallucination score (model contradicts its own logic)
- Consistency score
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    f1_score, precision_score, recall_score,
    accuracy_score, confusion_matrix, classification_report
)
from typing import List, Dict


FIELDS = [
    "product_match", "design_match", "color_match",
    "quantity_is_one", "is_damaged", "is_used", "accept_return"
]


def yn_to_int(val: str) -> int:
    return 1 if str(val).lower().strip() == "yes" else 0


def compute_field_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
) -> pd.DataFrame:
    """Compute F1, Precision, Recall per field."""
    rows = []
    for field in FIELDS:
        y_true = [yn_to_int(gt.get(field, "no")) for gt in ground_truths]
        y_pred = [yn_to_int(pr.get(field, "no")) for pr in predictions]

        f1   = f1_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec  = recall_score(y_true, y_pred, zero_division=0)
        acc  = accuracy_score(y_true, y_pred)

        rows.append({
            "field":     field,
            "accuracy":  round(acc,  4),
            "precision": round(prec, 4),
            "recall":    round(rec,  4),
            "f1":        round(f1,   4),
        })

    return pd.DataFrame(rows).set_index("field")


def compute_hallucination_score(predictions: List[Dict]) -> Dict:
    """
    Hallucination = model says accept_return=yes but logic says no.
    A hallucinated accept means the model bypassed its own reasoning.
    """
    total = len(predictions)
    hallucinated = 0
    logic_errors = []

    for i, pred in enumerate(predictions):
        # Recompute what accept_return SHOULD be based on other fields
        should_accept = (
            pred.get("product_match")   == "yes" and
            pred.get("design_match")    == "yes" and
            pred.get("color_match")     == "yes" and
            pred.get("quantity_is_one") == "yes" and
            pred.get("is_damaged")      == "no"  and
            pred.get("is_used")         == "no"
        )
        model_accept = pred.get("accept_return") == "yes"

        if should_accept != model_accept:
            hallucinated += 1
            logic_errors.append({
                "index":          i,
                "model_accept":   pred.get("accept_return"),
                "logical_accept": "yes" if should_accept else "no",
                "fields":         pred,
            })

    hallucination_rate = hallucinated / total if total > 0 else 0
    consistency_score  = 1 - hallucination_rate

    return {
        "total_cases":        total,
        "hallucinated":       hallucinated,
        "hallucination_rate": round(hallucination_rate, 4),
        "consistency_score":  round(consistency_score, 4),
        "logic_errors":       logic_errors,
    }


def compute_fraud_detection_metrics(
    predictions: List[Dict],
    ground_truths: List[Dict],
) -> Dict:
    """
    Fraud-specific metrics for accept_return field.
    False Negative = accepted a fraudulent return (costly!)
    False Positive = rejected a valid return (customer dissatisfaction)
    """
    y_true = [yn_to_int(gt.get("accept_return", "no")) for gt in ground_truths]
    y_pred = [yn_to_int(pr.get("accept_return", "no")) for pr in predictions]

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

    return {
        "accuracy":              round(accuracy_score(y_true, y_pred), 4),
        "f1":                    round(f1_score(y_true, y_pred, zero_division=0), 4),
        "precision":             round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":                round(recall_score(y_true, y_pred, zero_division=0), 4),
        "true_positives":        int(tp),   # correctly accepted valid returns
        "true_negatives":        int(tn),   # correctly rejected fraud
        "false_positives":       int(fp),   # rejected valid returns
        "false_negatives":       int(fn),   # accepted fraudulent returns (COSTLY)
        "fraud_slip_rate":       round(fn / (fn + tn) if (fn + tn) > 0 else 0, 4),
    }


def plot_field_metrics(df: pd.DataFrame, save_path: str = None):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(df))
    w = 0.25
    ax.bar(x - w,   df["precision"], w, label="Precision", color="#3498db", alpha=0.85)
    ax.bar(x,       df["recall"],    w, label="Recall",    color="#e74c3c", alpha=0.85)
    ax.bar(x + w,   df["f1"],        w, label="F1",        color="#27ae60", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(df.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Per-Field Metrics — Return Validation", fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    for i, (p, r, f) in enumerate(zip(df["precision"], df["recall"], df["f1"])):
        ax.text(i - w, p + 0.02, f"{p:.2f}", ha="center", fontsize=7)
        ax.text(i,     r + 0.02, f"{r:.2f}", ha="center", fontsize=7)
        ax.text(i + w, f + 0.02, f"{f:.2f}", ha="center", fontsize=7)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def plot_confusion_matrix(predictions, ground_truths, save_path=None):
    y_true = [yn_to_int(gt.get("accept_return", "no")) for gt in ground_truths]
    y_pred = [yn_to_int(pr.get("accept_return", "no")) for pr in predictions]
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Reject", "Accept"],
                yticklabels=["Reject", "Accept"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("accept_return Confusion Matrix", fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=130, bbox_inches="tight")
    plt.show()


def full_report(predictions: List[Dict], ground_truths: List[Dict]):
    print("\n" + "="*55)
    print("  RETURN VALIDATION — FULL EVALUATION REPORT")
    print("="*55)

    field_df = compute_field_metrics(predictions, ground_truths)
    print("\nPer-Field Metrics:")
    print(field_df.to_string())

    fraud = compute_fraud_detection_metrics(predictions, ground_truths)
    print(f"\nFraud Detection (accept_return):")
    for k, v in fraud.items():
        print(f"  {k:<25}: {v}")

    hall = compute_hallucination_score(predictions)
    print(f"\nHallucination Analysis:")
    print(f"  Hallucination Rate : {hall['hallucination_rate']:.2%}")
    print(f"  Consistency Score  : {hall['consistency_score']:.2%}")
    print(f"  Logic Errors       : {hall['hallucinated']} / {hall['total_cases']}")

    return {"field_metrics": field_df, "fraud_metrics": fraud, "hallucination": hall}
