"""
evaluate_last_run.py – compute metrics & plots for the latest (or a given) MLflow run,
then log those artifacts back into that run for visualization in run_report.ipynb.

This script generates all evaluation artifacts that are consumed by the 
run_report.ipynb notebook for comprehensive visual analysis.

Usage examples
--------------
# Evaluate the most-recent run in the default experiment name
python evaluate_last_run.py

# Specify a different experiment
python evaluate_last_run.py --experiment MyExperiment

# Evaluate a specific run ID
python evaluate_last_run.py --run-id abc123456789
"""
import argparse, json, tempfile
import numpy as np
import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
    multilabel_confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns
import os
from datetime import datetime

# ──── CLI args ─────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--experiment", default="photo-tagger-experiment",
                    help="MLflow experiment name (default: photo-tagger-experiment)")
parser.add_argument("--run-id", help="Specific MLflow run_id (optional)")
args = parser.parse_args()

# ──── Find the run to evaluate ─────────────────────────────────────────
client = MlflowClient()
exp = client.get_experiment_by_name(args.experiment) \
      or client.get_experiment_by_name("Default")
if exp is None:
    raise SystemExit(f"Experiment '{args.experiment}' not found — nothing to evaluate.")

run_id = args.run_id
if run_id is None:
    runs = client.search_runs([exp.experiment_id],
                              run_view_type=ViewType.ACTIVE_ONLY,
                              order_by=["attributes.start_time DESC"],
                              max_results=1)
    if not runs:
        raise SystemExit(f"No runs in experiment '{exp.name}'.")
    run_id = runs[0].info.run_id

print(f"Evaluating run {run_id} in experiment {exp.name}")

# ──── Get label names from run params ──────────────────────────────────
run = client.get_run(run_id)
classes_str = run.data.params.get("classes")
if not classes_str:
    raise SystemExit(f"'classes' param not found in run {run_id}.")
try:
    # Safely parse tuple string to list
    import ast
    label_names = list(ast.literal_eval(classes_str))
except Exception as e:
    raise SystemExit(f"Failed to parse 'classes' param: {e}")

# ──── Download arrays produced by run_train.py ─────────────────────────
with tempfile.TemporaryDirectory() as tmpdir:
    def safe_download(run_id, artifact_path, tmpdir):
        local_path = client.download_artifacts(run_id, artifact_path, tmpdir)
        if local_path is None:
            raise SystemExit(f"Artifact '{artifact_path}' not found in run {run_id}.")
        return local_path

    try:
        y_val_path = safe_download(run_id, "y_val.npy", tmpdir)
        y_pred_prob_path = safe_download(run_id, "y_pred_prob.npy", tmpdir)
        y_val = np.load(y_val_path)
        y_pred_prob = np.load(y_pred_prob_path)
    except Exception as e:
        raise SystemExit(f"Failed to download or load artifacts: {e}")

# ──── Compute metrics & visualisations ─────────────────────────────────
print("Computing evaluation metrics and visualizations...")
threshold = 0.5
y_pred = (y_pred_prob >= threshold).astype(int)

# Use label names in classification report
report = classification_report(
    y_val, y_pred, target_names=label_names, output_dict=True, zero_division=0
)

# Flatten for micro metrics
y_true_flat = y_val.ravel()
y_pred_flat = y_pred.ravel()
y_prob_flat = y_pred_prob.ravel()

# ROC curve and AUC for micro-average
fpr, tpr, _ = roc_curve(y_true_flat, y_prob_flat)
roc_auc = auc(fpr, tpr)

# Multilabel confusion matrix (per class)
multilabel_cm = multilabel_confusion_matrix(y_val, y_pred)

# Create results directory
results_dir = "results/evaluation"
os.makedirs(results_dir, exist_ok=True)

print(f"Logging evaluation artifacts to MLflow run: {run_id}")

# Set the run as active without creating a new nested run
mlflow.set_tracking_uri(mlflow.get_tracking_uri())

# ──── 1. Classification Report (JSON) ──────────────────────────────
report_path = os.path.join(results_dir, "classification_report.json")
with open(report_path, "w") as f:
    json.dump(report, f, indent=2)
client.log_artifact(run_id, report_path)
print("✓ Classification report logged")

# ──── 2. Combined Confusion Matrix (Overall visualization) ─────────
# Create a combined confusion matrix visualization for all classes
plt.figure(figsize=(12, 8))

# Calculate overall confusion matrix (binary classification for all labels combined)
cm_overall = confusion_matrix(y_true_flat, y_pred_flat)

# Create heatmap
sns.heatmap(
    cm_overall,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Negative", "Positive"],
    yticklabels=["Negative", "Positive"],
    cbar_kws={'label': 'Count'}
)
plt.xlabel("Predicted", fontsize=14, fontweight='bold')
plt.ylabel("Actual", fontsize=14, fontweight='bold')
plt.title("Overall Confusion Matrix (All Labels Combined)", fontsize=16, fontweight='bold')
plt.tight_layout()

confusion_matrix_path = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_path, dpi=300, bbox_inches='tight')
plt.close()
client.log_artifact(run_id, confusion_matrix_path)
print("✓ Combined confusion matrix logged")

# ──── 3. Individual Class Confusion Matrices ──────────────────────
for i, label in enumerate(label_names):
    cm_label = multilabel_cm[i]
    confusion_matrix_img_path = os.path.join(
        results_dir, f"confusion_matrix_{label}.png"
    )
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_label,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[f"Not {label}", label],
        yticklabels=[f"Not {label}", label],
        cbar_kws={'label': 'Count'}
    )
    plt.xlabel("Predicted", fontsize=14, fontweight='bold')
    plt.ylabel("Actual", fontsize=14, fontweight='bold')
    plt.title(f"Confusion Matrix: {label.title()}", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(confusion_matrix_img_path, dpi=300, bbox_inches='tight')
    plt.close()
    client.log_artifact(run_id, confusion_matrix_img_path)
print(f"✓ Individual confusion matrices for {len(label_names)} classes logged")

# ──── 4. ROC Curve Data and Plot ───────────────────────────────────
roc_curve_json_path = os.path.join(results_dir, "roc_curve.json")
with open(roc_curve_json_path, "w") as f:
    json.dump({
        "fpr": fpr.tolist(), 
        "tpr": tpr.tolist(),
        "auc": float(roc_auc),
        "threshold": threshold
    }, f, indent=2)
client.log_artifact(run_id, roc_curve_json_path)
print("✓ ROC curve data logged")

# ──── 5. Log Key Metrics ───────────────────────────────────────────
client.log_metric(run_id, "roc_auc_micro", roc_auc)

if "accuracy" in report:
    client.log_metric(run_id, "accuracy", report["accuracy"])
if "macro avg" in report:
    client.log_metric(run_id, "macro_f1", report["macro avg"]["f1-score"])
if "weighted avg" in report:
    client.log_metric(run_id, "weighted_f1", report["weighted avg"]["f1-score"])

# Log per-class F1 scores
for label in label_names:
    if label in report:
        client.log_metric(run_id, f"f1_{label}", report[label]["f1-score"])

print("✓ Key metrics logged")

# ──── 6. Evaluation Summary ────────────────────────────────────────
evaluation_summary = {
    "timestamp": datetime.now().isoformat(),
    "run_id": run_id,
    "experiment": exp.name,
    "threshold": threshold,
    "num_classes": len(label_names),
    "classes": label_names,
    "overall_metrics": {
        "accuracy": report.get("accuracy", 0),
        "macro_f1": report.get("macro avg", {}).get("f1-score", 0),
        "weighted_f1": report.get("weighted avg", {}).get("f1-score", 0),
        "roc_auc_micro": roc_auc
    },
    "artifacts_generated": [
        "classification_report.json",
        "confusion_matrix.png",
        "roc_curve.json"
    ]
}

summary_path = os.path.join(results_dir, "evaluation_summary.json")
with open(summary_path, "w") as f:
    json.dump(evaluation_summary, f, indent=2)
client.log_artifact(run_id, summary_path)
print("✓ Evaluation summary logged")

print("\n" + "="*60)
print("EVALUATION COMPLETE!")
print("="*60)
print(f"Run ID: {run_id}")
print(f"Overall Accuracy: {report.get('accuracy', 0):.1%}")
print(f"Macro F1-Score: {report.get('macro avg', {}).get('f1-score', 0):.3f}")
print(f"AUC Score: {roc_auc:.3f}")
print(f"Results saved to: {results_dir}")
print("\nRun the 'run_report.ipynb' notebook to visualize these results!")
print("="*60)
