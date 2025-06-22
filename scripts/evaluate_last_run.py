"""
evaluate_latest_run.py – compute metrics & plots for the latest (or a given) MLflow run,
then log those artifacts back into that run.

Usage examples
--------------
# Evaluate the most-recent run in the default experiment name
python evaluate_latest_run.py

# Specify a different experiment
python evaluate_latest_run.py --experiment MyExperiment

# Evaluate a specific run ID
python evaluate_latest_run.py --run-id abc123456789
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
)
import matplotlib.pyplot as plt
import seaborn as sns

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
threshold = 0.5
y_pred = (y_pred_prob >= threshold).astype(int)

report = classification_report(y_val, y_pred, output_dict=True)

# Flatten for micro metrics
y_true_flat = y_val.ravel()
y_pred_flat = y_pred.ravel()

cm        = confusion_matrix(y_true_flat, y_pred_flat)
fpr, tpr, _ = roc_curve(y_true_flat, y_pred_flat)
roc_auc   = auc(fpr, tpr)

# ──── Log artifacts back into the same run ─────────────────────────────
with mlflow.start_run(run_id=run_id):
    # 1 · JSON classification report
    with open("classification_report.json", "w") as f:
        json.dump(report, f, indent=2)
    mlflow.log_artifact("classification_report.json")

    # 2 · Confusion-matrix heat-map
    plt.figure(figsize=(4,4))
    sns.heatmap(cm, annot=True, fmt="d", cbar=False, cmap="Blues")
    plt.xlabel("Predicted"); plt.ylabel("Actual"); plt.title("Confusion Matrix")
    plt.tight_layout();  plt.savefig("confusion_matrix.png"); plt.close()
    mlflow.log_artifact("confusion_matrix.png")

    # 3 · ROC data (and AUC metric)
    with open("roc_curve.json", "w") as f:
        json.dump({"fpr": fpr.tolist(), "tpr": tpr.tolist()}, f)
    mlflow.log_artifact("roc_curve.json")
    mlflow.log_metric("roc_auc_micro", roc_auc)

    print("Evaluation artifacts logged successfully.")
