# 🚀 Quick Start Guide: Multi-Label Photo Tagger Evaluation Pipeline

This guide shows you how to use the integrated evaluation pipeline consisting of `evaluate_last_run.py` and `run_report.ipynb`.

## ⚡ Quick Usage

### 1. Complete Training Run
First, ensure you have a completed training run in MLflow with the required artifacts:
```bash
# Your training should save these artifacts to MLflow:
# - y_val.npy (validation labels)
# - y_pred_prob.npy (prediction probabilities)
# - classes parameter (list of class names)
```

### 2. Generate Evaluation Artifacts
Run the evaluation script to compute metrics and create visualizations:
```bash
python scripts/evaluate_last_run.py
```

**What this does:**
- Downloads prediction data from the latest MLflow run
- Computes classification metrics (precision, recall, F1-score, accuracy)
- Generates confusion matrices and ROC curves
- Saves all results back to MLflow as artifacts

### 3. Visualize Results
Open and run the Jupyter notebook:
```bash
jupyter notebook notebooks/run_report.ipynb
```

**What you'll see:**
- 🎯 Interactive performance cards for each class
- 🔥 Beautiful confusion matrix heatmaps
- 📈 ROC curve analysis with performance interpretation
- 📊 Training progress charts (if metrics were logged)
- 🎯 Comprehensive performance dashboard
- 📤 HTML export functionality

## 🔧 Advanced Usage

### Evaluate Specific Run
```bash
python scripts/evaluate_last_run.py --run-id YOUR_RUN_ID
```

### Evaluate Different Experiment
```bash
python scripts/evaluate_last_run.py --experiment "your-experiment-name"
```

### Test Integration
```bash
python test_integration.py
```

## 📁 Generated Files

After running `evaluate_last_run.py`, these artifacts are available in MLflow:

| Artifact | Description | Used by Notebook |
|----------|-------------|------------------|
| `classification_report.json` | Per-class metrics (precision, recall, F1) | ✅ Performance cards, summary tables |
| `confusion_matrix.png` | Combined confusion matrix visualization | ✅ Confusion matrix section |
| `confusion_matrix_{class}.png` | Individual class confusion matrices | ✅ Referenced in analysis |
| `roc_curve.json` | ROC curve data and AUC score | ✅ ROC analysis section |
| `evaluation_summary.json` | Complete evaluation metadata | ✅ Integration status |

## 🎯 Key Features

### 🔗 Perfect Integration
- **Separation of Concerns**: Computation (script) vs. Visualization (notebook)
- **MLflow Integration**: All artifacts stored centrally
- **Error Handling**: Clear messages when components are missing

### 📊 Comprehensive Analysis
- **Per-Class Metrics**: Detailed breakdown for each label
- **Visual Representations**: Charts, heatmaps, and interactive elements
- **Performance Insights**: Automated recommendations
- **Export Capability**: Professional HTML reports

### 🛠️ Developer Friendly
- **Modular Design**: Run evaluation and visualization independently
- **Extensible**: Easy to add new metrics or visualizations
- **Documented**: Clear error messages and usage instructions

## 🚨 Troubleshooting

### "No evaluation artifacts found"
**Solution**: Run `python scripts/evaluate_last_run.py` first

### "Missing y_val.npy or y_pred_prob.npy"
**Cause**: Training script didn't save prediction arrays
**Solution**: Ensure your training script includes:
```python
mlflow.log_artifact("y_val.npy")
mlflow.log_artifact("y_pred_prob.npy")
```

### "Training History Not Available"
**Cause**: No loss metrics logged during training
**Solution**: Add to your training loop:
```python
mlflow.log_metric("train_loss", loss_value, step=epoch)
mlflow.log_metric("val_loss", val_loss_value, step=epoch)
```

## 📈 Sample Output

The pipeline generates professional visualizations including:

- **Performance Dashboard**: Color-coded metric cards
- **Confusion Matrices**: Heatmaps with accuracy breakdown
- **ROC Analysis**: Curves with performance rating gauge
- **Training Curves**: Loss progression and overfitting analysis
- **Recommendations**: Actionable insights for model improvement

## 🎊 Next Steps

1. **Customize Visualizations**: Modify the notebook cells to add your own analysis
2. **Add New Metrics**: Extend `evaluate_last_run.py` with additional computations
3. **Automate Pipeline**: Create scripts to run evaluation after each training
4. **Share Results**: Use the HTML export feature for team collaboration

---

**🔥 Pro Tip**: Add this workflow to your training pipeline automation for continuous model evaluation and beautiful reporting!
