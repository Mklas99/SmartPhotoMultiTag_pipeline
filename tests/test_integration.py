#!/usr/bin/env python3
"""
Test script to verify the integration between evaluate_last_run.py and run_report.ipynb
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType
import json
import os

def test_integration():
    """Test that the evaluation artifacts are properly generated and accessible"""
    
    print("🧪 Testing Multi-Label Photo Tagger Evaluation Pipeline Integration")
    print("=" * 70)
    
    # Set up MLflow
    mlflow.set_tracking_uri("file:///workspaces/MultilabelPhotoTag_Pipeline_MKlas/mlruns")
    client = MlflowClient()
    
    # Get the latest run
    exp = client.get_experiment_by_name("photo-tagger-experiment") or client.get_experiment_by_name("Default")
    if not exp:
        print("❌ No experiment found")
        return False
    
    runs = client.search_runs([exp.experiment_id], run_view_type=ViewType.ACTIVE_ONLY,
                              order_by=["attributes.start_time DESC"], max_results=1)
    if not runs:
        print("❌ No runs found")
        return False
    
    run_id = runs[0].info.run_id
    print(f"🔍 Testing latest run: {run_id}")
    
    # Check required artifacts
    required_artifacts = [
        "classification_report.json",
        "confusion_matrix.png", 
        "roc_curve.json",
        "evaluation_summary.json"
    ]
    
    all_artifacts_present = True
    
    for artifact in required_artifacts:
        try:
            artifact_path = client.download_artifacts(run_id, artifact, ".")
            if os.path.exists(artifact_path):
                print(f"✅ {artifact} - Found and downloadable")
                
                # Validate JSON content
                if artifact.endswith('.json'):
                    try:
                        with open(artifact_path, 'r') as f:
                            data = json.load(f)
                        print(f"   📄 {artifact} - Valid JSON with {len(data)} keys")
                    except json.JSONDecodeError:
                        print(f"   ❌ {artifact} - Invalid JSON format")
                        all_artifacts_present = False
                        
            else:
                print(f"❌ {artifact} - Downloaded but not found locally")
                all_artifacts_present = False
                
        except Exception as e:
            print(f"❌ {artifact} - Not found: {e}")
            all_artifacts_present = False
    
    # Test notebook requirements
    print("\n🔍 Testing notebook visualization requirements...")
    
    try:
        # Test classification report loading
        report_path = client.download_artifacts(run_id, "classification_report.json", ".")
        with open(report_path, 'r') as f:
            report = json.load(f)
        
        # Check for required metrics
        required_metrics = ['macro avg', 'weighted avg']
        for metric in required_metrics:
            if metric in report:
                print(f"✅ Classification report contains '{metric}'")
            else:
                print(f"❌ Classification report missing '{metric}'")
                all_artifacts_present = False
        
        # Test ROC curve data
        roc_path = client.download_artifacts(run_id, "roc_curve.json", ".")
        with open(roc_path, 'r') as f:
            roc_data = json.load(f)
        
        required_roc_keys = ['fpr', 'tpr', 'auc']
        for key in required_roc_keys:
            if key in roc_data:
                print(f"✅ ROC data contains '{key}'")
            else:
                print(f"❌ ROC data missing '{key}'")
                all_artifacts_present = False
    
    except Exception as e:
        print(f"❌ Error testing artifact content: {e}")
        all_artifacts_present = False
    
    # Final result
    print("\n" + "=" * 70)
    if all_artifacts_present:
        print("🎉 INTEGRATION TEST PASSED!")
        print("✅ All evaluation artifacts are properly generated")
        print("✅ run_report.ipynb should work perfectly")
        print("\n💡 Next steps:")
        print("   1. Open notebooks/run_report.ipynb")
        print("   2. Run all cells to see beautiful visualizations")
        print("   3. Export HTML report for sharing")
        return True
    else:
        print("❌ INTEGRATION TEST FAILED!")
        print("⚠️  Some artifacts are missing or invalid")
        print("💡 Try running: python scripts/evaluate_last_run.py")
        return False

if __name__ == "__main__":
    test_integration()
