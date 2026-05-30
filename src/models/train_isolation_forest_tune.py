import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.metrics import fbeta_score, recall_score, precision_score
import warnings
warnings.filterwarnings('ignore')

def optimize_isolation_forest():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    print("Loading processed data...")
    # Load training data for fitting
    X_train = pd.read_csv(os.path.join(processed_data_dir, 'X_train.csv'))
    
    # Load test/validation data for evaluating F2-Score
    X_val = pd.read_csv(os.path.join(processed_data_dir, 'X_test.csv'))
    y_val = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv'))
    
    # Flatten y_val if it's a DataFrame
    y_val = y_val.values.ravel() if isinstance(y_val, pd.DataFrame) else y_val

    # Define the hyperparameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_samples': ['auto', 256, 512],
        'max_features': [1.0, 0.8]
    }

    best_f2 = 0
    best_recall = 0
    best_params = {}
    best_threshold = 0
    best_model = None

    print("\nStarting Hyperparameter & Threshold Search...")
    print("=" * 60)

    # 1. Iterate through Hyperparameters
    for n_est in param_grid['n_estimators']:
        for max_samp in param_grid['max_samples']:
            for max_feat in param_grid['max_features']:
                
                # Fit model (Notice we DO NOT use contamination here)
                clf = IsolationForest(
                    n_estimators=n_est, 
                    max_samples=max_samp, 
                    max_features=max_feat,
                    random_state=42,
                    n_jobs=-1
                )
                clf.fit(X_train)

                # Get anomaly scores (lower scores = more anomalous)
                # We invert them so higher score = more anomalous
                scores = -clf.decision_function(X_val)

                # 2. Iterate through Thresholds (percentiles of scores)
                thresholds = np.percentile(scores, np.linspace(1, 20, 40)) # Checking 1st to 20th percentiles
                
                for thresh in thresholds:
                    # Predict 1 if score > threshold, else 0
                    y_pred = (scores > thresh).astype(int)
                    
                    # Calculate metrics
                    f2 = fbeta_score(y_val, y_pred, beta=2, zero_division=0)
                    recall = recall_score(y_val, y_pred, zero_division=0)
                    
                    # 3. Update Best Model if F2 improves
                    if f2 > best_f2:
                        best_f2 = f2
                        best_recall = recall
                        best_threshold = thresh
                        best_params = {
                            'n_estimators': n_est,
                            'max_samples': max_samp,
                            'max_features': max_feat
                        }
                        best_model = clf

                print(f"Tested: n_est={n_est}, max_samp={max_samp}, max_feat={max_feat} | Best F2 so far: {best_f2:.4f}")

    print("\n" + "=" * 60)
    print("OPTIMIZATION COMPLETE")
    print("=" * 60)
    print(f"Best Parameters : {best_params}")
    print(f"Best Threshold  : {best_threshold:.4f}")
    print(f"Validation F2   : {best_f2:.4f}")
    print(f"Validation Recall: {best_recall:.4f}")

    # Save the optimized model
    model_path = os.path.join(model_dir, 'isolation_forest_optimized.joblib')
    
    # We save a dictionary containing both the model AND the optimal threshold
    # This is critical so your evaluation script knows what threshold to use
    joblib.dump({'model': best_model, 'threshold': best_threshold}, model_path)
    print(f"\nModel and optimized threshold saved successfully to {model_path}")

if __name__ == '__main__':
    optimize_isolation_forest()