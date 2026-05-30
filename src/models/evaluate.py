import pandas as pd
import joblib
from sklearn.metrics import confusion_matrix, recall_score, fbeta_score, average_precision_score
import os


def evaluate_model(model_path, model_name, X_test, y_test):
    if not os.path.exists(model_path):
        print(f"[{model_name}] Model file not found at {model_path}")
        return None

    model = joblib.load(model_path)

    predictions = model.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in predictions]

    cm = confusion_matrix(y_test, y_pred)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
    pr_auc = average_precision_score(y_test, y_pred)

    print(f"\n{'=' * 40}")
    print(f"{model_name} Detailed Results")
    print(f"{'=' * 40}")
    print("Confusion Matrix:")
    print(cm)
    print(f"Recall:   {recall:.4f}")
    print(f"F2-Score: {f2:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")

    return {
        'Model': model_name,
        'Recall': recall,
        'F2_Score': f2,
        'PR_AUC': pr_auc
    }


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')

    X_test_path = os.path.join(processed_data_dir, 'X_test.csv')
    y_test_path = os.path.join(processed_data_dir, 'y_test.csv')

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    models_to_evaluate = [
        (os.path.join(base_dir, 'models', 'isolation_forest.joblib'), 'Isolation Forest'),
        (os.path.join(base_dir, 'models', 'lof_model.joblib'), 'Local Outlier Factor'),
        (os.path.join(base_dir, 'models', 'svm_model.joblib'), 'One-Class SVM')
    ]

    results = []
    for path, name in models_to_evaluate:
        res = evaluate_model(path, name, X_test, y_test)
        if res:
            results.append(res)

    if results:
        print("\n" + "=" * 50)
        print("FINAL SUMMARY REPORT (Sorted by Recall)")
        print("=" * 50)
        results_df = pd.DataFrame(results).sort_values(by='Recall', ascending=False)
        print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == '__main__':
    main()