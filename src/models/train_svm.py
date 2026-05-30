import pandas as pd
import joblib
from sklearn.metrics import classification_report, confusion_matrix
import os

def evaluate_model(model_path, model_name, X_test, y_test):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return

    print(f"\nLoading {model_name}...")
    model = joblib.load(model_path)
    
    predictions = model.predict(X_test)
    y_pred = [1 if x == -1 else 0 for x in predictions]

    print("\n" + "="*40)
    print(f"{model_name} Evaluation Results")
    print("="*40)
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Normal (0)', 'Fraud (1)']))

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    
    X_test_path = os.path.join(processed_data_dir, 'X_test.csv')
    y_test_path = os.path.join(processed_data_dir, 'y_test.csv')

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path)

    lof_path = os.path.join(base_dir, 'models', 'lof_model.joblib')
    if_path = os.path.join(base_dir, 'models', 'isolation_forest.joblib')
    svm_path = os.path.join(base_dir, 'models', 'svm_model.joblib')

    evaluate_model(lof_path, "Local Outlier Factor", X_test, y_test)
    evaluate_model(if_path, "Isolation Forest", X_test, y_test)
    evaluate_model(svm_path, "One-Class SVM", X_test, y_test)

if __name__ == '__main__':
    main()
