import pandas as pd
from sklearn.svm import OneClassSVM
import joblib
import os

def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'models')

    os.makedirs(model_dir, exist_ok=True)

    print("Loading processed data...")
    X_train_path = os.path.join(processed_data_dir, 'X_train.csv')
    X_train = pd.read_csv(X_train_path)

    print("Training One-Class SVM model...")
    clf = OneClassSVM(kernel='rbf', gamma='scale', nu=0.015)
    clf.fit(X_train)

    model_path = os.path.join(model_dir, 'svm_model.joblib')
    joblib.dump(clf, model_path)

    print(f"Model saved successfully to {model_path}")

if __name__ == '__main__':
    main()