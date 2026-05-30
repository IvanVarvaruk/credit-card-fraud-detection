import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
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

    print("Training Local Outlier Factor model...")
    clf = LocalOutlierFactor(
        n_neighbors=20,
        contamination=0.015,
        novelty=True
    )

    clf.fit(X_train)

    model_path = os.path.join(model_dir, 'lof_model.joblib')
    joblib.dump(clf, model_path)

    print(f"Model saved successfully to {model_path}")


if __name__ == '__main__':
    main()