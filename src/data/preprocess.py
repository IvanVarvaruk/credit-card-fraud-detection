import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
import os


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'credit_card_fraud_10k.csv')
    processed_data_dir = os.path.join(base_dir, 'data', 'processed')

    os.makedirs(processed_data_dir, exist_ok=True)

    df = pd.read_csv(raw_data_path)

    X = df.drop(columns=['transaction_id', 'is_fraud'])
    y = df['is_fraud']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    numeric_features = ['amount', 'transaction_hour', 'device_trust_score', 'velocity_last_24h', 'cardholder_age']
    categorical_features = ['merchant_category']
    passthrough_features = ['foreign_transaction', 'location_mismatch']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', RobustScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features),
            ('pass', 'passthrough', passthrough_features)
        ]
    )

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = (
            numeric_features +
            preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features).tolist() +
            passthrough_features
    )

    X_train_df = pd.DataFrame(X_train_processed, columns=feature_names)
    X_test_df = pd.DataFrame(X_test_processed, columns=feature_names)

    X_train_df.to_csv(os.path.join(processed_data_dir, 'X_train.csv'), index=False)
    X_test_df.to_csv(os.path.join(processed_data_dir, 'X_test.csv'), index=False)
    y_train.to_csv(os.path.join(processed_data_dir, 'y_train.csv'), index=False)
    y_test.to_csv(os.path.join(processed_data_dir, 'y_test.csv'), index=False)

    joblib.dump(preprocessor, os.path.join(processed_data_dir, 'preprocessor.joblib'))

    print("Preprocessing complete.")
    print(f"Training data: {X_train_df.shape[0]} rows, {X_train_df.shape[1]} features.")
    print(f"Testing data: {X_test_df.shape[0]} rows, {X_test_df.shape[1]} features.")


if __name__ == '__main__':
    main()