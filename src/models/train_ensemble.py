import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
import os
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    precision_recall_curve, recall_score, precision_score, f1_score
)

SEEDS = [42, 123, 456, 789, 1024]

FEATURES = [
    # original
    'amount', 'transaction_hour', 'foreign_transaction', 'location_mismatch',
    'device_trust_score', 'velocity_last_24h', 'cardholder_age', 'merchant_category_enc',
    # engineered
    'amount_log', 'trust_deficit', 'risk_score', 'amount_velocity_ratio',
    'trust_velocity', 'is_night', 'combined_risk', 'high_velocity',
    'low_trust', 'risk_x_amount',
]


def build_features(df: pd.DataFrame, le: LabelEncoder, amount_max: float,
                   velocity_90th: float, trust_10th: float) -> pd.DataFrame:
    data = df.copy()
    data['merchant_category_enc'] = le.transform(
        data['merchant_category'].apply(
            lambda x: x if x in le.classes_ else le.classes_[0]
        )
    )
    data['amount_log']            = np.log1p(data['amount'])
    data['risk_score']            = (data['amount'] / amount_max
                                     + data['location_mismatch']
                                     + data['foreign_transaction'])
    data['amount_velocity_ratio'] = data['amount'] / (data['velocity_last_24h'] + 1)
    data['trust_deficit']         = 100 - data['device_trust_score']
    data['trust_velocity']        = (100 - data['device_trust_score']) * data['velocity_last_24h']
    data['is_night']              = (data['transaction_hour'] < 6).astype(int)
    data['combined_risk']         = data['foreign_transaction'] * data['location_mismatch']
    data['high_velocity']         = (data['velocity_last_24h'] > velocity_90th).astype(int)
    data['low_trust']             = (data['device_trust_score'] < trust_10th).astype(int)
    data['risk_x_amount']         = data['risk_score'] * data['amount_log']
    return data[FEATURES]


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    raw_data_path = os.path.join(base_dir, 'data', 'raw', 'credit_card_fraud_10k.csv')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    print("Loading data...")
    df = pd.read_csv(raw_data_path)
    y = df['is_fraud']

    # ── Fit-time statistics ──────────────────────────────────────────────────
    le = LabelEncoder()
    le.fit(df['merchant_category'])
    amount_max    = float(df['amount'].max())
    velocity_90th = float(df['velocity_last_24h'].quantile(0.9))
    trust_10th    = float(df['device_trust_score'].quantile(0.1))

    print("Engineering features...")
    X_raw = build_features(df, le, amount_max, velocity_90th, trust_10th)

    scaler = StandardScaler()
    X = scaler.fit_transform(X_raw)
    print(f"Feature matrix: {X.shape}  ({len(FEATURES)} features)")

    # ── Train ensemble ───────────────────────────────────────────────────────
    print("\nTraining ensemble (5 models)...")
    models       = []
    scores_list  = []
    score_ranges = []   # (train_max, train_min) per model for inference normalisation

    for seed in SEEDS:
        clf = IsolationForest(
            n_estimators=500,
            contamination=0.015,
            max_samples='auto',
            max_features=0.8,
            bootstrap=True,
            random_state=seed,
            n_jobs=-1,
        )
        clf.fit(X)
        models.append(clf)

        raw = clf.decision_function(X)
        score_ranges.append((float(raw.max()), float(raw.min())))

        norm = (raw.max() - raw) / (raw.max() - raw.min())
        scores_list.append(norm)

        auc = roc_auc_score(y, norm)
        print(f"  seed={seed:4d}  ROC-AUC={auc:.4f}")

    ensemble_score = np.mean(scores_list, axis=0)
    print(f"\nEnsemble ROC-AUC       : {roc_auc_score(y, ensemble_score):.4f}")
    print(f"Ensemble Avg Precision : {average_precision_score(y, ensemble_score):.4f}")

    # ── Threshold: Recall ≥ 0.99 ─────────────────────────────────────────────
    prec_curve, rec_curve, thresholds = precision_recall_curve(y, ensemble_score)
    idx_r99 = np.where(rec_curve[:-1] >= 0.99)[0]
    idx_r99 = idx_r99[np.argmax(prec_curve[idx_r99])]
    final_threshold = float(thresholds[idx_r99])

    y_pred = (ensemble_score >= final_threshold).astype(int)
    print(f"\nThreshold (Recall≥0.99) : {final_threshold:.6f}")
    print(f"Recall                  : {recall_score(y, y_pred):.4f}")
    print(f"Precision               : {precision_score(y, y_pred):.4f}")
    print(f"F1                      : {f1_score(y, y_pred):.4f}")
    print(f"FN (missed fraud)       : {((y_pred==0) & (y==1)).sum()}")
    print(f"FP (false alarms)       : {((y_pred==1) & (y==0)).sum()}")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_path = os.path.join(model_dir, 'ensemble_isolation_forest.joblib')
    joblib.dump({
        'models':        models,
        'score_ranges':  score_ranges,
        'scaler':        scaler,
        'label_encoder': le,
        'amount_max':    amount_max,
        'velocity_90th': velocity_90th,
        'trust_10th':    trust_10th,
        'threshold':     final_threshold,
        'features':      FEATURES,
    }, save_path)
    print(f"\nSaved → {save_path}")


if __name__ == '__main__':
    main()
