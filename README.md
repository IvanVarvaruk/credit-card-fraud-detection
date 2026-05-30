# Credit Card Fraud Detection 🕵️‍♂️💳

An end-to-end Machine Learning project dedicated to detecting anomalous and fraudulent financial transactions. The primary business objective of this model is to **minimize false negatives (maximize Recall)**, ensuring that actual fraud is not missed, while maintaining a reasonable threshold for false positives.

## 📊 Project Artifacts
* **Data Analysis:** Detailed Exploratory Data Analysis (EDA) and feature correlation insights can be found in our [Google Sheet Analysis](https://docs.google.com/spreadsheets/d/197rQL9u0Sw4f5jvWMkzkU3DuL34HOdQbLRJcoGDAFqY/edit?usp=sharing).
* **Data Storage:** Raw and processed datasets are securely stored on [Google Drive](https://drive.google.com/drive/folders/1NtZ8ovEtXQoexJ5G2Yo1taZ23A49AjZK?usp=sharing) and versioned using [DVC](https://dvc.org/).

## 🛠️ Tech Stack
* **Modeling:** `scikit-learn` (Isolation Forest, Local Outlier Factor, One-Class SVM)
* **API:** `FastAPI`, `Uvicorn`, `Pydantic`
* **Data Versioning:** `DVC` (Google Drive remote)
* **Infrastructure:** `Docker`

## 📁 Project Structure
```text
fraud-detection-ml/
├── api/                    # FastAPI application
├── data/
│   ├── raw/                # Original immutable datasets (tracked via DVC)
│   └── processed/          # Cleaned data and preprocessor pipeline
├── models/                 # Serialized joblib models
├── src/                    # Source code
│   ├── data/               # Preprocessing scripts
│   └── models/             # Training, evaluation, and tuning scripts
├── .dvc/                   # DVC configuration
├── Dockerfile              # Docker image configuration
├── requirements.txt        # Python dependencies
└── README.md
```
🚀 Quick Start (Local Setup)
1. Clone & Environment
```Bash
git clone [https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git](https://github.com/YOUR_USERNAME/credit-card-fraud-detection.git)
cd credit-card-fraud-detection
python -m venv venv
source venv/Scripts/activate  # On Windows: .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
2. DVC & Data Access (Google Drive)
Since data is tracked with DVC and stored on Google Drive, you need to pull it before running any scripts. Ask the repository owner for the gdrive_client_secret and configure your local DVC:

```Bash
# Set up your local DVC secrets
dvc remote modify --local myremote gdrive_client_id YOUR_CLIENT_ID
dvc remote modify --local myremote gdrive_client_secret YOUR_CLIENT_SECRET

# Pull the datasets from Google Drive
dvc pull
```
3. Model Pipeline
If you want to retrain the models from scratch, run the pipeline in this order:

```Bash
python src/data/preprocess.py
python src/models/tune_isolation_forest.py
python src/models/evaluate.py
```
🌐 Running the REST API
Option A: Local Python
Start the FastAPI server locally:

```Bash
uvicorn src.api.main:app --reload
```
Access the interactive Swagger UI documentation at: http://127.0.0.1:8000/docs

Option B: Docker (Recommended)
Build and run the containerized application:

```Bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
The API will be available at http://localhost:8000.
````
🧪 API Usage Example
Send a POST request to /predict to evaluate a transaction:

```Bash
curl -X 'POST' \
  'http://localhost:8000/predict' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "amount": 5000,
  "transaction_hour": 0,
  "merchant_category": "string",
  "foreign_transaction": 0,
  "location_mismatch": 1,
  "device_trust_score": 0,
  "velocity_last_24h": 0,
  "cardholder_age": 0
}'
```
Expected Response:

```JSON
{
  "is_fraud": 1,
  "anomaly_score": 0.1433
}
```