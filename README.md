# Data Engineering & MLOps Projects

This repository demonstrates end-to-end data engineering and machine learning operations (MLOps) with a focus on **reproducible, version-controlled workflows**. It combines programmatic SQL transformations with predictive modeling and orchestrated data pipelines.

## Project Overview

### Core Components

**1. Programmatic SQL & Data Versioning via dbt**
- Define data transformations as version-controlled SQL code using dbt
- Leverage Jinja templating for dynamic, reusable SQL logic
- Express model dependencies explicitly with `ref()`
- Track all data transformations in Git for auditability and reproducibility

**2. ML Model Versioning & Car Sales Prediction**
- End-to-end ML pipeline for predicting car sales based on Federal Reserve rates
- Model code and training notebooks tracked in Git
- Feature engineering integrated with dbt transformations
- Multiple model variations (Ridge, XGBoost) with parameter tracking

**3. Airflow DAG Orchestration**
Two distinct data pipelines serve different operational needs:

- **`fed_rates_to_features_pipeline.py`**: Scheduled ETL workflow
  - Fetches latest Federal Reserve rates via API
  - Stages current data into PostgreSQL
  - Triggers dbt transformations to generate features
  - Runs on a schedule to maintain fresh training data
  
- **`fed_rates_to_features_for_predictions_pipeline.py`**: On-demand prediction pipeline
  - Uses staged current data from the ETL pipeline
  - Applies ML models for real-time/batch predictions
  - Different timing and trigger requirements than training pipeline

### Secrets Management

This project uses **Airflow native secret management**, which is well-suited for small to mid-sized deployments:
- **Connections**: Managed via Airflow's connection registry (database credentials, API keys)
- **Variables**: Managed via Airflow's variable store (environment configs, feature parameters)
- Local development uses `.env` files (excluded from Git) with `python-dotenv`

## Architecture & Data Flow

```
Federal Reserve API
        ↓
   DAG: ETL Pipeline
        ↓
  PostgreSQL Database
        ↓
   dbt Transformations (SQL)
        ↓
  Feature Engineering Models
        ↓
   DAG: Prediction Pipeline
        ↓
  ML Model (Ridge/XGBoost)
        ↓
  Predictions & Outputs
```

## Data Model Workflow

1. **API Integration**: Fetch current Fed rates and market data via FRED API
2. **Data Staging**: Import and store raw data in PostgreSQL
3. **Feature Engineering**: dbt builds analytical features from staged data
4. **Model Training**: ML notebooks train and evaluate models
5. **Predictions**: Airflow DAG invokes trained models for predictions
6. **Monitoring**: Track model performance and data drift over time

## Project Structure

```
dbt_projects/
├── dags/                           # Airflow DAGs for orchestration
│   ├── dag_utils.py                # Shared utilities (secrets, DB, API calls)
│   ├── fed_rates_to_features_pipeline.py          # ETL scheduling
│   └── fed_rates_to_features_for_predictions_pipeline.py  # Predictions
├── my_dbt_projects/
│   └── car_sales_predictor/        # dbt project for car sales features
├── predict_car_sales_ML_modeling/  # ML notebooks and models
└── visualizations/                 # Analysis and dashboards
```

## Running the Pipelines

### Prerequisites
- Airflow environment configured with PostgreSQL connection
- Federal Reserve API key (FRED) in Airflow Variables
- dbt profile configured for your database

### Triggering Pipelines
- ETL pipeline: Runs on schedule (see DAG for cron syntax)
- Prediction pipeline: Triggered after ETL completes or on-demand

## Why Two DAGs?

The separation enables:
- **Independent scaling**: ETL can run frequently; predictions run as needed
- **Different SLAs**: Training data freshness ≠ prediction latency requirements
- **Clear ownership**: Easy to monitor, alert, and troubleshoot each workflow
- **Code reusability**: Both DAGs share common utilities (`dag_utils.py`)
