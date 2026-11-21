import os
from datetime import date, datetime, time, timedelta

from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

from dag_utils import (
    determine_api_call_start_date,
    dicts_of_params,
    first_day_of_month,
    import_multi_table_import_data,
    make_sql_db_params,
    subtract_months,
    update_frequencies,
    N_LAGS,
    retrieve_fed_rates_save_to_csv,
    DATE_COLUMNS,
    TABLE_NAMES,
    OLD_FREQUENCIES,
    NEW_FREQUENCY,
    TARGET_FREQUENCIES,
    fetch_fed_rates_data,
    transform_and_save_fed_rates_data_as_features,
)

# ----------------------------------------


# WHERE N LAGS IS N FEATURE MONTHS; N_MONTHS_LAG_PREDICTIONS POINTS TO FIRST NEEDED PREDICTION
N_MONTHS_LAG_PREDICTIONS = 148
HISTORY_YEARS = (N_MONTHS_LAG_PREDICTIONS/12) + (N_LAGS/12)
# First of the month exactly HISTORY_YEARS years before the current month
SCHEDULE_START_DATE = subtract_months(
    first_day_of_month(date.today()), HISTORY_YEARS * 12
)
DEFAULT_OBSERVATION_START = SCHEDULE_START_DATE.isoformat()





# ---------------------------------------
# the DAG
default_args: dict[str, object] = {
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="fed_rates_features_for_predictions_pipeline",
    start_date=datetime.combine(SCHEDULE_START_DATE, time.min),
    schedule_interval="@monthly",
    default_args=default_args,
    catchup=False,                
    max_active_runs=1,
    is_paused_upon_creation=False
) as dag:

    docker_postgres_task = BashOperator(
        task_id="start_postgres_container",
        bash_command=(
    'cd "$HOME/Projects/dbt_projects/postgres" && '
    'docker compose up -d --force-recreate --wait db'
    ),
)
    fetch_task = PythonOperator(
        task_id="fetch_fed_rates_data",
        python_callable=fetch_fed_rates_data,
    )

    dbt_task = BashOperator(
        task_id="run_dbt_models",
        bash_command=(
    'cd "$HOME/Projects/dbt_projects/my_dbt_projects/car_sales_predictor" && '
    '"$HOME/Projects/dbt_projects/.venv-dbt/bin/dbt" run --select '
    'effective_rate_fed_rates four_year_fed_rates base_data_fed_rates'
),
    )

    feature_engineering_task = PythonOperator(
        task_id="save_a_features_csv",
        python_callable=transform_and_save_fed_rates_data_as_features,
    )

    
    docker_postgres_task >> fetch_task >> dbt_task >> feature_engineering_task
