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



# -----------------------------------------------
# create the database
@task
def fetch_fed_rates_data() -> int:
    imported_params = BaseHook.get_connection("postgres_params")
    postgres_params = make_sql_db_params(
        dbname=imported_params["POSTGRES_DB"],
        user=imported_params["POSTGRES_USER"],
        password=imported_params["POSTGRES_PASSWORD"],
        port=int(imported_params["POSTGRES_PORT"]),
        host=imported_params["POSTGRES_HOST"],
    )
    SERIES_API_IDS = Variable.get("series_api_ids")
    FREQUENCIES = update_frequencies(
        TARGET_FREQUENCIES,
        old_freqs=OLD_FREQUENCIES,
        new_freq=NEW_FREQUENCY,
    )
    observation_start = determine_api_call_start_date(   # default date is used only when no table has data; otherwise start follows the existing data
        pg_df_params=postgres_params,
        default_date=DEFAULT_OBSERVATION_START,
        table_names=TABLE_NAMES,
        date_cols_for_table_names=DATE_COLUMNS,
        max_period=3, # enter number of months such as 3 for fiscal quarters in api data. 
    )
    api_params = dicts_of_params(
        keys_list=list(SERIES_API_IDS),
        ids_dict=SERIES_API_IDS,
        frequencies_dict=FREQUENCIES,
        FRED_API_KEY=Variable.get("fred_api_key"),
        observation_start=observation_start.isoformat(),
    )
    return import_multi_table_import_data(
        fred_api_url=Variable.get("fred_url"),
        dict_of_api_params=api_params,
        postgres_params=postgres_params,
        auto_rename_columns=True,
    )


# ---------------------------------------
# save the file as .csv
DATE_COLUMN = "observation_date"
EFFECTIVE_TABLE = "effective_rate"
FOUR_YEAR_TABLE = "four_year_installment"
# epsilon is added to features for use in ridge regression
EPSILON = 0.0000000001

@task
def transform_and_save_fed_rates_data_as_features():
    imported_params = BaseHook.get_connection("postgres_params")
    NEW_FILE_DESTINATION_DIR = "../predict_car_sales_ML_modeling"
    retrieve_fed_rates_save_to_csv(
                            DB_KEY = imported_params["POSTGRES_DB"],
                            USER_KEY = imported_params["POSTGRES_USER"],
                            PG_PW = imported_params["POSTGRES_PASSWORD"],
                            PG_HOST = imported_params["POSTGRES_HOST"],
                            PG_PORT = int(imported_params["POSTGRES_PORT"]),
                            DATE_COLUMN = DATE_COLUMN,
                            EFFECTIVE_TABLE = EFFECTIVE_TABLE,
                            FOUR_YEAR_TABLE = FOUR_YEAR_TABLE,
                            EPSILON = EPSILON,
                            NEW_FILE_DESTINATION_DIR = NEW_FILE_DESTINATION_DIR
)
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
