
from collections.abc import Sequence
from datetime import date, datetime
from decimal import Decimal

import psycopg2
from psycopg2 import sql
import requests
import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv
import numpy as np

from airflow.decorators import task
from airflow.models import Variable
from airflow.hooks.base import BaseHook


# Variables Shared by other files
N_LAGS = 24
DATE_COLUMNS = {
    "four_year_installment":"observation_date",
    "effective_rate":"observation_date"
    }
TABLE_NAMES = ["four_year_installment", "effective_rate"]
OLD_FREQUENCIES=["d","bw","wef","weth","wew","wetu","wem","wesu","wesa","bwew","bwem"]
NEW_FREQUENCY="m"
TARGET_FREQUENCIES = {
            "four_year_installment": "q",
            "effective_rate": "m",
        }

# -----------------------------------------
# params for 
DATE_COLUMN = "observation_date"
EFFECTIVE_TABLE = "effective_rate"
FOUR_YEAR_TABLE = "four_year_installment"
# epsilon is added to features for use in ridge regression
EPSILON = 0.0000000001
# -----------------------------------------

# TRANSFORM FEATURE DATA AND CAPTURE AS CSV
def get_lag_chng_pctchng(data:pd.DataFrame=None,
                         colname:str=None,
                         i:int=None):
    data[f"lag_{i}_{colname}"] = data[colname].shift(periods=i)
    data[f"chng_lag_{i}_{colname}"] = data[colname]-data[f"lag_{i}_{colname}"]
    data[f"pct_chng_lag_{i}_{colname}"] = data[f"chng_lag_{i}_{colname}"]/data[colname]
    return data


def read_table(conn, 
               table:str,
               DATE_COLUMN:str):
    with conn.cursor() as cur:
        cur.execute(f'SELECT * FROM "{table}" ORDER BY "{DATE_COLUMN}"')
        columns = [desc[0] for desc in cur.description]
        frame = pd.DataFrame(cur.fetchall(), columns=columns)
    frame[DATE_COLUMN] = pd.to_datetime(frame[DATE_COLUMN])
    value_columns = [c for c in frame.columns if c != DATE_COLUMN]
    for c in value_columns:
        frame[c] = pd.to_numeric(frame[c], errors="coerce")
    return frame

def connect_to_postgreSQL_to_retrieve_fed_rates(
        DB_KEY:str,
        USER_KEY:str,
        PG_PW:str,
        PG_HOST:str,
        PG_PORT:str,
        DATE_COLUMN:str,
        EFFECTIVE_TABLE:str,
        FOUR_YEAR_TABLE:str,
        EPSILON:float
) -> pd.DataFrame:
    """
    """

    try:   
            # Find the project .env by walking up from the notebook's folder.
        env_path = next(
            (p / ".env" for p in [Path.cwd(), *Path.cwd().parents] if (p / ".env").exists()),
            None,
        )
        if env_path is None:
            raise FileNotFoundError("No .env file found in this folder or any parent.")
        load_dotenv(env_path) 

        conn = psycopg2.connect(
            dbname=os.environ[DB_KEY],
            user=os.environ[USER_KEY],
            password=os.environ[PG_PW],
            host=os.environ[PG_HOST],
            port=int(os.environ[PG_PORT]),
        )
    except:
        raise ValueError("PostgreSQL connection failed. Please ensure the database and .env exist, and that all keys are correct.")
    
    try:
        effective_rate = read_table(conn, 
                                    EFFECTIVE_TABLE,
                                    DATE_COLUMN)
        four_year_installment = read_table(conn, 
                                           FOUR_YEAR_TABLE,
                                           DATE_COLUMN)
    finally:
        conn.close()

    imported_features = (effective_rate[[DATE_COLUMN, EFFECTIVE_TABLE]]
                         .merge(four_year_installment[[DATE_COLUMN, FOUR_YEAR_TABLE]],
                                on=DATE_COLUMN, 
                                how="left"))
    imported_features = imported_features[[DATE_COLUMN, EFFECTIVE_TABLE, FOUR_YEAR_TABLE]].reset_index(drop=True)
    imported_features[DATE_COLUMN] = pd.to_datetime(imported_features[DATE_COLUMN],yearfirst=True,errors='raise')
    imported_features = imported_features.sort_values(DATE_COLUMN)

    for column in imported_features.columns:
        if column != DATE_COLUMN:
            imported_features[column] = imported_features[column].ffill()


    # 
    imported_features = imported_features.dropna(axis=0)

    # feature_engineered_rate_metrics (ferm)
    ferm = imported_features.copy() 
    ferm.index = ferm[DATE_COLUMN]
    ferm = ferm.drop(columns=DATE_COLUMN)  

    for colname in [FOUR_YEAR_TABLE,EFFECTIVE_TABLE]:
        for i in range(1, N_LAGS +1):
            if ferm.isna().sum().sum()>0:
                raise ValueError('There are NaN in the Data. Cannot proceed.')
            ferm = get_lag_chng_pctchng(ferm,
                            colname=colname,
                            i=i)
            ferm = ferm.fillna(0)

    features = ferm.copy()

    # cyclical encoding of month, so December (12) sits next to January (1)
    month = features.index.month
    features['month_sin'] = np.sin(2 * np.pi * month / 12)
    features['month_cos'] = np.cos(2 * np.pi * month / 12)
    day = features.index.dayofyear
    features['day_sin']   = np.sin(2 * np.pi * day / 365.25)
    for col in features.columns:
        for pow in [2,3,4]:
            features[f"power{pow}_{col}"] = features[col]**pow

    # add epsilon
    features += EPSILON

    return features




def retrieve_fed_rates_save_to_csv(
        DB_KEY:str,
        USER_KEY:str,
        PG_PW:str,
        PG_HOST:str,
        PG_PORT:str,
        DATE_COLUMN:str,
        EFFECTIVE_TABLE:str,
        FOUR_YEAR_TABLE:str,
        EPSILON:float,
        NEW_FILE_DESTINATION_DIR:str
): 
    """
    """
    df = connect_to_postgreSQL_to_retrieve_fed_rates(
                                                DB_KEY = DB_KEY,
                                                USER_KEY = USER_KEY,
                                                PG_PW = PG_PW,
                                                PG_HOST = PG_HOST,
                                                PG_PORT = PG_PORT,
                                                DATE_COLUMN = DATE_COLUMN,
                                                EFFECTIVE_TABLE = EFFECTIVE_TABLE,
                                                FOUR_YEAR_TABLE = FOUR_YEAR_TABLE,
                                                EPSILON = EPSILON
                                                )
    df[DATE_COLUMN] = pd.to_datetime(df.index,yearfirst=True,errors='raise')
    tsmin,tsmax = str(df[DATE_COLUMN].min()) , str(df[DATE_COLUMN].max()) 
    filename = f"car_sales_prediction_features_{tsmin}_to_{tsmax}.csv"
    filepath = Path(NEW_FILE_DESTINATION_DIR) / filename

    df.to_csv(filepath,index=True,encoding='utf-8')



def first_day_of_month(value: date) -> date:
    return date(value.year, value.month, 1)


################################################
# manage api params
def make_params(
    series_id: str,
    observation_start: str,
    file_type: str = "json",
    api_key: str | None = None,
    observation_end: str | None = None,
    units: str | None = "lin",
    aggregation_method: str | None = None,
    frequency: str | None = None,
) -> dict[str, str | None]:
    """
    FRED data is requested as JSON.
    units can be:
        'lin' = Levels (No transformation)
        'chg' = Change
        'ch1' = Change from Year Ago
        'pch' = Percent Change
        'pc1' = Percent Change from Year Ago
        'pca' = Compounded Annual Rate of Change
        'cch' = Continuously Compounded Rate of Change
        'cca' = Continuously Compounded Annual Rate of Change
        'log' = Natural Log
    observation_start
        Required YYYY-MM-DD formatted string.
    observation_end
        YYYY-MM-DD formatted string, optional, default: 9999-12-31 (latest available)
    aggregation_method
        This parameter has no affect if the frequency parameter is not set
        One of  'avg', 'sum', 'eop' (end of period)
        default: avg
    frequency
        string, optional, default: no value for no frequency aggregation
        One of the following values: 'd', 'w', 'bw', 'm', 'q', 'sa', 'a', 'wef', 'weth', 'wew', 'wetu', 'wem', 'wesu', 'wesa', 'bwew', 'bwem'
        Frequencies without period descriptions:
        d = Daily
        w = Weekly
        bw = Biweekly
        m = Monthly
        q = Quarterly
        sa = Semiannual
        a = Annual
        Frequencies with period descriptions:
        wef = Weekly, Ending Friday
        weth = Weekly, Ending Thursday
        wew = Weekly, Ending Wednesday
        wetu = Weekly, Ending Tuesday
        wem = Weekly, Ending Monday
        wesu = Weekly, Ending Sunday
        wesa = Weekly, Ending Saturday
        bwew = Biweekly, Ending Wednesday
        bwem = Biweekly, Ending Monday
        For most cases, it will be sufficient to specify a lower frequency without a period description (e.g. 'd', 'w', 'bw', 'm', 'q', 'sa', 'a') as opposed to frequencies with period descriptions (e.g. 'wef', 'weth', 'wew', 'wetu', 'wem', 'wesu', 'wesa', 'bwew', 'bwem') which only exist for the weekly and biweekly frequencies.
        The weekly and biweekly frequencies with periods exist to offer more options and override the default periods implied by values 'w' and 'bw'.
        The value 'w' defaults to frequency and period 'Weekly, Ending Friday' when aggregating daily series.
        The value 'bw' defaults to frequency and period 'Biweekly, Ending Wednesday' when aggregating daily and weekly series.
        Consider the difference between values 'w' for 'Weekly' and 'wef' for 'Weekly, Ending Friday'. When aggregating observations from daily to weekly, the value 'w' defaults to frequency and period 'Weekly, Ending Friday' which is the same as 'wef'. Here, the difference is that the period 'Ending Friday' is implicit for value 'w' but explicit for value 'wef'. However, if a series has native frequency 'Weekly, Ending Monday', an error will be returned for value 'wef' but not value 'w'.
        Note that frequency aggregation is currently only available for JSON or XML.
        Note that an error will be returned if a frequency is specified that is higher than the native frequency of the series.
    """

    params = {
    "series_id": series_id,
    "api_key": api_key,
    "file_type": file_type,
    "observation_start": observation_start,
    "units": units,
    }

    if observation_end is not None:
        params.update({'observation_end':observation_end})
    if frequency is not None:
        params.update({'frequency':frequency})
    if aggregation_method is not None:
        params.update({'aggregation_method':aggregation_method})

    return params




def dicts_of_params(
    keys_list: list[str],
    ids_dict: dict[str, str],
    frequencies_dict: dict[str, str],
    api_key: str,
    observation_start: str,
    args: dict[str, str | None] | None = None,
) -> dict[str, dict[str, str | None]]:
    '''
    returns a dictionary of parameter dictionaries for every key in ids_dict
    where ids_dict keys point to api keys, table names in the postgress database, and value column names in the respective tables
    '''
    default_args = {
        'observation_end':None,
               'units':"lin",
               'aggregation_method':None,
               'frequency':None,
               'file_type':"json",
    }
    default_args.update(args or {})
    new_args = default_args

    params_dict = {}
    for key in keys_list:
        params = make_params(series_id=ids_dict[key],
               observation_start = observation_start,
               api_key=api_key,
               **new_args)
        # DATA VALIDATION STEP
        if key in frequencies_dict.keys():
              u_date = frequencies_dict[key]
              params.update({'frequency':u_date})
        params_dict[key] = params
    return params_dict

################################################
# fetch data over API
def make_api_call(
    url: str,
    params: dict[str, str | None],
) -> dict:
    if params.get("file_type", "").lower() != "json":
        raise ValueError("Only JSON API responses are supported.")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()
    return response.json()





################################################
# STORE/MANAGE RETRIEVED DATA

def data_to_pg_db_fed_rate_data(
    data,
    pg_db_params: dict,
    new_table_name: str,
    old_data_key: str | None = None,
    old_column_keys: list[str] | None = None,
    column_key_rename_map: dict[str, str] | None = None,
):
    """Insert records from a FRED JSON response into PostgreSQL.

    pg_db_params must contain host, port, dbname, user, and password.
    Selected fields must be 'date' and/or 'value'.
    Returns the number of source rows processed after monthly deduplication.
    """
    if old_data_key is None:
        old_data_key = "observations"

    if old_data_key not in data:
        raise ValueError(f"Data key {old_data_key!r} was not found.")

    records = data[old_data_key]

    if not isinstance(records, list):
        raise ValueError("Expected a list of observation dictionaries.")

    if old_column_keys is None:
        old_column_keys = ["date", "value"]

    column_key_rename_map = column_key_rename_map or {}

    # Types are based on original field names, before renaming.
    field_types = {
        "date": sql.SQL("DATE"),
        "value": sql.SQL("NUMERIC"),
    }

    if not old_column_keys or any(
        col not in field_types for col in old_column_keys
    ):
        raise ValueError("Select 'date' and/or 'value'.")

    if set(column_key_rename_map) - set(old_column_keys):
        raise ValueError("Rename map contains unselected columns.")

    column_names = [
        column_key_rename_map.get(col, col)
        for col in old_column_keys
    ]

    if any(not isinstance(col, str) or not col for col in column_names):
        raise ValueError("Column names must be nonempty strings.")

    if len(set(column_names)) != len(column_names):
        raise ValueError("Renaming produces duplicate column names.")

    # Normalize and deduplicate the API batch at the monthly grain before
    # sending it to PostgreSQL's unique-key upsert.
    rows_by_date = {}
    for record in records:
        values = []
        for col in old_column_keys:
            value = record[col]

            if value is None or value == ".":
                value = None
            elif col == "date":
                value = first_day_of_month(date.fromisoformat(value))
            elif col == "value":
                value = Decimal(str(value))

            values.append(value)

        if "date" in old_column_keys:
            date_index = old_column_keys.index("date")
            if values[date_index] is None:
                raise ValueError("Observation dates cannot be null.")
            rows_by_date[values[date_index]] = tuple(values)
        else:
            rows_by_date[len(rows_by_date)] = tuple(values)

    rows = list(rows_by_date.values())

    create_query = sql.SQL(
        "CREATE TABLE IF NOT EXISTS {} ({})"
    ).format(
        sql.Identifier(new_table_name),
        sql.SQL(", ").join(
            (
                sql.SQL("{} {} NOT NULL").format(
                    sql.Identifier(new_col),
                    field_types[old_col],
                )
                if old_col == "date"
                else sql.SQL("{} {}").format(
                    sql.Identifier(new_col),
                    field_types[old_col],
                )
            )
            for old_col, new_col in zip(old_column_keys, column_names)
        ),
    )

    date_column = (
        column_key_rename_map.get("date", "date")
        if "date" in old_column_keys
        else None
    )
    insert_query = sql.SQL(
        "INSERT INTO {} ({}) VALUES ({})"
    ).format(
        sql.Identifier(new_table_name),
        sql.SQL(", ").join(map(sql.Identifier, column_names)),
        sql.SQL(", ").join(
            sql.Placeholder() for _ in column_names
        ),
    )
    if date_column is not None:
        update_columns = [
            column for column in column_names if column != date_column
        ]
        conflict_action = (
            sql.SQL("DO UPDATE SET {}").format(
                sql.SQL(", ").join(
                    sql.SQL("{} = EXCLUDED.{}").format(
                        sql.Identifier(column),
                        sql.Identifier(column),
                    )
                    for column in update_columns
                )
            )
            if update_columns
            else sql.SQL("DO NOTHING")
        )
        insert_query += sql.SQL(" ON CONFLICT ({}) ").format(
            sql.Identifier(date_column)
        ) + conflict_action

    conn = psycopg2.connect(**pg_db_params)

    try:
        with conn:
            with conn.cursor() as cur:
                cur.execute(create_query)

                if date_column is not None:
                    date_identifier = sql.Identifier(date_column)
                    table_identifier = sql.Identifier(new_table_name)
                    cur.execute(
                        sql.SQL(
                            """
                            UPDATE {table}
                            SET {date_col} = date_trunc(
                                'month', {date_col}::timestamp
                            )::date
                            WHERE {date_col} IS NOT NULL
                              AND {date_col}::date <> date_trunc(
                                  'month', {date_col}::timestamp
                              )::date
                            """
                        ).format(
                            table=table_identifier,
                            date_col=date_identifier,
                        )
                    )
                    cur.execute(
                        sql.SQL(
                            "DELETE FROM {} WHERE {} IS NULL"
                        ).format(table_identifier, date_identifier)
                    )
                    cur.execute(
                        sql.SQL(
                            """
                            DELETE FROM {table} AS duplicate
                            USING {table} AS keeper
                            WHERE duplicate.{date_col} = keeper.{date_col}
                              AND duplicate.ctid < keeper.ctid
                            """
                        ).format(
                            table=table_identifier,
                            date_col=date_identifier,
                        )
                    )
                    cur.execute(
                        sql.SQL(
                            "ALTER TABLE {} ALTER COLUMN {} SET NOT NULL"
                        ).format(table_identifier, date_identifier)
                    )
                    constraint_name = (
                        f"{new_table_name}_{date_column}_month_check"[:63]
                    )
                    cur.execute(
                        sql.SQL(
                            "SELECT EXISTS ("
                            "SELECT 1 FROM pg_constraint "
                            "WHERE conrelid = to_regclass(%s) "
                            "AND conname = %s)"
                        ),
                        (
                            sql.Identifier(new_table_name).as_string(conn),
                            constraint_name,
                        ),
                    )
                    if not cur.fetchone()[0]:
                        cur.execute(
                            sql.SQL(
                                "ALTER TABLE {} ADD CONSTRAINT {} "
                                "CHECK (EXTRACT(DAY FROM {}) = 1)"
                            ).format(
                                table_identifier,
                                sql.Identifier(constraint_name),
                                date_identifier,
                            )
                        )
                    index_name = (
                        f"{new_table_name}_{date_column}_month_uidx"[:63]
                    )
                    cur.execute(
                        sql.SQL(
                            "CREATE UNIQUE INDEX IF NOT EXISTS {} "
                            "ON {} ({})"
                        ).format(
                            sql.Identifier(index_name),
                            table_identifier,
                            date_identifier,
                        )
                    )

                if rows:
                    cur.executemany(insert_query, rows)
    finally:
        conn.close()

    return len(rows)

def get_table_max_dates(
    pg_db_params: dict[str, str | int],
    table_names: list[str],
    date_cols_for_table_names: dict[str, str],
) -> dict[str, date | None]:
    """Return each table's maximum date; absent tables have a None value.

    Table names are unqualified names resolved through PostgreSQL's
    search path. Missing tables and empty/all-NULL columns return None.
    Missing columns and invalid date values raise database errors.
    """
    table_max_dates: dict[str, date | None] = {}
    conn = psycopg2.connect(**pg_db_params)

    try:
        with conn:
            with conn.cursor() as cur:
                for table_name in table_names:
                    date_col = date_cols_for_table_names[table_name]

                    # Check whether the table exists.
                    # Identifier quoting preserves case and special characters.
                    cur.execute(
                        sql.SQL("SELECT to_regclass(%s)"),
                        (sql.Identifier(table_name).as_string(conn),),
                    )

                    if cur.fetchone()[0] is None:
                        table_max_dates[table_name] = None
                        continue

                    # Cast before MAX so text dates are compared as dates.
                    query = sql.SQL(
                        "SELECT MAX({date_col}::date) FROM {table}"
                    ).format(
                        date_col=sql.Identifier(date_col),
                        table=sql.Identifier(table_name),
                    )

                    cur.execute(query)
                    table_max_date = cur.fetchone()[0]
                    if isinstance(table_max_date, datetime):
                        table_max_date = table_max_date.date()
                    table_max_dates[table_name] = table_max_date
    finally:
        conn.close()

    return table_max_dates

def subtract_months(value: date, months: int) -> date:
    total = value.year * 12 + (value.month - 1) - months
    return date(total // 12, total % 12 + 1, 1)

def get_start_date(
    database_dates: dict[str, date | None],
    input_date: str,
    max_period: int | None = None, 
) -> date:
    """Return the first-of-month date to start the API fetch from, inclusive.

    - If no table has data, return the month of input_date (the default).
    - Otherwise return the month of the oldest of the tables' latest dates,
      so the least-advanced table is caught up. input_date is ignored here.
    - If some tables are empty, step back a further max_period + 1 months
      from that month so periodic (e.g. quarterly) data is fully refetched.

    max_period is the longest source period in months (3 for quarterly,
    12 for yearly); None defaults to 3, the longest period in the current
    FRED series.
    """
    if max_period is None:
        max_period = 3  
    default_month = first_day_of_month(date.fromisoformat(input_date))
    if not database_dates or all(
        value is None for value in database_dates.values()
    ):
        return default_month

    populated_dates = [value for value in database_dates.values() if value]
    earliest_table_max = min(populated_dates)
    earliest_table_month = first_day_of_month(earliest_table_max)
    if any(
        value is None for value in database_dates.values()
    ):
        earliest_table_month = subtract_months(earliest_table_month, max_period+1)
    return earliest_table_month


def determine_api_call_start_date(
    pg_df_params: dict[str, str | int],
    default_date: str,
    table_names: list[str],
    date_cols_for_table_names: dict[str, str],
    max_period: int | None = None, # max period in months such as 3 for fiscal quarter
) -> date:
    table_max_dates = get_table_max_dates(
        pg_db_params=pg_df_params,
        table_names=table_names,
        date_cols_for_table_names=date_cols_for_table_names,
    )
    return get_start_date(
        database_dates=table_max_dates,
        input_date=default_date,
        max_period=max_period
    )

######################################
# postgress params
def make_sql_db_params(
    dbname: str,
    user: str,
    password: str,
    port: int,
    host: str,
) -> dict[str, str | int]:
    return {
        "dbname": dbname,
        "user": user,
        "password": password,
        "port": port,
        "host": host,
    }



######################################
# normalize frequencies
def update_frequencies(
    old_series_id_map: dict[str, str],
    old_freqs: Sequence[str] | str = ("d",),
    new_freq: str = "d",
) -> dict[str, str]:
    '''
    replaces frequencies in old
    with the new freq
    '''

    old_frequency_set = (
        {old_freqs} if isinstance(old_freqs, str) else set(old_freqs)
    )
    return {
        key: new_freq if frequency in old_frequency_set else frequency
        for key, frequency in old_series_id_map.items()
    }


def import_data(
    api_url: str,
    api_params: dict[str, str | None],
    postgres_params: dict[str, str | int],
    new_table_name: str,
    old_data_key: str | None = None,
    old_column_keys: list[str] | None = None,
    column_key_rename_map: dict[str, str] | None = None,
) -> int:
    data = make_api_call(api_url,api_params)
    n_rows_loaded = data_to_pg_db_fed_rate_data(data,
                                                postgres_params,
                                                new_table_name=new_table_name,
                                                old_data_key=old_data_key,
                                                old_column_keys=old_column_keys,
                                                column_key_rename_map=column_key_rename_map)
    return n_rows_loaded

def generate_column_key_rename_map(new_table_name):
    return {'value':new_table_name,'date':'observation_date'}

def import_multi_table_import_data(
    api_url: str,
    dict_of_api_params: dict[str, dict[str, str | None]],
    postgres_params: dict[str, str | int],
    table_keyed_import_data_params: dict[str, dict[str, object]] | None = None,
    auto_rename_columns: bool = False,
) -> int:
    '''
    where table_keyed_import_data_params should be such as {'table_name':{'new_table_name':table_name,
                                                                        'old_data_key':'observations',
                                                                        'old_column_keys':['date','value']},
                                                            'table2_name':{...}}
    '''
    total_rows_loaded = 0
    for table_name, api_params in dict_of_api_params.items():
        if table_keyed_import_data_params:
            import_params = dict(
                table_keyed_import_data_params[table_name]
            )
        else:
            import_params = {'new_table_name':table_name,
                             'old_data_key':None,
                             'old_column_keys':None}
        if auto_rename_columns==True:
            column_key_rename = generate_column_key_rename_map(table_name)
            new_map = {'column_key_rename_map':column_key_rename}
            import_params.update(new_map)
        n_rows_loaded = import_data(api_url,
                        api_params,
                        postgres_params,
                        **import_params)
        total_rows_loaded += n_rows_loaded
    return total_rows_loaded


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