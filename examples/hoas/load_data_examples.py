import os
import pandas as pd
import pathlib
import datetime as dt

# SQL mode reads connection details from environment variables instead of
# hardcoding them here. Set these in your own environment before running
# with source="sql", e.g.:
#   export AMP_DB_USER=... AMP_DB_PASSWORD=... AMP_DB_HOST=... AMP_DB_NAME=... AMP_DB_TABLE=...
db_user = os.environ.get('AMP_DB_USER')
db_password = os.environ.get('AMP_DB_PASSWORD')
db_host = os.environ.get('AMP_DB_HOST', 'localhost')
db_name = os.environ.get('AMP_DB_NAME')
db_table = os.environ.get('AMP_DB_TABLE')


def load_data(source="csv", **kwargs):
    """
    Load data from CSV (default) or PostgreSQL.

    Args:
        source (str): "csv" or "sql".
        **kwargs: Additional arguments for SQL mode.

    Returns:
        pd.DataFrame
    """
    if source == "csv":
        filepath = pathlib.Path('data/hoas_example_data.csv')
        return pd.read_csv(filepath, index_col='timestamp', parse_dates=True)

    elif source == "sql":
        try:
            from sqlalchemy import create_engine
            from ema.data_utils.db_utils import load_data_from_postgres
        except ImportError:
            raise ImportError(
                "SQL mode requires `sqlalchemy` and `psycopg2`. Install with: "
                "`pip install sqlalchemy psycopg2-binary`"
            )

        sql_user = kwargs.get("user")
        sql_password = kwargs.get("password")
        sql_host = kwargs.get("host", "localhost")
        sql_dbname = kwargs.get("dbname")
        sql_table = kwargs.get("table")
        start_date = kwargs.get("start_date")
        end_date = kwargs.get("end_date")
        columns = kwargs.get("columns")

        connection_string = (
            f"postgresql+psycopg2://{sql_user}:{sql_password}@{sql_host}/{sql_dbname}"
        )
        engine = create_engine(connection_string)

        return load_data_from_postgres(
            engine,
            table_name=sql_table,
            index_col="timestamp",
            start_date=start_date,
            end_date=end_date,
            columns=columns
        )

    else:
        raise ValueError(f"Unknown source type: {source}")


# Default CSV loading
df = load_data()

# SQL loading (optional) - only runs if DB connection details were provided
# via environment variables (see top of file); otherwise this is skipped so
# importing/running this example doesn't fail or attempt a real connection
# with empty credentials.
if all([db_user, db_password, db_name, db_table]):
    df_sql = load_data(
        source="sql",
        user=db_user,
        password=db_password,
        host=db_host,
        dbname=db_name,
        table=db_table,
        start_date=dt.datetime(2022, 1, 1),
        end_date=dt.datetime.now(),
        columns=['ele', 'dh', 't_out'],
    )
