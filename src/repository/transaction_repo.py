from typing import Optional
import pandas as pd
from sqlalchemy import text, inspect
from src.repository.common import get_session

SQL_TO_PANDAS_TYPES = {
    "integer": "Int64",
    "bigint": "Int64",
    "smallint": "Int64",
    "numeric": "float64",
    "double precision": "float64",
    "real": "float32",
    "text": "string",
    "character varying": "string",
    "varchar": "string",
    "boolean": "bool",
    "date": "datetime64[ns]",
    "timestamp without time zone": "datetime64[ns]",
    "timestamp with time zone": "datetime64[ns]",
    "time without time zone": "string",
    "uuid": "string",
}

# Get all the rows from the table labelled_transaction and infer column types
def get_multiple_rows(limit: Optional[int] = 100000) -> pd.DataFrame:
    """
    Get multiple rows from the labelled_transaction table and infer column types
    """
    with next(get_session()) as session:
        # 1. Get the data
        if limit:
            result = session.execute(text(f"SELECT * FROM labelled_transaction limit {limit}"))
        else:
            result = session.execute(text("SELECT * FROM labelled_transaction"))
        rows = result.fetchall()
        columns = result.keys()
        df = pd.DataFrame(rows, columns=columns)

        # 2. Get the data types
        inspector = inspect(session.bind)
        column_info = inspector.get_columns("labelled_transaction")
        column_types_sql = {col["name"]: str(col["type"]).lower() for col in column_info}
        column_types = {col["name"]: str(col["type"]) for col in column_info}

        # 3. Add the column types to the DataFrame as an attribute
        df.attrs["sql_column_types"] = column_types

        # 4. Convert the SQL types to pandas types
        pandas_dtypes = {}
        for col, sql_type in column_types_sql.items():
            for sql_base, pandas_type in SQL_TO_PANDAS_TYPES.items():
                if sql_base in sql_type:  # match partiel (ex: character varying(255))
                    pandas_dtypes[col] = pandas_type
                    break
        
        df = df.astype(pandas_dtypes)

        return df
