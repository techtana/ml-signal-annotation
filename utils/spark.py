"""
PySpark query utilities for pulling data from Hive/Hadoop databases.

Configure the SparkSession with your own database names and Spark
settings before use.  All public query methods return pandas DataFrames
or PySpark DataFrames depending on the method.
"""

import re
import json
import pandas as pd
import numpy as np
from datetime import datetime as dt, timedelta

from utils.io import print_section


# ---------------------------------------------------------------------------
# Environment helpers
# ---------------------------------------------------------------------------

def show_environment():
    """Print Python and PySpark version / path info to stdout."""
    import os, sys
    from pyspark import __version__, __file__ as _pyspark_file
    print(f"python      = {sys.executable}")
    print(f"working dir = {os.getcwd()}")
    print(f"pyspark     = {__version__}  ({_pyspark_file})")


def show_auth_status():
    """Print the current Kerberos ticket status (Linux/Hadoop environments)."""
    from subprocess import check_output
    print("Kerberos ticket status (refresh with `kinit` if expired):")
    print(check_output(["klist"]).decode("utf-8"))


# ---------------------------------------------------------------------------
# SparkSession wrapper
# ---------------------------------------------------------------------------

class SparkSession:
    """Managed PySpark session with convenience query helpers.

    Parameters
    ----------
    app_name : str
        Spark application name.
    measurement_db : str
        Name of the Hive database that holds measurement / wafer-level data.
    sensor_db : str
        Name of the Hive database that holds sensor / FD data.
    schema_path : str
        Path to a local ``hive_schema.json`` file that maps
        ``{db: {table: [columns]}}`` — used to validate context keys.
    config : dict, optional
        Spark configuration key-value pairs.  Defaults are provided but
        should be replaced with values appropriate for your cluster.
    start_date : str, optional
        Default earliest data date in ``YYYY-MM-DD`` format.
        Defaults to 30 days before today.
    verbose : bool
        Print query statements and progress to stdout.
    """

    DEFAULT_LOOKBACK_DAYS = 30

    def __init__(
        self,
        app_name: str,
        measurement_db: str,
        sensor_db: str,
        schema_path: str = "hive_schema.json",
        config: dict = None,
        start_date: str = None,
        verbose: bool = True,
    ):
        self.measurement_db = measurement_db
        self.sensor_db = sensor_db
        self.verbose = verbose
        self.hive = None

        # Default column name used as the "data item identifier" in wide→long pivots.
        # Override this to match your database schema.
        self.default_item_col = "data_item_id"

        if config is None:
            config = {
                "spark.executor.memory": "8g",
                "spark.executor.instances": "8",
                "spark.driver.memory": "8g",
                "spark.driver.maxResultSize": "4G",
                "spark.ui.showConsoleProgress": str(verbose).lower(),
            }

        from pyspark.sql import SparkSession as _SparkSession
        builder = _SparkSession.builder.appName(app_name)
        for k, v in config.items():
            builder = builder.config(k, v)
        self.ss = builder.getOrCreate()

        if start_date:
            self.start_date = start_date
        else:
            self.start_date = (dt.today() - timedelta(days=self.DEFAULT_LOOKBACK_DAYS)).strftime("%Y-%m-%d")

        with open(schema_path) as f:
            self.schema = json.load(f)

    def stop(self):
        """Stop the underlying Spark session."""
        self.ss.stop()

    def use(self, database: str):
        """Set the active Hive database."""
        from pyspark_llap import HiveWarehouseSession
        self.hive = HiveWarehouseSession.session(self.ss).build()
        self.hive.setDatabase(database)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _filter_valid_keys(self, db: str, table: str, keys: list) -> list:
        return [k for k in keys if k in self.schema.get(db, {}).get(table, [])]

    def _resolve_db(self, db: str) -> str:
        if db.lower() == "sensor":
            return self.sensor_db
        if db.lower() == "measurement":
            return self.measurement_db
        return db

    def _get_context_keys(self, db: str, table: str, context: dict):
        cntx = context.copy()
        start_date = cntx.pop("run_date", self.start_date)
        if db == self.sensor_db:
            date_clause = self._sensor_date_clause(start_date)
        else:
            date_clause = f"run_date >= '{start_date}'"
        keys = self._filter_valid_keys(db, table, list(cntx.keys()))
        if not keys:
            raise ValueError(f"No context keys valid for table '{table}' in db '{db}'")
        return keys, date_clause, cntx

    def _sensor_date_clause(self, start_date: str) -> str:
        """Generate a year/month/day partition clause for databases partitioned by date parts."""
        y, m, d = map(int, start_date.split("-"))
        return (
            f"((year={y} AND month={m} AND day>={d})"
            f" OR (year={y} AND month>{m})"
            f" OR (year>{y}))"
        )

    def _condition_clause(self, context_keys: list, context: dict, item_col: str) -> str:
        parts = []
        for key in context_keys:
            val = context[key]
            if isinstance(val, str):
                parts.append(f"AND {key} RLIKE '{val}'")
            elif isinstance(val, list):
                sub = " OR ".join(f"({item_col} RLIKE '{v}')" for v in val)
                parts.append(f"AND ({sub})")
        return " ".join(parts)

    # ------------------------------------------------------------------
    # Core query methods
    # ------------------------------------------------------------------

    def query(self, db: str, table: str, pk: list, context: dict, verbose=None):
        """Select rows matching `context` conditions from `table`.

        Returns a PySpark DataFrame.
        """
        if verbose is None:
            verbose = self.verbose
        db = self._resolve_db(db)
        keys, date_clause, cntx = self._get_context_keys(db, table, context)
        select_cols = list(set(pk + keys))
        cond = self._condition_clause(keys, cntx, self.default_item_col)

        sql = f"""
            SELECT {', '.join(select_cols)}
            FROM {table}
            WHERE {date_clause}
            {cond}
        """
        if verbose:
            print(f"[query] {table}\n{sql}")
        return self.hive.executeQuery(sql)

    def pivot(self, query, pk: list, pivot_col: str, value_col: str,
              dtype: str, values=None, verbose=None):
        """Pivot a long-format PySpark DataFrame to wide format.

        Parameters
        ----------
        dtype : {"string_first", "float_mean", "integer_mean", "float_sum", "integer_sum"}
        """
        if verbose is None:
            verbose = self.verbose
        if verbose:
            print(f"Pivoting on '{pivot_col}' with dtype='{dtype}' ...")

        from pyspark.sql.types import StringType, DoubleType, IntegerType
        from pyspark.sql.functions import first

        dtype_map = {
            "string_first":  (StringType(),  lambda q: q.groupBy(pk).pivot(pivot_col, values).agg(first(value_col))),
            "float_mean":    (DoubleType(),   lambda q: q.groupBy(pk).pivot(pivot_col, values).agg({value_col: "mean"})),
            "integer_mean":  (IntegerType(),  lambda q: q.groupBy(pk).pivot(pivot_col, values).agg({value_col: "mean"})),
            "float_sum":     (DoubleType(),   lambda q: q.groupBy(pk).pivot(pivot_col, values).sum(value_col)),
            "integer_sum":   (IntegerType(),  lambda q: q.groupBy(pk).pivot(pivot_col, values).sum(value_col)),
        }
        if dtype not in dtype_map:
            raise ValueError(f"Unsupported pivot dtype: {dtype!r}. Choose from {list(dtype_map)}")

        spark_type, pivot_fn = dtype_map[dtype]
        query = query.withColumn(value_col, query[value_col].cast(spark_type))
        result = pivot_fn(query)
        print("Pivot complete.")
        return result

    def query_pivot(self, db: str, table: str, pk: list, context: dict,
                    pivot_col: str = "default", value_col: str = "value",
                    dtype: str = "string_first", verbose=None):
        """Query then immediately pivot the result."""
        if pivot_col == "default":
            pivot_col = self.default_item_col
        q = self.query(db, table, pk + [pivot_col, value_col], context, verbose=verbose)
        return self.pivot(q, pk, pivot_col, value_col, dtype, verbose=verbose)

    # ------------------------------------------------------------------
    # Sensor-database helpers
    # ------------------------------------------------------------------

    def get_tool_ids(self, tool_name_pattern: str) -> list:
        """Resolve tool IDs from a name regex.

        Queries the tool registry table in the sensor database.
        Adapt the table/column names to match your schema.
        """
        self.use(self.sensor_db)
        print_section("Query: tool IDs")
        sql = f"""
            SELECT id AS tool_id, name AS tool_name
            FROM tool_registry
            WHERE name RLIKE '{tool_name_pattern}'
        """
        if self.verbose:
            print(sql)
        df = self.hive.executeQuery(sql).toPandas()
        if self.verbose:
            print(df)
        return df["tool_id"].tolist()

    def get_run_ids(self, tool_ids: list, item_ids: list = None, batch_ids: list = None,
                    workflow_step: str = None) -> pd.DataFrame:
        """Query run-context records to get run IDs for subsequent sensor queries.

        Adapt table / column names to match your schema.
        """
        self.use(self.sensor_db)
        tool_list = ", ".join(str(i) for i in tool_ids)

        item_clause  = f"AND item_id IN ('{\"','\".join(item_ids)}')" if item_ids else ""
        batch_clause = f"AND batch_id IN ('{\"','\".join(batch_ids)}')" if batch_ids else ""
        step_clause  = f"AND workflow_step RLIKE '{workflow_step}'" if workflow_step else ""

        print_section("Query: run IDs")
        sql = f"""
            SELECT run_id, tool_id, item_id, batch_id, workflow_step
            FROM run_context
            WHERE {self._sensor_date_clause(self.start_date)}
            AND tool_id IN ({tool_list})
            {item_clause}
            {batch_clause}
            {step_clause}
        """
        if self.verbose:
            print(sql)
        df = self.hive.executeQuery(sql).toPandas()
        print(f"Found {len(df)} rows across {df['run_id'].nunique()} unique runs.")
        return df

    def get_sensor_summary(self, runid_df: pd.DataFrame,
                           aggregations: list = None,
                           include_step_names: bool = True) -> pd.DataFrame:
        """Query aggregated sensor data for the given run IDs and pivot to wide format.

        Parameters
        ----------
        runid_df : DataFrame
            Output of ``get_run_ids``. Must have ``tool_id`` and ``run_id`` columns.
        aggregations : list of str
            Aggregation types to include (e.g. ``["mean", "max"]``).
        include_step_names : bool
            Whether to join step name data into the result.

        Returns
        -------
        pandas.DataFrame
            Wide-format DataFrame with one row per (run_id, step_id, step_occurrence).
        """
        self.use(self.sensor_db)
        agg_list = ", ".join(f"'{a}'" for a in (aggregations or ["mean"]))

        run_groups = runid_df.groupby("tool_id")["run_id"].apply(list).to_dict()
        run_clause = " OR ".join(
            f"(tool_id = {tid} AND run_id IN ({', '.join(str(r) for r in rids)}))"
            for tid, rids in run_groups.items()
        )

        print_section("Query: sensor summary")
        sql = f"""
            SELECT tool_id, run_id, step_id, step_occurrence, aggregation_name, sensor_name, value
            FROM sensor_summary
            WHERE {self._sensor_date_clause(self.start_date)}
            AND aggregation_name IN ({agg_list})
            AND ({run_clause})
        """
        if self.verbose:
            print(sql)
        sensor_q = self.hive.executeQuery(sql)

        # Cast types
        from pyspark.sql.types import IntegerType, StringType, DoubleType
        for col, ctype in [("tool_id", IntegerType()), ("run_id", IntegerType()),
                            ("step_occurrence", IntegerType()), ("value", DoubleType())]:
            sensor_q = sensor_q.withColumn(col, sensor_q[col].cast(ctype))

        pk_sensor = ["tool_id", "run_id", "step_id", "step_occurrence", "aggregation_name"]
        Sensor_DF = sensor_q.groupBy(pk_sensor).pivot("sensor_name").mean("value")
        Sensor_DF = Sensor_DF.dropna(subset=["tool_id", "run_id"], how="any").orderBy(["tool_id", "run_id"])
        Sensor_DF = Sensor_DF.toPandas()

        # Merge run context
        Sensor_DF = pd.merge(runid_df, Sensor_DF, how="inner", on=["tool_id", "run_id"])

        if include_step_names:
            print_section("Query: step names")
            step_sql = f"""
                SELECT tool_id, run_id, step_id, step_occurrence, value AS step_name
                FROM sensor_point
                WHERE {self._sensor_date_clause(self.start_date)}
                AND sensor_name = 'StepName'
                AND ({run_clause})
            """
            step_df = self.hive.executeQuery(step_sql).dropDuplicates().toPandas()
            if len(step_df) > 0:
                Sensor_DF = pd.merge(Sensor_DF, step_df, how="inner",
                                     on=["tool_id", "run_id", "step_id", "step_occurrence"])

        Sensor_DF = Sensor_DF.drop_duplicates()
        return Sensor_DF

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    def validate_items(self, db: str, table: str, context: dict,
                       item_col: str, items: list, verbose=None) -> tuple:
        """Verify that `items` exist in `table` under `context`.

        Returns
        -------
        matched_tables : dict
            ``{item_name: table_name}``
        available_items : dict
            ``{table_name: [available_item_names]}``
        status : str
            ``""`` on success or ``"failed lookup"`` if any item is missing.
        """
        if verbose is None:
            verbose = self.verbose
        status = ""
        available = {}
        for table_name in [table] if isinstance(table, str) else table:
            q = self.query_list_items(db, table_name, context, item_col, verbose=False)
            available[table_name] = list(q.toPandas()[item_col])

        matched = {}
        for item in items:
            found = False
            for table_name, avail in available.items():
                if any(item in w or re.match(item, w) for w in avail):
                    matched[item] = table_name
                    found = True
                    break
            if not found:
                print(f"Item not found: '{item}'")
                print("Available items:", sorted(sum(available.values(), [])))
                status = "failed lookup"

        return matched, available, status

    def query_list_items(self, db: str, table: str, context, item_col: str = "default", verbose=None):
        """Return distinct values of `item_col` matching `context`."""
        if verbose is None:
            verbose = self.verbose
        if item_col == "default":
            item_col = self.default_item_col
        db = self._resolve_db(db)

        if isinstance(context, dict):
            keys, date_clause, cntx = self._get_context_keys(db, table, context)
            cond = self._condition_clause(keys, cntx, item_col)
            sql = f"""
                SELECT {item_col}
                FROM {table}
                WHERE {date_clause}
                {cond}
                GROUP BY {item_col}
            """
        else:
            sql = f"SELECT {item_col} FROM {table} WHERE {context} GROUP BY {item_col}"

        if verbose:
            print(f"[list_items] {sql}")
        return self.hive.executeQuery(sql)
