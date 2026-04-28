"""
ML Regression Pipeline

Reads a JSON config file listing one or more modeling jobs. For each job:
  1. Validates that the specified data items exist in the databases
  2. Queries and merges data from multiple sources
  3. Engineers features and pivots sensor data into a wide feature matrix
  4. Trains a PLS regression model (optionally split by a grouping column)
  5. Exports models, metrics, and diagnostic plots
  6. Writes results back into the config JSON for resumable runs

Usage
-----
    python -m modeling.pipeline [config.json] [--verbose] [--reset] [--reset-model]

Flags
-----
--verbose       Print detailed progress.
--reset         Re-query all data and retrain from scratch.
--reset-model   Retrain using cached data without re-querying.
"""

import os
import re
import sys
import json
import math
import pickle
import pathlib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pprint import pprint
from datetime import datetime as dt, date
from contextlib import redirect_stdout
from functools import partial, reduce
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score

# Local imports — adjust sys.path so the project root is on the path
_project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import utils.spark as utq
import utils.preprocess as utprep
from utils.io import update_json, print_section


# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

RUN_PARAM_PATTERN = "^run_param::"
DEFAULT_CONFIG_PATH = "config.json"
RESET_DAYS = 14
MISSING_THRESHOLD = 0.5   # drop columns with > this fraction missing
STDEV_THRESHOLD   = 0.05  # drop columns with stddev < this value


# ---------------------------------------------------------------------------
# CLI argument parsing
# ---------------------------------------------------------------------------

reset_override = False
redo_model     = False
verbose        = False
config_path    = DEFAULT_CONFIG_PATH

args = [a for a in sys.argv[1:] if not a.startswith("--")]
opts = [a for a in sys.argv[1:] if a.startswith("--")]

if args:
    config_path = args[0]
    if not re.match(r"(.*).json$", config_path):
        raise ValueError(f"Invalid config path: {config_path!r} — expected a .json file")

for opt in opts:
    if opt == "--verbose":
        verbose = True
    elif opt == "--reset":
        reset_override = True
    elif opt == "--reset-model":
        redo_model = True


class _Colors:
    BLUE  = "\033[94m"
    GREEN = "\033[92m"
    WARN  = "\033[93m"
    FAIL  = "\033[91m"
    END   = "\033[0m"
    BOLD  = "\033[1m"


print(f"config_path : {_Colors.BLUE}{config_path}{_Colors.END}")
print(f"args        : {_Colors.BLUE}{args}{_Colors.END}")
print(f"opts        : {_Colors.BLUE}{opts}{_Colors.END}")


# ---------------------------------------------------------------------------
# Spark / database setup
# ---------------------------------------------------------------------------

# Instantiate the SparkSession wrapper.
# Set measurement_db and sensor_db to your actual database names.
spark = utq.SparkSession(
    app_name="ml-regression-pipeline",
    measurement_db=os.environ.get("MEASUREMENT_DB", "measurement_database"),
    sensor_db=os.environ.get("SENSOR_DB", "sensor_database"),
    schema_path=os.path.join(_project_root, "hive_schema.json"),
    verbose=verbose,
)


# ---------------------------------------------------------------------------
# Load config
# ---------------------------------------------------------------------------

try:
    with open(config_path, "r") as f:
        job_list = json.load(f)
except json.JSONDecodeError as e:
    raise json.JSONDecodeError(
        f"Error reading {config_path}. Check that the file is valid JSON.\n{e.msg}",
        e.doc, e.pos
    )

# Set to True to use manual_job_list instead of the JSON file.
manual_job_list = None
if manual_job_list is not None:
    job_list = manual_job_list
    json_update = False
else:
    json_update = True

# Cache for validated item lookups shared across jobs
_validation_cache = {}


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

for ndx, job in enumerate(job_list):

    # ------------------------------------------------------------------
    # Status and parameter extraction
    # ------------------------------------------------------------------

    status = ""

    raw_status = job.get("status", "")
    job_status = raw_status.lower() if isinstance(raw_status, str) else "failed"

    # Re-query if the last data extract is older than RESET_DAYS
    extract_date = job.get("extract_date", "")
    if extract_date:
        if not re.match(r"^\d{4}-\d{2}-\d{2}$", extract_date):
            raise ValueError(f"extract_date format error: {extract_date!r} — expected YYYY-MM-DD")
        delta = date.today() - date(*map(int, extract_date.split("-")))
        if delta.days > RESET_DAYS:
            job_status = "reset"
    else:
        if json_update:
            update_json(config_path, ndx, "extract_date", date.today().strftime("%Y-%m-%d"))

    # Read all job parameters
    start_date        = job["start_date"]
    source_type       = job["source_type"]        # position strategy: "chamber", "slot", etc.
    position_type     = job["position_type"]       # "chamber" or "slot"
    source_tool       = job["source_tool"]         # tool name / regex
    source_name       = job["source_name"]         # recipe / process name regex
    source_step       = job["source_step"]         # workflow step identifier regex
    sensor_aggregation= job["sensor_aggregation"]  # list: ["mean", "max", ...]
    modelgroup        = job["modelgroup"]           # column to split models; "" = one model
    run_params_enable = bool(job["run_params_enable"])
    position_column   = job["position_column"]     # column name that holds position info
    target_items      = job["target_items"]        # list of {source_step, name, limits}
    feature_items     = job["feature_items"]       # list of additional X column names
    target_tables     = job["target_tables"]       # {item_name: table_name} — filled by validation
    feature_tables    = job["feature_tables"]
    data_conversion   = job["data_conversion"]

    # Build canonical target item names: "{source_step}__{name}"
    target_item_names = [f"{item['source_step']}__{item['name']}" for item in target_items]

    # Normalise sensor_aggregation to a SQL IN-clause string
    if isinstance(sensor_aggregation, str):
        sensor_agg_sql = f"'{sensor_aggregation}'"
    elif isinstance(sensor_aggregation, list):
        sensor_agg_sql = ", ".join(f"'{a}'" for a in sensor_aggregation)
    else:
        raise ValueError(f"sensor_aggregation must be str or list; got {type(sensor_aggregation)}")

    # Cached parquet paths (populated after first successful extract)
    TargetDF_path    = job.get("TargetDF", "")
    FeatureDF_path   = job.get("FeatureDF", "")
    RunParamDF_path  = job.get("RunParamDF", "")
    SensorDF_path    = job.get("SensorDF", "")
    MergedDF_path    = job.get("MergedDF", "")
    ProcessedDF_path = job.get("ProcessedDF", "")
    PivotedDF_path   = job.get("PivotedDF", "")

    # Decide which cached files can be reused
    _target_missing  = not all(item["name"] in target_tables for item in target_items)
    _feature_missing = not all(item in feature_tables for item in feature_items)

    if reset_override:
        job_status = "reset"
        print("** Override: job_status = RESET")

    load_pivoted   = job_status != "reset" and bool(PivotedDF_path)
    load_processed = job_status != "reset" and bool(ProcessedDF_path)
    load_merged    = job_status != "reset" and bool(MergedDF_path)
    load_target    = job_status != "reset" and bool(TargetDF_path)  and not _target_missing
    load_feature   = job_status != "reset" and bool(FeatureDF_path) and not _feature_missing
    load_runparam  = job_status != "reset" and bool(RunParamDF_path)
    load_sensor    = job_status != "reset" and bool(SensorDF_path)

    # Apply status-driven cache invalidations
    if job_status == "reset":
        if json_update:
            update_json(config_path, ndx, "status", job_status)
            for key in ["TargetDF", "FeatureDF", "RunParamDF", "SensorDF", "PreMergedDF",
                        "MergedDF", "ProcessedDF", "PivotedDF", "FilteredDF",
                        "ModelFile", "N", "R2", "RMSE", "Y-Range", "error", "plots"]:
                update_json(config_path, ndx, key, "")

    elif redo_model or job_status in ["reset data_processing"] or (
            any([load_merged, load_processed, load_pivoted]) and
            not all([load_target, load_feature, load_runparam, load_sensor])
    ):
        job_status = "reset data_processing"
        load_merged = load_processed = load_pivoted = False
        if json_update:
            for key in ["MergedDF", "ProcessedDF", "PivotedDF", "FilteredDF",
                        "ModelFile", "R2", "RMSE", "Y-Range", "error", "plots"]:
                update_json(config_path, ndx, key, "")
            update_json(config_path, ndx, "status", job_status)

    elif job_status == "reset model":
        if json_update:
            for key in ["ModelFile", "R2", "RMSE", "Y-Range", "error", "plots"]:
                update_json(config_path, ndx, key, "")
            update_json(config_path, ndx, "status", job_status)

    # ------------------------------------------------------------------
    # Skip jobs that are already done or broken
    # ------------------------------------------------------------------

    if re.match(r"^success|^skip|^failed(.+)", job_status):
        print_section(f"Skipping job #{ndx} — status: {job_status}")
        if job_status == "failed lookup":
            print("Resolve by reviewing target_items or feature_items in the config.")
        elif job_status == "failed zero entry":
            print("Datasets were queried but returned zero rows. Check start_date and filters.")
        elif job_status == "failed invalid sensor data":
            print("Neither 'step_name' nor 'step_id' found in ProcessedDF. Check sensor data collection.")
        else:
            print(f"Status: {job_status}")
        continue

    # ------------------------------------------------------------------
    # Helper functions (scoped to this job iteration)
    # ------------------------------------------------------------------

    def load_cached(df_name: str, df_path: str):
        """Load a cached parquet file; return (True, df) or (False, None)."""
        try:
            print_section(f"LOAD {df_name}")
            print(f"path = {df_path}")
            df = pd.read_parquet(df_path, engine="pyarrow")
            df.info()
            return True, df
        except Exception:
            if json_update:
                update_json(config_path, ndx, df_name, "")
            return False, None

    def context_summary():
        print(
            f"\nJob #{ndx}",
            f"  status          = {job_status}",
            f"  source_name     = {source_name}",
            f"  source_step     = {source_step}",
            f"  source_tool     = {source_tool}",
            f"  run_params      = {run_params_enable}",
            f"  load_target     = {load_target}",
            f"  load_feature    = {load_feature}",
            f"  load_runparam   = {load_runparam}",
            f"  load_sensor     = {load_sensor}",
            f"  load_merged     = {load_merged}",
            f"  load_processed  = {load_processed}",
            f"  load_pivoted    = {load_pivoted}",
            sep="\n"
        )
        if _target_missing:
            print("  [!] target_tables missing some target_items — will validate")
        if _feature_missing:
            print("  [!] feature_tables missing some feature_items — will validate")
        sys.stdout.flush()

    def zero_check(df: pd.DataFrame, name: str = "DataFrame"):
        """Return ("failed zero entry", error) if df is empty, else ("", None)."""
        if len(df) == 0:
            if json_update:
                update_json(config_path, ndx, "status", "failed zero entry")
            return "failed zero entry", ValueError(f"{name} has zero rows. Check setup or data source.")
        print(f"{name} passed zero-entry check ({len(df)} rows)")
        return "", None

    # ------------------------------------------------------------------
    # Reporting setup
    # ------------------------------------------------------------------

    date_str   = dt.today().strftime("%Y-%m-%d-%H%M%S")
    start_time = dt.today()
    response_name = str(target_item_names[0])

    output_dir = os.path.join(
        "output",
        f"{source_name}__{source_step}__{source_tool}"
    )
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)

    report_path = os.path.join(output_dir, f"pipeline_report_{response_name}_{date_str}.txt")
    print_section(f"Job #{ndx}: writing report to {report_path}")
    context_summary()

    if json_update:
        update_json(config_path, ndx, "last_report", report_path)
        update_json(config_path, ndx, "report_date", date_str)

    # ------------------------------------------------------------------
    # All query/model work inside the report file redirect
    # ------------------------------------------------------------------

    with open(report_path, "w") as report_file:
        with redirect_stdout(report_file):

            try:
                print(f"ML Regression Pipeline Report — {date_str}\n")
                utq.show_environment()
                context_summary()
                print_section("Data Query Starts")

                # ------------------------------------------------------
                # 0. Validate item names
                # ------------------------------------------------------
                print_section("[0] Validate Inputs")

                if _target_missing:
                    print_section("0.1 Validating target items")
                    spark.use(spark.measurement_db)
                    tables_to_search = ["measurement_wafer_summary", "measurement_summary"]

                    for ti in target_items:
                        step = ti["source_step"]
                        if step not in _validation_cache:
                            names = [i["name"] for i in target_items if i["source_step"] == step]
                            context = {"workflow_step": step}
                            matched, avail, vstatus = spark.validate_items(
                                "measurement", tables_to_search, context,
                                "data_item_id", names, verbose=True
                            )
                            target_tables.update(matched)
                            _validation_cache[step] = avail
                        else:
                            for i in target_items:
                                if i["source_step"] == step:
                                    for tbl, avail_items in _validation_cache[step].items():
                                        if i["name"] in avail_items:
                                            target_tables[i["name"]] = tbl
                                            break

                    if vstatus:
                        raise ValueError("Target item(s) not found in measurement tables")
                    if json_update:
                        update_json(config_path, ndx, "target_tables", target_tables)

                if _feature_missing:
                    print_section("0.2 Validating feature items")
                    spark.use(spark.measurement_db)
                    tables_to_search = ["measurement_wafer_summary", "measurement_summary", "measurement_run_summary"]
                    context = {"workflow_step": source_step, "tool_id": source_tool}
                    matched, avail, vstatus = spark.validate_items(
                        "measurement", tables_to_search, context,
                        "data_item_id", feature_items, verbose=True
                    )
                    feature_tables.update(matched)
                    if vstatus:
                        raise ValueError("Feature item(s) not found in measurement tables")
                    if json_update:
                        update_json(config_path, ndx, "feature_tables", feature_tables)

                # Separate run-level from wafer-level feature items
                run_level_features  = [k for k, v in feature_tables.items() if v == "measurement_run_summary"]
                wafer_level_features = [k for k, v in feature_tables.items() if v != "measurement_run_summary"]
                feature_tables_wafer = {k: v for k, v in feature_tables.items() if v != "measurement_run_summary"}

                # ------------------------------------------------------
                # 1. Target (output measurement) data
                # ------------------------------------------------------
                if load_target:
                    load_target, Target_DF = load_cached("Target_DF", TargetDF_path)

                if not load_target:
                    print_section("[1] Query Target Data")
                    spark.use(spark.measurement_db)

                    pk = ["run_date", "batch_id", "workflow_step", "item_id", "data_item_id", "value"]
                    queries = []
                    for ti in target_items:
                        name = ti["name"]
                        step = ti["source_step"]
                        tbl  = target_tables[name]
                        sql  = f"""
                            SELECT workflow_step, batch_id, run_date, item_id, data_item_id, value
                            FROM {tbl}
                            WHERE run_date >= '{start_date}'
                            AND workflow_step RLIKE '{step}'
                            AND data_item_id = '{name}'
                        """
                        print(sql)
                        queries.append(spark.hive.executeQuery(sql))

                    # Pivot each query, rename columns to include step prefix, then join
                    PK_target = ["batch_id", "item_id"]
                    pivoted = []
                    for i, q in enumerate(queries):
                        step = target_items[i]["source_step"]
                        pq   = spark.pivot(q, PK_target, "data_item_id", "value", "float_mean").toPandas()
                        extra_cols = list(set(pq.columns) - set(PK_target))
                        pq.rename(columns={c: f"{step}__{c}" for c in extra_cols}, inplace=True)
                        pivoted.append(pq)

                    merge_fn  = partial(pd.merge, on=PK_target, how="outer")
                    Target_DF = reduce(merge_fn, pivoted).drop_duplicates()
                    st, err   = zero_check(Target_DF, "Target_DF")
                    if st:
                        raise err

                    path = os.path.join(output_dir, f"pipeline_data_TargetDF_{date_str}.parquet")
                    Target_DF.to_parquet(path, engine="pyarrow")
                    if json_update:
                        update_json(config_path, ndx, "TargetDF", path)
                        update_json(config_path, ndx, "extract_date", date.today().strftime("%Y-%m-%d"))

                # ------------------------------------------------------
                # 2. Feature (wafer-level context) data
                # ------------------------------------------------------
                if load_feature:
                    load_feature, Feature_DF = load_cached("Feature_DF", FeatureDF_path)

                if not load_feature:
                    print_section("[2] Query Feature Data")
                    spark.use(spark.measurement_db)

                    if run_params_enable and position_column and position_column not in wafer_level_features:
                        wafer_level_features.append(position_column)

                    pk = ["run_date", "batch_id", "workflow_step", "item_id", "data_item_id", "value"]
                    query_parts = []
                    for item in wafer_level_features:
                        tbl = feature_tables_wafer[item]
                        sql = f"""
                            SELECT {', '.join(pk)}
                            FROM {tbl}
                            WHERE run_date >= '{start_date}'
                            AND workflow_step RLIKE '{source_step}'
                            AND data_item_id = '{item}'
                        """
                        print(sql)
                        query_parts.append(spark.hive.executeQuery(sql))

                    if len(query_parts) > 1:
                        from pyspark.sql import DataFrame
                        query_union = reduce(DataFrame.unionAll, query_parts)
                    elif len(query_parts) == 1:
                        query_union = query_parts[0]
                    else:
                        raise ValueError("Zero feature queries generated")

                    PK_feature = ["run_date", "batch_id", "item_id", "workflow_step"]
                    Feature_DF = spark.pivot(query_union, PK_feature, "data_item_id", "value", "string_first").toPandas()
                    Feature_DF.rename(columns={"workflow_step": "source_step", "run_date": "run_date"}, inplace=True)
                    Feature_DF = Feature_DF.drop_duplicates()
                    st, err = zero_check(Feature_DF, "Feature_DF")
                    if st:
                        raise err

                    path = os.path.join(output_dir, f"pipeline_data_FeatureDF_{date_str}.parquet")
                    Feature_DF.to_parquet(path, engine="pyarrow")
                    if json_update:
                        update_json(config_path, ndx, "FeatureDF", path)
                        update_json(config_path, ndx, "extract_date", date.today().strftime("%Y-%m-%d"))

                # ------------------------------------------------------
                # 3. Run-parameter data (recipe parameters + run context)
                # ------------------------------------------------------
                if load_runparam:
                    load_runparam, RunParam_DF = load_cached("RunParam_DF", RunParamDF_path)

                if not load_runparam:
                    print_section("[3] Query Run-Parameter Data")
                    spark.use(spark.measurement_db)

                    # 3.1 Recipe name + run-level items
                    PK_run = ["tool_id", "run_batch_id", "run_id_internal"]
                    items_to_query = ["RECIPE_NAME"] + run_level_features
                    context = {
                        "run_date":   start_date,
                        "tool_id":    source_tool,
                        "data_item_id": "|".join(items_to_query),
                    }
                    RunData_q = spark.query_pivot(
                        "measurement", "measurement_run_summary",
                        PK_run, context,
                        pivot_col="data_item_id", value_col="value",
                        dtype="string_first"
                    )
                    RunData_q = RunData_q.filter(RunData_q["RECIPE_NAME"].rlike(source_name))

                    if run_params_enable:
                        # 3.2 Optionally join recipe parameters
                        print_section("3.2 Query Run Parameters")
                        run_param_context = {
                            "run_date":     start_date,
                            "tool_id":      source_tool,
                            "data_item_id": RUN_PARAM_PATTERN,
                        }
                        RunParam_q = spark.query_pivot(
                            "measurement", "measurement_run_summary",
                            PK_run, run_param_context,
                            pivot_col="data_item_id", value_col="value",
                            dtype="float_mean"
                        )
                        cond = ["run_id_internal", "run_batch_id", "tool_id"]
                        RunData_q = RunData_q.alias("run").join(RunParam_q.alias("params"), cond, how="left")

                    # 3.3 Join with item-level context to get batch_id and item_id
                    print_section("3.3 Join item context")
                    PK_items = ["run_id_internal", "run_batch_id", "batch_id", "item_id"]
                    item_context = {"run_date": start_date, "workflow_step": source_step}
                    ItemCtx_q = spark.query(
                        "measurement", "measurement_items",
                        PK_items, item_context
                    )
                    join_keys  = ["run_id_internal", "run_batch_id"]
                    RunParam_DF = (
                        ItemCtx_q.alias("items")
                        .join(RunData_q.alias("run"), join_keys, how="inner")
                        .select("items.batch_id", "items.item_id", "run.*")
                        .toPandas()
                        .dropna(axis=1, how="all")
                        .drop_duplicates()
                    )
                    st, err = zero_check(RunParam_DF, "RunParam_DF")
                    if st:
                        raise err

                    path = os.path.join(output_dir, f"pipeline_data_RunParamDF_{date_str}.parquet")
                    RunParam_DF.to_parquet(path, engine="pyarrow")
                    if json_update:
                        update_json(config_path, ndx, "RunParamDF", path)
                        update_json(config_path, ndx, "extract_date", date.today().strftime("%Y-%m-%d"))

                # ------------------------------------------------------
                # 4. Sensor summary data
                # ------------------------------------------------------
                if load_sensor:
                    load_sensor, Sensor_DF = load_cached("Sensor_DF", SensorDF_path)

                if not load_sensor:
                    print_section("[4] Query Sensor Data")
                    spark.use(spark.sensor_db)

                    item_ids  = Target_DF["item_id"].unique().tolist()
                    batch_ids = Target_DF["batch_id"].unique().tolist()
                    print(f"Unique items: {len(item_ids)} | Unique batches: {len(batch_ids)}")

                    tool_ids = spark.get_tool_ids(source_tool)
                    print(f"Tool IDs: {tool_ids}")

                    runid_df = spark.get_run_ids(
                        tool_ids,
                        item_ids=item_ids,
                        workflow_step=source_step if source_step != ".*" else None,
                    )

                    Sensor_DF = spark.get_sensor_summary(
                        runid_df,
                        aggregations=sensor_aggregation if isinstance(sensor_aggregation, list) else [sensor_aggregation],
                        include_step_names=True,
                    )
                    Sensor_DF = Sensor_DF.drop_duplicates()
                    st, err = zero_check(Sensor_DF, "Sensor_DF")
                    if st:
                        raise err

                    path = os.path.join(output_dir, f"pipeline_data_SensorDF_{date_str}.parquet")
                    Sensor_DF.to_parquet(path, engine="pyarrow")
                    if json_update:
                        update_json(config_path, ndx, "SensorDF", path)
                        update_json(config_path, ndx, "extract_date", date.today().strftime("%Y-%m-%d"))

                # ------------------------------------------------------
                # 5. Merge all DataFrames
                # ------------------------------------------------------
                if load_merged:
                    load_merged, Merged_DF = load_cached("Merged_DF", MergedDF_path)

                if not load_merged:
                    print_section("[5] Merging DataFrames")

                    for df_name, df_obj in [("Target_DF", Target_DF),
                                            ("Feature_DF", Feature_DF),
                                            ("RunParam_DF", RunParam_DF)]:
                        st, err = zero_check(df_obj, df_name)
                        if st:
                            raise err

                    Sigma_DF   = pd.merge(Target_DF, RunParam_DF, how="inner", on=["batch_id", "item_id"])
                    Sigma_DF   = pd.merge(Sigma_DF, Feature_DF, how="left", on=["batch_id", "item_id"])
                    Sigma_DF   = Sigma_DF.drop_duplicates()
                    st, err    = zero_check(Sigma_DF, "Sigma_DF after target+runparam+feature join")
                    if st:
                        raise err

                    path = os.path.join(output_dir, f"pipeline_data_PreMergedDF_{date_str}.parquet")
                    Sigma_DF.to_parquet(path, engine="pyarrow")
                    if json_update:
                        update_json(config_path, ndx, "PreMergedDF", path)

                    Merged_DF = pd.merge(Sigma_DF, Sensor_DF, how="inner", on=["batch_id", "item_id"])
                    Merged_DF = Merged_DF.drop_duplicates()
                    st, err   = zero_check(Merged_DF, "Merged_DF after sensor join")
                    if st:
                        raise err

                    path = os.path.join(output_dir, f"pipeline_data_MergedDF_{date_str}.parquet")
                    Merged_DF.to_parquet(path, engine="pyarrow")
                    if json_update:
                        update_json(config_path, ndx, "MergedDF", path)

                # ------------------------------------------------------
                # 6. Feature engineering
                # ------------------------------------------------------
                run_param_origincols = utprep.get_run_param_cols(Merged_DF, RUN_PARAM_PATTERN) if run_params_enable else []
                run_param_shortnames = utprep.get_run_param_shortnames(run_param_origincols) if run_params_enable else []

                if load_processed:
                    load_processed, Processed_DF = load_cached("Processed_DF", ProcessedDF_path)

                if not load_processed:
                    print_section("[6] Feature Engineering")
                    Processed_DF = Merged_DF.copy()

                    if run_params_enable:
                        # 6.1 Match run-parameter columns to the correct position
                        print_section("6.1 Run-parameter position matching")
                        _skip_matching = False
                        method = position_type.lower() if position_type else "chamber"

                        try:
                            matched_values = [
                                utprep.get_run_param_by_position(
                                    row[position_column], row, run_param_shortnames, method=method
                                )
                                for _, row in Processed_DF.iterrows()
                            ]
                        except LookupError as e:
                            _skip_matching = True
                            run_param_shortnames = run_param_origincols
                            print(f"Run-param position matching failed: {e}")
                            print("Continuing with original (unmatched) column names.")

                        if not _skip_matching:
                            matched_df = pd.DataFrame(matched_values, columns=run_param_shortnames)
                            Processed_DF = pd.concat([Processed_DF, matched_df], axis=1)
                            Processed_DF = Processed_DF.drop(columns=run_param_origincols)

                        # 6.2 Data type conversions for run-parameter columns
                        if isinstance(data_conversion, dict):
                            print_section("6.2 Run-parameter data conversion")
                            for pattern, conv_type in data_conversion.items():
                                matching = [c for c in Processed_DF.columns if re.match(pattern, c)]
                                if matching:
                                    for col in matching:
                                        Processed_DF[col] = Processed_DF[col].apply(
                                            utprep.convert_value, conv_type=conv_type
                                        )
                                    print(f"Converted {matching} → {conv_type}")
                                else:
                                    print(f"No columns matched pattern: {pattern!r}")

                    # 6.3 Ensure step_name column is present
                    if any(re.match(r"^step_name$", c) for c in Processed_DF.columns):
                        print("`step_name` found — will use for sensor step pivot")
                    elif "step_id" in Processed_DF.columns:
                        Processed_DF["step_name"] = Processed_DF["step_id"].apply(lambda x: f"StepID{x}")
                    else:
                        update_json(config_path, ndx, "status", "failed invalid sensor data")
                        raise ValueError("Neither `step_name` nor `step_id` in Processed_DF. Check sensor data.")

                    # Update sensor_steps list in config
                    sensor_steps = Processed_DF["step_name"].unique().tolist()
                    if json_update:
                        update_json(config_path, ndx, "SensorSteps", sensor_steps)

                    path = os.path.join(output_dir, f"pipeline_data_ProcessedDF_{date_str}.parquet")
                    Processed_DF.to_parquet(path, engine="pyarrow")
                    if json_update:
                        update_json(config_path, ndx, "ProcessedDF", path)

                # ------------------------------------------------------
                # 7. Data transformation (pivot sensor steps)
                # ------------------------------------------------------
                print_section("[7] Data Transformation")

                if "step_name" in Processed_DF.columns:
                    sensor_steps = Processed_DF["step_name"].unique().tolist()
                    config_steps = job.get("SensorSteps")
                    if config_steps and isinstance(config_steps, list) and set(config_steps).issubset(sensor_steps):
                        sensor_steps = config_steps
                    elif json_update:
                        update_json(config_path, ndx, "SensorSteps", sensor_steps)
                else:
                    update_json(config_path, ndx, "status", "reset model")
                    raise ValueError("`step_name` not in Processed_DF. Reset required.")

                Processed_DF = Processed_DF[Processed_DF["step_name"].isin(sensor_steps)]

                context_cols = ["run_date", "batch_id", "item_id", "source_step",
                                "tool_id", "RECIPE_NAME", "run_id", "workflow_step"]
                drop_cols    = ["step_id", "step_occurrence", "workflow_step_raw"]
                run_param_cols = [c for c in Processed_DF.columns if re.match(RUN_PARAM_PATTERN, c)]
                feature_cols_in_df = [c for c in feature_items if c in Processed_DF.columns]

                PK_final = (context_cols + target_item_names + feature_cols_in_df + run_param_cols)
                Processed_DF = Processed_DF.drop(columns=drop_cols, errors="ignore")

                Pivoted_DF = Processed_DF.pivot_table(
                    index=PK_final,
                    columns=["step_name", "aggregation_name"],
                    aggfunc=np.mean
                )
                # Flatten multi-level column: (step_name, agg, sensor) → "STEP.SENSOR.Agg"
                Pivoted_DF.columns = [
                    ".".join([str(c[1]), str(c[0]), str(c[2]).capitalize()])
                    for c in Pivoted_DF.columns
                ]
                Pivoted_DF = Pivoted_DF.reset_index()

                st, err = zero_check(Pivoted_DF, "Pivoted_DF")
                if st:
                    raise err

                path = os.path.join(output_dir, f"pipeline_data_PivotedDF_{date_str}.parquet")
                Pivoted_DF.to_parquet(path, engine="pyarrow")
                if json_update:
                    update_json(config_path, ndx, "PivotedDF", path)

                # ------------------------------------------------------
                # 8. Data filtering
                # ------------------------------------------------------
                print_section("[8] Data Filtering")

                inspect_cols = list(set(Pivoted_DF.columns) - set(context_cols + target_item_names))
                inspect_df   = Pivoted_DF[inspect_cols]

                # Drop high-missing columns
                inspect_df, dropped_cols = utprep.dropna_column_percent(inspect_df, MISSING_THRESHOLD)
                print(f"Columns after dropping high-missing: {inspect_df.shape[1]}")

                # Drop rows with any remaining missing values
                inspect_df = inspect_df.dropna()
                print(f"Rows after dropping NaN rows: {len(inspect_df)}")

                # Drop low-variance columns
                stdev = inspect_df.std(axis=0, skipna=True, ddof=0)
                retained = stdev[stdev > STDEV_THRESHOLD].index.tolist()
                print(f"Columns after low-variance filter (threshold={STDEV_THRESHOLD}): {len(retained)}")
                if not retained:
                    raise ValueError("No sensor columns retained after filtering — relax STDEV_THRESHOLD.")

                # Apply target variable limits
                Filtered_DF = Pivoted_DF[context_cols + target_item_names + retained]
                for ti in target_items:
                    col = f"{ti['source_step']}__{ti['name']}"
                    limits = sorted(ti.get("limits", []))
                    if len(limits) == 2:
                        Filtered_DF = Filtered_DF[
                            (Filtered_DF[col] > limits[0]) & (Filtered_DF[col] < limits[1])
                        ]
                        print(f"Applied limits {limits} to {col} → {len(Filtered_DF)} rows remain")

                path = os.path.join(output_dir, f"pipeline_data_FilteredDF_{date_str}.parquet")
                Filtered_DF.to_parquet(path, engine="pyarrow")
                if json_update:
                    update_json(config_path, ndx, "FilteredDF", path)

                # ------------------------------------------------------
                # 9. Model training (PLS Regression)
                # ------------------------------------------------------
                print_section("[9] Model Training — PLS Regression")

                dataset = Filtered_DF
                Y_cols  = target_item_names[:1]
                X_cols  = retained + target_item_names[1:]

                print(f"Targets (Y): {Y_cols}")
                print(f"Features (X): {len(X_cols)} columns")
                print(f"Samples: {len(dataset)}")

                if len(Y_cols) != 1:
                    raise ValueError("This pipeline supports exactly one target variable.")
                if len(X_cols) < 2:
                    raise ValueError("Fewer than 2 feature columns after filtering — relax filter thresholds.")

                try:
                    if modelgroup and modelgroup in dataset.columns:
                        print(f"Model groups defined by: {modelgroup}")
                        data_groups = list(dataset[X_cols + Y_cols + [modelgroup]].groupby(modelgroup))
                    else:
                        raise AttributeError
                except AttributeError:
                    print("No model grouping defined — training one model.")
                    data_groups = [("All", dataset[X_cols + Y_cols])]

                models         = {}
                r2_scores      = []
                rmse_scores    = []
                n_samples_list = []

                for group_key, gdata in data_groups:
                    gdata = gdata.dropna(axis=0)
                    if len(gdata) == 0:
                        print(f"Group {group_key!r}: skipped (zero rows after dropna)")
                        continue

                    n_comp = min(len(gdata), len(X_cols), 5)
                    reg    = PLSRegression(n_components=n_comp).fit(gdata[X_cols], gdata[Y_cols])
                    preds  = reg.predict(gdata[X_cols])

                    r2   = r2_score(gdata[Y_cols], preds)
                    rmse = math.sqrt(mean_squared_error(gdata[Y_cols], preds))

                    r2_scores.append(r2)
                    rmse_scores.append(rmse)
                    n_samples_list.append(len(gdata))
                    models[group_key] = reg

                    print(f"Group {group_key!r}: n={len(gdata)} | components={n_comp} | R²={r2:.4f} | RMSE={rmse:.4f}")

                # Persist model
                model_path = os.path.join(output_dir, f"pipeline_model_{response_name}_{date_str}.pkl")
                with open(model_path, "wb") as f:
                    pickle.dump(models, f)
                print(f"Model saved to {model_path}")
                if json_update:
                    update_json(config_path, ndx, "ModelFile", model_path)
                    update_json(config_path, ndx, "N",    n_samples_list)
                    update_json(config_path, ndx, "R2",   r2_scores)
                    update_json(config_path, ndx, "RMSE", rmse_scores)
                    last_group_data = gdata
                    update_json(config_path, ndx, "Y-Range",
                                last_group_data[Y_cols].agg(["min", "max"]).T.values.tolist())

                # 9.1 Print regression equation
                print_section("9.1 Regression Equations")
                for group_key, _ in data_groups:
                    if group_key not in models:
                        continue
                    coef = models[group_key].coef_
                    print(f"Group {group_key!r}:")
                    print(utprep.generate_linear_equation(X_cols, coef))

                # 9.2 Actual vs predicted plots
                plot_paths = []
                for order, (group_key, gdata) in enumerate(data_groups):
                    if group_key not in models:
                        continue
                    gdata = gdata.dropna(axis=0)
                    preds = models[group_key].predict(gdata[X_cols])

                    fig, ax = plt.subplots(figsize=(8, 8))
                    ax.scatter(preds, gdata[Y_cols].values)
                    ax.set_title(
                        f"Actual vs Predicted — {group_key}\n"
                        f"Target={Y_cols[0]}\n"
                        f"R²={r2_scores[order]:.3f}  RMSE={rmse_scores[order]:.3f}\n"
                        f"Recipe={source_name}  Step={source_step}  Tool={source_tool}\n"
                        f"{date_str}"
                    )
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    fig.tight_layout()

                    plot_path = os.path.join(output_dir, f"pipeline_model_{response_name}_{date_str}_{group_key}.png")
                    fig.savefig(plot_path)
                    plt.close()
                    plot_paths.append(plot_path)

                status = "success"
                print(status)
                if json_update:
                    update_json(config_path, ndx, "plots", plot_paths)
                    update_json(config_path, ndx, "error", "")

            except Exception as exc:
                if not status:
                    status = "failed"
                print_section("! Exception", "debug")
                error_msg = f"{type(exc).__name__}: {exc}"
                print(error_msg)
                if json_update:
                    update_json(config_path, ndx, "error", error_msg)

            # Elapsed time
            elapsed = dt.today() - start_time
            total_s = elapsed.seconds
            h, rem  = divmod(total_s, 3600)
            m, s    = divmod(rem, 60)
            print_section(f"Elapsed: {h}h {m:02}m {s:02}s")
            if json_update:
                update_json(config_path, ndx, "status", status)

    print(f"\nJob #{ndx} done  |  elapsed {h}h {m:02}m {s:02}s  |  status: {status}")

spark.stop()
