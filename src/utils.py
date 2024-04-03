""" Contains the functions used across the project.

    @author: mikhail.galkin
"""

# %% Import needed python libraryies and project config info
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path
from IPython.display import display

from google.cloud import bigquery
from google.cloud.exceptions import NotFound


# ------------------------------------------------------------------------------
# -------------------------------- U T I L S -----------------------------------
# ------------------------------------------------------------------------------
def timing(tic):
    min, sec = divmod(time.time() - tic, 60)
    return f"for: {int(min)}min {int(sec)}sec"


def read_all_csv_in_folder(parent_folder, **kwargs):
    print("Starting read the data :")
    tic = time.time()
    parent_folder = Path(parent_folder)
    df = pd.DataFrame()  # empty DF for interim result
    for path in sorted(parent_folder.rglob("*")):
        if path.is_file() and path.suffix == ".csv":
            print(f"reading '{path}' ...", end=" ")
            _df = pd.read_csv(path, low_memory=False, **kwargs)
            print(f"{len(_df):,} records")
            # Inject state (sometime it is NA)
            state = path.stem[:2]
            print(f"... inject [state] = {state} ...")
            _df["state"] = state
            # Concatenate the data
            df = pd.concat([df, _df], axis=0, ignore_index=True)
            del _df
    print(f"\nAll data has {len(df):,} records ...")
    print(f"All read {timing(tic)}.")
    return df


# ------------------------------------------------------------------------------
# ----------------------------- P A R A M E T E R S ----------------------------
# ------------------------------------------------------------------------------
def pd_set_options():
    """Set parameters for PANDAS to InteractiveWindow"""

    display_settings = {
        "max_columns": 40,
        "max_rows": 400,  # default: 60
        "width": 500,
        "max_info_columns": 500,
        "expand_frame_repr": True,  # Wrap to multiple pages
        "float_format": lambda x: "%.5f" % x,
        "pprint_nest_depth": 4,
        "precision": 4,
        "show_dimensions": True,
    }
    print("\nPandas options established are:")
    for op, value in display_settings.items():
        pd.set_option(f"display.{op}", value)
        option = pd.get_option(f"display.{op}")
        print(f"\tdisplay.{op}: {option}")


def pd_reset_options():  # -------------------------------------------------------
    """Set parameters for PANDAS to InteractiveWindow"""
    pd.reset_option("all")
    print("Pandas all options re-established.")


# ------------------------------------------------------------------------------
# -------------------- P R O C E S S   D A T A  --------------------------------
# ------------------------------------------------------------------------------
def drop_duplicated(df, subset=None):
    len_df = len(df)
    print(f"\nOriginal data has {len_df:,} records ...")
    dupes = df.duplicated(subset).sum()
    if dupes == 0:
        print(f"Original data has no duplicated records ...")
    else:
        print(f"Original data has # {dupes:,} duplicated records ...")
        print(f"\t % {dupes/len(df)*100} of duplicated records ...")
        df.drop_duplicates(subset, keep="first", inplace=True)
        print(f"Dropped # {dupes:,} of duplicated records.")
        print(f"Final data has {len(df):,} records.")
    return df


def clean_currency(pd_series):
    """If the value is a string, then remove currency symbol and delimiters
    otherwise, the value is numeric and can be converted
    """
    x = pd_series.replace({"\$": "", ",": ""}, regex=True).astype("Int64")
    return x


def outliers_get_zscores(df, cols_to_check=None, sigma=3):
    print(f"\nGet columns w/ outliers w/ {sigma}-sigma...")
    cols = []
    nums = []
    df_out = None
    for col in cols_to_check:
        mean = df[col].mean()
        std = df[col].std()
        z = np.abs(df[col] - mean) > (sigma * std)
        num_outliers = z.sum()

        if num_outliers > 0:
            print(f"\t{col}: {num_outliers} ouliers.")
            display(df.loc[z, col])
            cols.append(col)
            nums.append(num_outliers)
            df_out = pd.concat([df_out, z], axis=1)
    display(df_out.sum())
    return df_out


def outliers_get_quantiles(df, cols_to_check=None, treshold=[0.01, 0.99]):
    print(f"\nGet columns w/ outliers w/ {treshold} quantile tresholds...")
    cols = []
    nums = []
    df_out = pd.DataFrame()
    for col in cols_to_check:
        cutoffs = df[col].quantile(treshold).to_list()
        q = (df[col] < cutoffs[0]) | (df[col] > cutoffs[1])
        num_outliers = q.sum()

        if num_outliers > 0:
            print(f"\t{col}: cutoffs = {cutoffs}: {num_outliers} ouliers.")
            display(df.loc[q, col])
            cols.append(col)
            nums.append(num_outliers)
            df_out = pd.concat([df_out, q], axis=1)

    display(df_out.sum())
    return df_out


def outliers_ridoff(df, df_out):
    print(f"\nRid off outliers via z-score..")
    idx = df_out.sum(axis=1) > 0
    df = df[~idx]
    print(f"Totally deleted {sum(idx)} outliers...")
    print(f"Data w/o outliers has: {len(df)} rows X {len(df.columns)} cols")
    return df

# ------------------------------------------------------------------------------
# -------------------- G O O G L E   B I G Q U E R Y ---------------------------
# ------------------------------------------------------------------------------
def bq_list_datasets(bq_client):
    # Make an API request.
    datasets = list(bq_client.list_datasets())
    project_id = bq_client.project

    if datasets:
        print(f"Datasets in project <{project_id}> :")
        for dataset in datasets:
            print(f"\t{dataset.dataset_id}")
    else:
        print(f"<{project_id}> project does not contain any datasets.")


def bq_get_dataset_info(bq_client, gcp_project_id, bq_dataset_id):
    ## (developer): Set dataset_id to the ID of the dataset to fetch.
    # dataset_id = 'your-project.your_dataset'

    dataset_id = f"{gcp_project_id}.{bq_dataset_id}"
    dataset = bq_client.get_dataset(dataset_id)  # Make an API request.
    full_dataset_id = f"{dataset.project}.{dataset.dataset_id}"
    friendly_name = dataset.friendly_name
    print(
        f"Got dataset <{full_dataset_id}> with friendly_name '{friendly_name}'."
    )

    # View dataset properties.
    print(f"Description: {dataset.description}")
    print("Labels:")
    labels = dataset.labels
    if labels:
        for label, value in labels.items():
            print(f"\t{label}: {value}")
    else:
        print("\tDataset has no labels defined.")

    # View tables in dataset.
    print("Tables:")
    tables = list(bq_client.list_tables(dataset))  # Make an API request(s).
    if tables:
        for table in tables:
            print(f"\t{table.table_id}")
    else:
        print("\tThis dataset does not contain any tables.")


def bq_check_dataset_existence(bq_client, gcp_project_id, bq_dataset_id):
    ## Set dataset_id to the ID of the dataset to determine existence.
    # dataset_id = "your-project.your_dataset"
    dataset_id = f"{gcp_project_id}.{bq_dataset_id}"

    try:
        bq_client.get_dataset(dataset_id)  # Make an API request.
        print(f"Dataset <{dataset_id}> already exists")
        return True
    except NotFound:
        print(f"Dataset <{dataset_id}> is not found")
        return False


def bq_check_table_existence(
    bq_client,
    gcp_project_id,
    bq_dataset_id,
    bq_table_id,
):
    ## Set table_id to the ID of the dataset to determine existence.
    # table_id = "your-project.your_dataset.your_table"
    table_id = f"{gcp_project_id}.{bq_dataset_id}.{bq_table_id}"

    try:
        bq_client.get_table(table_id)  # Make an API request.
        print(f"Table <{table_id}> already exists")
        return True
    except NotFound:
        print(f"Table <{table_id}> is not found")
        return False


def bq_create_dataset(
    bq_client,
    gcp_location,
    gcp_project_id,
    bq_dataset_id,
):
    """_summary_

    Args:
        gcp_project_id (_type_): _description_
        gcp_location (_type_): _description_
        bq_client (_type_): _description_
        bq_dataset_id (_type_): Letters, numbers, and underscores allowed only
    """
    ## (developer): Set dataset_id to the ID of the dataset to create.
    dataset_id = f"{gcp_project_id}.{bq_dataset_id}"

    # Construct a full Dataset object to send to the API.
    dataset = bigquery.Dataset(dataset_id)

    # Specify the geographic location where the dataset should reside.
    dataset.location = gcp_location

    # Send the dataset to the API for creation, with an explicit timeout.
    # Raises google.api_core.exceptions.Conflict if the Dataset already
    # exists within the project.
    try:
        # Make an API request.
        dataset = bq_client.create_dataset(dataset, timeout=30)
        print(f"Created dataset <{gcp_project_id}.{dataset.dataset_id}>")
    except:
        print(f"Dataset <{dataset_id}> already exists")


def bq_create_table_schema(saved_schema):
    print(f"Create BQ table schema from ...")
    bq_table_schema = []
    for field in saved_schema["fields"]:
        if field["type"] == "number":
            field_type = "FLOAT"
        else:
            field_type = field["type"].upper()

        bq_table_schema.append(bigquery.SchemaField(field["name"], field_type))
    return bq_table_schema


def df_create_table_schema(saved_schema):
    print(f"Create table schema from for pushing data with Pandas ...")
    df_table_schema = []
    for field in saved_schema["fields"]:
        if field["type"] == "number":
            field["type"] = "FLOAT"
        else:
            field["type"] = field["type"].upper()

        df_field = {"name": field["name"], "type": field["type"]}
        df_table_schema.append(df_field)
    return df_table_schema


def bq_create_table(
    bq_client,
    gcp_project_id,
    bq_dataset_id,
    bq_table_id,
    bq_table_schema,
):
    # (developer): Set table_id to the ID of the table to create.
    table_id = f"{gcp_project_id}.{bq_dataset_id}.{bq_table_id}"

    table = bigquery.Table(table_id, schema=bq_table_schema)
    table.clustering_fields = ["addr_state"]
    table = bq_client.create_table(table)
    print(f"Created clustered table <{table_id}>")
    print(f"  with clustering fields: {table.clustering_fields}")


def bq_delete_dataset(
    bq_client,
    gcp_project_id,
    bq_dataset_id,
):
    ## Set dataset_id to the ID of the dataset to fetch.
    dataset_id = f"{gcp_project_id}.{bq_dataset_id}"

    # Use the delete_contents parameter to delete a dataset and its contents.
    # Use the not_found_ok parameter to not receive an error if the dataset has already been deleted.
    bq_client.delete_dataset(
        dataset_id,
        delete_contents=True,
        not_found_ok=True,
    )
    print(f"Deleted dataset <{dataset_id}>.")
