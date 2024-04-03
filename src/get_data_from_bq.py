""" Queries data from the BQ table with local python

    @author: mikhail.galkin
"""

# %% Setup ---------------------------------------------------------------------
import pandas as pd
from sklearn import model_selection
from google.cloud import bigquery
from IPython.display import display

# %% Params --------------------------------------------------------------------
RND_STATE = 42

GCP_LOCATION = ""
GCP_PROJECT_ID = ""
BQ_DATASET_ID = ""
BQ_TABLE_ID = ""

DIR_TO_SAVE_DATA = "Z:/S3/ft-model/companies-us-sales/data/"
FILE_TO_SAVE_TRAIN_DATA = "companies_us_sales_train.csv.gz"
FILE_TO_SAVE_TEST_DATA = "companies_us_sales_test.csv"

EM_NUM = [
    "1 to 4",
    "5 to 9",
    "10 to 19",
    "20 to 49",
    "50 to 99",
    "100 to 249",
    "250 to 499",
    "500 to 999",
    "Over 1000",
]


# %% Custom functions ----------------------------------------------------------
def outliers_get_quantiles(df, cols_to_check=None, treshold=0.999):
    print(f"\nGet columns w/ outliers w/ {treshold} quantile treshold...")
    cols = []
    nums = []
    df_out = None
    for col in cols_to_check:
        cutoff = df[col].quantile(treshold)
        q = df[col] > cutoff
        num_outliers = q.sum()

        if num_outliers > 0:
            print(f"\t{col}: cutoff = {cutoff}: {num_outliers} ouliers.")
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


# %% Setup ---------------------------------------------------------------------
client = bigquery.Client(project=GCP_PROJECT_ID)
table_id = f"{GCP_PROJECT_ID}.{BQ_DATASET_ID}.{BQ_TABLE_ID}"

# %% Pull the data -------------------------------------------------------------
print(f"Pull the dev data from Google BigQuery <{table_id}> ...")
sql = f"""
    SELECT
        addr_city
        , addr_state
        , addr_zip_4
        # , addr_zip_5 #* Matches with 'addr_city'
        , business_specialty #* Distinct: 8348
        , naics_code #* Distinct: 1047
        # , sic_2_code #* Distinct: 83 - Possible to use 'naics_code' is better
        # , sic_4_code #* Distinct: 1005  - to use 'business_specialty' is better
        , company_started_year
        , company_employees
        # , company_employee_num  #* Contains None: 7136 - 'company_employees' is better
        , company_sales_exact
        , company_sale_volume_exact
    FROM `{table_id}`
    WHERE
        addr_city IS NOT NULL
        AND addr_state IS NOT NULL
        AND addr_zip_4 IS NOT NULL
        AND business_specialty IS NOT NULL
        AND naics_code IS NOT NULL
        AND company_started_year IS NOT NULL
        AND company_employees IS NOT NULL
        AND (company_sales_exact IS NOT NULL OR company_sale_volume_exact IS NOT NULL)
    # LIMIT 100
"""
df = client.query(sql).to_dataframe()
print(f"Downloaded dataset has {df.shape[0]:,} rows and {df.shape[1]} cols.")

# %% Clean data ----------------------------------------------------------------
df = df.drop_duplicates()

# Force a NAICS into numeric
df["naics_code"] = pd.to_numeric(df["naics_code"], errors="coerce")

# Clean ZIPs
df["addr_zip_4"] = df["addr_zip_4"].str.zfill(4)

# Clean Employee amount
df = df[df["company_employees"].isin(EM_NUM)]

# Clean Sale Volume Exactly
df[["company_sales_exact", "company_sale_volume_exact"]] = df[
    ["company_sales_exact", "company_sale_volume_exact"]
].fillna(0)
df["company_sales"] = df[["company_sales_exact", "company_sale_volume_exact"]].max(axis=1)
df["company_sales"] = df["company_sales"].astype("int64")
df.drop(columns=["company_sales_exact", "company_sale_volume_exact"], inplace=True)

print(f"Final dataset has {df.shape[0]:,} rows and {df.shape[1]} cols.")

# %% Split processed data ------------------------------------------------------
# Split the data with stratified target
print(f"\nSplit development data on train & test sets ...")
df_train, df_test = model_selection.train_test_split(
    df,
    train_size=0.9,
    random_state=RND_STATE,
    shuffle=True,
    stratify=df["addr_state"],
)
print(f"Dev set: {df.shape[1]} vars & {df.shape[0]:,} rows.")
print(f"\tdf_train: {df_train.shape[1]} vars & {df_train.shape[0]:,} rows.")
print(f"\tdf_test: {df_test.shape[1]} vars & {df_test.shape[0]:,} rows.")

# %% Save processed data -------------------------------------------------------
print(f"Save DEV dataset to", end=" ")
path_to_save = DIR_TO_SAVE_DATA + FILE_TO_SAVE_TRAIN_DATA
print(f"{path_to_save}")
df_train.to_csv(
    path_to_save,
    index=False,
    encoding="utf-8-sig",
    compression="gzip",
)

print(f"Save TEST dataset to", end=" ")
path_to_save = DIR_TO_SAVE_DATA + FILE_TO_SAVE_TEST_DATA
print(f"{path_to_save}")
df_test.to_csv(
    path_to_save,
    index=False,
    encoding="utf-8-sig",
)

print("Done.")
