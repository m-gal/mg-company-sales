# %% Setup ---------------------------------------------------------------------
import sys
import pandas as pd
from sklearn import model_selection
from pprint import pprint
from IPython.display import display


# %% Setup the Tensorflow ------------------------------------------------------
import tensorflow as tf

print("tensorflow ver.:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        # Currently, memory growth needs to be the same across GPUs
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices("GPU")
        cuda = tf.test.is_built_with_cuda()
        gpu_support = tf.test.is_built_with_gpu_support()
        print(f"\tPhysical GPUs: {len(gpus)}\n\tLogical GPUs: {len(logical_gpus)}")
        print(f"\tIs built with GPU support: {gpu_support}")
        print(f"\tIs built with CUDA: {cuda}")
    except RuntimeError as e:
        # Memory growth must be set before GPUs have been initialized
        print(e)

# %% Params --------------------------------------------------------------------
RND_STATE = 42

DIR_WITH_DATA = "Z:/fishtailS3/ft-model/companies-us-sales/data/"
FILE_WITH_TRAIN_DATA = "companies_us_sales_train.csv.gz"
FILE_WITH_TEST_DATA = "companies_us_sales_test.csv"

COLS_DTYPE = {
    "addr_city": "object",
    "addr_state": "object",
    "addr_zip_4": "object",
    "business_specialty": "object",
    "naics_code": "int64",
    "company_started_year": "int64",
    "company_employees": "object",
    "company_sales": "int64",
}
USECOLS = list(COLS_DTYPE.keys())

X_cols = [
    "addr_city",
    "addr_state",
    "addr_zip_4",
    "business_specialty",
    "naics_code",
    "company_started_year",
    "company_employees",
]
Y_cols = ["company_sales"]

MODEL_PARAMS = {
    "_batch_size": 8184,
    "_epochs": 20,
    "_optimizer": tf.keras.optimizers.RMSprop(learning_rate=0.001),
    "_loss": tf.keras.losses.MeanSquaredError(name="mse"),
    "_metrics": tf.keras.metrics.MeanAbsolutePercentageError(name="mape"),
}

# %%
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from tensorflow.keras import layers

# %% Load processed data -------------------------------------------------------
print(f"Load DEV dataset to", end=" ")
path_to_data = DIR_WITH_DATA + FILE_WITH_TRAIN_DATA
print(f"{path_to_data}")
df_train = pd.read_csv(path_to_data, dtype=COLS_DTYPE, nrows=1_000_000)
print(f"Loaded dataset has {df_train.shape[0]:,} rows and {df_train.shape[1]} cols.")

# %%
val_dataframe = df_train.sample(frac=0.2, random_state=1337)
train_dataframe = df_train.drop(val_dataframe.index)

print(
    "Using %d samples for training and %d for validation"
    % (len(train_dataframe), len(val_dataframe))
)


# %%
def dataframe_to_dataset(dataframe):
    dataframe = dataframe.copy()
    labels = dataframe.pop("company_sales")
    ds = tf.data.Dataset.from_tensor_slices((dict(dataframe), labels))
    ds = ds.shuffle(buffer_size=len(dataframe))
    return ds


train_ds = dataframe_to_dataset(train_dataframe)
val_ds = dataframe_to_dataset(val_dataframe)

print(f"\nInspect the train dataset's elements ...")
pprint(train_ds.element_spec)

# %%
for x, y in train_ds.take(1):
    print("Input:", x)
    print("Target:", y)

# %%
from tensorflow.keras.layers import IntegerLookup
from tensorflow.keras.layers import Normalization
from tensorflow.keras.layers import StringLookup


def encode_numerical_feature(feature, name, dataset):
    # Create a Normalization layer for our feature
    normalizer = Normalization()

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the statistics of the data
    normalizer.adapt(feature_ds)

    # Normalize the input feature
    encoded_feature = normalizer(feature)
    return encoded_feature


def encode_categorical_feature(feature, name, dataset, is_string):
    lookup_class = StringLookup if is_string else IntegerLookup
    # Create a lookup layer which will turn strings into integer indices
    lookup = lookup_class(output_mode="binary")

    # Prepare a Dataset that only yields our feature
    feature_ds = dataset.map(lambda x, y: x[name])
    feature_ds = feature_ds.map(lambda x: tf.expand_dims(x, -1))

    # Learn the set of possible string values and assign them a fixed integer index
    lookup.adapt(feature_ds)

    # Turn the string input into integer indices
    encoded_feature = lookup(feature)
    return encoded_feature


# %% Build a model --------------------------------------------------------------
# Categorical features encoded as integers
naics_code = keras.Input(shape=(1,), name="naics_code", dtype="int64")


# Categorical feature encoded as string
addr_city = keras.Input(shape=(1,), name="addr_city", dtype="string")
addr_state = keras.Input(shape=(1,), name="addr_state", dtype="string")
addr_zip_4 = keras.Input(shape=(1,), name="addr_zip_4", dtype="string")
business_specialty = keras.Input(shape=(1,), name="business_specialty", dtype="string")
company_employees = keras.Input(shape=(1,), name="company_employees", dtype="string")

# Numerical features
company_started_year = keras.Input(shape=(1,), name="company_started_year")


all_inputs = [
    naics_code,
    addr_city,
    addr_state,
    addr_zip_4,
    business_specialty,
    company_employees,
    company_started_year,
]

# Integer categorical features
naics_code_encoded = encode_categorical_feature(naics_code, "naics_code", train_ds, False)
print(f"<naics_code> encoded.")

# String categorical features
addr_city_encoded = encode_categorical_feature(addr_city, "addr_city", train_ds, True)
print(f"<addr_city> encoded.")
addr_state_encoded = encode_categorical_feature(addr_state, "addr_state", train_ds, True)
print(f"<addr_state> encoded.")
addr_zip_4_encoded = encode_categorical_feature(addr_zip_4, "addr_zip_4", train_ds, True)
print(f"<addr_zip_4> encoded.")
business_specialty_encoded = encode_categorical_feature(
    business_specialty, "business_specialty", train_ds, True
)
print(f"<business_specialty> encoded.")
company_employees_encoded = encode_categorical_feature(
    company_employees, "company_employees", train_ds, True
)
print(f"<company_employees> encoded.")

# Numerical features
company_started_year_encoded = encode_numerical_feature(
    company_started_year, "company_started_year", train_ds
)
print(f"<company_started_year> encoded.")

all_features = layers.concatenate(
    [
        naics_code_encoded,
        addr_city_encoded,
        addr_state_encoded,
        addr_zip_4_encoded,
        business_specialty_encoded,
        company_employees_encoded,
        company_started_year_encoded,
    ]
)
x = layers.Dense(32, activation="relu")(all_features)
x = layers.Dropout(0.5)(x)
output = layers.Dense(1, activation="relu")(x)
model = keras.Model(all_inputs, output)
model.compile("adam", "mse", "mape")

# %%
# `rankdir='LR'` is to make the graph horizontal.
keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# %%
model.fit(train_ds, epochs=50, validation_data=val_ds)


#%%
m = tf.keras.metrics.MeanAbsolutePercentageError()
m.update_state([[1, 1]], [[0.9, 0.9]])
m.result().numpy()
