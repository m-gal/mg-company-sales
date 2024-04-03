"""
    Train a Transformer DL model

    1. All the categorical features are encoded as embeddings,
    using the same embedding_dims.
    This means that each value in each categorical feature
    will have its own embedding vector.
    2. A column embedding, one embedding vector for each categorical feature,
    is added (point-wise) to the categorical feature embedding.
    3. The embedded categorical features are fed into a stack of Transformer blocks.
    Each Transformer block consists of a multi-head self-attention layer
    followed by a feed-forward layer.
    4. The outputs of the final Transformer layer,
    which are the contextual embeddings of the categorical features,
    are concatenated with the input numerical features,
    and fed into a final MLP block.

    @author: mikhail.galkin
"""

#### WARNING: the code that follows would make you cry.
#### A Safety Pig is provided below for your benefit
#                            _
#    _._ _..._ .-',     _.._(`))
#   '-. `     '  /-._.-'    ',/
#      )         \            '.
#     / _    _    |             \
#    |  a    a    /              |
#    \   .-.                     ;
#     '-('' ).-'       ,'       ;
#        '-;           |      .'
#           \           \    /
#           | 7  .__  _.-\   \
#           | |  |  ``/  /`  /
#          /,_|  |   /,_/   /
#            /,_/      '`-'


# %% Setup ---------------------------------------------------------------------
import sys
import inspect
import datetime
import pandas as pd
import numpy as np
from pathlib import Path
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

tf.keras.utils.set_random_seed(42)


# %% Load project's stuff ------------------------------------------------------
sys.path.extend([".", "./.", "././.", "../..", "../../.."])

from src.dl.models import transformer_model
from src.dl.utils import df_to_ds
from src.dl.utils import create_model_inputs_and_features
from src.utils import outliers_get_quantiles
from src.utils import outliers_get_zscores
from src.utils import outliers_ridoff


# %% Custom functions ----------------------------------------------------------
def scheduler_exp(epoch, lr):
    import math

    n_epoch = 8
    if epoch < n_epoch:
        return lr
    else:
        return lr * math.exp(-0.3)


def scheduler_drop(epoch, lr):
    n_epoch = 10
    lr_drop_rate = 0.75
    if epoch < n_epoch:
        return lr
    else:
        return lr * lr_drop_rate


# %% Params --------------------------------------------------------------------
RND_STATE = 42

DIR_WITH_DATA = "Z:/fishtailS3/ft-model/companies-us-sales/data/"
FILE_WITH_TRAIN_DATA = "companies_us_sales_train.csv.gz"
FILE_WITH_TEST_DATA = "companies_us_sales_test.csv"

TENSORBOARD_DIR = "D:/dprojects/mgal-for-github/mg-company-sales/tensorboard/"
DL_DIR = "D:/dprojects/mgal-for-github/mg-company-sales/src/dl/"
LOG_DIR = TENSORBOARD_DIR + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

COLS_DTYPE = {
    "addr_city": "object",
    "addr_state": "object",
    "addr_zip_4": "object",
    "business_specialty": "object",
    # "naics_code": "int64",
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
    # "naics_code",
    "company_started_year",
    "company_employees",
]
Y_cols = ["company_sales"]

MODEL_PARAMS = {
    "_batch_size": 130_944,
    "_epochs": 25,
    "_optimizer": tf.keras.optimizers.RMSprop(learning_rate=0.003),
    "_loss": tf.keras.losses.MeanSquaredError(name="mse"),
    "_metrics": tf.keras.metrics.MeanAbsoluteError(name="mae"),
    "_callbacks": [
        tf.keras.callbacks.TensorBoard(log_dir=LOG_DIR),
        # tf.keras.callbacks.LearningRateScheduler(scheduler_exp, verbose=1),
    ],
    "_num_transformer_blocks": 2,
    "_num_heads": 5,
    "_embedding_dim": 16,
    "_multi_head_attention_dropout_rate": 0.0,
}


# %% Load processed data -------------------------------------------------------
print(f"Load DEV dataset to", end=" ")
path_to_data = DIR_WITH_DATA + FILE_WITH_TRAIN_DATA
print(f"{path_to_data}")
df_train = pd.read_csv(path_to_data, dtype=COLS_DTYPE, usecols=USECOLS)
print(f"Loaded dataset has {df_train.shape[0]:,} rows and {df_train.shape[1]} cols.")

print(f"Load TEST dataset to", end=" ")
path_to_data = DIR_WITH_DATA + FILE_WITH_TEST_DATA
print(f"{path_to_data}")
df_test = pd.read_csv(path_to_data, dtype=COLS_DTYPE, usecols=USECOLS)
print(f"Loaded dataset has {df_test.shape[0]:,} rows and {df_test.shape[1]} cols.")


# %% Handle outliers -----------------------------------------------------------
OUTLIERS_TRESHOLD = [0.01, 0.995]
# Handling outliers
train_out = outliers_get_quantiles(
    df_train,
    cols_to_check=["company_sales"],
    treshold=OUTLIERS_TRESHOLD,
)
df_train = outliers_ridoff(df_train, train_out)

test_out = outliers_get_quantiles(
    df_test,
    cols_to_check=["company_sales"],
    treshold=OUTLIERS_TRESHOLD,
)
df_test = outliers_ridoff(df_test, test_out)


# %% Slit DEV set onto TRAIN and VAL -------------------------------------------
# Split the development dataset with stratified target
print(f"\nSplit development data on train & validation sets ...")
df_train, df_val = model_selection.train_test_split(
    df_train,
    train_size=0.9,
    random_state=RND_STATE,
    shuffle=True,
    stratify=df_train["addr_state"],
)

print(f"\tdf_train: {df_train.shape[1]} vars & {df_train.shape[0]:,} rows.")
print(f"\tdf_val: {df_val.shape[1]} vars & {df_val.shape[0]:,} rows.")
print(f"\tdf_test: {df_test.shape[1]} vars & {df_test.shape[0]:,} rows.")


# %% Handle target -------------------------------------------------------------
# Save separatedly the true target values
Y_true_train = df_train[Y_cols].values
Y_true_val = df_val[Y_cols].values
Y_true_test = df_test[Y_cols].values

# Log transformation for target values
df_train[Y_cols] = np.log(df_train[Y_cols])
df_val[Y_cols] = np.log(df_val[Y_cols])
df_test[Y_cols] = np.log(df_test[Y_cols])


# %% Convert pd.DataFrame to tf.Dataset ----------------------------------------
#! Do not use the shuffle=True! It leads to wrong accuracy on the test set
#! also a runtime increases x10 times
print("Train set:", end=" ")
ds_train, y_train_true = df_to_ds(
    df_train,
    X_cols,
    Y_cols,
    MODEL_PARAMS["_batch_size"],
    shuffle=False,
)
print("Val set:", end=" ")
ds_val, y_val_true = df_to_ds(
    df_val,
    X_cols,
    Y_cols,
    MODEL_PARAMS["_batch_size"],
    shuffle=False,
)
print("Test set:", end=" ")
ds_test, y_test_true = df_to_ds(
    df_test,
    X_cols,
    Y_cols,
    MODEL_PARAMS["_batch_size"],
    shuffle=False,
)

print(f"\nInspect the train dataset's elements ...")
pprint(ds_train.element_spec)


# %% Setup a FeatureSpace ------------------------------------------------------
# A lists of the binary feature names. Feature like "is_animal"
BINARY_FEATURES = []  # ["is_animal"]

# A list of the numerical feature names.
NUMERIC_FEATURES_TO_NORMALIZED = ["company_started_year"]
NUMERIC_FEATURES_TO_DISCRETIZED = []

# A list of INT categorical features w/o vocabulary we want be one-hot encoding.
CATEGORICAL_INTEGER_FEATURES_TO_OHE = []
# A dict of INT categorical features with vocabulary we want be one-hot encoding.
CATEGORICAL_INTEGER_FEATURES_TO_EMBEDD = {}

# A list of OBJECT categorical features w/o vocabulary we want be one-hot encoding.
CATEGORICAL_STRING_FEATURES_TO_OHE = []
# A dict of OBJECT categorical features with vocabulary we want be one-hot encoding.
CATEGORICAL_STRING_FEATURES_TO_EMBEDD = {
    "addr_zip_4": sorted(df_train["addr_zip_4"].unique().tolist()),
    "addr_state": sorted(df_train["addr_state"].unique().tolist()),
    "addr_city": sorted(df_train["addr_city"].unique().tolist()),
    "business_specialty": sorted(df_train["business_specialty"].unique().tolist()),
    "company_employees": sorted(df_train["company_employees"].unique().tolist()),
}

# A list of textual features w/o vocabulary we want be embedded
TEXT_FEATURES_TO_EMBEDD = []

# Targets' names and types
REGRESSION_TARGETS = ["company_sales"]
BINARY_TARGETS = []
MULTICLASS_TARGET_W_CLASSES = {}  # {MULTICLASS_TARGET_NAME: classes}
MULTILABEL_TARGET_W_LABELS = {}  # {MULTILABEL_TARGET_NAME: labels}


# A dictionary of all the input columns\features.
set_of_features = {
    "x": X_cols,
    "y": Y_cols,
    "type_x": {
        "binary": BINARY_FEATURES,
        "numeric_to_normalized": NUMERIC_FEATURES_TO_NORMALIZED,
        "numeric_to_discretized": NUMERIC_FEATURES_TO_DISCRETIZED,
        "categorical_int_to_ohe": CATEGORICAL_INTEGER_FEATURES_TO_OHE,
        "categorical_int_to_embedd": CATEGORICAL_INTEGER_FEATURES_TO_EMBEDD,
        "categorical_str_to_ohe": CATEGORICAL_STRING_FEATURES_TO_OHE,
        "categorical_str_to_embedd": CATEGORICAL_STRING_FEATURES_TO_EMBEDD,
        "text_to_embedd": TEXT_FEATURES_TO_EMBEDD,
    },
    "type_y": {
        "regression": REGRESSION_TARGETS,
        "binary": BINARY_TARGETS,
        "multiclass": MULTICLASS_TARGET_W_CLASSES,
        "multilabel": MULTILABEL_TARGET_W_LABELS,
    },
}


# %%Create inputs and encoded features -----------------------------------------
inputs, encoded_features, embedded_features, _ = create_model_inputs_and_features(
    set_of_features,
    ds_train,
    embedding_dims=MODEL_PARAMS["_embedding_dim"],
    print_result=True,
)


# %% Build the model -----------------------------------------------------------
# Create folder for logging
if not Path(LOG_DIR).is_dir():
    Path(LOG_DIR).mkdir()

# Build & compile model
model = transformer_model(
    inputs,
    encoded_features,
    embedded_features,
    embedding_dims=MODEL_PARAMS["_embedding_dim"],
    num_transformer_blocks=MODEL_PARAMS["_num_transformer_blocks"],
    num_heads=MODEL_PARAMS["_num_heads"],
    dropout_rate=MODEL_PARAMS["_multi_head_attention_dropout_rate"],
    optimizer=MODEL_PARAMS["_optimizer"],
    loss=MODEL_PARAMS["_loss"],
    metrics=MODEL_PARAMS["_metrics"],
    print_summary=False,
)

# Plot the model
fig_model = tf.keras.utils.plot_model(
    model,
    to_file=LOG_DIR + f"/dl_{model.name}_arch.png",
    show_shapes=True,
    show_dtype=True,
    show_layer_names=True,
    expand_nested=False,
    dpi=100,
    rankdir="TB",
    show_layer_activations=True,
)
display(fig_model)


# %% Get & save model's code ---------------------------------------------------
with open(LOG_DIR + f"/dl_{model.name}_code.txt", "w") as f:
    print(inspect.getsource(transformer_model), file=f)

with open(LOG_DIR + f"/dl_{model.name}_params.txt", "w") as f:
    print(MODEL_PARAMS, file=f)
    print(model.optimizer.lr, file=f)
    print(inspect.getsource(scheduler_exp), file=f)
    print(inspect.getsource(scheduler_drop), file=f)


# %% Train the model -----------------------------------------------------------
# Train the model
tf.keras.backend.clear_session()

history = model.fit(
    ds_train,
    batch_size=MODEL_PARAMS["_batch_size"],
    epochs=MODEL_PARAMS["_epochs"],
    verbose=1,
    validation_data=ds_val,
    use_multiprocessing=True,
    callbacks=MODEL_PARAMS["_callbacks"],
)


# %% Make prediction & define true values --------------------------------------
print(f"\nMake predictions for performance calculating ...")
y_pred_train = np.exp(model.predict(ds_train))
y_pred_val = np.exp(model.predict(ds_val))
y_pred_test = np.exp(model.predict(ds_test))

print(f"\nCalculate metrics for performance evaluating ...")
mae = tf.keras.losses.MeanAbsoluteError()
mape = tf.keras.losses.MeanAbsolutePercentageError()

# Train set
y_train_mae = mae(Y_true_train, y_pred_train).numpy()
y_train_mape = mape(Y_true_train, y_pred_train).numpy()
print(f"TRAIN set:\n\t MAE = {y_train_mae}\n\t MAPE = {y_train_mape}")

# Validation set
y_val_mae = mae(Y_true_val, y_pred_val).numpy()
y_val_mape = mape(Y_true_val, y_pred_val).numpy()
print(f"VAL set:\n\t MAE = {y_val_mae}\n\t MAPE = {y_val_mape}")

# Test set
y_test_mae = mae(Y_true_test, y_pred_test).numpy()
y_test_mape = mape(Y_true_test, y_pred_test).numpy()
print(f"TEST set:\n\t MAE = {y_test_mae}\n\t MAPE = {y_test_mape}")

with open(LOG_DIR + f"/dl_{model.name}_performance.txt", "w") as f:
    print(f"OUTLIERS_TRESHOLD={OUTLIERS_TRESHOLD}", file=f)
    print(history.history, file=f)
    print(f"TRAIN set:\n\t MAE = {y_train_mae}\n\t MAPE = {y_train_mape}", file=f)
    print(f"VAL set:\n\t MAE = {y_val_mae}\n\t MAPE = {y_val_mape}", file=f)
    print(f"TEST set:\n\t MAE = {y_test_mae}\n\t MAPE = {y_test_mape}", file=f)

print("Done.")

# %% Save whole model ----------------------------------------------------------
model_dir = LOG_DIR + "/model"
if not Path(model_dir).is_dir():
    Path(model_dir).mkdir()
tf.keras.saving.save_model(model, model_dir, overwrite=True, save_format=None)
