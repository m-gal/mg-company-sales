"""
    Helps to reload project's module and get its inspections
    w\o reloading working space

    @author: mikhail.galkin
"""

#%% Import libs
import sys
import inspect
import importlib

sys.path.extend([".", "./.", "././.", "../..", "../../.."])
# import src

#%% ------------------------------ CONFIG --------------------------------------
import src.config

importlib.reload(src.config)
from src.config import project_dir
print(project_dir)


#%% ------------------------------ UTILS DL-------------------------------------
import src.dl.utils

importlib.reload(src.dl.utils)
print(inspect.getsource(src.dl.utils.embedd_text_without_vocab))

#%% ----------------------------- MODELS DL-------------------------------------
import src.dl.utils

importlib.reload(src.dl.models)
print(inspect.getsource(src.dl.models.transformer_model))
