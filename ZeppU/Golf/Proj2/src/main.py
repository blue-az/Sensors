import warnings
import base64
import io
from datetime import date
import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import wrangle
from category_encoders import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import AutoReg
from statsmodels.tsa.arima.model import ARIMA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_is_fitted
from sklearn import set_config


# PC path
file_path = "/home/blueaz/Downloads/SensorDownload/May2024/Golf3.db"

warnings.simplefilter("ignore", UserWarning)

df = wrangle.wrangle(file_path)
# make single user
mask = df['USER_HEIGHT'] < 180
df = df[mask]
mask = df['HAND_SPEED'] < 100
df = df[mask]

# plt.hist(df["SCORE"])
# Label axes
# plt.xlabel("score")
# plt.ylabel("Count")
# Add title
# plt.title("Score Distribution")
# plt.scatter(x=df["HAND_SPEED"], y=df["SCORE"])
# plt.xlabel("HAND_SPEED")
# plt.ylabel("SCORE")
# plt.title("Score vs Hand Speed")

#Split data
X_train = df.drop("SCORE", axis=1)
target = "SCORE"
y_train = df[target]

y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean score:", y_mean)
print("Baseline MAE:", baseline_mae)

model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()   
) 

set_config(display="text") 
model
# Fit model
model.fit(X_train, y_train)

print("hello")
