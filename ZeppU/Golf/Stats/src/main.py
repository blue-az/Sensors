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
from sklearn.model_selection import train_test_split
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
# # Label axes
# plt.xlabel("score")
# plt.ylabel("Count")
# # Add title
# plt.title("Score Distribution")
# plt.scatter(x=df["HAND_SPEED"], y=df["SCORE"])
# plt.xlabel("HAND_SPEED")
# plt.ylabel("SCORE")
# plt.title("Score vs Hand Speed")

# #Split data
X_data = df.drop("SCORE", axis=1)
target = "SCORE"
y_data = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)
# 
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

model


# Fit model
model.fit(X_train, y_train)

y_pred_test = pd.Series(model.predict(X_test))

print(X_test.info())
X_test.head()


coefficients = model.named_steps["ridge"].coef_
features = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index=features)
feat_imp

# Build bar chart
# feat_imp.sort_values(key=abs).tail(15).plot(kind="barh")
top5 = feat_imp.sort_values(key=abs).tail(5).keys()

sns.pairplot(df[top5])
plt.show()

# Label axes
# plt.xlabel("importance")
# plt.ylabel("score")

# Add title
# plt.title("importance vs score")

print("hello")
