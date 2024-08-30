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
from scipy.stats.mstats import trimmed_var
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from category_encoders import OneHotEncoder, OrdinalEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import set_config
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.validation import check_is_fitted
from sklearn.tree import DecisionTreeClassifier, plot_tree

# input data path
file_path = "/home/blueaz/Downloads/SensorDownload/May2024/ztennis.db" 


warnings.simplefilter("ignore", UserWarning)

df = wrangle.wrangle(file_path)

# df = df.sort_values("date")
corr = df.select_dtypes("number").corr()
pd.set_option('display.max_columns', 40)
# df["client_created"] = pd.to_datetime(df["client_created"])
session = ['l_id' , 'swing_type', 'swing_side', 'hand_type',
       'backswing_type', 'backswing_time', 'power', 'stroke',
       'dbg_acc_1', 'dbg_acc_2', 'dbg_acc_3',
       'dbg_gyro_1', 'dbg_gyro_2', 'dbg_var_1', 'dbg_var_2',
       'dbg_var_3', 'dbg_var_4',  'dbg_sum_gx',
       'dbg_sum_gy', 'dbg_sv_ax', 'dbg_sv_ay', 'dbg_max_ax', 'dbg_max_ay',
       'dbg_min_az', 'dbg_max_az', 'impact_region', 'diffxy',
       'ball_spin', 'impact_position_x', 'impact_position_y', 'racket_speed']
serves = ['l_id', 'impact_vel', 'ball_vel', 'spin', 'upswing_time',
         'impact_time', 'service_court']
df_session = df[session]
df_serves = df[serves]

sensor = [ 'dbg_acc_1', 'dbg_acc_2', 'dbg_acc_3', 'dbg_gyro_1',
       'dbg_gyro_2', 'dbg_var_1', 'dbg_var_2', 'dbg_var_3', 'dbg_var_4',
       'dbg_sum_gx', 'dbg_sum_gy', 'dbg_sv_ax', 'dbg_sv_ay', 'dbg_max_ax',
          'dbg_max_ay', 'dbg_min_az', 'dbg_max_az' ]
calc = [ 'backswing_time', 'power', 'ball_spin',
        'impact_position_x', 'impact_position_y',
       'racket_speed', 'impact_region']
df_session_sensor = df_session[sensor]
df_session_calc = df_session[calc]


#Split data
X_data = df_session_sensor
target = "impact_position_y"
y_data = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

# Baseline
y_mean = y_train.mean()
y_pred_baseline = [y_mean] * len(y_train)
baseline_mae = mean_absolute_error(y_train, y_pred_baseline)
print("Mean score:", y_mean)
print("Baseline MAE:", baseline_mae)

# Fit model
model = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    SimpleImputer(),
    Ridge()   
) 
model
model.fit(X_train, y_train)
y_pred_test = pd.Series(model.predict(X_test))

# get model features
coefficients = model.named_steps["ridge"].coef_
features = model.named_steps["onehotencoder"].get_feature_names()
feat_imp = pd.Series(coefficients, index=features)
feat_imp

# Build bar chart
feat_imp.sort_values(key=abs).tail(15).plot(kind="barh")
# Label axes
plt.xlabel("importance")
plt.ylabel("feature")
# Add title
plt.title("importance vs feature")

#RMSE
TestMAE = metrics.mean_absolute_error(y_test, y_pred_test)
print("Test MAE:", TestMAE)

print("hello")
