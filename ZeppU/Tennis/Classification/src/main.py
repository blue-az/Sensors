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
target = "hand_type"
y_data = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

acc_baseline = y_train.value_counts(normalize=True).max()
print("Baseline Accuracy:", round(acc_baseline, 2))

model_lr = make_pipeline(
    OneHotEncoder(use_cat_names=True),
    LogisticRegression(max_iter=1000)
)
# Fit model to training data
model_lr.fit(X_train, y_train)

lr_train_acc = accuracy_score(y_train, model_lr.predict(X_train))
lr_val_acc = model_lr.score(X_test, y_test)

print("Logistic Regression, Training Accuracy Score:", lr_train_acc)
print("Logistic Regression, Validation Accuracy Score:", lr_val_acc)

depth_hyperparams = range(1, 16)
training_acc = []
validation_acc = []

for d in depth_hyperparams:
    # Create model with `max_depth` of `d`
    model_dt = make_pipeline(
        OrdinalEncoder(),
        DecisionTreeClassifier(max_depth=d, random_state=42)
    )
    # Fit model to training data
    model_dt.fit(X_train, y_train)
    # Calculate training accuracy score and append to `training_acc`
    training_acc.append(model_dt.score(X_train, y_train))
    # Calculate validation accuracy score and append to `training_acc`
    validation_acc.append(model_dt.score(X_test, y_test))

# plt.plot(depth_hyperparams, training_acc, label="training")
# plt.plot(depth_hyperparams, validation_acc, label="validation")
# plt.xlabel("max")
# plt.ylabel("acc")
# plt.title("Val curve")
# plt.legend(); 


final_model_dt = make_pipeline(
        OrdinalEncoder(),
        DecisionTreeClassifier(max_depth=6, random_state=42)
    )

final_model_dt.fit(X_train, y_train)

y_test_pred = final_model_dt.predict(X_train)
fm_val_acc = final_model_dt.score(X_train, y_train)

features = X_train.columns
importances = final_model_dt.named_steps["decisiontreeclassifier"].feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values()

feat_imp.plot(kind="barh")
plt.xlabel("importance")
plt.ylabel("Feature")

# RF classifier
# clf = RandomForestClassifier(random_state=42, n_jobs=-1)
# Fit model to training data
# clf.fit(X_train, y_train)
# clg_acc = clf.score(X_test, y_test)

# X_train_scaled = scaler.fit_transform(X_train)
# X_pca = pca.fit_transform(X_train_scaled)
# X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])
pipe = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        RandomForestClassifier()
    )
pipe.fit(X_train,y_train)
pipe_acc = pipe.score(X_test, y_test)

# pipe = Pipeline([(‘scaler’, StandardScaler()),
#      (‘pca’, PCA(n_components=2)),
#      (‘clf’, RandomForestClassifier())])

print("hello")
