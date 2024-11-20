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
from category_encoders import OneHotEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge  # noqa F401
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

calc = [ 'backswing_time', 'power', 'ball_spin', 'impact_position_x',
       'impact_position_y', 'racket_speed', 'impact_region']

all_signals = sensor + calc

df_session_sensor = df_session[sensor]
df_session_calc = df_session[calc]
df_total = df_session[all_signals]

# Calculate variance
top_ten_var = df_total.var().sort_values().tail(15)
top_ten_var
# Create horizontal bar chart of `top_ten_trim_var`
fig = px.bar(
    x=top_ten_var,
    y=top_ten_var.index,
    title="High Variance Features"
)

fig.update_layout(xaxis_title="Trimmed Var", yaxis_title="Feat")
#Split data
X_data = df_total
target = "swing_side"
y_data = df[target]
X_train, X_test, y_train, y_test = train_test_split(
    X_data, y_data, test_size=0.2, random_state=42
)

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

y_pred_test = pd.Series(model.predict(X_test))

print(X_test.info())
X_test.head()


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

# Calculate accuracy
# accuracy = accuracy_score(y_test, y_pred_test)
# print("Accuracy:", accuracy)
# RMSE
print(np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
# Build bar chart
# top15 = feat_imp.sort_values(key=abs).tail(15).plot(kind="barh")
top15 = feat_imp.sort_values(key=abs).tail(15).index.to_list()
top5 = feat_imp.sort_values(key=abs).tail(5).index.to_list()

fig.update_layout(xaxis_title="Trimmed Var", yaxis_title="Feat")

X = df_total
# X = df_total[top15]
n_clusters = range(2,13)
inertia_errors = []
silhouette_scores = []

# Add `for` loop to train model and calculate inertia, silhouette score.
for k in n_clusters:
    model = make_pipeline(StandardScaler(), KMeans(n_clusters=k, random_state=42))
    model.fit(X)
    inertia_errors.append(model.named_steps["kmeans"].inertia_)
    silhouette_scores.append(
            silhouette_score(X, model.named_steps["kmeans"].labels_)
    )

fig = px.line(
    x=n_clusters, y=inertia_errors, title="Kmeans"
)
fig.update_layout(xaxis_title="clust", yaxis_title="Inertia")
# Create a line plot of `silhouette_scores` vs `n_clusters`
fig = px.line(
    x=n_clusters, y=silhouette_scores, title="Kmeans"
)
fig.update_layout(xaxis_title="clust", yaxis_title="ss")

final_model = make_pipeline(
    StandardScaler(),
    KMeans(n_clusters=4, random_state=42)
)
final_model.fit(X)

labels = final_model.named_steps["kmeans"].labels_
xgb = X.groupby(labels).mean()
xgb

# Create side-by-side bar chart of `xgb`
fig = px.bar(
    xgb,
    barmode="group",
    title="MEAN by cluster"
)

fig.update_layout(xaxis_title = "Cluster", yaxis_title="$")
# Instantiate transformer
pca = PCA(n_components=2, random_state=42)

# Transform `X`
X_t = pca.fit_transform(X)

# Put `X_t` into DataFrame
X_pca = pd.DataFrame(X_t, columns=["PC1", "PC2"])

print("X_pca shape:", X_pca.shape)
X_pca.head()

fig = px.scatter(
    data_frame=X_pca,
    x="PC1",
    y="PC2",
    color=labels.astype(str),
    title="PCA representation"
)
fig.update_layout(xaxis_title="PC1", yaxis_title="PC2")

print("complete")
