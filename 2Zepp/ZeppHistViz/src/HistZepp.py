import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import pytz

# Build your `wrangle` function here
def wrangle(db_path):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Construct query
    query = """
    SELECT _id, HAPPENED_TIME, SWING_TYPE, HAND_TYPE,
    SPIN, BALL_SPEED, HEAVINESS,
    POSITION_X,POSITION_Y,
    L_PLAY_SESSION_ID, IS_HIT_FRAME from SWING
    """

    # Read query results into DataFrame
    df = pd.read_sql(query, conn, index_col="_id")
    # Remove HR outliers
    # df = df[df["AVGHR"] > 50]
    # Create duration column from timestamps
    # Convert Unix timestamps to datetime objects

    df['HAPPENED_TIME'] = pd.to_datetime(df['HAPPENED_TIME'], unit='ms')
    az_timezone = pytz.timezone('America/Phoenix')
    df['HAPPENED_TIME'] = df['HAPPENED_TIME'].dt.tz_localize('UTC').dt.tz_convert(az_timezone)
    df['HAPPENED_TIME'] = df['HAPPENED_TIME'].dt.strftime('%m-%d-%Y %I:%M:%S %p')

    
    #Replace type with sport 
    hand_type = {2: "BH", 1: "FH"}
    swing_type = {4: "VOLLEY", 3: "SERVE", 2: "TOPSPIN", 0: "SLICE", 1: "FLAT", 5: "SMASH"}
    df['SWING_TYPE'] = df['SWING_TYPE'].replace(swing_type)
    df['HAND_TYPE'] = df['HAND_TYPE'].replace(hand_type)
    
    conn.close()
    
    return df


# db_path = "/mnt/g/My Drive/FitnessData/SensorDownload/Sep14/"
db_path = "/mnt/chromeos/GoogleDrive/MyDrive/FitnessData/SensorDownload/Sep14/"
df = wrangle(db_path + "ZeppTennis.db")
corr = df.select_dtypes("number").corr()
# sns.heatmap(corr)
summary_stats = df[["SWING_TYPE",
                    "HAND_TYPE",
                    "IS_HIT_FRAME",
                    "SPIN",
                    "BALL_SPEED",
                    "HEAVINESS"]].describe()
pd.set_option('display.max_columns', 10)
print(summary_stats)
# plt.hist(df["BALL_SPEED"], bins=15)

# conn = sqlite3.connect(db_path + "OutZepp.db")
# df.to_sql('mytable', conn, if_exists='replace', index=False)

# Create a FacetGrid object
g = sns.FacetGrid(df, row="HAND_TYPE", col="SWING_TYPE", hue="IS_HIT_FRAME")


# Create a list of row and column labels
row_labels = df['HAND_TYPE'].unique()
col_labels = df['SWING_TYPE'].unique()


# Define a function to create a histogram
def histplot(data, **kwargs):
    sns.histplot(data=df, x="HEAVINESS")

# Map the histogram to the FacetGrid object
g.map_dataframe(histplot)

plt.show()

print(summary_stats)
