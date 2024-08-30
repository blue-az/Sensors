import sqlite3
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Build your `wrangle` function here
def wrangle(db_path):
    # Connect to database
    conn = sqlite3.connect(db_path)

    # Construct query
    query = """
    SELECT _id, date, TYPE, TRACKID, ENDTIME, CAL, AVGHR, MAX_HR from TRACKRECORD
    """

    # Read query results into DataFrame
    df = pd.read_sql(query, conn, index_col="_id")
    # Remove HR outliers
    df = df[df["AVGHR"] > 50]
    df = df[df["MAX_HR"] > 50]
    # Create duration column from timestamps
    # Convert Unix timestamps to datetime objects
    df['TRACKID'] = pd.to_datetime(df['TRACKID'], unit='s')
    df['ENDTIME'] = pd.to_datetime(df['ENDTIME'], unit='s')
    
    # Calculate the duration in minutes
    df['duration_minutes'] = (df['ENDTIME'] - df['TRACKID']).dt.total_seconds() / 60
    df['duration_minutes'] = df['duration_minutes'].round()
    #remove duration outliersl
    df = df[df["duration_minutes"] > 10]

    #Replace type with sport 
    new_type = {16: "Free", 10: "IndCyc", 9: "OutCyc", 12: "Elliptical", 60: "Yoga", 14: "Swim" }
    df['TYPE'] = df['TYPE'].replace(new_type)
    
    return df


df = wrangle("/home/blueaz/Downloads/SensorDownload/Sep14/MiiFit.db")
print(df.columns)
print(df.select_dtypes("object").nunique())
print(df.head())
corr = df.select_dtypes("number").corr()
print(corr)
# sns.heatmap(corr)
summary_stats = df[["AVGHR","MAX_HR", "CAL"]].describe()
print(summary_stats)

# create time series object
y = df[["TRACKID", "CAL"]]
y = pd.DataFrame(y).set_index("TRACKID")

# plt.hist(df["MAX_HR"], bins=15)
# cmap = plt.cm.tab20(np.arange(len(df["TYPE"].unique())))
# plt.plot(np.array(df["TRACKID"]), np.array(df["MAX_HR"]))
# plt.scatter(x=df["AVGHR"], y=df["MAX_HR"], c=cmap[df["TYPE"]])
# plt.show()

# Convert the TYPE column to a sequence of integers
types_int, types_str = pd.factorize(df["TYPE"])

# Create a colormap for the TYPE column
cmap = plt.cm.tab20(np.arange(len(types_int)))

# Create the scatter plot
plt.scatter(x=df["AVGHR"], y=df["MAX_HR"], c=cmap[types_int])
plt.xlabel("AVGHR")
plt.ylabel("MAX_HR")
# Create the legend
# legend_elements = [plt.Line2D([0], [0], color=cmap[i], label=types_str[i]) for i in range(len(types_int))]
# plt.legend(legend_elements, loc="upper left")
# plt.legend(types_int)
plt.show()


print(df.head())
