import json
import sqlite3

# Connect to the SQLite database
conn = sqlite3.connect("example.db")
c = conn.cursor()

# Create a table
c.execute("""CREATE TABLE IF NOT EXISTS data (feelType text, playingPlace text,
    surfaceType text, type text, createdDate text, effectiveTime integer,
    fwVersion text, location text, nativeId text, piqScore integer,
    sensor text)""")

# Load the JSON data
with open("data.json") as f:
    data = json.load(f)

# Insert the data into the table
c.execute(
    "INSERT INTO data VALUES (:feelType, :playingPlace,\
    :surfaceType, :type, :createdDate, :effectiveTime,\
    :fwVersion, :location, :nativeId, :piqScore, :sensor)",
    data,
)

# Commit the changes and close the connection
conn.commit()
conn.close()
