import json
import sqlite3

def extract_motions(json_file, session_counter):
    motions_data = []

    if "storage" in json_file:
        storage_values = json_file["storage"]
        if "mStorageValues" in storage_values:
            training_stats = storage_values["mStorageValues"]["TRAINING_STATISTICS"]
            if "mStorageArrayValues" in training_stats:
                motions = training_stats["mStorageArrayValues"]["motions"]
                stroke_counter = 1  # Initialize the stroke counter

                for motion in motions:
                    motion_data = {
                        "time": motion["mLongValues"]["time"],
                        "type": motion["mStringValues"]["type"],
                        "spin": motion["mStringValues"]["spin"],
                        "stroke_counter": stroke_counter,
                        "session_counter": session_counter,
                    }

                    # Extract KPI data if available
                    kpi = motion.get("mStorageValues", {}).get("kpi", {}).get("mStorageValues", {})

                    motion_data["style_score"] = kpi.get("style", {}).get("mFloatValues", {}).get("score")
                    motion_data["style_value"] = kpi.get("style", {}).get("mFloatValues", {}).get("value")
                    motion_data["effect_score"] = kpi.get("effect", {}).get("mFloatValues", {}).get("score")
                    motion_data["effect_value"] = kpi.get("effect", {}).get("mFloatValues", {}).get("value")
                    motion_data["speed_score"] = kpi.get("speed", {}).get("mFloatValues", {}).get("score")
                    motion_data["speed_value"] = kpi.get("speed", {}).get("mFloatValues", {}).get("value")

                    motions_data.append(motion_data)

                    # Check if we need to reset the stroke counter
                    if "mStorageArrayValues" in motion:
                        stroke_counter = 1
                    else:
                        stroke_counter += 1

    return motions_data

def extract_and_save_motions_list(json_list, database_filename):
    # Connect to the database.
    conn = sqlite3.connect(database_filename)
    
    # Create a cursor.
    c = conn.cursor()
    
    # Create a table to store the motions with separate columns for KPI values.
    c.execute('''
        CREATE TABLE IF NOT EXISTS motions (
          time INTEGER,
          type TEXT,
          spin TEXT,
          StyleScore REAL,
          StyleValue REAL,
          EffectScore REAL,
          EffectValue REAL,
          SpeedScore REAL,
          SpeedValue REAL,
          stroke_counter INTEGER,
          session_counter INTEGER
        )
    ''')

    # Commit the changes.
    conn.commit()

    # Initialize the session counter
    session_counter = 1

    for json_file in json_list:
        # Extract the motions from each JSON object in the list.
        motions = extract_motions(json_file, session_counter)

        if motions:
            # The JSON object had a `motions` field, so the `motions` list contains all of
            # the motions in the JSON object.
            print(motions)
        else:
            # The JSON object did not have a `motions` field, so the `motions` list is
            # empty.
            print("The JSON object did not have a `motions` field.")

        for motion in motions:
            time = motion.get("time", None)
            motion_type = motion["type"]
            spin = motion["spin"]
            style_score = motion.get("style_score", None)
            style_value = motion.get("style_value", None)
            effect_score = motion.get("effect_score", None)
            effect_value = motion.get("effect_value", None)
            speed_score = motion.get("speed_score", None)
            speed_value = motion.get("speed_value", None)
            stroke_counter = motion["stroke_counter"]
            session_counter = motion["session_counter"]

            c.execute('''
                INSERT INTO motions (time, type, spin, StyleScore, StyleValue, EffectScore, EffectValue, SpeedScore, SpeedValue, stroke_counter, session_counter)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (time, motion_type, spin, style_score, style_value, effect_score, effect_value, speed_score, speed_value, stroke_counter, session_counter))

        # Increment the session counter
        session_counter += 1

    # Commit the changes and close the connection.
    conn.commit()
    conn.close()

# Load the JSON data
with open("/home/blueaz/Python/JSONExtract/src/output.json") as f:
    json_list = json.load(f)

database_filename = "BabPopExt.db"
extract_and_save_motions_list(json_list, database_filename)
print(database_filename)
