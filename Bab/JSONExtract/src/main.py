import extract_json_objects
import json

# Get the path to the text file that contains the JSON files.
text_file = "/home/blueaz/Downloads/SensorDownload/Compare/log_13.06.24_191127.txt"

json_objects = extract_json_objects.extract_json_objects(text_file)


def write_json_objects(json_objects, output_file):
    with open(output_file, "w") as f:
        json.dump(json_objects, f)


def extract_fields(output_file, fields):
    extracted_data = []
    with open(output_file, "r") as f:
        json_objects = json.load(f)
        for json_object in json_objects:
            data = {}
            for field in fields:
                if field in json_object:
                    data[field] = json_object[field]
            extracted_data.append(data)
    return extracted_data


output_file = "output.json"
write_json_objects(json_objects, output_file)
fields = ["time", "piqScore", "type"]
extracted_data = extract_fields(output_file, fields)

print("complete")
