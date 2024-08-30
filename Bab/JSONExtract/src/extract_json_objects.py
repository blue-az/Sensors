import json


def extract_json_objects(text_file):
    json_objects = []
    recording = False
    json_str = ''
    t = 0
    with open(text_file, 'r') as f:
        for line in f:
            if 'json:' in line:
                recording = True
            if recording:
                json_str += line
                if '6eb361ebee6a"}' or '}]}' in line:
                    try:
                        start_index = json_str.index('{')
                        end_index = json_str.index('6eb361ebee6a"}') + 15
                        json_object =\
                            json.loads(json_str[start_index:end_index])
                        json_objects.append(json_object)
                    except (json.JSONDecodeError, ValueError):
                        t += 1
                        pass
                    recording = False
                    json_str = ''
    return json_objects
