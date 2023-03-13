import json


with open('D:/EuroSAT/EuroSAT/label_map.json') as user_file:
  file_contents = user_file.read()
  
print(file_contents)
# parsed_json = json.loads(file_contents)
# print(parsed_json)