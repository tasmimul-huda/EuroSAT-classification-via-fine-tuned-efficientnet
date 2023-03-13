import json


with open('D:/EuroSAT/EuroSAT/label_map.json') as user_file:
  file_contents = user_file.read()
  
# print(file_contents)
dict_obj = json.loads(file_contents)

swapped_dict = {value: key for key, value in dict_obj.items()}
print(swapped_dict.keys())
# parsed_json = json.loads(file_contents)
# print(parsed_json)