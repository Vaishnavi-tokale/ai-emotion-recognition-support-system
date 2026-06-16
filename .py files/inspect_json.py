import json

JSON_FILE = "E:/archive/nslt_100.json"

with open(JSON_FILE, 'r') as f:
    data = json.load(f)

print("TYPE:", type(data))

# print first item
if isinstance(data, dict):
    first_key = list(data.keys())[0]
    print("\nFIRST KEY:", first_key)
    print("\nFIRST ENTRY:")
    print(data[first_key])

elif isinstance(data, list):
    print("\nFIRST ENTRY:")
    print(data[0])