import json
import random

# Load your JSON file
with open(r'dataset/test.json', 'r') as f:
    data = json.load(f)

# Check the number of filenames
print(f"Original number of filenames: {len(data)}")

# Randomly select 7000 filenames
reduced_data = random.sample(data, 3000)

# Check the number of filenames after reduction
print(f"Reduced number of filenames: {len(reduced_data)}")

# Save the reduced list back to a JSON file
with open('test.json', 'w') as f:
    json.dump(reduced_data, f)

