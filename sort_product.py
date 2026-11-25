import json

# Load products
with open("products.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# List priority IDs in the exact order you want them to appear first
priority_ids = ["uno", "nano", "esp32", "tcrt", "hcsr04", "dht11", "l298n"]

items = data.get("items", [])

# Split into priority items and remaining items
featured = [item for item in items if item["id"] in priority_ids]
# Sort featured to match the exact order of priority_ids
featured.sort(key=lambda x: priority_ids.index(x["id"]))

others = [item for item in items if item["id"] not in priority_ids]

# Combine featured items at top + all other items
data["items"] = featured + others

# Save back to products.json
with open("products.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=4)

print("products.json reordered successfully!")