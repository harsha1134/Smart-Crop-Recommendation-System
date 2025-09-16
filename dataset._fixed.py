import pandas as pd

# Load your dataset
df = pd.read_csv("Crop_recommendation_upgraded_fixed.csv")

# Step 1: Standardize text columns
for col in ["Soil Type", "Season", "Region"]:
    if col in df.columns:
        df[col] = df[col].str.title().str.strip()

# Step 2: Expand crop labels
mapping = {
    "Vegetables": ["Tomato", "Potato", "Onion", "Cabbage", "Brinjal"],
    "Cereals": ["Rice", "Wheat", "Maize", "Barley", "Millets"],
    "Pulses": ["Green Gram", "Black Gram", "Red Gram"],
    "Cash Crops": ["Cotton", "Sugarcane", "Tobacco", "Groundnut"],
    "Oilseeds": ["Mustard", "Sunflower", "Soybean", "Sesame"],
    "Fruits": ["Mango", "Banana", "Papaya", "Guava"]
}

import random
df["Label"] = df["Label"].apply(lambda x: random.choice(mapping[x]) if x in mapping else x)

# Step 3: Save cleaned dataset
df.to_csv("Crop_recommendation_cleaned.csv", index=False)
print("✅ Cleaned dataset saved as Crop_recommendation_cleaned.csv")
