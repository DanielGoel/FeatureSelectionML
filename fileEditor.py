
import pandas as pd

# Load the dataset
df = pd.read_csv("GaspipelineDatasets/NewGasFilteredcommandMinMax.csv")

# ✅ Define the mapping of old values to new values
label_mapping = {3: 1, 4: 2, 5: 3, 6: 3, 7: 5}

# ✅ Apply the mapping to the 'Label' column
df["Label"] = df["Label"].replace(label_mapping)

# ✅ Save the modified dataset
df.to_csv("GaspipelineDatasets/NewGasFilteredCommandMinMax_Remapped.csv", index=False)

print("✅ Label values updated and file saved!")
