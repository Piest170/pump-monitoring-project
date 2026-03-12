import pandas as pd

# Load dataset
df = pd.read_csv("sensor.csv")
print("Original dataset size:", df.shape)

df_broken = df[df["machine_status"] == "BROKEN"]
df_normal = df[df["machine_status"] != "BROKEN"]
df_sample_normal = df_normal.sample(n=9993, random_state=42)
df_sample = pd.concat([df_broken, df_sample_normal])
print("Sample dataset size:", df_sample.shape)
print(df_sample["machine_status"].value_counts())

# Save
df_sample.to_csv("sensor_sample.csv", index=False)
print("New dataset saved as sensor_sample.csv")
