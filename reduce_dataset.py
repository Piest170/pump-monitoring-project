import pandas as pd

# Load original dataset
df = pd.read_csv("sensor.csv")

print("Original dataset size:", df.shape)

# แยก broken ออกมา
df_broken = df[df["machine_status"] == "BROKEN"]

# แยก normal + recovering
df_normal = df[df["machine_status"] != "BROKEN"]

# สุ่ม normal ให้เหลือประมาณ 9993 rows
df_sample_normal = df_normal.sample(n=9993, random_state=42)

# รวม dataset
df_sample = pd.concat([df_broken, df_sample_normal])

print("Sample dataset size:", df_sample.shape)
print(df_sample["machine_status"].value_counts())

# Save
df_sample.to_csv("sensor_sample.csv", index=False)

print("New dataset saved as sensor_sample.csv")