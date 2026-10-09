import pandas as pd
import numpy as np

df = pd.read_csv("insurance.csv")
print("shape:", df.shape)
print(df.dtypes)

# make the text columns consistent
df.columns = df.columns.str.strip().str.lower()
for col in ["sex", "smoker", "region"]:
    df[col] = df[col].str.strip().str.lower()
    print(col, df[col].unique())

print("missing values:")
print(df.isnull().sum())
df = df.dropna()

print("duplicates:", df.duplicated().sum())
df = df.drop_duplicates().reset_index(drop=True)

# rows that can't be real
bad = df[(df["age"] <= 0) | (df["age"] > 120) | (df["bmi"] <= 0) | (df["bmi"] > 80)
         | (df["children"] < 0) | (df["charges"] <= 0)]
print("impossible values:", len(bad))

df["smoker_bin"] = (df["smoker"] == "yes").astype(int)

# WHO bmi classes
df["bmi_category"] = pd.cut(df["bmi"], bins=[0, 18.5, 25, 30, 40, np.inf],
                            labels=["underweight", "normal", "overweight", "obese", "extreme_obese"],
                            right=False)

print("clean shape:", df.shape)
df.to_csv("clean_data.csv", index=False)
