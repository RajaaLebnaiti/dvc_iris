import pandas as pd

df = pd.read_csv("data/iris.csv")

df = df[df["sepal_length"]< 5]
df.to_csv("data/clean_iris.csv", index= False)

print("Preprocessing Finished")