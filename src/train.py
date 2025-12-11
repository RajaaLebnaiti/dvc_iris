import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import json
import pickle
df = pd.read_csv("data/clean_iris.csv")
X = df.drop("species", axis= 1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.2, random_state= 42)
hyper_pram = json.load(open("models/rfc_best_params.json"))

n_estimators = hyper_pram["n_estimators"]
max_depth = hyper_pram["max_depth"]
model = RandomForestClassifier(n_estimators, max_depth)
model.fit(X_train, y_train)
y_preds = model.predict(X_test)
acc = accuracy_score(y_test, y_preds)


# save model
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

# save metrics
with open("metrics.json", "w") as f:
    json.dump({"accuracy": acc}, f)

print("Model accuracy:", acc)