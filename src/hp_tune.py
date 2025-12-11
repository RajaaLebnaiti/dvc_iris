import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import json

df = pd.read_csv("data/clean_iris.csv")
X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

model = RandomForestClassifier()

param_grid = json.load(open("configs/hp_config.json", "r"))

grid_search = GridSearchCV(model, param_grid, cv= 5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

y_preds = grid_search.predict(X_test)
acc = accuracy_score(y_test, y_preds)


with open("models/rfc_best_params", "w") as outfile:
    json.dump(best_params, outfile)