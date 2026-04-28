import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def train(csv_file, model_name):
    df = pd.read_csv(csv_file)

    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=150,
        random_state=42
    )

    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    acc = accuracy_score(y_test, pred)

    print(model_name, "Accuracy:", round(acc * 100, 2), "%")

    pickle.dump(model, open(model_name, "wb"))

train("features_isl.csv", "model_isl.pkl")
train("features_asl.csv", "model_asl.pkl")