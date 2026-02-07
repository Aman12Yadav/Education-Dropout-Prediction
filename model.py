import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

def train_model():
    # Load dataset (semicolon separated)
    df = pd.read_csv("data.csv", sep=";")

    # Encode target
    le = LabelEncoder()
    df["Status"] = le.fit_transform(df["Status"])

    X = df.drop("Status", axis=1)
    y = df["Status"]

    # 🔹 NEW: calculate mean values for each feature
    feature_means = X.mean().to_dict()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 🔹 UPDATED RETURN
    return model, X.columns.tolist(), le, feature_means
