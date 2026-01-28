import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import pickle
from sklearn.preprocessing import StandardScaler



df = pd.read_csv("Disease_Prediction_Symptoms_1000.csv")

le = LabelEncoder()
df["Disease"] = le.fit_transform(df["Disease"])

X = df.drop("Disease", axis=1)
y = df["Disease"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = SVC(
    C=1.0,
    kernel="rbf",
    degree=3,
    gamma="scale",
)

model.fit(X_train_scaled, y_train)

print("Accuracy:", model.score(X_test, y_test))


pickle.dump(model, open("disease_model.pkl", "wb"))
pickle.dump(le, open("label_encoder.pkl", "wb"))
pickle.dump(scaler, open("scaler.pkl", "wb"))

print("model saved successfully")

