import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("database.csv", error_bad_lines=False)

data = data.dropna()

data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})

x = np.array(data["password"])
y = np.array(data["strength"])

tdif = TfidfVectorizer(analyzer='char')  # Use 'char' analyzer directly
X = tdif.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestClassifier()
model.fit(x_train, y_train)

accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

# Plotting Feature Importance
feature_importances = model.feature_importances_
feature_names = tdif.get_feature_names_out()

sorted_idx = np.argsort(feature_importances)[::-1]

plt.figure(figsize=(10, 6))
plt.bar(range(20), feature_importances[sorted_idx][:20], align="center")
plt.xticks(range(20), np.array(feature_names)[sorted_idx][:20], rotation=45, ha="right")
plt.xlabel("Feature")
plt.ylabel("Feature Importance")
plt.title("Top 20 Feature Importances")
plt.tight_layout()
plt.show()

usrpwd = "password"
data = tdif.transform([usrpwd]).toarray()
pwd_strength = model.predict(data)

print("Predicted Password Strength:", pwd_strength[0])
