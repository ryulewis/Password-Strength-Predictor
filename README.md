# Password Strength Prediction using Random Forest Classifier

This repository contains a Python script that demonstrates how to predict the strength of passwords using a Random Forest Classifier based on their character compositions. The script uses the `pandas`, `numpy`, `sklearn`, and `matplotlib` libraries for data manipulation, machine learning, and visualization.

## Prerequisites

Before running the script, make sure you have the following prerequisites installed:

- Python 3.x
- Pandas
- NumPy
- Scikit-learn (sklearn)
- Matplotlib

You can install these libraries using the following command:

```bash
pip install pandas numpy scikit-learn matplotlib
```

## Usage

1. Clone or download this repository to your local machine.

2. Place a CSV file named `database.csv` containing password data with a "password" column and a "strength" column (0 for Weak, 1 for Medium, 2 for Strong) in the same directory as the script.

3. Run the script using the following command:

```bash
python password_strength_prediction.py
```

## Script Explanation

The script performs the following steps:

1. Import necessary libraries:

```python
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
```

2. Read and preprocess data:

```python
data = pd.read_csv("selected.csv", error_bad_lines=False)
data = data.dropna()
data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})
```

3. Perform TF-IDF vectorization:

```python
x = np.array(data["password"])
y = np.array(data["strength"])
tdif = TfidfVectorizer(analyzer='char')
X = tdif.fit_transform(x)
```

4. Split data into training and testing sets:

```python
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

5. Train a Random Forest Classifier:

```python
model = RandomForestClassifier()
model.fit(x_train, y_train)
```

6. Evaluate the model's accuracy:

```python
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)
```

7. Plot the top 20 feature importances:

```python
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
```

8. Predict the strength of a user-provided password:

```python
usrpwd = "password"
data = tdif.transform([usrpwd]).toarray()
pwd_strength = model.predict(data)
print("Predicted Password Strength:", pwd_strength[0])
```

## Disclaimer

This script provides a basic demonstration of password strength prediction using a Random Forest Classifier. However, it's important to note that real-world password strength assessment involves more complex considerations, including dictionary attacks, common patterns, and other security measures. Always use strong and unique passwords to enhance your online security.

## Conclusion
In conclusion, this script showcases a practical approach to predicting password strength using a Random Forest Classifier and character-based TF-IDF vectorization. By leveraging machine learning techniques, the script offers insights into the underlying features that contribute to password strength assessment. However, it's essential to emphasize that this demonstration is a simplified illustration for educational purposes. Real-world password security is a multifaceted challenge that involves various factors beyond character composition. For robust online security, individuals should follow best practices such as using strong, unique passwords, enabling multi-factor authentication, and staying informed about evolving cybersecurity threats.
