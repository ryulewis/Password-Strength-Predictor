# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Load the dataset from a CSV file
data = pd.read_csv("data.csv", error_bad_lines=False)

# Remove rows with missing values
data = data.dropna()

# Map numerical strength labels to corresponding categories
data["strength"] = data["strength"].map({0: "Weak", 1: "Medium", 2: "Strong"})

# Extract password texts and corresponding strength labels
x = np.array(data["password"])
y = np.array(data["strength"])

# Initialize a TF-IDF vectorizer with 'char' analyzer
tdif = TfidfVectorizer(analyzer='char')

# Transform password texts into TF-IDF features
X = tdif.fit_transform(x)

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize a Random Forest Classifier model
model = RandomForestClassifier()

# Train the model on the training data
model.fit(x_train, y_train)

# Calculate and print the accuracy of the model on the test data
accuracy = model.score(x_test, y_test)
print("Accuracy:", accuracy)

# Test a user-provided password for its strength
usrpwd = "password"
data = tdif.transform([usrpwd]).toarray()
pwd_strength = model.predict(data)

# Print the predicted strength of the user's password
print("Predicted Password Strength:", pwd_strength[0])
