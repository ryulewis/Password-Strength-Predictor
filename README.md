# Password Strength Predictor using Random Forest Classifier

This repository contains a Python script that utilizes the Random Forest Classifier to predict the strength of passwords based on their character composition. The script uses the **scikit-learn** library for creating the classifier and the **pandas** library for data manipulation.

## Requirements

Before using the script, make sure you have the following libraries installed:

- pandas
- numpy
- scikit-learn

You can install them using the following command:

```bash
pip install pandas numpy scikit-learn
```

## Usage

1. Clone this repository to your local machine or download the script directly.

2. Prepare your password dataset in a CSV format similar to the `data.csv` file used in the script. The dataset should have a column named "password" containing the passwords and a column named "strength" with corresponding strength labels (0 for Weak, 1 for Medium, 2 for Strong).

3. Update the script to point to your dataset CSV file:

   ```python
   data = pd.read_csv("your_dataset.csv", error_bad_lines=False)
   ```

4. Run the script using your Python interpreter:

   ```bash
   python password_strength_predictor.py
   ```

5. The script will perform the following steps:

   - Load the dataset from the CSV file and drop any rows with missing values.
   - Map the numeric password strength labels to human-readable labels.
   - Transform the password text data using the TF-IDF vectorizer.
   - Split the dataset into training and testing sets.
   - Train a Random Forest Classifier on the training data.
   - Calculate and display the accuracy of the model on the testing data.
   - Prompt for a password input and predict its strength using the trained model.

## Disclaimer

Keep in mind that password strength prediction is a complex task and may not be perfectly accurate in all cases. This script serves as a demonstration of using machine learning for password strength estimation and should not be used as the sole criteria for assessing password security.
