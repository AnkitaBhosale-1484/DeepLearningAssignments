'''Create a neural network model to predict loan approval.
Features:
Applicant income
Credit score
Loan amount
Existing EMI
Employment status'''

# ============================================================
# DEEP LEARNING ASSIGNMENT
# Loan Approval Prediction using MLPClassifier
# ============================================================


# ============================================================
# 1.  CREATE THE DATASET
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix


# Create dataset
X = np.array([
    [25000, 600, 200000, 10000, 0],
    [40000, 700, 300000, 8000, 1],
    [60000, 750, 150000, 12000, 1],
    [20000, 550, 150000, 15000, 0],
    [80000, 700, 700000, 10000, 1],
    [35000, 650, 250000, 9000, 1],
    [18000, 500, 100000, 12000, 0],
    [90000, 850, 800000, 15000, 1],
    [30000, 580, 200000, 14000, 0],
    [70000, 780, 600000, 10000, 1]
])

# Target values
y = np.array([
    0, 1, 1, 0, 1,
    1, 0, 1, 0, 1
])


# ============================================================
# 2. DISPLAY DATASET INFORMATION
# ============================================================

print("\n============================================================")
print("DATASET INFORMATION")
print("============================================================")

print("\nDataset Shape:")
print(X.shape)

print("\nFirst 5 Records:")
print(X[:5])


# ============================================================
# 3. PREPROCESS CATEGORICAL VALUES
# ============================================================

print("\n============================================================")
print("PREPROCESSING CATEGORICAL VALUES")
print("============================================================")

print("\nEmployment Status:")
print("0 = Not Stable")
print("1 = Stable")

print("\nCategorical preprocessing completed.")


# ============================================================
# 4. DISPLAY TARGET VALUES
# ============================================================

print("\n============================================================")
print("TARGET VALUES")
print("============================================================")

print("\nTarget Values:")
print(y)


# ============================================================
# 5. SEPARATE INDEPENDENT AND DEPENDENT VARIABLES
# ============================================================

print("\n============================================================")
print("SEPARATING X AND Y")
print("============================================================")

# Independent variables
print("\nFeatures X:")
print(X)

# Dependent / target variable
print("\nTarget y:")
print(y)

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)


# ============================================================
# 6. DIVIDE DATA INTO TRAINING AND TESTING DATA
# ============================================================

print("\n============================================================")
print("TRAIN TEST SPLIT")
print("============================================================")

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# ============================================================
# 7. APPLY FEATURE SCALING
# ============================================================

print("\n============================================================")
print("FEATURE SCALING")
print("============================================================")

scaler = StandardScaler()

# Fit only on training data
X_train_scaled = scaler.fit_transform(X_train)

# Transform testing data
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed.")


# ============================================================
# 8. DESIGN AN MLP WITH TWO HIDDEN LAYERS
# ============================================================

print("\n============================================================")
print("MLP MODEL")
print("============================================================")

mlp = MLPClassifier(
    hidden_layer_sizes=(16, 8),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)

print("\nMLP Model:")
print(mlp)


# ============================================================
# 9. TRAIN THE NETWORK
# ============================================================

print("\n============================================================")
print("MODEL TRAINING")
print("============================================================")

mlp.fit(X_train_scaled, y_train)

print("\nModel training completed successfully.")


# ============================================================
# 10. DISPLAY NUMBER OF ITERATIONS
# ============================================================

print("\n============================================================")
print("NUMBER OF ITERATIONS")
print("============================================================")

print("\nNumber of iterations required:")
print(mlp.n_iter_)


# ============================================================
# 11. CALCULATE TRAINING ACCURACY
# ============================================================

print("\n============================================================")
print("TRAINING ACCURACY")
print("============================================================")

y_train_pred = mlp.predict(X_train_scaled)

training_accuracy = accuracy_score(
    y_train,
    y_train_pred
)

print("\nTraining Accuracy:")
print(training_accuracy)

print("\nTraining Accuracy Percentage:")
print(training_accuracy * 100)


# ============================================================
# TEST DATA PREDICTION
# ============================================================

y_test_pred = mlp.predict(X_test_scaled)


# ============================================================
# 12. CALCULATE TESTING ACCURACY
# ============================================================

print("\n============================================================")
print("TESTING ACCURACY")
print("============================================================")

testing_accuracy = accuracy_score(
    y_test,
    y_test_pred
)

print("Testing Accuracy:", testing_accuracy)
print("Testing Accuracy Percentage:", testing_accuracy * 100)


# ============================================================
# 13. GENERATE CONFUSION MATRIX
# ============================================================

print("\n============================================================")
print("CONFUSION MATRIX")
print("============================================================")

cm = confusion_matrix(
    y_test,
    y_test_pred
)

print("\nConfusion Matrix:")
print(cm)


# Plot confusion matrix
plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Loan Approval - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Rejected", "Approved"]
)

plt.yticks(
    [0, 1],
    ["Rejected", "Approved"]
)

for i in range(2):
    for j in range(2):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()
plt.show()


# ============================================================
# 14. PLOT THE LOSS CURVE
# ============================================================

print("\n============================================================")
print("LOSS CURVE")
print("============================================================")

plt.figure(figsize=(8, 5))

plt.plot(
    mlp.loss_curve_
)

plt.title("MLP Training Loss Curve")
plt.xlabel("Iterations")
plt.ylabel("Loss")

plt.grid()

plt.show()


# ============================================================
# 15. CREATE A FUNCTION PredictLoanApproval()
# ============================================================

print("\n============================================================")
print("PREDICTION FUNCTION")
print("============================================================")


def PredictLoanApproval(applicant_data):

    # Convert input into DataFrame
    applicant_df = pd.DataFrame(
        applicant_data,
        columns=[
            "Income",
            "CreditScore",
            "LoanAmount",
            "ExistingEMI",
            "EmploymentStatus"
        ]
    )

    # Scale input data
    applicant_scaled = scaler.transform(applicant_df)

    # Make prediction
    prediction = mlp.predict(applicant_scaled)

    # Get probability
    probability = mlp.predict_proba(applicant_scaled)

    if prediction[0] == 0:
        result = "Loan Rejected"
    else:
        result = "Loan Approved"

    print("\nPrediction:", result)

    print(
        "Probability of Approval:",
        probability[0][1]
    )

    return result


# ============================================================
# 16. TEST THE SYSTEM USING NEW APPLICANT
# ============================================================

print("\n============================================================")
print("TESTING WITH NEW APPLICANT")
print("============================================================")

new_applicant = [
    [55000, 720, 400000, 10000, 1]
]

print("\nNew Applicant:")
print(new_applicant)

PredictLoanApproval(new_applicant)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL MODEL SUMMARY")
print("============================================================")

print("Training Accuracy:", training_accuracy * 100, "%")
print("Testing Accuracy :", testing_accuracy * 100, "%")
print("Iterations        :", mlp.n_iter_)

print("\nLoan Approval Prediction System Completed!")