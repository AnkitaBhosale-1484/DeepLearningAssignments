'''Create a neural network model to predict whether a customer will leave a service.
Features:
Age
Monthly charges
Tenure
Number of complaints
Customer support calls'''

# ============================================================
# DEEP LEARNING ASSIGNMENT
# Customer Leave Prediction using MLPClassifier
# ============================================================


# ============================================================
# 1. CREATE THE DATASET
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
    [25, 500, 12, 1, 2],
    [30, 700, 24, 0, 1],
    [45, 1200, 6, 5, 8],
    [50, 1500, 5, 6, 10],
    [28, 600, 18, 1, 1],
    [35, 800, 30, 0, 0],
    [48, 1400, 4, 7, 9],
    [52, 1600, 3, 8, 12],
    [27, 550, 20, 0, 1],
    [42, 1300, 8, 4, 7]
])

# Target values
y = np.array([
    0, 0, 1, 1, 0,
    0, 1, 1, 0, 1
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
# 3. DISPLAY TARGET VALUES
# ============================================================

print("\n============================================================")
print("TARGET VALUES")
print("============================================================")

print("\nTarget Values:")
print(y)


# ============================================================
# 4. SEPARATE INDEPENDENT AND DEPENDENT VARIABLES
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
# 5. DIVIDE DATA INTO TRAINING AND TESTING DATA
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
# 6. APPLY FEATURE SCALING
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
# 7. DESIGN AN MLP WITH TWO HIDDEN LAYERS
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
# 8. TRAIN THE NETWORK
# ============================================================

print("\n============================================================")
print("MODEL TRAINING")
print("============================================================")

mlp.fit(X_train_scaled, y_train)

print("\nModel training completed successfully.")


# ============================================================
# 9. DISPLAY NUMBER OF ITERATIONS
# ============================================================

print("\n============================================================")
print("NUMBER OF ITERATIONS")
print("============================================================")

print("\nNumber of iterations required:")
print(mlp.n_iter_)


# ============================================================
# 10. CALCULATE TRAINING ACCURACY
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
# 11. CALCULATE TESTING ACCURACY
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
# 12. GENERATE CONFUSION MATRIX
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

plt.title("Customer Leave - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")

plt.xticks(
    [0, 1],
    ["Stay", "Leave"]
)

plt.yticks(
    [0, 1],
    ["Stay", "Leave"]
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
# 13. PLOT THE LOSS CURVE
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
# 14. CREATE A FUNCTION PredictCustomerLeave()
# ============================================================

print("\n============================================================")
print("PREDICTION FUNCTION")
print("============================================================")


def PredictCustomerLeave(customer_data):

    # Convert input into DataFrame
    customer_df = pd.DataFrame(
        customer_data,
        columns=[
            "Age",
            "MonthlyCharges",
            "Tenure",
            "Complaints",
            "SupportCalls"
        ]
    )

    # Scale input data
    customer_scaled = scaler.transform(customer_df)

    # Make prediction
    prediction = mlp.predict(customer_scaled)

    # Get probability
    probability = mlp.predict_proba(customer_scaled)

    if prediction[0] == 0:
        result = "Customer will stay"
    else:
        result = "Customer may leave"

    print("\nPrediction:", result)

    print(
        "Probability of Leaving:",
        probability[0][1]
    )

    return result


# ============================================================
# 15. TEST THE SYSTEM USING NEW CUSTOMER
# ============================================================

print("\n============================================================")
print("TESTING WITH NEW CUSTOMER")
print("============================================================")

new_customer = [
    [46, 1450, 5, 6, 9]
]

print("\nNew Customer:")
print(new_customer)

PredictCustomerLeave(new_customer)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n============================================================")
print("FINAL MODEL SUMMARY")
print("============================================================")

print("Training Accuracy:", training_accuracy * 100, "%")
print("Testing Accuracy :", testing_accuracy * 100, "%")
print("Iterations        :", mlp.n_iter_)

print("\nCustomer Leave Prediction System Completed!")