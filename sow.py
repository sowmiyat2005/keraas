import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier

# Sample data as an alternative to loading from CSV
data = pd.DataFrame({
    'Age': [25, 30, 35, 40, 45],
    'Gender': ['Male', 'Female', 'Male', 'Female', 'Male'],
    'Subscription_Type': ['Basic', 'Premium', 'Basic', 'Premium', 'Basic'],
    'Account_Type': ['Free', 'Paid', 'Paid', 'Free', 'Paid'],
    'Churn': [0, 1, 0, 1, 0]  # 0 = not churned, 1 = churned
})

# Preprocessing the data
X = data.drop('Churn', axis=1)  # Features
y = data['Churn']  # Target variable

# Encode categorical variables if present
X = pd.get_dummies(X, drop_first=True)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the data (important for neural networks)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Build the neural network model
model = Sequential()

# Input layer (size = number of features)
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dropout(0.2))  # Dropout to prevent overfitting

# Hidden layer 1
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.2))

# Hidden layer 2
model.add(Dense(16, activation='relu'))

# Output layer (binary classification: churn or not churn)
model.add(Dense(1, activation='sigmoid'))

# Compile the model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

# Evaluate the model on the test set
y_pred_nn = (model.predict(X_test) > 0.5).astype("int32")

# Print classification metrics for the neural network model
print("Neural Network - Classification Report:")
print(classification_report(y_test, y_pred_nn))

# Print the confusion matrix for the neural network
print("Neural Network - Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_nn))

# Output the churn prediction for each customer
for i in range(len(X_test)):
    if y_pred_nn[i] == 1:
        print(f"Customer {i+1} churned")
    else:
        print(f"Customer {i+1} did not churn")