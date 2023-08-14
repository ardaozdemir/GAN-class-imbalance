import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

#PREPROCESSING

#Read the dataset
df = pd.read_csv("olink_covid_outcome.csv", header =None)

#Take transpose of the dataset
new_df = df.T

#Drop attributes that has missing values more than 30%
new_df = new_df.dropna(thresh=len(new_df)*0.7, axis=1)

#Get column names
column_names = new_df.iloc[0]

#Take only proteins for scaling
proteins = new_df.iloc[:, :-21]

# Exclude the first column
data_to_scale = proteins.iloc[:, 1:]

# Exclude the first row from the data to scale
data_to_scale = data_to_scale.iloc[1:]

# Apply standard scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_scale)

# Update the original DataFrame with the scaled values
proteins.iloc[1:, 1:] = scaled_data

#Combining scaled and other values
all_data = pd.concat([proteins, new_df.iloc[:, -21:]], axis=1)

#Getting attributes
df_values = all_data.iloc[:, 1:]

# Create new values for NaNs using KNN imputation
imputer = KNNImputer(n_neighbors=20)
new_values = imputer.fit_transform(df_values.iloc[1:])

# Update the original DataFrame with the new values
all_data.iloc[1:, 1:] = new_values

all_data = all_data.iloc[1:]

#Get label and features
y = all_data.iloc[:, 0]
X = all_data.iloc[:, 1:]

#Convert to float
X = X.astype(float)
y = y.astype(float)

#Train-test-split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)

#Create SMOTE object
sm = SMOTE(random_state=None)

#Create new samples
X_res, y_res = sm.fit_resample(X_train, y_train)

balanced_df = pd.concat([pd.DataFrame(y_res), pd.DataFrame(X_res)], axis=1)

#Create SVM model
svm = SVC()

#Get label and features
X_train_new = balanced_df.iloc[:, 1:]
y_train_new = balanced_df.iloc[:, 0]

#Convert to float
X_train_new = X_train_new.astype(float)
y_train_new = y_train_new.astype(float)

# Fit the model on the training data
svm.fit(X_train_new.values, y_train_new)

# Make predictions on the testing data
y_pred = svm.predict(X_test)

print("SVM Result")
# Calculate and print accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Calculate and print F1 score
f1 = f1_score(y_test, y_pred)
print("F1 Score:", f1)

