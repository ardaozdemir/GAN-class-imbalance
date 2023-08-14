#Conditional Tabular GAN Application

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np
from sklearn.model_selection import train_test_split
from ctgan import CTGAN
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score

#PREPROCESSING

#Read the dataset
df = pd.read_csv("olink_covid_outcome.csv", header =None)

#Take transpose of the dataset
new_df = df.T

#Drop attributes that has missing values more than 30%
new_df = new_df.dropna(thresh=len(new_df)*0.7, axis=1)

column_names = new_df.iloc[0]

#Take only proteins for scaling
proteins = new_df.iloc[:, :-21]

#column_names = proteins.iloc[0]

# Exclude the first column
data_to_scale = proteins.iloc[:, 1:]

# Exclude the first row from the data to scale
data_to_scale = data_to_scale.iloc[1:]

# Apply standard scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data_to_scale)

# Update the original DataFrame with the scaled values
proteins.iloc[1:, 1:] = scaled_data

#Create dataframe with scaled proteins and other features
all_data = pd.concat([proteins, new_df.iloc[:, -21:]], axis=1)

#Getting attributes
df_values = all_data.iloc[:, 1:]

# Create new values for NaNs using KNN imputation excluding the first row
imputer = KNNImputer(n_neighbors=20)
new_values = imputer.fit_transform(df_values.iloc[1:])

# Update the original DataFrame with the new values
all_data.iloc[1:, 1:] = new_values

# Split the data into features (X) and labels (y)
X = all_data.iloc[:, 1:]  # Assuming features start from column 1
y = all_data.iloc[:, 0]   # Assuming labels are in column 0

X = X.drop(0)
y = y.drop(0)

#Conver to float
X = X.astype(float)
y = y.astype(float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)

#GAN Application

all_dead = X_train[X_train.iloc[:, 0] != 0].copy()

# Drop the first row since it's now the column names
all_dead = all_dead[1:]

#alldead_ct = all_dead.iloc[:, 1:]
alldead_ct = all_dead

#Take column names
column_names_new = column_names[1:]

#Create CTGAN model
model = CTGAN(epochs=50, verbose=True)
model.fit(alldead_ct.values)

# Create synthetic data
synthetic_data = model.sample(220)

#Give new dead samples label "1"
new_column = np.ones((synthetic_data.shape[0], 1))
new_samples = np.concatenate((new_column, synthetic_data), axis=1)


all_data = all_data.iloc[1:]
alldata_num = all_data.to_numpy()

#Combine the numpy array with new samples
final = np.vstack((alldata_num, new_samples))

# Create the dataframe
final_df = pd.DataFrame(final, columns=column_names)

# Create an SVM model
svm = SVC()

#Get label and features
X_train_new = final_df.iloc[:, 1:]
y_train_new = final_df.iloc[:, 0]

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

# Print the predicted labels
#print("Predicted Labels:", y_pred)

from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

final_df = final_df.drop(final_df[final_df['Patient outcome'] == 0].index)

#Original dead samples (there was only 42 at first)
df1 = final_df.iloc[:42]
df1 = df1.drop(df1.columns[0], axis=1)

#GAN dead samples
df2 = final_df.iloc[42:84]
df2 = df2.drop(df2.columns[0], axis=1)

#Create TSNE graph for 2 different data
tsne = TSNE(n_components=2, random_state=42, perplexity=5)

tsne_result1 = tsne.fit_transform(df1)
tsne_result2 = tsne.fit_transform(df2)

plt.figure(figsize=(10, 5))
plt.scatter(tsne_result1[:, 0], tsne_result1[:, 1], label='Original')
plt.scatter(tsne_result2[:, 0], tsne_result2[:, 1], label='GAN Generated')
plt.title('t-SNE Graph Comparison')
plt.legend()
plt.show()