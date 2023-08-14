import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd
from geneticalgorithm2 import geneticalgorithm2 as ga
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

#PREPROCESSING

#Read the dataset
df = pd.read_csv("olink_covid_outcome.csv", header =None)

#Take transpose of the dataset
new_df = df.T

#Drop attributes that has missing values more than 30%
new_df = new_df.dropna(thresh=len(new_df)*0.7, axis=1)

#Take column names
column_names = new_df.iloc[0]

#Take only proteins
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

# Update the original DataFrame with the new values
all_data = pd.concat([proteins, new_df.iloc[:, -21:]], axis=1)

#Getting attributes
df_values = all_data.iloc[:, 1:]

# Create new values for NaNs using KNN imputation
imputer = KNNImputer(n_neighbors=20)
new_values = imputer.fit_transform(df_values.iloc[1:])

# Update the original DataFrame with the new values
all_data.iloc[1:, 1:] = new_values

# Split the data into features (X) and labels (y)
X = all_data.iloc[:, 1:]
y = all_data.iloc[:, 0]

X = X.drop(0)
y = y.drop(0)

#Convert to float
X = X.astype(float)
y = y.astype(float)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=None)

#GAN Application

all_dead = X[X.iloc[:, 0] != 0].copy()

#Get attributes
df_values = all_dead

# Exclude the first row
df_values = df_values.iloc[1:]

# Function to generate synthetic data using a simple GAN
def generate_synthetic_data(num_samples, data_dim):
    generator = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(data_dim,)),
        layers.Dense(data_dim, activation='linear'),
    ])

    discriminator = tf.keras.Sequential([
        layers.Dense(128, activation='relu', input_shape=(data_dim,)),
        layers.Dense(1, activation='sigmoid'),
    ])

    gan = tf.keras.Sequential([generator, discriminator])

    gan.compile(loss='binary_crossentropy', optimizer='adam')

    real_data = np.random.randn(num_samples, data_dim)
    real_labels = np.ones((num_samples, 1))

    fake_data = generator.predict(np.random.randn(num_samples, data_dim))
    fake_labels = np.zeros((num_samples, 1))

    x_combined = np.vstack([real_data, fake_data])
    y_combined = np.vstack([real_labels, fake_labels])

    gan.fit(x_combined, y_combined, epochs=200, batch_size=num_samples)

    synthetic_data = generator.predict(np.random.randn(num_samples, data_dim))
    return synthetic_data

# Function to optimize the synthetic data using a genetic algorithm
def optimize_data(original_data, synthetic_data):
    def fitness_function(coefficients):
        # Scale the synthetic data using the coefficients
        scaled_data = synthetic_data * coefficients
        # Calculate the mean squared error between the original and scaled data
        mse = np.mean(np.square(original_data - scaled_data))
        return mse

    varbound = np.array([[0.5, 2.0]] * original_data.shape[1])  # Bounds for scaling coefficients

    algorithm_param = {'max_num_iteration': 100, 'population_size': 10, 'mutation_probability': 0.1,
                       'elit_ratio': 0.01, 'parents_portion': 0.3, 'crossover_probability': 0.5,
                       'max_iteration_without_improv': None}

    model = ga(function=fitness_function, dimension=original_data.shape[1], variable_type='real', variable_boundaries=varbound, algorithm_parameters=algorithm_param)
    model.run(no_plot=True)
    return model.output_dict['variable']

# Generating synthetic data
num_samples = 220
data_dim = 1107
synthetic_data = generate_synthetic_data(num_samples, data_dim)

#Original data
original_data_df = all_data[all_data.iloc[:, 0] != 0].copy()

#Exclude first column
original_data_df = original_data_df.iloc[:, 1:]

#Exclude first row
original_data = original_data_df.iloc[1:].values

# Optimizing the synthetic data using a genetic algorithm
scaling_coefficients = optimize_data(original_data, synthetic_data[:42])

# Scale the synthetic data using the obtained coefficients
scaled_synthetic_data = synthetic_data * scaling_coefficients

#Give new dead samples label "1"
new_column = np.ones((scaled_synthetic_data.shape[0], 1))
new_samples = np.concatenate((new_column, scaled_synthetic_data), axis=1)

all_data = all_data.iloc[1:]

alldata_num = all_data.to_numpy()

#Combine the numpy array with new samples
final = np.vstack((alldata_num, new_samples))

# Create the dataframe
final_df = pd.DataFrame(final, columns=column_names)

#Create an SVM model
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
