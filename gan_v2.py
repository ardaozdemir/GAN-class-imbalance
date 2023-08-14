#GAN Version 2

#Use different learning rates for the discriminator and the generator.
#Introduce some noise to the labels when training the discriminator.
#Train the generator more often. (Train generator 2 times more than discriminator)

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

#PREPROCESSING

#Read the dataset
df = pd.read_csv("olink_covid_outcome.csv", header =None)

#Take transpose of the dataset
new_df = df.T

#Drop attributes that has missing values more than 30%
new_df = new_df.dropna(thresh=len(new_df)*0.7, axis=1)

#Take column names
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

#Create dataframe with scaled proteins and other features
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=None)

#GAN Application

all_dead = X_train[X_train.iloc[:, 0] != 0].copy()

#Get attributes
df_values = all_dead

# Exclude the first row
df_values = df_values.iloc[1:]

#Convert df to numpy
data = df_values.to_numpy()
data = np.asarray(data).astype('float32')

g_losses = []
d_losses = []

# Define the discriminator model
# Use different learning rates for the discriminator and generator
d_optimizer = Adam(lr=0.0002)
g_optimizer = Adam(lr=0.0001)

def define_discriminator(n_inputs=data.shape[1]):
    model = Sequential()
    model.add(Dense(25, kernel_initializer='he_uniform', input_dim=n_inputs))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=d_optimizer, metrics=['accuracy'])
    return model

def define_generator(latent_dim, n_outputs=data.shape[1]):
    model = Sequential()
    model.add(Dense(15, kernel_initializer='he_uniform', input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.01))
    model.add(Dense(n_outputs, activation='linear'))
    return model

def define_gan(generator, discriminator):
    discriminator.trainable = False
    model = Sequential()
    model.add(generator)
    model.add(discriminator)
    model.compile(loss='binary_crossentropy', optimizer=g_optimizer)
    return model

def train(g_model, d_model, gan_model, data, latent_dim, n_epochs=200, n_batch=128):
    half_batch = int(n_batch / 2)
    for i in range(n_epochs):
        # Get randomly selected 'real' samples
        X_real = data[np.random.randint(0, data.shape[0], half_batch)]
        y_real = np.ones((half_batch, 1))
        # Introduce noise to the labels
        y_real += 0.05 * np.random.normal(size=y_real.shape)

        # Update discriminator model weights
        d_loss_real = d_model.train_on_batch(X_real, y_real)

        # Generate 'fake' examples
        X_fake = g_model.predict(np.random.normal(size=(half_batch, latent_dim)))
        y_fake = np.zeros((half_batch, 1))
        # Introduce noise to the labels
        y_fake += 0.05 * np.random.normal(size=y_fake.shape)

        # Update discriminator model weights
        d_loss_fake = d_model.train_on_batch(X_fake, y_fake)

        # Prepare points in latent space as input for the generator
        X_gan = np.random.normal(size=(n_batch, latent_dim))
        y_gan = np.ones((n_batch, 1))
        # Update the generator via the discriminator's error
        g_loss = gan_model.train_on_batch(X_gan, y_gan)

        # Train the generator twice for every time you train the discriminator
        if i % 2 == 0:
            g_loss = gan_model.train_on_batch(X_gan, y_gan)

        g_losses.append(g_loss)
        d_losses.append(0.5 * np.add(d_loss_real, d_loss_fake))

        # Print the losses
        print(  f"Epoch {i + 1}/{n_epochs} - Discriminator Loss: {0.5 * np.add(d_loss_real, d_loss_fake)}, Generator Loss: {g_loss}")


# Size of the latent space
latent_dim = 5
# Create the discriminator
discriminator = define_discriminator()
# Create the generator
generator = define_generator(latent_dim)
# Create the GAN
gan = define_gan(generator, discriminator)
# Train the GAN
train(generator, discriminator, gan, data, latent_dim)

# Generate new samples
num_samples = 110
new_samples = generator.predict(np.random.normal(size=(num_samples, latent_dim)))

#Give new dead samples label "1"
new_column = np.ones((new_samples.shape[0], 1))
new_samples = np.concatenate((new_column, new_samples), axis=1)

all_data = all_data.iloc[1:]
alldata_num = all_data.to_numpy()

#Combine the numpy array with new samples
final = np.vstack((alldata_num, new_samples))

# Create the dataframe
final_df = pd.DataFrame(final, columns=column_names)

# Print the dataframe
#print(final_df.tail(n=15))

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

plt.figure(figsize=(10,5))
plt.title("Generator and Discriminator Loss During Training")
plt.plot(g_losses,label="G")
plt.plot([subarray[0] for subarray in d_losses],label="D")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

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



