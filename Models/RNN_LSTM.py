#Recurrent Neural Network implemented with Long-Short Term Memory cells
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report
from tensorflow.keras.utils import plot_model

# Load the dataset
data = pd.read_csv('data1.csv')

# Separate features and target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Encode string labels into numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Feature selection
feature_selector = SelectKBest(score_func=f_classif, k=10)  # Select top 10 features
X_train_selected = feature_selector.fit_transform(X_train, y_train)
X_test_selected = feature_selector.transform(X_test)

# Reshape data for LSTM input (samples, timesteps, features)
X_train_selected = np.reshape(X_train_selected, (X_train_selected.shape[0], 1, X_train_selected.shape[1]))
X_test_selected = np.reshape(X_test_selected, (X_test_selected.shape[0], 1, X_test_selected.shape[1]))

# Define the model
model = Sequential([
    LSTM(64, input_shape=(X_train_selected.shape[1], X_train_selected.shape[2])),
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with number of classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train_selected, y_train, epochs=250, batch_size=32, validation_split=0.2)

# Evaluate the model
#y_pred = np.argmax(model.predict(X_test_selected), axis=-1)
#print(classification_report(y_test, y_pred))
# Visualize the model
plot_model(model, to_file='model_structure.png', show_shapes=True, show_layer_names=True)