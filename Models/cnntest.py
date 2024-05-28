#Convolutional Neural Network
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import classification_report

# Load the dataset
try:
    data = pd.read_csv('data1.csv')
except FileNotFoundError:
    print("File 'data1.csv' not found. Please provide the correct file path.")
    exit()
except Exception as e:
    print("Error loading the dataset:", e)
    exit()

# Separate features and target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Convert categorical labels to numerical labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Define a custom transformer to select numerical features and include target variable
class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, num_features):
        self.num_features = num_features

    def fit(self, X, y):
        self.selector = SelectKBest(score_func=f_classif, k=self.num_features)
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        return self.selector.transform(X)

# Define the preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', FeatureSelector(num_features=10), list(X.columns))  # Select top 10 features
    ])

# Define the model
model = Sequential([
    Dense(64, activation='relu', input_shape=(10,)),  # Adjust input shape based on selected features
    Dense(32, activation='relu'),
    Dense(len(label_encoder.classes_), activation='softmax')  # Output layer with number of classes
])

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
model.fit(preprocessor.fit_transform(X_train, y_train), y_train, epochs=200, batch_size=32, validation_split=0.2)

# Evaluate the model
# y_pred = np.argmax(model.predict(preprocessor.transform(X_test)), axis=-1)
# print(classification_report(y_test, y_pred))
