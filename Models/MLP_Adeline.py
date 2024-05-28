#MLP trained with adeline which uses SGD optimizer
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('data1.csv')

# Separate features and target
X = data.iloc[:, :-1]  # Features
y = data.iloc[:, -1]   # Target

# Perform feature selection
k_best_features = 10  # Number of features to select
selector = SelectKBest(score_func=f_classif, k=k_best_features)
X_selected = selector.fit_transform(X, y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=40)

# Initialize and train the Adaline Classifier
adaline = SGDClassifier(loss='perceptron', learning_rate='constant', eta0=0.02, max_iter=1000, random_state=40)
adaline.fit(X_train, y_train)

# Predict labels for the test set
y_pred = adaline.predict(X_test)

# Print classification report
print(classification_report(y_test, y_pred))
