#Traditional ML models
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Load the dataset
data = pd.read_csv('data1.csv', header=None)

# Separate features and target variable
X = data.iloc[1:, :-1]  # Assuming the features are in columns 0 to second-to-last
y = data.iloc[1:, -1]   # Assuming the target variable is in the last column

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Perform feature selection using SelectKBest with mutual information score
selector = SelectKBest(mutual_info_classif, k=15)  # Select 15 features
selector.fit(X_train, y_train)

# Get the selected features
selected_features = X.columns[selector.get_support()]

# Use only the selected features for training and testing
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

# Initialize classifiers
classifiers = {
    "SVM": SVC(),
    "k-NN": KNeighborsClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(),
    "Logistic Regression": LogisticRegression()
}

# Train and evaluate each classifier
for name, classifier in classifiers.items():
    # Train the classifier using the selected features
    classifier.fit(X_train_selected, y_train)

    # Predict the labels of the test data
    y_predicted = classifier.predict(X_test_selected)

    # Evaluate the classifier
    print(f"Classifier: {name}")
    print(confusion_matrix(y_test, y_predicted))
    print(classification_report(y_test, y_predicted))
    print("="*50)
