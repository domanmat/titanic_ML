import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import io

# Load the df_processed
file_path = r"C:\Users\Mateusz\Downloads\titanic\train.csv"
data_frame = pd.read_csv(file_path)
# df_processed=data_frame.copy()

# Preprocess the df_processed
# Select relevant features
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
X = data_frame[features].copy()
y = data_frame['Survived'].copy()

# Convert Sex to numeric (male: 0, female: 1)
X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

# # Handle missing values in Age (fill with median)
# X['Age'] = X['Age'].fillna(X['Age'].median())
X['Pclass'] = X['Pclass'].fillna(0)
X['Sex'] = X['Sex'].fillna(-1)
# Handle missing values in Age (fill with -1)
X['Age'] = X['Age'].fillna(-1)
X['SibSp'] = X['SibSp'].fillna(-1)
X['Parch'] = X['Parch'].fillna(-1)
# Handle missing values in Fare (fill with -1)
X['Fare'] = X['Fare'].fillna(-1)

# Ensure all features are numeric and handle any remaining missing values
X = X.fillna(0)

# Split data_frame into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Optimize max_depth
best_depth = 1
best_accuracy = 0
for depth in range(1, 11):
    model = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_depth = depth

# Train final model with best max_depth
final_model = DecisionTreeClassifier(max_depth=best_depth, criterion='gini', random_state=42)
final_model.fit(X, y)

# Function to export tree as text with probabilities
def export_tree(model, feature_names):
    tree = model.tree_
    def recurse(node, depth, output):
        indent = "  " * depth
        if tree.feature[node] != -2:  # Not a leaf
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            output.append(f"{indent}{name} < {threshold:.2f} (gini {tree.impurity[node]:.4f}, samples {tree.n_node_samples[node]}, class counts [{tree.value[node][0][0]:.0f}, {tree.value[node][0][1]:.0f}])")
            output.append(f"{indent}-- True:")
            recurse(tree.children_left[node], depth + 1, output)
            output.append(f"{indent}-- False:")
            recurse(tree.children_right[node], depth + 1, output)
        else:  # Leaf node
            total = sum(tree.value[node][0])
            prob_survived = tree.value[node][0][1] / total if total > 0 else 0
            prediction = 1 if prob_survived >= 0.5 else 0
            output.append(f"{indent}Leaf: Predict {prediction} (prob {prob_survived:.2f}, samples {tree.n_node_samples[node]}, class counts [{tree.value[node][0][0]:.0f}, {tree.value[node][0][1]:.0f}])")
    output = []
    recurse(0, 0, output)
    return "\n".join(output)

# Print the decision tree
print(f"Best max_depth: {best_depth} with validation accuracy: {best_accuracy:.4f}")
print("Decision Tree Structure:")
print(export_tree(final_model, features))

# Example prediction
example_passenger = pd.DataFrame({
    'Pclass': [3], 'Sex': [0], 'Age': [25], 'SibSp': [0], 'Parch': [0], 'Fare': [7.25]
})
prediction = final_model.predict(example_passenger)
prob = final_model.predict_proba(example_passenger)[0][1]
print(f"\nExample Prediction (Pclass=3, Sex=male, Age=25, SibSp=0, Parch=0, Fare=7.25):")
print(f"Predicted: {'Survived' if prediction[0] == 1 else 'Died'} (probability of survival: {prob:.2f})")