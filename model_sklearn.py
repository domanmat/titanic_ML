
import time
loading_time = time.time()
print("Loading modules...")
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.tree import plot_tree
import io
import os
import sys

# Start timing#
start_time = time.time()
print("Executing the code...")

# Load the df_processed
# file_path = r"C:\Users\Mateusz\Downloads\titanic\train.csv"
directory = r"C:\Users\Mateusz\Downloads\titanic"
train_file = "train.csv"
result_file = os.path.join(directory, 'result_train.csv')
path = os.path.join(directory, train_file)

data_frame = pd.read_csv(path)
# df_processed=data_frame.copy()

# Select relevant features_train
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
y = data_frame['Survived'].copy()


#Preprocessing the data_frame
def preprocess(data,features):
    # Preprocess the df_processed
    X = data[features].copy()
    # Convert Sex to numeric (male: 0, female: 1)
    X['Sex'] = X['Sex'].map({'male': 0, 'female': 1})

    # # Handle missing values in Age (fill with median)
    # X['Age'] = X['Age'].fillna(X['Age'].median())

    # Handle missing values in Pclass (fill with 0)
    X['Pclass'] = X['Pclass'].fillna(0)
    X['Sex'] = X['Sex'].fillna(-1)
    # Handle missing values in Age (fill with -1)
    X['Age'] = X['Age'].fillna(-1)
    X['SibSp'] = X['SibSp'].fillna(-1)
    X['Parch'] = X['Parch'].fillna(-1)
    # Handle missing values in Fare (fill with -1)
    X['Fare'] = X['Fare'].fillna(-1)

    # Ensure all features_train are numeric and handle any remaining missing values
    X = X.fillna(0)
    return(X)


X = preprocess(data_frame,features)

# Split data_frame into training and validation sets
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.5)

best_depth = 1
best_accuracy = 0
#iterate to get better statistics
for i in range(1):
    # Optimize max_depth
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=1)
    # X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=i)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    # print(X_train)
    for depth in range(1, 15):
        # train_model = DecisionTreeClassifier(max_depth=depth, criterion='gini', random_state=i)
        train_model = DecisionTreeClassifier(splitter='best',max_depth=depth, criterion='gini')
        train_model.fit(X_train, y_train)
        y_pred = train_model.predict(X_val)
        # print(best_depth)
        # print(len(y_pred))
        # print(y_pred)
        accuracy = accuracy_score(y_val, y_pred)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_depth = depth
            trained_model=train_model
    print(f"Current best max_depth: {best_depth} with validation accuracy: {best_accuracy:.4f}")

# Results from the optimization
print(f"Best max_depth: {best_depth} with validation accuracy: {best_accuracy:.4f}")
# Train final model with best max_depth
# final_model = DecisionTreeClassifier(max_depth=best_depth, criterion='gini', random_state=1)
best_depth=10
final_model = DecisionTreeClassifier(max_depth=best_depth, criterion='gini')
final_model.fit(X, y)
# final_model=trained_model


# Function to export tree as text with probabilities
def export_tree(model, feature_names):
    tree = model.tree_

    def recurse(node, depth, output):
        indent = "  " * depth
        if tree.feature[node] != -2:  # Not a leaf
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            output.append(
                f"{indent}{name} < {threshold:.2f} (gini {tree.impurity[node]:.4f}, samples {tree.n_node_samples[node]}, "
                f"class counts [{tree.value[node][0][0]:.0f}, {tree.value[node][0][1]:.0f}])")
            output.append(f"{indent}-- True:")
            recurse(tree.children_left[node], depth + 1, output)
            output.append(f"{indent}-- False:")
            recurse(tree.children_right[node], depth + 1, output)
        else:  # Leaf node
            total = sum(tree.value[node][0])
            prob_survived = tree.value[node][0][1] / total if total > 0 else 0
            prediction = 1 if prob_survived >= 0.5 else 0
            output.append(
                f"{indent}Leaf: Predict {prediction} (prob {prob_survived:.2f}, samples {tree.n_node_samples[node]}, "
                f"class counts [{tree.value[node][0][0]:.0f}, {tree.value[node][0][1]:.0f}])")

    output = []
    recurse(0, 0, output)
    return "\n".join(output)


# Print the decision tree
# print("Decision Tree Structure:")
# print(export_tree(final_model, features_train))
model_tree = (export_tree(final_model, features))
tree_file = os.path.join(directory, 'result_tree.txt')
with open(tree_file, "w") as txt:
    txt.write(export_tree(final_model, features))


# Function to plot and save the decision tree as a PDF
def save_tree_plot(model, feature_names, dir, filename):
    try:
        plt.figure(figsize=(30, 20))  # Set figure size for better readability
        plot_tree(model, feature_names=feature_names, class_names=['Died', 'Survived'], filled=True, rounded=True)
        output_path = os.path.join(dir, filename)
        plt.savefig(output_path, format='pdf', bbox_inches='tight')
        plt.close()
        print(f"Decision tree plot saved to {output_path}")
    except Exception as e:
        print(f"Error saving decision tree plot to {output_path}: {e}")


def example_prediction(model_test, PassengerId):
    row=PassengerId-1
    example_df=data_frame.loc[[row]] # data extracted for analysis
    example=data_frame.iloc[row] # data extracted
    # features_train = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    processed=preprocess(example_df,features)
    prediction = model_test.predict(processed)
    prob = model_test.predict_proba(processed)[0][1]

    print(f"\nExample Prediction:")
    # print(processed)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    print(example_df.head(1))
    print(f"Predicted: {'Survived' if prediction[0] == 1 else 'Died'} (probability of survival: {prob:.2f})")

    actual = example['Survived']
    actual_text = "SURVIVED" if actual == 1 else "DIED"
    correct = "✓ CORRECT" if prediction == actual else "✗ INCORRECT"
    print(f"Actual: {actual} ({actual_text}) - {correct}")

def final_test(model):
    # Final test on full data
    predictions = final_model.predict(X)
    full_accuracy = accuracy_score(y, predictions)

    # Summary of model efficiency
    # print(f"\nModel Efficiency Summary:")
    print(f"Final accuracy on full data: {full_accuracy:.4f}")

    # Create CSV with PassengerId and predicted Survived
    result_df = pd.DataFrame({
        'PassengerId': data_frame['PassengerId'],
        'Survived': predictions
    })

    # # Print the CSV content in terminal
    # print("\nPredicted Survival CSV:")
    # sys.stdout.write(result_df.to_csv(index=False))
    try:
        result_df.to_csv(result_file, index=False)
        print(f"\nPredictions saved to {result_file}")
    except Exception as e:
        print(f"\nError saving CSV to {result_file}: {e}")

# # Saving the tree plot - lasts long
# save_tree_plot(final_model, features_train, directory, 'plot_tree.pdf')
# # Calculating for an example
# example_prediction(final_model, 1)
final_test(final_model)
final_test(trained_model)

# End timing
end_time = time.time()
execution_time = end_time - start_time
t_execution_time = end_time - loading_time
print(f"\nCode execution time: {execution_time:.4f} seconds")
print(f"\nTotal execution time: {t_execution_time:.4f} seconds")
