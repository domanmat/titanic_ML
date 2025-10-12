import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations
GREEN = "\033[32m"
RESET = "\033[0m"

import time
# Start timing
start_time = time.time()
print("Executing the code...")

total_score=0
def counter(survived, len_df, survival_r):
    """
    Add a value to the global counter variable.
    """
    global total_score  # Declare that we're using the global variable
    if survival_r >= 0.50:
        total_score += survived
    else:
        total_score += len_df-survived
    return total_score

def inspect_data(df, detailed=True):
    """
    Inspect the dataset for missing df_processed.

    Parameters:
    - df: pandas DataFrame to inspect
    - detailed: bool, if True shows full analysis, if False shows only summary
    """

    if detailed:
        print("=" * 60 )
        # Display basic information about the dataset
        print("Dataset Shape:", df.shape)
        print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
        print("=" * 60)

        # Display first 10 rows with all columns
        print("First 10 rows of the dataset:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.head(10))
        print("=" * 60)

        # Missing df_processed analysis as a single row with column headers
        print("Missing Data Analysis:")
        print("-" * 60)

        missing_count = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

        # Create DataFrame and transpose so columns are headers
        missing_analysis = pd.DataFrame({
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_pct
        }).T

        print(missing_analysis)
        print("=" * 60)

        # Summary statistics
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        print(f"Total cells in dataset: {total_cells}")
        print(f"Total missing cells: {total_missing}")
        print(f"Overall missing percentage: {(total_missing / total_cells * 100):.2f}%")
        print("=" * 60)

    # Display columns with missing df_processed (shown in both modes)
    print("Columns with missing data:")
    # print("-" * 60)

    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

    # Transpose so columns are headers
    missing_analysis = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_pct
    }).T

    # Filter to only columns with missing df_processed
    cols_with_missing = missing_analysis.loc[:, (missing_analysis.loc['Missing_Count'] > 0)]

    if not cols_with_missing.empty:
        print(cols_with_missing)
    else:
        print("No missing data found in any column!")
    print("=" * 60)


def process_data(df,detailed):
    """
    Process the Titanic dataset by handling missing values and optimizing df_processed types.

    Parameters:
    - df: pandas DataFrame to process

    Returns:
    - Processed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    print("\nProcessing data...")
    # print("=" * 60)

    # Handle missing values
    print("1. Filling missing values...")
    processed_df['Age'] = processed_df['Age'].fillna(-1)
    processed_df['Cabin'] = processed_df['Cabin'].fillna('None')
    processed_df['Embarked'] = processed_df['Embarked'].fillna('None')

    # Fill remaining columns with 'None'
    for col in processed_df.columns:
        if col not in ['Age', 'Cabin', 'Embarked']:
            if processed_df[col].isnull().any():
                processed_df[col] = processed_df[col].fillna('None')

    print("   Missing values filled.")

    # Process and optimize df_processed types
    print("2. Optimizing df_processed types...")

    # Cabin - string with <=10 symbols
    processed_df['Cabin'] = processed_df['Cabin'].astype(str).str[:10]

    # Age - int below 120
    processed_df['Age'] = processed_df['Age'].astype(int)
    processed_df['Age'] = processed_df['Age'].clip(upper=119)

    # Embarked - string below 5 symbols
    processed_df['Embarked'] = processed_df['Embarked'].astype(str).str[:5]

    # PassengerId - int below 10000
    processed_df['PassengerId'] = processed_df['PassengerId'].astype(int)
    processed_df['PassengerId'] = processed_df['PassengerId'].clip(upper=9999)

    # Name - text string, below 100 symbols
    processed_df['Name'] = processed_df['Name'].astype(str).str[:100]

    # Pclass - small int, below 10
    processed_df['Pclass'] = processed_df['Pclass'].astype('int8')
    processed_df['Pclass'] = processed_df['Pclass'].clip(upper=9)

    # Survived - small int, 0 or 1
    processed_df['Survived'] = processed_df['Survived'].astype('int8')
    processed_df['Survived'] = processed_df['Survived'].clip(lower=0, upper=1)

    # Sex - string shorter than 10 symbols
    processed_df['Sex'] = processed_df['Sex'].map({'male': 0, 'female': 1})
    processed_df['Sex'] = processed_df['Sex'].astype('int8')

    # Parch - small int below 100
    processed_df['Parch'] = processed_df['Parch'].astype('int8')
    processed_df['Parch'] = processed_df['Parch'].clip(upper=99)

    # SibSp - small int below 100
    processed_df['SibSp'] = processed_df['SibSp'].astype('int8')
    processed_df['SibSp'] = processed_df['SibSp'].clip(upper=99)

    # Fare - int below 100000
    processed_df['Fare'] = processed_df['Fare'].astype(int)
    processed_df['Fare'] = processed_df['Fare'].clip(upper=99999)

    # Ticket - string shorter than 100 symbols
    processed_df['Ticket'] = processed_df['Ticket'].astype(str).str[:100]

    if detailed:
        # Display first 10 rows of processed df_processed with all columns
        print("\n### FIRST 10 ROWS OF PROCESSED DATA ###")
        print("=" * 60)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(processed_df.head(10))
        print("=" * 60)

        print("   Data types optimized.")

        # Display df_processed type summary
        print("\n3. Data type summary:")
        print("-" * 60)
        print(processed_df.dtypes)

        print("=" * 60)
        print("Processing complete!")
        print("=" * 60)
    print()
    return processed_df


def visualize_survival_data(df, enable_visualization=True):
    """
    Create scatter plots for all 2-element combinations of parameters,
    showing Survived status with color and marker coding.

    Parameters:
    - df: pandas DataFrame with processed df_processed
    - enable_visualization: bool, if False the function returns without creating plots
    """
    if not enable_visualization:
        print("\nVisualization is disabled.")
        return

    print("\n### VISUALIZING SURVIVAL DATA ###")
    print("=" * 60)

    # Parameters to create combinations from
    parameters = ['Age', 'Pclass', 'Fare', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']

    # Generate all 2-element combinations
    param_pairs = list(combinations(parameters, 2))

    print(f"Creating {len(param_pairs)} scatter plots...")
    print("=" * 60)

    # Calculate grid dimensions
    n_plots = len(param_pairs)
    n_cols = 4  # 4 plots per row for better visibility
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create figure with subplots (normal size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    # Prepare df_processed - convert categorical variables to numeric for plotting
    plot_df = df.copy()

    # Encode categorical variables
    if plot_df['Sex'].dtype == 'object':
        plot_df['Sex'] = plot_df['Sex'].map({'male': 0, 'female': 1, 'None': -1})

    if plot_df['Cabin'].dtype == 'object':
        plot_df['Cabin_encoded'] = plot_df['Cabin'].apply(lambda x: ord(x[0]) if x != 'None' else -1)
    else:
        plot_df['Cabin_encoded'] = -1

    if plot_df['Embarked'].dtype == 'object':
        plot_df['Embarked_encoded'] = plot_df['Embarked'].map({'S': 0, 'C': 1, 'Q': 2, 'None': -1})
    else:
        plot_df['Embarked_encoded'] = -1

    # Create scatter plots
    for idx, (param1, param2) in enumerate(param_pairs):
        ax = axes[idx]

        # Use encoded versions for categorical variables
        x_param = 'Cabin_encoded' if param1 == 'Cabin' else param1
        y_param = 'Cabin_encoded' if param2 == 'Cabin' else param2
        x_param = 'Embarked_encoded' if param1 == 'Embarked' else x_param
        y_param = 'Embarked_encoded' if param2 == 'Embarked' else y_param

        # Separate df_processed by survival status
        survived = plot_df[plot_df['Survived'] == 1]
        died = plot_df[plot_df['Survived'] == 0]

        # Plot died (red crosses)
        ax.scatter(died[x_param], died[y_param],
                   c='red', marker='x', s=50, alpha=0.6, label='Died')

        # Plot survived (green circles)
        ax.scatter(survived[x_param], survived[y_param],
                   c='green', marker='o', s=50, alpha=0.6, label='Survived')

        # Set labels and title
        ax.set_xlabel(param1, fontsize=10)
        ax.set_ylabel(param2, fontsize=10)
        ax.set_title(f'{param1} vs {param2}', fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    # Increase spacing between plots by 1.5x
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)

    # Save as PDF in the specified directory
    output_path = r"C:\Users\Mateusz\Downloads\titanic\Figure_1.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    plt.close(fig)  # Close the figure instead of showing it

    print("=" * 60)
    print(f"Generated {n_plots} scatter plots showing survival patterns")
    print("=" * 60)


def gini_impurity(group):
    # Calculate Gini impurity for each group
    if len(group) == 0:
        return 0
    survived_ratio = group['Survived'].sum() / len(group)
    return 2 * survived_ratio * (1 - survived_ratio)


def decision_tree_node(df, features):
    """
    Create a single-node decision tree that splits passengers by Feature
    to maximize homogeneity of the 'Survived' parameter.

    Parameters:
    - df: pandas DataFrame with processed df_processed

    Returns:
    - best_split_param: optimal value for splitting
    - best_w_gini: weighted Gini impurity at the optimal split
    """
    df_valid = df[features].copy()

    features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    best_gini_global = float('inf')
    best_split_param_global = None
    # best_feature = None
    # best_split_feature = None
    # best_split_info_global = None

    for i in features:
        print(i)
        # if df_valid[i].dtype != 'object':
        #     print(f"Parameter range: {df_valid[i].min()} to {df_valid[i].max()}")

        # Get unique age values to test as thresholds
        unique_values = sorted(df_valid[i].unique())

        best_w_gini = float('inf')
        best_split_param = None
        best_split_info = None

        # Iterate through all possible age thresholds
        for limit in unique_values:

            # Split df_processed into two groups
            left_group = df_valid[df_valid[i] <= limit]
            right_group = df_valid[df_valid[i] > limit]

            # Skip if either group is empty
            if len(left_group) == 0 or len(right_group) == 0:
                continue

            left_gini = gini_impurity(left_group)
            right_gini = gini_impurity(right_group)

            # Calculate weighted Gini impurity
            n_left = len(left_group)
            n_right = len(right_group)
            n_total = n_left + n_right

            weighted_gini = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini

            # Update best split if this is better
            if (weighted_gini < best_w_gini):
            # if (weighted_gini < best_w_gini and len(left_group)>1 and len(right_group)>1): #gives an error
                best_w_gini = weighted_gini
                best_split_param = limit
                left_group_local = left_group
                right_group_local = right_group
                best_split_info = {
                    'left_group': left_group,
                    'right_group': right_group,
                    'left_gini': left_gini,
                    'right_gini': right_gini,
                    'n_left': n_left,
                    'n_right': n_right
                }
            # print(f"{best_split_param:2n}, {best_w_gini:.4f}, {limit:2n}, {weighted_gini:.4f}, {left_gini:.4f}, {right_gini:.4f}")
        if best_w_gini < best_gini_global:
            best_gini_global = best_w_gini
            best_split_param_global = best_split_param
            left_group_best = left_group_local
            right_group_best = right_group_local
            best_feature=i
            best_split_info_global = {
                'left_group': left_group,
                'right_group': right_group,
                'left_gini': left_gini,
                'right_gini': right_gini,
                'n_left': n_left,
                'n_right': n_right
            }

        # Display results
        print(f"Optimal Splitting Value of {i}: {best_split_param}")
        print(f"Weighted Gini Impurity: {best_w_gini:.4f}")
        if best_split_info:
            print("-" * 60)
            left = best_split_info['left_group']
            right = best_split_info['right_group']

            print(f"Left Group ({i} <= {best_split_param}):")
            print(f"  Size: {best_split_info['n_left']} passengers")
            print(f"  Correct prediction: {left['Survived'].sum()} out of {len(left)}, so {(left['Survived'].mean()) * 100:.1f}% Survived")
            print(f"  Gini Impurity: {best_split_info['left_gini']:.4f}")

            print(f"Right Group ({i} > {best_split_param}):")
            print(f"  Size: {best_split_info['n_right']} passengers")
            print(f"  Correct prediction: {len(right) - right['Survived'].sum()} out of {len(right)}, so {(1 - right['Survived'].mean()) * 100:.1f}% Survived")
            print(f"  Gini Impurity: {best_split_info['right_gini']:.4f}")


    # print(best_feature, best_split_param_global, f"{best_gini_global:.4f}")
    return best_feature, best_split_param_global, best_gini_global, left_group_best, right_group_best


def predict_survival(PassengerId, age_threshold, df=None):
    """
    Predict survival probability for a given passenger based on optimized age threshold.

    Parameters:
    - PassengerId: ID of the passenger to predict
    - age_condition: optimal age threshold from decision tree
    - df: DataFrame containing passenger df_processed (default: processed_df if available)

    Returns:
    - prediction: 0 (died) or 1 (survived)
    - probability: probability of survival
    """
    print("\n### SURVIVAL PREDICTION ###")
    print("=" * 60)

    # Check if passenger exists in dataframe
    passenger_row = df[df['PassengerId'] == PassengerId]

    if passenger_row.empty:
        print(f"Error: PassengerId {PassengerId} not found in dataset")
        return None, None

    # Get passenger df_processed
    passenger_data = passenger_row.iloc[0]
    passenger_age = passenger_data['Age']

    print(f"Passenger ID: {PassengerId}")
    print(f"Name: {passenger_data['Name']}")
    print(f"Sex: {passenger_data['Sex']}, Age: {passenger_age}, Class: {passenger_data['Pclass']}")
    print("-" * 60)

    # Filter out missing age values (-1) from training df_processed
    df_valid = df[df['Age'] >= 0].copy()

    # Check if passenger has valid age
    if passenger_age < 0:
        print("Warning: Passenger has missing age df_processed (-1)")
        print("Using overall survival rate for prediction...")
        survival_probability = df_valid['Survived'].mean()
    else:
        # Determine which group the passenger belongs to
        if passenger_age <= age_threshold:
            group = df_valid[df_valid['Age'] <= age_threshold]
            group_name = f"Age <= {age_threshold}"
        else:
            group = df_valid[df_valid['Age'] > age_threshold]
            group_name = f"Age > {age_threshold}"

        # Calculate survival probability for this group
        survival_probability = group['Survived'].mean()

        print(f"Group: {group_name}")
        print(f"Group Size: {len(group)} passengers")

    # Display probability
    print(f"Survival Probability: {survival_probability * 100:.2f}%")

    # Make prediction based on 50% threshold
    if survival_probability >= 0.5:
        prediction = 1
        prediction_text = "SURVIVED"
    else:
        prediction = 0
        prediction_text = "DIED"

    print(f"Prediction: {prediction} ({prediction_text})")

    # Show actual survival status if available
    if 'Survived' in passenger_data:
        actual = passenger_data['Survived']
        actual_text = "SURVIVED" if actual == 1 else "DIED"
        correct = "✓ CORRECT" if prediction == actual else "✗ INCORRECT"
        print(f"Actual: {actual} ({actual_text}) - {correct}")

    print("=" * 60)

    return prediction, survival_probability


def build_decision_tree(df, features, max_depth, gini_threshold, min_group, current_depth=0):
    """
    Recursively build a decision tree by splitting data until Gini threshold is met or max depth reached.

    Parameters:
    - df: DataFrame to split
    - features: list of feature column names
    - max_depth: maximum depth of the tree
    - gini_threshold: stop splitting if Gini impurity <= this value
    - current_depth: current depth in the tree (used for recursion)

    Returns:
    - tree: dictionary representing the decision tree structure
    """

    # Statistical functions
    gini_branch=gini_impurity(df)
    global total_score
    survival_r = df['Survived'].mean()
    if survival_r >= 0.50:
        accuracy = df['Survived'].sum() / len(df)
    else:
        accuracy = (len(df) - df['Survived'].sum()) / len(df)

    # Base cases: stop splitting
    print(f"Depth {current_depth}: Splitting {len(df)} samples...")
    if current_depth >= max_depth:
        print(f"Depth {current_depth}: Reached maximum depth with gini={gini_branch:.4f}")
        print("=" * 60)
        counter(df['Survived'].sum(), len(df), survival_r)
        return {
            'leaf': True,
            'depth': current_depth,
            'size': len(df),
            'gini': gini_branch,
            'survival_rate': survival_r if len(df) > 0 else 0,
            'accuracy': accuracy
        }
    # if node is already small, len<4, then leaves it
    # min_group parameter
    if len(df) <= min_group:
        counter(df['Survived'].sum(), len(df), survival_r)
        print(f"Depth {current_depth}: Sample of 4, it is too small")
        print("=" * 60)
        return {
            'leaf': True,
            'depth': current_depth,
            'size': len(df),
            'gini': gini_branch,
            'survival_rate': survival_r if len(df) > 0 else 0,
            'accuracy': accuracy
        }
    if len(df) == 0:
        print(f"Depth {current_depth}: Empty dataframe")
        return {
            'leaf': True,
            'depth': current_depth,
            'size': 0,
            'survival_rate': 0,
            'accuracy': accuracy
        }

    # Perform split
    best_feature, best_split_param, best_w_gini, left_group, right_group = decision_tree_node(df, features)
    print(f"Depth {current_depth}: Splitted into {len(left_group)} and {len(right_group)} samples, "
          f"using {best_feature} = {best_split_param}, yielding w_gini = {best_w_gini:.4f}")

    # Check if Gini threshold for splitting is met and if it is, creates Leaf boolean
    # if best_w_gini <= gini_threshold:
    if gini_branch <= gini_threshold:
        # Creates Leaf with split information
        print(f"Depth {current_depth}: Gini of splitting {best_w_gini:.4f} <= threshold {gini_threshold}")
        print("=" * 90)
        counter(df['Survived'].sum(), len(df), survival_r)
        return {
            'leaf': True,
            'depth': current_depth,
            'size': len(df),
            'gini': gini_branch,
            'survival_rate': survival_r,
            'accuracy': accuracy
        }
    print("=" * 90)

    # Create node with split information
    node = {
        'leaf': False,
        'depth': current_depth,
        'feature': best_feature,
        'split_value': best_split_param,
        'gini': gini_branch,
        'w_gini': best_w_gini,
        'size': len(df),
        'survival_rate': survival_r,
        'accuracy': accuracy
    }

    # Recursively build left and right branches
    print(f"Depth {current_depth}: Creating branches ({best_feature} <= {best_split_param})")
    node['left'] = build_decision_tree(left_group, features, max_depth, gini_threshold, min_group, current_depth + 1)
    node['right'] = build_decision_tree(right_group, features, max_depth, gini_threshold, min_group, current_depth + 1)
    #rozbudowuje do końca węzły, aż nie napotka któregoś limitera - albo threshold, albo max_depth
    return node


def print_tree(node, prefix="", is_left=True):
    """
    Print the decision tree in a readable format.

    Parameters:
    - node: tree node dictionary
    - prefix: string prefix for formatting
    - is_left: whether this is a left branch
    """
    if node['leaf']:
        print(f"{GREEN}"
              f"{prefix}{'└─ L' if is_left else '└─ R'} LEAF: size={node['size']}, "
              f"depth={node['depth']}, survival_rate={node['survival_rate']:.2%}, "
              f"accuracy={node['accuracy']:.2%}, "
              # f"gini={node.get('gini', 'N/A')}"
              f"gini={node.get('gini'):.4f}"
              # f"gini={node['gini']:.4f}"
              f"{RESET}")
    else:
        connector = '└─ L' if is_left else '└─ R'
        print(f"{prefix}{connector} [{node['feature']} <= {node['split_value']}] "
              f"size={node['size']}, gini={node['gini']:.4f}, w_gini={node['w_gini']:.4f}")

        new_prefix = prefix + ("    " if is_left else "    ")
        print_tree(node['left'], new_prefix, True)
        print_tree(node['right'], new_prefix, False)



# Load the CSV file
file_path = r"C:\Users\Mateusz\Downloads\titanic\train.csv"
df = pd.read_csv(file_path)

features = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# Inspect original df_processed
print(("=" * 60)+"\n### ORIGINAL DATA INSPECTION ###")
inspect_data(df, detailed=False)  # Change to False for summary only

# Process the df_processed
processed_df = process_data(df, detailed=False)

# Inspect processed df_processed
print(("=" * 60)+"\n### PROCESSED DATA INSPECTION ###")
inspect_data(processed_df, detailed=False)  # Change to False for summary only

# Visualize the survival df_processed (set to False to disable)
visualize_survival_data(processed_df, enable_visualization=False)

# Build the tree, using build_decision_tree
print("\n" + "=" * 60 + "\n### BUILDING DECISION TREE ###" + "\n"+ ("=" * 60))
min_group=4
max_depth = 10
gini_threshold = 0.05
tree = build_decision_tree(processed_df, features, max_depth, gini_threshold=gini_threshold, min_group=min_group)

# Print the tree structure
print("\n" + "=" * 60 + "\n### DECISION TREE STRUCTURE ###" + "\n" + ("=" * 60))
print(f"Root: size={tree['size']}, survival_rate={tree['survival_rate']:.2%}")
# use print_tree function
if not tree['leaf']:
    print_tree(tree['left'], "", True)
    print_tree(tree['right'], "", False)
print("=" * 60)

print(f"\nTotal score: {total_score} out of {tree['size']}, giving {total_score/tree['size']:.2%} accuracy")

# End timing
end_time = time.time()
execution_time = end_time - start_time
print(f"\nTotal execution time: {execution_time:.4f} seconds")

