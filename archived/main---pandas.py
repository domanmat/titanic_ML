import pandas as pd
import numpy as np

import survival_counter
import gini_Y_impurity
import data_inspection

# import seaborn as sns

import visualize_survival_data

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

import time
# Start timing
start_time = time.time()
print("Executing the code...")

total_score=0


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
    processed_df['Age'] = processed_df['Age'].astype('float32')

    # Embarked - string below 5 symbols
    processed_df['Embarked'] = processed_df['Embarked'].astype(str).str[:5]
    processed_df['PassengerId'] = processed_df['PassengerId'].astype(int)

    # Name - text string, below 100 symbols
    processed_df['Name'] = processed_df['Name'].astype(str).str[:100]

    # Pclass - small int, below 10
    processed_df['Pclass'] = processed_df['Pclass'].astype('int8')
    # processed_df['Pclass'] = processed_df['Pclass'].clip(upper=9)

    # Survived - small int, 0 or 1
    processed_df['Survived'] = processed_df['Survived'].astype('int8')
    # processed_df['Survived'] = processed_df['Survived'].clip(lower=0, upper=1)

    # Sex - string shorter than 10 symbols
    processed_df['Sex'] = processed_df['Sex'].map({'male': 0, 'female': 1})
    processed_df['Sex'] = processed_df['Sex'].astype('int8')

    # Parch - small int below 100
    processed_df['Parch'] = processed_df['Parch'].astype('int8')

    # SibSp - small int below 100
    processed_df['SibSp'] = processed_df['SibSp'].astype('int8')

    # Fare - int below 100000
    processed_df['Fare'] = processed_df['Fare'].astype('float64')

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


def gini_X_impurity(group,features):
    # Calculate Gini impurity for each df
    if len(group) == 0:
        return 0
    # p_unique = df[features_train].value_counts()
    # print(p_unique)
    gini_x=[]
    group_len = len(group)
    df_x = pd.DataFrame({"gini"})
    for feature in features:
        # Vectorized calculation
        value_counts = group[feature].value_counts()
        p_squared_sum = ((value_counts / group_len) ** 2).sum()
        gini_x.append(1 - p_squared_sum)

    # for i in features_train:
    #     p_sq_sum=0
    #     unique_values = sorted(df[i].unique())
    #     for unique in unique_values:
    #         counts=df[i].value_counts()[unique]
    #         # print(unique, counts)
    #         p_unique = counts / len(df)
    #         p_sq_sum+=p_unique**2
    #     gini_x.append(1 - p_sq_sum)
    #     # df_x.append(1 - p_sq_sum)
    df_x = pd.DataFrame({"gini": gini_x})
    return df_x


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

    # features_train = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    best_w_gini = float('inf')
    best_gini_global = float('inf')
    best_split_param = None
    best_split_param_global = None
    best_feature = None
    left_group_best = None
    right_group_best = None

    print(time.time()-start_time)
    for feature in features:
        # # Get unique age values to test as thresholds --- old
        # unique_values = sorted(df[feature].unique())

        # Sort df once
        sorted_df = df.sort_values(feature)
        survived = sorted_df['Survived'].values
        feature_vals = sorted_df[feature].values

        # Get unique feature values to test as thresholds
        # n_total = len(sorted_df)
        unique_values = np.unique(feature_vals)

        # # Binning idea - not improving much
        # n_unique = len(df_valid[i].unique())
        # n_bins = 20
        # if (n_unique >= n_bins):
        #     unique_values = pd.qcut(df_valid[i].unique(), q=n_bins, labels=False)
        # else:
        #     unique_values = sorted(df_valid[i].unique())
        # # print(unique_values)


        # Iterate through all possible age thresholds

        print(time.time() - start_time)
        for limit in unique_values[:-1]:  # Skip last value, then right df is empty

            # # Find split point
            # split_idx = np.searchsorted(feature_vals, limit, side='right')
            #
            # if split_idx == 0 or split_idx == n_total:
            #     continue

            # Split df into two groups
            left_group = df[df[feature] <= limit]
            right_group = df[df[feature] > limit]

            # Skip if either df is empty
            if len(left_group) == 0 or len(right_group) == 0:
                continue

            # # Calculate gini using counts, not dataframes
            # n_left = split_idx
            # n_right = n_total - split_idx
            #
            # survived_left = survived[:split_idx].sum()
            # survived_right = survived[split_idx:].sum()


            left_gini = gini_Y_impurity.calc(left_group)
            right_gini = gini_Y_impurity.calc(right_group)

            # Calculate weighted Gini impurity
            n_left = len(left_group)
            n_right = len(right_group)
            n_total = n_left + n_right

            weighted_gini = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini

            print(time.time() - start_time)
            # Update best split if this is better
            if (weighted_gini < best_w_gini):
            # if (weighted_gini < best_w_gini and len(left_group)>1 and len(right_group)>1): #gives an error
                best_w_gini = weighted_gini
                best_split_param = limit
                left_group_local = left_group
                right_group_local = right_group
                # best_split_info = {
                #     'left_group': left_group,
                #     'right_group': right_group,
                #     'left_gini': left_gini,
                #     'right_gini': right_gini,
                #     'group_size': group_size,
                #     'n_right': n_right
                # }
            # print(f"{best_split_param:2n}, {best_w_gini:.4f}, {limit:2n}, {weighted_gini:.4f}, {left_gini:.4f}, {right_gini:.4f}")
        if best_w_gini < best_gini_global:
            best_gini_global = best_w_gini
            best_split_param_global = best_split_param
            left_group_best = left_group_local
            right_group_best = right_group_local
            best_feature=feature
            best_split_info_global = {
                'left_group': left_group,
                'right_group': right_group,
                'left_gini': left_gini,
                'right_gini': right_gini,
                'group_size': n_left,
                'n_right': n_right
            }

        print(time.time() - start_time)
        # Display results
        print(f"Optimal Splitting Value of {feature}: {best_split_param_global}")
        print(f"Weighted Gini Impurity: {best_w_gini:.4f}")

    # print(best_feature, best_split_param_global, f"{best_gini_global:.4f}")
    print(best_feature)
    return best_feature, best_split_param_global, best_gini_global, left_group_best, right_group_best


def build_decision_tree(df, features, max_depth, gini_threshold, min_group, current_depth=0):
    """
    Recursively build a decision tree by splitting data until Gini threshold is met or max depth reached.

    Parameters:
    - df: DataFrame to split
    - features_train: list of feature column names
    - max_depth: maximum depth of the tree
    - gini_threshold: stop splitting if Gini impurity <= this value
    - current_depth: current depth in the tree (used for recursion)

    Returns:
    - tree: dictionary representing the decision tree structure
    """

    print(time.time()-start_time)
    # Statistical functions
    gini_branch=gini_Y_impurity.calc(df)
    global total_score
    # Zakładamy że wszyscy z komórki umierają albo przeżywają
    survival_r = df['Survived'].mean()
    if survival_r >= 0.50:
        accuracy = df['Survived'].sum() / len(df)
    else:
        accuracy = (len(df) - df['Survived'].sum()) / len(df)

    print(time.time()-start_time)
    # Base cases: stop splitting
    print(f"Depth {current_depth}: Splitting {len(df)} samples...")

    # Calculating gini impurity of the parameters set
    df_x=gini_X_impurity(df,features)
    x_gini=df_x.sum().iloc[0]

    print(time.time()-start_time)
    if current_depth >= max_depth:
        print(f"Depth {current_depth}: Reached maximum depth with gini={gini_branch:.4f}")
        print("=" * 60)
        total_score = survival_counter.calc(df['Survived'].sum(), total_score, len(df), survival_r)
        return {
            'leaf': True,
            'depth': current_depth,
            'size': len(df),
            'gini': gini_branch,
            'survival_rate': survival_r if len(df) > 0 else 0,
            'accuracy': accuracy
        }
    print(time.time()-start_time)
    # if node is already small, len<4, then leaves it
    # min_group parameter
    if len(df) <= min_group:
        total_score = survival_counter.calc(df['Survived'].sum(), total_score, len(df), survival_r)
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


    print(time.time()-start_time)
    # Check if Gini threshold for splitting is met and if it is, creates Leaf boolean
    # if best_w_gini <= gini_threshold:
    if gini_branch <= gini_threshold:
        # Creates Leaf with gini information
        print(f"Depth {current_depth}: Gini before splitting {gini_branch:.4f} <= threshold {gini_threshold}")
        print("=" * 90)
        total_score = survival_counter.calc(df['Survived'].sum(), total_score, len(df), survival_r)
        return {
            'leaf': True,
            'depth': current_depth,
            'size': len(df),
            'gini': gini_branch,
            'survival_rate': survival_r,
            'accuracy': accuracy
        }
    elif x_gini<= gini_threshold:
        print(df)
        # Creates Leaf where there is a set of the same X parameters, but different Survive values
        print(f"Depth {current_depth}: Gini of branch {gini_branch:.4f} > {gini_threshold} "
              f"but set can't be split as g_impurity of parameters = {x_gini:.3f}")
        print("=" * 90)
        total_score = survival_counter.calc(df['Survived'].sum(), total_score, len(df), survival_r)
        return {
            'leaf': True,
            'depth': current_depth,
            'size': len(df),
            'gini': gini_branch,
            'survival_rate': survival_r,
            'accuracy': accuracy
        }
    else:
        # Perform split
        best_feature, best_split_param, best_w_gini, left_group, right_group = decision_tree_node(df, features)
        print(f"Depth {current_depth}: Splitted into {len(left_group)} and {len(right_group)} samples, "
              f"using {best_feature} = {best_split_param}, yielding w_gini = {best_w_gini:.4f}")
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
    print(time.time()-start_time)
    # exit()
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


def df_random_slice(df, percent, random_seed):
    """ Select a percentage of random rows.
        Parameters:
    - df: pandas loaded DataFrame
    - percent: fraction of rows to select (0.0 to 1.0)
    - random_seed: optional seed
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = int(len(df) * percent)
    indices = np.random.choice(len(df), size=n_samples, replace=False)
    sliced_df = df.iloc[indices].copy()

    return sliced_df, indices




# Load the CSV file
file_path = r"C:\Users\Mateusz\Downloads\titanic\train.csv"
df = pd.read_csv(file_path)

features_all = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
# features_train = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked'] #Embarked nawet pogorszyło fit
# features_train = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# Inspect original df_processed
print(("=" * 60)+"\n### ORIGINAL DATA INSPECTION ###")
data_inspection.check(df, detailed=False)  # Change to False for summary only

# Process the df_processed
processed_df = process_data(df, detailed=False)

# Inspect processed df_processed
print(("=" * 60)+"\n### PROCESSED DATA INSPECTION ###")
data_inspection.check(processed_df, detailed=False)  # Change to False for summary only

# # Visualize the survival df_processed (set to False to disable)
# visualize_survival_data.figure(processed_df, enable_visualization=False)

# Get a slice, usage
df, train_indices = df_random_slice(processed_df, percent=1, random_seed=21)

time1 = time.time()
# Build the tree, using build_decision_tree
print("\n" + "=" * 60 + "\n### BUILDING DECISION TREE ###" + "\n"+ ("=" * 60))
min_group=1
max_depth = 11
gini_threshold = 0.01
tree = build_decision_tree(df, features, max_depth, gini_threshold=gini_threshold, min_group=min_group)

time2 = time.time()
# Print the tree structure
print("\n" + "=" * 60 + "\n### DECISION TREE STRUCTURE ###" + "\n" + ("=" * 60))
print(f"Root: size={tree['size']}, survival_rate={tree['survival_rate']:.2%}")
# use print_tree function
if not tree['leaf']:
    print_tree(tree['left'], "", True)
    print_tree(tree['right'], "", False)
print("=" * 60)

print(f"\nTotal score: {total_score} out of {tree['size']}, giving {total_score/tree['size']:.2%} accuracy")


time3 = time.time()

def predict_single(tree, sample):
    """
    Predict survival for a single sample using the decision tree.

    Parameters:
    - tree: decision tree dictionary structure
    - sample: dictionary with feature values (e.g., {'Pclass': 3, 'Sex': 0, 'Age': 22, ...})

    Returns:
    - prediction: predicted survival (0 or 1)
    """
    node = tree

    while not node['leaf']:
        feature = node['feature']
        split_value = node['split_value']
        sample_value = sample[feature]

        if sample_value <= split_value:
            node = node['left']
        else:
            node = node['right']

    # Predict based on survival rate at leaf
    prediction = 1 if node['survival_rate'] >= 0.5 else 0
    real_status = sample['Survived']
    pass_id = sample['PassengerId']
    if prediction == real_status:
        accuracy = 1
    else:
        accuracy = 0
    return pass_id, prediction, real_status, accuracy



def predict_batch(tree, df, features):
    """
    Predict survival for multiple samples.

    Parameters:
    - tree: decision tree dictionary structure
    - df: DataFrame with samples to predict
    - features_train: list of feature names used in the model

    Returns:
    - predictions: list of predictions
    """
    predictions = []
    real_status = []
    col_accuracy = []
    col_pass_id = []


    for idx, row in df.iterrows():
        sample = {feature: row[feature] for feature in features}
        pass_id, pred, status, accuracy = predict_single(tree, sample)
        predictions.append(pred)
        real_status.append(status)
        col_accuracy.append(accuracy)
        col_pass_id.append(pass_id)
    df_predictions=pd.DataFrame({"PassengerId":col_pass_id,"predictions":predictions, "real status":real_status, "accuracy":col_accuracy})

    return df_predictions

# Predict for test dataset
# test_df = pd.read_csv("test.csv")
# test_processed = process_data(test_df, detailed=False)

# # Get a slice, usage
# df, train_indices = df_random_slice(processed_df, percent=0.9, random_seed=21)
# df, train_indices = df_random_slice(processed_df, percent=0.2, random_seed=42)
# df_predictions = predict_batch(tree, df, features_all)

time4 = time.time()

df_predictions = predict_batch(tree, processed_df, features_all)
# print(df_predictions)
# print(len(df_predictions))
# df['Predicted_Survival'] =

accuracy_ratio=df_predictions['accuracy'].sum()/len(df_predictions)
print(f"Accuracy on full data = {accuracy_ratio:.2%}")

#ADD PRINTING TO A FILE

# # for idx, row in df.iterrows():
# #     df_predictions['predictions']==df_predictions['real status']:
# if df_predictions['predictions'] == df_predictions['real status']:


# End timing
end_time = time.time()
exec1 = time1 - start_time
exec2 = time2 - time1
exec3 = time3 - time2
exec4 = time4 - time3
execution_time = end_time - start_time
print(f"Time part 1: {exec1:.4f} seconds")
print(f"Time part 2: {exec2:.4f} seconds")
print(f"Time part 3: {exec3:.4f} seconds")
print(f"Time part 4: {exec4:.4f} seconds")
print(f"Total execution time: {execution_time:.4f} seconds")

