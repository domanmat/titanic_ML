import pandas as pd
import numpy as np

import correlation_analysis
import survival_counter
import gini_Y_impurity
import data_inspection
import data_process
import split_data

# import seaborn as sns

import visualize_survival_data

GREEN = "\033[32m"
RED = "\033[31m"
RESET = "\033[0m"

from time import time
# Start timing
start_time = time()
print("Executing the code...")

total_score = 0
time_predictions = 0


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
    df_x = pd.DataFrame({"gini": gini_x})
    return df_x


def calc_tree_node(df, features):
    """
    Create a single-node decision tree that splits passengers by Feature
    to maximize homogeneity of the 'Survived' parameter.

    Parameters:
    - df: pandas DataFrame with processed df_processed
    - features_train: search features_train included in the ML process

    Returns:
    - best_split_param: optimal value for splitting
    - best_w_gini: weighted Gini impurity at the optimal split
    """

    # features_train = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
    best_w_gini = float('inf')
    best_feature = None
    best_split_param = None
    left_group_best = None
    right_group_best = None

    for feature in features:

        # Sort the dataframe once
        sorted_df = df.sort_values(feature)
        survived = sorted_df['Survived'].values
        feature_vals = sorted_df[feature].values

        # Get unique feature values to test as thresholds
        n_total = len(sorted_df)
        unique_values = np.unique(feature_vals)

        # # Binning idea - not improving much
        # n_unique = len(df_valid[i].unique())
        # n_bins = 20
        # if (n_unique >= n_bins):
        #     unique_values = pd.qcut(df_valid[i].unique(), q=n_bins, labels=False)
        # else:
        #     unique_values = sorted(df_valid[i].unique())
        # # print(unique_values)


        # Iterate through all possible feature thresholds
        for limit in unique_values[:-1]:  # Skip last value, then right df is empty

            # Find split point, iterating through all the values
            # Split index is the count of sorted values <= limit
            split_index = np.searchsorted(feature_vals, limit, side='right')

            # Skip if either df is empty
            if split_index == 0 or split_index == n_total:
                continue
            #
            # if split_index <= 2 or split_index >= n_total-2:
            #     continue

            # Calculate gini using counts, not whole dataframes
            n_left = split_index
            n_right = n_total - split_index

            survived_left = survived[:split_index].sum()
            survived_right = survived[split_index:].sum()

            left_gini = gini_Y_impurity.calc(n_left, survived_left)
            right_gini = gini_Y_impurity.calc(n_right, survived_right)

            # Calculate weighted Gini impurity - parameter of the split
            weighted_gini = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini
            # # Data inspection
            # print(f"Current Splitting Value of {feature}: {limit}")
            # print(f"Weighted Gini Impurity: {weighted_gini:.4f}")

            if weighted_gini < best_w_gini:
                best_w_gini = weighted_gini
                best_split_param = limit
                best_feature = feature
                # create dataframe splits only when we find a split with better w_gini
                left_group_best = sorted_df.iloc[:split_index]
                right_group_best = sorted_df.iloc[split_index:]

    # # Display detailed results
    # print(f"Optimal Splitting Value of {feature}: {best_split_param}")
    # print(f"Weighted Gini Impurity: {best_w_gini:.4f}")
    return best_feature, best_split_param, best_w_gini, left_group_best, right_group_best


def build_decision_tree(df, features, max_depth, gini_threshold, min_group, death_threshold, current_depth=0):
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

    # Statistical functions
    survived = df['Survived'].values.sum()
    n_total = len(df)
    gini_branch = gini_Y_impurity.calc(n_total, survived)

    global total_score
    # Zakładamy że wszyscy z komórki umierają albo przeżywają
    survival_r = df['Survived'].mean()
    if survival_r >= death_threshold:
        accuracy = df['Survived'].sum() / len(df)
    else:
        accuracy = (len(df) - df['Survived'].sum()) / len(df)


    # # Base cases: stop splitting
    # print(f"Depth {current_depth}: Splitting {len(df)} samples...")

    # Calculating gini impurity of the parameters set
    df_x=gini_X_impurity(df,features)
    x_gini=df_x.sum().iloc[0]


    if current_depth > max_depth:
        # print(f"Depth {current_depth}: Reached maximum depth with gini={gini_branch:.4f}")
        # print("=" * 60)
        total_score = survival_counter.calc(df['Survived'].sum(), total_score, len(df), survival_r)
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
    elif len(df) < min_group:
        total_score = survival_counter.calc(df['Survived'].sum(), total_score, len(df), survival_r)
        # print(f"Depth {current_depth}: Group of {len(df)} <= {min_group}, it is too small")
        # print("=" * 60)
        return {
            'leaf': True,
            'depth': current_depth,
            'size': len(df),
            'gini': gini_branch,
            'survival_rate': survival_r if len(df) > 0 else 0,
            'accuracy': accuracy
        }
    elif len(df) == 0:
        # print(f"Depth {current_depth}: Empty dataframe")
        return {
            'leaf': True,
            'depth': current_depth,
            'size': 0,
            'survival_rate': 0,
            'accuracy': accuracy
        }

    # Check if Gini threshold for splitting is met and if it is, creates Leaf boolean
    # if best_w_gini <= gini_threshold:
    elif gini_branch <= gini_threshold:
        # # Creates Leaf with gini information
        # print(f"Depth {current_depth}: Gini before splitting {gini_branch:.4f} <= threshold {gini_threshold}")
        # print("=" * 90)
        total_score = survival_counter.calc(df['Survived'].sum(), total_score, len(df), survival_r)
        return {
            'leaf': True,
            'depth': current_depth,
            'size': len(df),
            'gini': gini_branch,
            'survival_rate': survival_r,
            'accuracy': accuracy
        }
    elif x_gini <= gini_threshold:
        # # Creates Leaf where there is a set of the same X parameters, but different Survive values
        # # Can't be split based on Features
        # print(f"Depth {current_depth}: Gini of branch {gini_branch:.4f} > {gini_threshold} "
        #       f"but set can't be split as g_impurity of parameters = {x_gini:.3f}")
        # print("=" * 90)
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
        best_feature, best_split_param, best_w_gini, left_group, right_group = calc_tree_node(df, features)
        #prevent too small groups after splitting
        if len(left_group) < min_group or len (right_group) < min_group:
            # print(f"Depth {current_depth + 1}: Splitting down to group of lengths {len(left_group)} "
            #       f"and {len(right_group)} <= {min_group}, too small")
            # print("=" * 90)
            return {
                'leaf': True,
                'depth': current_depth,
                'size': len(df),
                'gini': gini_branch,
                'survival_rate': survival_r,
                'accuracy': accuracy
            }
        else:
            # L=len(left_group)
            # R=len(right_group)
            # print(f"Depth {current_depth}: Splitted into "
            #       f"L={len(left_group)} samples ({best_feature}<={best_split_param}) and "
            #       f"R={len(right_group)} samples ({best_feature}>{best_split_param}), "
            #       # f"using {best_feature} = {best_split_param}, "
            #       f"yielding w_gini = {best_w_gini:.4f}")

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
            # print("=" * 90)
            # # exit()
            # # Recursively build left and right branches
            # print(f"Depth {current_depth+1}: Splitting left branch: ({best_feature} <= {best_split_param})")
            # print(f"Depth {current_depth+1}: Splitting right branch: ({best_feature} > {best_split_param})")
            node['left'] = build_decision_tree(left_group, features, max_depth, gini_threshold, min_group, death_threshold, current_depth + 1)
            node['right'] = build_decision_tree(right_group, features, max_depth, gini_threshold, min_group, death_threshold, current_depth + 1)
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


def train_df_slicing(df, percent, random_seed):
    """
    PASTING - sampling with replacement - losowanie ze zwracaniem
    Select a percentage of random rows.
        Parameters:
    - df: pandas loaded DataFrame
    - percent: fraction of rows to select (0.0 to 1.0)
    - random_seed: optional seed
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    n_samples = int(len(df) * percent/100)
    indices = np.random.choice(len(df), size=n_samples, replace=True)
    sliced_df = df.iloc[indices].copy()

    return sliced_df, indices


def predict_batch(tree, df, features, death_threshold):
    """
    Predict survival for all data using whole tree.

    Parameters:
    - tree: decision tree dictionary structure
    - df: DataFrame with samples to predict
    - features_train: list of feature names used in the model

    Returns:
    - predictions: list of predictions
    """
    time_pred_start = time()
    n_samples = len(df)
    # global death_threshold

    # Convert to numpy arrays once
    feature_arrays = {feature: df[feature].values for feature in features}
    survived = df['Survived'].values
    pass_ids = df['PassengerId'].values

    predictions = np.empty(n_samples, dtype=np.int8)

    # Process each sample
    for i in range(n_samples):
        node = tree

        # Traverse tree
        while not node['leaf']:
            if feature_arrays[node['feature']][i] <= node['split_value']:
                node = node['left']
            else:
                node = node['right']

        predictions[i] = 1 if node['survival_rate'] >= death_threshold else 0

    accuracy = (predictions == survived).astype(np.int8)

    time_pred_end = time()
    global time_predictions
    time_predictions += (time_pred_end-time_pred_start)

    return pd.DataFrame({
        "PassengerId": pass_ids,
        "predictions": predictions,
        "real status": survived,
        "accuracy": accuracy
    })


def train_trees(train_df, percent, sessions, max_depth, gini_threshold, min_group, death_threshold):

    accuracy_ratio_best = 0
    tree_best = None
    best_index = 0
    df_trained = None
    global features_train

    # Store all trained trees with their metadata
    trained_trees = []

    # Pre-generate all data slices
    print("Generating data slices...")
    df_slices = []
    for session in range(sessions):
        # random_seed if set would be useless
        df_tmp, train_indices = train_df_slicing(train_df, percent=percent, random_seed=None)
        df_slices.append(df_tmp)

    for session in range(sessions):
        print(f'Tree number {session + 1}')
        total_accuracy=0

        # Build tree
        tree = build_decision_tree(df_slices[session], features_train, max_depth,
                                   gini_threshold, min_group, death_threshold)

        # Evaluate on all test sets
        accuracies = []
        for testing in range(sessions):
            tree_predictions = predict_batch(tree, df_slices[testing], features_all, death_threshold)
            accuracy_ratio = tree_predictions['accuracy'].mean()  # Faster than sum/len
            accuracies.append(accuracy_ratio)
            if testing==session:
                print(f'Accuracy of tree {session + 1:2d} on set {testing + 1:2d}:  {accuracy_ratio:.2%}')

        # Calculate average accuracy on all training data sets
        final_accuracy = sum(accuracies) / sessions
        print(f'Average accuracy of tree {session+1:2d}:    {final_accuracy:.2%}')
        print("=" * 90)

        # Store the trained tree with its metadata
        trained_trees.append({
            'tree': tree,
            'session_index': session,
            'training_data': df_slices[session],
            'accuracies': accuracies,
            'average_accuracy': final_accuracy
        })

        # Update the best tree
        if final_accuracy > accuracy_ratio_best:
            print('Found a more accurate tree!')
            accuracy_ratio_best = final_accuracy
            tree_best = tree
            best_index = session + 1
            df_trained = df_slices[session]

    return tree_best, df_trained, accuracy_ratio_best, best_index, trained_trees


def predict_ensemble(trained_trees, df, features_all, threshold):
    """
    Make predictions using ensemble of trained trees with majority voting.

    Parameters:
    - trained_trees: list of tree dictionaries from train_trees()
                     (can be list of dicts with 'tree' key, or just list of trees)
    - df: DataFrame with samples to predict
    - features_all: list of feature names used in the model
    - threshold: fraction of trees that must agree (default 0.5 for majority)

    Returns:
    - DataFrame with predictions, including vote counts and agreement percentage
    """
    n_trees = len(trained_trees)
    n_samples = len(df)

    print(f"Generating ensemble predictions using {n_trees} trees...")

    # Collect predictions from all trees
    all_predictions = []

    for i, tree_data in enumerate(trained_trees):
        # Handle both formats: list of dicts or list of trees
        if isinstance(tree_data, dict):
            tree = tree_data['tree']
        else:
            tree = tree_data

        # Get predictions from this tree
        predictions = predict_batch(tree, df, features_all, death_threshold)
        all_predictions.append(predictions['predictions'].values)

        # if (i + 1) % 5 == 0 or (i + 1) == n_trees:
        #     print(f"  Processed {i + 1}/{n_trees} trees...")

    # Stack all predictions: shape (n_trees, n_samples)
    all_predictions = np.array(all_predictions)

    # Count votes for survival (1) for each passenger
    votes_for_survival = all_predictions.sum(axis=0)  # Sum across trees

    # Calculate voting percentage
    vote_percentage = votes_for_survival / n_trees

    # Majority voting: if >= threshold of trees predict survival, predict 1
    ensemble_predictions = (vote_percentage >= threshold).astype(int)

    # Get real status and calculate accuracy
    real_status = df['Survived'].values
    accuracy = (ensemble_predictions == real_status).astype(int)

    # Create detailed results DataFrame
    df_predictions = pd.DataFrame({
        "PassengerId": df['PassengerId'].values,
        "predictions": ensemble_predictions,
        "votes_for_survival": votes_for_survival,
        "votes_for_death": n_trees - votes_for_survival,
        "vote_percentage": vote_percentage,
        "agreement": np.maximum(vote_percentage, 1 - vote_percentage),  # How much trees agree
        "real_status": real_status,
        "accuracy": accuracy
    })

    # Calculate statistics for each category
    unanimous = df_predictions['agreement'] >= 0.95
    high_agreement = (df_predictions['agreement'] > 0.7) & (df_predictions['agreement'] < 0.95)
    split_decisions = (df_predictions['agreement'] >= 0.5) & (df_predictions['agreement'] <= 0.7)

    unanimous_count = unanimous.sum()
    unanimous_mistakes = (unanimous & (df_predictions['accuracy'] == 0)).sum()

    high_agreement_count = high_agreement.sum()
    high_agreement_mistakes = (high_agreement & (df_predictions['accuracy'] == 0)).sum()

    split_count = split_decisions.sum()
    split_mistakes = (split_decisions & (df_predictions['accuracy'] == 0)).sum()

    total_mistakes = split_mistakes+high_agreement_mistakes+unanimous_mistakes

    print(f"Unanimous decisions (>95%):       {unanimous_count:4d} samples, "
          f"{unanimous_mistakes:3d} mistakes "
          f"({unanimous_mistakes / unanimous_count * 100 if unanimous_count > 0 else 0:.1f}%)")
    print(f"High agreement (95-70%):          {high_agreement_count:4d} samples, "
          f"{high_agreement_mistakes:3d} mistakes "
          f"({high_agreement_mistakes / high_agreement_count * 100 if high_agreement_count > 0 else 0:.1f}%)")
    print(f"Split decisions (50-70%):         {split_count:4d} samples, "
          f"{split_mistakes:3d} mistakes "
          f"({split_mistakes / split_count * 100 if split_count > 0 else 0:.1f}%)")
    print(f"Total samples: {n_samples} with {total_mistakes} mistakes")

    return df_predictions


def print_tree_structure(tree):
    # Print the first example tree structure
    print("\n" + "=" * 60 + "\n### DECISION TREE STRUCTURE ###" + "\n" + ("=" * 60))
    print(f"Root: size={tree['size']}, survival_rate={tree['survival_rate']:.2%}")
    # use print_tree function
    if not tree['leaf']:
        print_tree(tree['left'], "", True)
        print_tree(tree['right'], "", False)
    print("=" * 60)



'''
#####################################
### MAIN PART OF THE CODE         ###
#####################################
'''
# Load the CSV file
file_path = r"C:\Users\Mateusz\Downloads\titanic\train.csv"
df = pd.read_csv(file_path)

# Inspect original df_processed
print(("=" * 60)+"\n### ORIGINAL DATA INSPECTION ###")
data_inspection.check(df, detailed=True)  # Change to False for summary only

# Process the df_processed
processed_df = data_process.calc(df, detailed=False)

# Inspect processed df_processed
print(("=" * 60)+"\n### PROCESSED DATA INSPECTION ###")
data_inspection.check(processed_df, detailed=False)  # Change to False for summary only

# features_all is a list of all variable parameters including the target
features_all = list(df.head())

print("\n### FOUND PARAMETERS IN THE DATA ###")
print(features_all)
# features_all = ['PassengerId', 'Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']

corr_results, features_train = correlation_analysis.comprehensive_correlation_analysis(
    processed_df,
    target='Survived',
    corr_threshold=0.01,
    visualize=False,
    # categorical_columns=['Pclass', 'Sex', 'Cabin', 'Embarked', 'Deck', 'Title']
    # categorical_columns=['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
    categorical_columns=['Cabin', 'Embarked']
    # categorical_columns=None
)
# Manual features train
# features_train = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin']
features_train = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

pd.set_option('display.max_columns', None)
print(processed_df.head())

print("\n### TRAINING ON ###")
print(features_train)
# print(corr_results['feature_importance'])
# # Visualize the survival df_processed (set to False to disable)
# visualize_survival_data.figure(processed_df, enable_visualization=False)


'''
#####################################
### HYPER-PARAMETERS FOR TRAINING ###
#####################################
'''
# # manually set hyperparameters
# min_group = 3
# max_depth = 8
# gini_threshold = 0.01
# death_threshold = 0.50
# n_estimators = 10

# trained hyperparameters
min_group = 3
max_depth = 6
gini_threshold = 0.01
death_threshold = 0.6
n_estimators = 10
rand_seed = None

rand_percent = 50 #percent of how much df_train is used for each single tree
train_share = 0.6
test_share = 0.2
validation_share = 1 - train_share - test_share
'''
#####################################
### HYPER-PARAMETERS FOR TRAINING ###
#####################################
'''
time1 = time()

'''
#####################################
########## DATA SPLITTING to sets 
#####################################
'''
# Get a slice, usage
# df, train_indices = df_random_slice(processed_df, percent=rand_percent, random_seed=None) # no random seed
# df, train_indices = df_random_slice(processed_df, percent=rand_percent, random_seed=rand_seed)
# df
df_train, df_test, df_validation, dict_indices = split_data.calc(processed_df, train_share, test_share, random_seed=rand_seed)
df = df_train
'''
################################################
########## BUILD A SINGLE TREE ON A TRAINING SET
################################################'''
# Build A SINGLE tree, using build_decision_tree on the whole training set
print("\n" + "=" * 60 + "\n### BUILDING A SINGLE TREE ###" + "\n"+ ("=" * 60))
single_tree = build_decision_tree(df_train,
                                  features_train,
                                  max_depth,
                                  gini_threshold,
                                  min_group,
                                  death_threshold)
print_tree_structure(single_tree)
single_tree_prediction = predict_batch(single_tree, df_train, features_train, death_threshold)
accuracy_ratio = single_tree_prediction['accuracy'].mean()
score = single_tree_prediction['accuracy'].sum()
print(f"Accuracy of a single tree on the training data = {accuracy_ratio:.2%}\n")

time2 = time()

'''
#####################################
########## TRAIN THE FOREST 
#####################################
'''
# Train set of trees on random data slices
# random_seed is here useless
# df_trained - df na którym trenował
tree_best, df_trained, final_accuracy, best_index, trained_trees = train_trees(df_train, percent=rand_percent,
                                                                               sessions=n_estimators,
                                                                               max_depth=max_depth,
                                                                               gini_threshold=gini_threshold,
                                                                               min_group=min_group,
                                                                               death_threshold=death_threshold )
tree_predictions = predict_batch(tree_best, df_trained, features_train, death_threshold)
accuracy_ratio = tree_predictions['accuracy'].mean()
score = tree_predictions['accuracy'].sum()
# Results for decision tree training
print(f"Final score of the obtained tree No.{best_index}: \n\t"
      f"on its own set No.{best_index}: {score} out of {tree_best['size']}, giving {accuracy_ratio:.2%} accuracy, \n\t "
      f"on all training data: {final_accuracy:.2%} averaged.")

time3 = time()

# # Print the best trained tree structure
# tree = tree_best
# print_tree_structure(tree)


# Test predictions of THE BEST TREE on the full dataset (processed_df)
best_tree_prediction = predict_batch(tree_best, df_validation, features_train, death_threshold)
accuracy_ratio=best_tree_prediction['accuracy'].mean()
print(f"Accuracy on the validation data = {accuracy_ratio:.2%}\n")

'''
############################################
#### FINAL Predictions of the random forest on the training data
############################################'''
predict_RF = predict_ensemble(trained_trees, df_train, features_train, threshold=0.5)
RF_score = predict_RF['accuracy'].mean()
print(f"Accuracy of the Random Forest algorithm on the training data = {RF_score:.2%}\n")
'''
############################################
#### FINAL Predictions of the random forest on the test data
############################################'''
predict_RF = predict_ensemble(trained_trees, df_test, features_train, threshold=0.5)
RF_score = predict_RF['accuracy'].mean()
print(f"Accuracy of the Random Forest algorithm on the test data = {RF_score:.2%}\n")
'''
############################################
#### FINAL Predictions of the random forest on the validation data
############################################'''
predict_RF = predict_ensemble(trained_trees, df_validation, features_train, threshold=0.5)
RF_score = predict_RF['accuracy'].mean()
print(f"Accuracy of the Random Forest algorithm on the validation data = {RF_score:.2%}\n")
'''
############################################
#### FINAL Predictions of the random forest on the full data
############################################'''
predict_RF = predict_ensemble(trained_trees, processed_df, features_train, threshold=0.5)
RF_score = predict_RF['accuracy'].mean()
print(f"Accuracy of the Random Forest algorithm on the full data = {RF_score:.2%}\n")


time4 = time()

#ADD PRINTING TO A FILE
# # for idx, row in df.iterrows():
# #     df_predictions['predictions']==df_predictions['real status']:
# if df_predictions['predictions'] == df_predictions['real status']:

'''
############################################
#### Random Search for the hyperparameters
############################################'''
def hyperparameter_search_1d(do=True, n_repeats=5):
    if not do:
        return None

    results = []

    global min_group, max_depth, gini_threshold, death_threshold, n_estimators

    BASE_PARAMS = {
        "min_group": min_group,
        "max_depth": max_depth,
        "gini_threshold": gini_threshold,
        "death_threshold": death_threshold,
        "n_estimators": n_estimators
    }

    PARAM_RANGES = {
        "min_group": range(2, 5, 1),
        "max_depth": range(4, 12, 1),
        "gini_threshold": np.arange(0.01, 0.05, 0.01),
        "death_threshold": np.arange(0.2, 0.7, 0.1),
        "n_estimators": range(6, 18, 2)
    }

    print("Skanowanie hiperparametrów (1D – jeden parametr naraz)")
    print("=" * 80)

    for param_name, param_values in PARAM_RANGES.items():
        print(f"\n>>> TESTOWANIE PARAMETRU: {param_name} o wartości {param_values}")
        print("-" * 80)

        for value in param_values:
            params = BASE_PARAMS.copy()
            params[param_name] = value

            print("-" * 80)
            print(f"Testowanie parametru {param_name} = {value}")

            accuracies = []

            for run in range(n_repeats):
                print(f"  → Run {run + 1}/{n_repeats}")

                try:
                    _, _, _, _, trained_trees = train_trees(
                        df_train,
                        min_group=int(params["min_group"]),
                        max_depth=int(params["max_depth"]),
                        percent=rand_percent,
                        gini_threshold=float(params["gini_threshold"]),
                        death_threshold=float(params["death_threshold"]),
                        sessions=int(params["n_estimators"])
                    )

                    test_accuracy = predict_ensemble(
                        trained_trees,
                        df_test,
                        features_train,
                        threshold=0.5
                    )["accuracy"].mean()

                    accuracies.append(test_accuracy)

                except Exception as e:
                    print(f"    Błąd w run {run + 1}: {e}")

            if len(accuracies) > 0:
                mean_acc = np.mean(accuracies)
                std_acc = np.std(accuracies)

                results.append({
                    "scanned_param": param_name,
                    "param_value": value,
                    **params,
                    "test_accuracy_mean": mean_acc,
                    "test_accuracy_std": std_acc
                })

                print(f"  → Mean accuracy: {mean_acc:.6f} ± {std_acc:.6f}")

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("test_accuracy_mean", ascending=False)

    print("\n" + "=" * 80)
    print("TOP 30 WYNIKÓW (GLOBALNIE)")
    print("=" * 80)
    print(results_df.head(30).to_string(index=False))

    return results_df

hyperparameter_search_1d(False)

time5 = time()

# summary
print("Trained on features: ")
print(features_train, "\n")


# End timing
end_time = time()
exec1 = time1 - start_time
exec2 = time2 - time1
exec3 = time3 - time2
exec4 = time4 - time3
exec5 = time5 - time4
execution_time = end_time - start_time
print("=" * 80)
print(f"Time part 1: {exec1:7.3f} seconds - loading modules and processing the data")
print(f"Time part 2: {exec2:7.3f} seconds - single tree build time")
print(f"Time part 3: {exec3:7.3f} seconds - full training of {n_estimators} trees on {rand_percent / 100:.1%} of data")
print(f"In part 3:   {time_predictions:7.3f} seconds - spent on calculating accuracy/predictions")
print(f"Time part 4: {exec4:7.3f} seconds - final testing")
print(f"Time part 5: {exec5:7.3f} seconds - for hyperparameter search")
print(f"Total execution time: {execution_time:.4f} seconds")


''' DODAĆ 
0. losowanie klasyfikatorów - wcale nie najlepiej dzielących
1. Dodać system rozpoznający kluczowe zmienne spośród zadanych i wybiera które bierze do analizy
2. Dodać samooptymalizację głębokości drzewa (da się?)
3. Rozdzielić funkcję na main i powiększyć opcje 
4. Dodać cross-validation matrix czy coś, korelacje pomiędzy zmiennymi
5. dodać, by dane do testowania były usuwane z danych do uczenia modelu - zbiór treningowy, walidacyjny, testowy 
'''