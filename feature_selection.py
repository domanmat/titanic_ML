import pandas as pd
import numpy as np
from correlation_analysis import calculate_correlation_matrix, cramers_v


def calculate_feature_importance_gini(df, features, target='Survived', n_iterations=10):
    """
    Calculate feature importance using Gini impurity reduction.
    Tests how much each feature reduces Gini when used for splitting.

    Parameters:
    - df: DataFrame with data
    - features: list of features to evaluate
    - target: target variable name
    - n_iterations: number of random samples to test

    Returns:
    - feature_importance: Series of importance scores (higher = more important)
    """
    from gini_Y_impurity import calc as calc_gini

    importance_scores = {feature: [] for feature in features}

    for _ in range(n_iterations):
        # Sample random subset for faster computation
        sample_df = df.sample(frac=0.7, random_state=None)

        # Calculate baseline Gini
        n_total = len(sample_df)
        survived = sample_df[target].sum()
        baseline_gini = calc_gini(n_total, survived)

        for feature in features:
            # Sort by feature
            sorted_df = sample_df.sort_values(feature)
            survived_vals = sorted_df[target].values
            feature_vals = sorted_df[feature].values

            # Try median split
            median_idx = len(sorted_df) // 2

            n_left = median_idx
            n_right = n_total - median_idx

            if n_left == 0 or n_right == 0:
                importance_scores[feature].append(0)
                continue

            survived_left = survived_vals[:median_idx].sum()
            survived_right = survived_vals[median_idx:].sum()

            left_gini = calc_gini(n_left, survived_left)
            right_gini = calc_gini(n_right, survived_right)

            weighted_gini = (n_left / n_total) * left_gini + (n_right / n_total) * right_gini

            # Importance = reduction in Gini
            gini_reduction = baseline_gini - weighted_gini
            importance_scores[feature].append(gini_reduction)

    # Average importance across iterations
    avg_importance = {feature: np.mean(scores) for feature, scores in importance_scores.items()}

    return pd.Series(avg_importance).sort_values(ascending=False)


def calculate_mutual_information(df, features, target='Survived', bins=10):
    """
    Calculate mutual information between features and target.
    Measures how much knowing a feature reduces uncertainty about target.

    Parameters:
    - df: DataFrame
    - features: list of features to evaluate
    - target: target variable name
    - bins: number of bins for discretizing continuous variables

    Returns:
    - mi_scores: Series of mutual information scores
    """

    def entropy(y):
        """Calculate entropy of a variable."""
        value_counts = pd.Series(y).value_counts()
        probabilities = value_counts / len(y)
        return -np.sum(probabilities * np.log2(probabilities + 1e-10))

    mi_scores = {}
    target_entropy = entropy(df[target])

    for feature in features:
        # Discretize continuous features
        if df[feature].dtype in ['float32', 'float64']:
            # Handle -1 (missing) specially
            feature_binned = pd.cut(df[feature][df[feature] != -1], bins=bins, labels=False)
            # Keep -1 as separate bin
            feature_vals = df[feature].copy()
            feature_vals[feature_vals != -1] = feature_binned
        else:
            feature_vals = df[feature]

        # Calculate conditional entropy H(Y|X)
        conditional_entropy = 0
        for x_val in feature_vals.unique():
            mask = feature_vals == x_val
            p_x = mask.sum() / len(df)
            y_given_x = df[target][mask]
            if len(y_given_x) > 0:
                conditional_entropy += p_x * entropy(y_given_x)

        # Mutual Information = H(Y) - H(Y|X)
        mi_scores[feature] = target_entropy - conditional_entropy

    return pd.Series(mi_scores).sort_values(ascending=False)


def calculate_variance_threshold(df, features, threshold=0.01):
    """
    Identify features with low variance (potentially uninformative).

    Parameters:
    - df: DataFrame
    - features: list of features to evaluate
    - threshold: minimum variance threshold

    Returns:
    - low_variance_features: list of features below threshold
    - feature_variances: Series of variance values
    """
    variances = {}
    for feature in features:
        if df[feature].dtype in ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']:
            # For numerical features, calculate standard variance
            variances[feature] = df[feature].var()
        else:
            # For categorical features, calculate normalized entropy
            value_counts = df[feature].value_counts()
            probabilities = value_counts / len(df)
            entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
            max_entropy = np.log2(len(value_counts))
            variances[feature] = entropy / max_entropy if max_entropy > 0 else 0

    variance_series = pd.Series(variances)
    low_variance = variance_series[variance_series < threshold].index.tolist()

    return low_variance, variance_series.sort_values(ascending=False)


def select_features_multi_method(df, features, target='Survived',
                                 top_n=None, threshold=0.1, methods='all'):
    """
    Select best features using multiple methods and combine results.

    Parameters:
    - df: DataFrame
    - features: list of candidate features
    - target: target variable name
    - top_n: number of top features to select (None = use threshold)
    - threshold: minimum score threshold (used if top_n is None)
    - methods: 'all', 'correlation', 'gini', 'mutual_info', or list of methods

    Returns:
    - selected_features: list of selected features
    - scores: DataFrame with scores from all methods
    """
    print("\n" + "=" * 70)
    print("FEATURE SELECTION - MULTI-METHOD ANALYSIS")
    print("=" * 70)

    scores_dict = {}

    # Available methods
    if methods == 'all':
        methods = ['correlation', 'gini', 'mutual_info']
    elif isinstance(methods, str):
        methods = [methods]

    # 1. Correlation-based
    if 'correlation' in methods:
        print("\n[1/3] Computing correlation-based importance...")
        numerical_features = [f for f in features if df[f].dtype in
                              ['float32', 'float64', 'int8', 'int16', 'int32', 'int64']]
        if numerical_features:
            _, target_corr = calculate_correlation_matrix(df[numerical_features + [target]], target)
            scores_dict['correlation'] = target_corr.abs()

        # Add categorical associations
        categorical_features = [f for f in features if f not in numerical_features]
        if categorical_features:
            cat_scores = {}
            for feature in categorical_features:
                try:
                    cat_scores[feature] = cramers_v(df[feature], df[target])
                except:
                    cat_scores[feature] = 0
            if 'correlation' in scores_dict:
                scores_dict['correlation'] = pd.concat([
                    scores_dict['correlation'],
                    pd.Series(cat_scores)
                ])
            else:
                scores_dict['correlation'] = pd.Series(cat_scores)

    # 2. Gini-based importance
    if 'gini' in methods:
        print("[2/3] Computing Gini-based importance...")
        scores_dict['gini'] = calculate_feature_importance_gini(df, features, target)

    # 3. Mutual Information
    if 'mutual_info' in methods:
        print("[3/3] Computing mutual information...")
        scores_dict['mutual_info'] = calculate_mutual_information(df, features, target)

    # Combine scores
    scores_df = pd.DataFrame(scores_dict)

    # Normalize each method to 0-1 range
    for col in scores_df.columns:
        max_val = scores_df[col].max()
        if max_val > 0:
            scores_df[col] = scores_df[col] / max_val

    # Calculate average score
    scores_df['average'] = scores_df.mean(axis=1)
    scores_df = scores_df.sort_values('average', ascending=False)

    # Select features
    if top_n is not None:
        selected_features = scores_df.head(top_n).index.tolist()
    else:
        selected_features = scores_df[scores_df['average'] >= threshold].index.tolist()

    # Print results
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE SCORES (normalized)")
    print("=" * 70)
    print(scores_df.to_string())
    print("\n" + "=" * 70)
    print(f"SELECTED FEATURES (top_n={top_n}, threshold={threshold}):")
    print("-" * 70)
    for i, feature in enumerate(selected_features, 1):
        avg_score = scores_df.loc[feature, 'average']
        print(f"{i:2d}. {feature:20s} (avg score: {avg_score:.4f})")
    print("=" * 70)

    return selected_features, scores_df


def auto_select_features(df, all_features, target='Survived',
                         selection_strategy='balanced', max_features=10):
    """
    Automatically select the most important features for modeling.

    Parameters:
    - df: DataFrame
    - all_features: list of all available features (excluding target)
    - target: target variable name
    - selection_strategy: 'aggressive' (fewer features), 'balanced', or 'conservative' (more features)
    - max_features: maximum number of features to select

    Returns:
    - selected_features: list of selected features
    - report: dictionary with selection details
    """
    print("\n" + "=" * 70)
    print(f"AUTOMATIC FEATURE SELECTION - {selection_strategy.upper()} STRATEGY")
    print("=" * 70)

    # Remove low-variance features
    print("\n[Step 1] Removing low-variance features...")
    variance_thresholds = {'aggressive': 0.05, 'balanced': 0.02, 'conservative': 0.01}
    low_var_features, variances = calculate_variance_threshold(
        df, all_features, threshold=variance_thresholds.get(selection_strategy, 0.02)
    )

    if low_var_features:
        print(f"Removed {len(low_var_features)} low-variance features: {low_var_features}")
        remaining_features = [f for f in all_features if f not in low_var_features]
    else:
        print("No low-variance features found.")
        remaining_features = all_features

    # Select based on multiple methods
    print(f"\n[Step 2] Selecting top features from {len(remaining_features)} candidates...")

    top_n_map = {'aggressive': min(5, len(remaining_features)),
                 'balanced': min(8, len(remaining_features)),
                 'conservative': min(max_features, len(remaining_features))}

    selected_features, scores = select_features_multi_method(
        df, remaining_features, target,
        top_n=top_n_map.get(selection_strategy, 8),
        methods='all'
    )

    # Create report
    report = {
        'strategy': selection_strategy,
        'original_features': len(all_features),
        'low_variance_removed': len(low_var_features),
        'final_selected': len(selected_features),
        'selected_features': selected_features,
        'feature_scores': scores,
        'variance_scores': variances
    }

    print(f"\n[Step 3] Feature selection complete!")
    print(f"Original features: {len(all_features)}")
    print(f"Low variance removed: {len(low_var_features)}")
    print(f"Final selected: {len(selected_features)}")
    print("=" * 70)

    return selected_features, report


# Convenience function for Titanic dataset
def select_titanic_features(processed_df, strategy='balanced', exclude_features=None):
    """
    Select best features for Titanic dataset.

    Parameters:
    - processed_df: processed Titanic DataFrame
    - strategy: 'aggressive', 'balanced', or 'conservative'
    - exclude_features: list of features to exclude (e.g., ['PassengerId', 'Name', 'Ticket'])

    Returns:
    - selected_features: list of selected features
    - report: selection report dictionary
    """
    # Get all potential features
    all_features = [col for col in processed_df.columns if col != 'Survived']

    # Exclude specified features
    if exclude_features is None:
        exclude_features = ['PassengerId', 'Name', 'Ticket']  # Usually not predictive

    all_features = [f for f in all_features if f not in exclude_features]

    print(f"Available features for selection: {all_features}")

    selected_features, report = auto_select_features(
        processed_df, all_features, target='Survived',
        selection_strategy=strategy, max_features=10
    )

    return selected_features, report