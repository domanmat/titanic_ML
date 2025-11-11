import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def calculate_correlation_matrix(df, target='Survived', method='pearson'):
    """
    Calculate correlation matrix for numerical features with target variable.

    Parameters:
    - df: DataFrame with processed data
    - target: target variable name
    - method: 'pearson', 'spearman', or 'kendall'

    Returns:
    - corr_matrix: full correlation matrix
    - target_corr: correlations with target variable (sorted)
    """
    # Select only numerical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Calculate correlation matrix
    corr_matrix = df[numerical_cols].corr(method=method)

    # Get correlations with target variable
    if target in corr_matrix.columns:
        target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
    else:
        target_corr = None

    return corr_matrix, target_corr


def visualize_correlation_matrix(corr_matrix, figsize=(12, 10), save_path=None):
    """
    Create a heatmap of the correlation matrix.

    Parameters:
    - corr_matrix: correlation matrix from calculate_correlation_matrix
    - figsize: figure size tuple
    - save_path: optional path to save figure
    """
    plt.figure(figsize=figsize)

    # Create heatmap
    sns.heatmap(corr_matrix,
                annot=True,  # Show correlation values
                fmt='.2f',  # Format to 2 decimal places
                cmap='coolwarm',  # Color scheme
                center=0,  # Center colormap at 0
                square=True,
                linewidths=0.5,
                cbar_kws={"shrink": 0.8})

    plt.title('Feature Correlation Matrix', fontsize=16, pad=20)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


def visualize_target_correlations(target_corr, target_name='Survived',
                                  threshold=0.1, figsize=(10, 6)):
    """
    Create bar plot of feature correlations with target variable.

    Parameters:
    - target_corr: Series of correlations with target
    - target_name: name of target variable
    - threshold: minimum absolute correlation to display
    - figsize: figure size tuple
    """
    # Filter by threshold
    significant_corr = target_corr[abs(target_corr) >= threshold]

    plt.figure(figsize=figsize)

    # Create color map (positive = green, negative = red)
    colors = ['green' if x > 0 else 'red' for x in significant_corr.values]

    # Create bar plot
    bars = plt.barh(range(len(significant_corr)), significant_corr.values, color=colors, alpha=0.7)
    plt.yticks(range(len(significant_corr)), significant_corr.index)
    plt.xlabel('Correlation with ' + target_name, fontsize=12)
    plt.title(f'Feature Correlations with {target_name}\n(threshold: |r| ≥ {threshold})',
              fontsize=14, pad=15)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    plt.grid(axis='x', alpha=0.3)

    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, significant_corr.values)):
        plt.text(value, i, f' {value:.3f}',
                 va='center', ha='left' if value > 0 else 'right',
                 fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.show()


def print_correlation_summary(target_corr, threshold=0.1):
    """
    Print text summary of correlations with target variable.

    Parameters:
    - target_corr: Series of correlations with target
    - threshold: minimum absolute correlation to report
    """
    print("=" * 70)
    print("CORRELATION ANALYSIS SUMMARY")
    print("=" * 70)

    significant = target_corr[abs(target_corr) >= threshold]

    print(f"\nFeatures with |correlation| ≥ {threshold}:")
    print("-" * 70)

    for feature, corr in significant.items():
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        direction = "Positive" if corr > 0 else "Negative"
        print(f"{feature:20s}: {corr:7.4f}  ({strength} {direction})")

    print("-" * 70)
    print(f"Total significant features: {len(significant)}")
    print("=" * 70)


def cramers_v(x, y):
    """
    Calculate Cramér's V statistic for categorical-categorical association.
    Ranges from 0 (no association) to 1 (perfect association).
    """
    confusion_matrix = pd.crosstab(x, y)
    chi2 = sum([(o - e) ** 2 / e for o, e in
                zip(confusion_matrix.values.flatten(),
                    np.outer(confusion_matrix.sum(axis=1),
                             confusion_matrix.sum(axis=0)).flatten() /
                    confusion_matrix.sum().sum())])
    n = confusion_matrix.sum().sum()
    min_dim = min(confusion_matrix.shape) - 1
    return np.sqrt(chi2 / (n * min_dim)) if min_dim > 0 else 0


def analyze_categorical_associations(df, target='Survived', categorical_cols=None):
    """
    Calculate associations between categorical variables and target.
    Uses Cramér's V for categorical-categorical relationships.

    Parameters:
    - df: DataFrame
    - target: target variable name
    - categorical_cols: list of categorical column names (auto-detected if None)

    Returns:
    - associations: Series of Cramér's V values
    """
    if categorical_cols is None:
        # Auto-detect categorical columns (object type or few unique values)
        categorical_cols = []
        for col in df.columns:
            if col != target:
                if df[col].dtype == 'object' or df[col].nunique() < 10:
                    categorical_cols.append(col)

    associations = {}
    for col in categorical_cols:
        if col != target and col in df.columns:
            try:
                associations[col] = cramers_v(df[col], df[target])
            except:
                associations[col] = np.nan

    return pd.Series(associations).sort_values(ascending=False)


def comprehensive_correlation_analysis(df, target, corr_threshold,
                                       visualize, categorical_cols=None):
    """
    Perform complete correlation analysis including both numerical and categorical features.

    Parameters:
    - df: DataFrame with processed data
    - target: target variable name
    - corr_threshold: minimum correlation to report
    - visualize: whether to create plots
    - categorical_cols: list of categorical columns for association analysis

    Returns:
    - results: dictionary with correlation matrix, target correlations, and associations
    """
    print("\n" + "=" * 70)
    print("COMPREHENSIVE CORRELATION ANALYSIS")
    print("=" * 70)

    results = {}
    final_features =[]

    # 1. Numerical correlations (Pearson)
    print("\n[1/3] Calculating numerical feature correlations...")
    corr_matrix, target_corr = calculate_correlation_matrix(df, target, method='pearson')
    results['corr_matrix'] = corr_matrix
    results['target_correlations'] = target_corr

    if target_corr is not None:
        print_correlation_summary(target_corr, threshold=corr_threshold)

        #  collect numeric features above threshold
        strong_num = [feat for feat, val in target_corr.items()
                      if abs(val) > corr_threshold]
        final_features.extend(strong_num)

        if visualize:
            visualize_correlation_matrix(corr_matrix)
            visualize_target_correlations(target_corr, target_name=target,
                                          threshold=corr_threshold)

    # 2. Categorical associations (Cramér's V)
    print("\n[2/3] Calculating categorical feature associations...")
    categorical_associations = analyze_categorical_associations(df, target, categorical_cols)
    results['categorical_associations'] = categorical_associations

    if len(categorical_associations) > 0:
        print("\nCategorical Feature Associations (Cramér's V):")
        print("-" * 70)
        for feature, assoc in categorical_associations.items():
            if not np.isnan(assoc):
                strength = "Strong" if assoc > 0.3 else "Moderate" if assoc > 0.1 else "Weak"
                print(f"{feature:20s}: {assoc:7.4f}  ({strength})")

                # collect categorical features above threshold
                if assoc > corr_threshold:
                    final_features.append(feature)

        print("-" * 70)

    # 3. Combined feature importance ranking
    print("\n[3/3] Creating combined feature importance ranking...")

    # Combine numerical and categorical measures
    all_features = {}
    if target_corr is not None:
        all_features.update({f: abs(v) for f, v in target_corr.items()})
    all_features.update({f: v for f, v in categorical_associations.items() if not np.isnan(v)})

    feature_importance = pd.Series(all_features).sort_values(ascending=False)
    results['feature_importance'] = feature_importance

    print("\nCombined Feature Importance Ranking:")
    print("-" * 70)
    for i, (feature, importance) in enumerate(feature_importance.items(), 1):
        print(f"{i:2d}. {feature:20s}: {importance:7.4f}")
    print("=" * 70)

    # Store & return the final list of strong features
    results['final_features'] = final_features

    print(f"\nFeatures with |corr|/assoc > {corr_threshold}:")
    print(final_features if final_features else "None")
    print("=" * 70)

    return results, final_features
    # return results


# Example usage function
def analyze_titanic_correlations(processed_df, enable_visualization=True):
    """
    Convenience function for Titanic dataset correlation analysis.

    Parameters:
    - processed_df: processed Titanic DataFrame
    - enable_visualization: whether to show plots

    Returns:
    - results dictionary with all correlation analyses
    """
    categorical_cols = ['Cabin', 'Embarked']  # Known categorical columns in Titanic

    results = comprehensive_correlation_analysis(
        processed_df,
        target='Survived',
        corr_threshold=0.01,
        visualize=enable_visualization,
        categorical_cols=categorical_cols
    )

    return results



# file_path = r"C:\Users\Mateusz\Downloads\titanic\train.csv"
# df = pd.read_csv(file_path)
#
# corr_results, features_train = comprehensive_correlation_analysis(
#     df,
#     target='Survived',
#     corr_threshold=0.05,
#     visualize=False,
#     categorical_cols=['Pclass', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked']
#     # categorical_cols=None
# )
#
# print(1)