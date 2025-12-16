"""
Improved Titanic Machine Learning Pipeline
Integrates automatic feature selection, depth optimization, and correlation analysis
"""

import pandas as pd
import numpy as np
import time

# Your existing imports
import survival_counter
import gini_Y_impurity
import data_inspection
import data_process

# New imports
import correlation_analysis
import feature_selection

# Colors for output
GREEN = "\033[32m"
RED = "\033[31m"
BLUE = "\033[34m"
RESET = "\033[0m"

# Start timing
start_time = time.time()
print("Executing improved ML pipeline...")


def run_improved_pipeline(file_path, enable_visualization=True,
                          feature_strategy='balanced', optimize_depth=False):
    """
    Run complete ML pipeline with automatic feature selection and optimization.

    Parameters:
    - file_path: path to training data CSV
    - enable_visualization: whether to show correlation plots
    - feature_strategy: 'aggressive', 'balanced', or 'conservative'
    - optimize_depth: whether to auto-optimize tree depth

    Returns:
    - results: dictionary with all results and models
    """
    results = {}
    times = {}

    # ==========================
    # PHASE 1: DATA LOADING AND PREPROCESSING
    # ==========================
    t1 = time.time()
    print("\n" + "=" * 90)
    print(f"{BLUE}PHASE 1: DATA LOADING AND PREPROCESSING{RESET}")
    print("=" * 90)

    # Load data
    df = pd.read_csv(file_path)
    results['original_shape'] = df.shape

    # Inspect original data
    print("\n### ORIGINAL DATA INSPECTION ###")
    data_inspection.check(df, detailed=False)

    # Process data
    processed_df = data_process.calc(df, detailed=False)

    # Feature engineering
    print("\n### FEATURE ENGINEERING ###")
    processed_df = feature_engineering(processed_df)

    # Inspect processed data
    print("\n### PROCESSED DATA INSPECTION ###")
    data_inspection.check(processed_df, detailed=False)

    results['processed_df'] = processed_df
    times['preprocessing'] = time.time() - t1

    # ==========================
    # PHASE 2: CORRELATION ANALYSIS
    # ==========================
    t2 = time.time()
    print("\n" + "=" * 90)
    print(f"{BLUE}PHASE 2: CORRELATION ANALYSIS{RESET}")
    print("=" * 90)

    corr_results = correlation_analysis.comprehensive_correlation_analysis(
        processed_df,
        target='Survived',
        corr_threshold=0.1,
        visualize=enable_visualization,
        categorical_columns=['Cabin', 'Embarked', 'Deck', 'Title']
    )
    results['correlation_analysis'] = corr_results
    times['correlation'] = time.time() - t2

    # ==========================
    # PHASE 3: AUTOMATIC FEATURE SELECTION
    # ==========================
    t3 = time.time()
    print("\n" + "=" * 90)
    print(f"{BLUE}PHASE 3: AUTOMATIC FEATURE SELECTION{RESET}")
    print("=" * 90)

    selected_features, feature_report = feature_selection.select_titanic_features(
        processed_df,
        strategy=feature_strategy,
        exclude_features=['PassengerId', 'Name', 'Ticket']
    )

    print(f"\n{GREEN}Selected features for training: {selected_features}{RESET}")
    results['selected_features'] = selected_features
    results['feature_report'] = feature_report
    times['feature_selection'] = time.time() - t3

    # ==========================
    # PHASE 4: HYPERPARAMETER OPTIMIZATION
    # ==========================
    t4 = time.time()
    print("\n" + "=" * 90)
    print(f"{BLUE}PHASE 4: HYPERPARAMETER OPTIMIZATION{RESET}")
    print("=" * 90)

    if optimize_depth:
        print("\n### OPTIMIZING TREE DEPTH ###")
        optimal_depth, cv_scores = optimize_tree_depth_cv(
            processed_df,
            selected_features,
            depth_range=(3, 12),
            cv_folds=5
        )
        results['optimal_depth'] = optimal_depth
        results['cv_scores'] = cv_scores
    else:
        optimal_depth = 10  # Default
        print(f"Using default depth: {optimal_depth}")

    times['optimization'] = time.time() - t4

    # ==========================
    # PHASE 5: MODEL TRAINING
    # ==========================
    t5 = time.time()
    print("\n" + "=" * 90)
    print(f"{BLUE}PHASE 5: MODEL TRAINING (Random Forest){RESET}")
    print("=" * 90)

    # Training parameters
    training_params = {
        'max_depth': optimal_depth,
        'min_group': 1,
        'gini_threshold': 0.01,
        'rand_percent': 50,
        'n_estimators': 10
    }

    print(f"\nTraining parameters:")
    for param, value in training_params.items():
        print(f"  {param}: {value}")

    # Train random forest
    from main import train_trees, predict_ensemble

    tree_best, df_trained, final_accuracy, best_index, trained_trees = train_trees(
        processed_df,
        percent=training_params['rand_percent'],
        sessions=training_params['n_estimators']
    )

    results['best_tree'] = tree_best
    results['best_tree_index'] = best_index
    results['best_tree_accuracy'] = final_accuracy
    results['trained_trees'] = trained_trees
    times['training'] = time.time() - t5

    # ==========================
    # PHASE 6: EVALUATION
    # ==========================
    t6 = time.time()
    print("\n" + "=" * 90)
    print(f"{BLUE}PHASE 6: MODEL EVALUATION{RESET}")
    print("=" * 90)

    # Evaluate ensemble on full dataset
    features_all = selected_features + ['PassengerId', 'Survived']
    predict_RF = predict_ensemble(trained_trees, processed_df, features_all, threshold=0.5)

    rf_accuracy = predict_RF['accuracy'].mean()
    results['rf_predictions'] = predict_RF
    results['rf_accuracy'] = rf_accuracy

    print(f"\n{GREEN}{'=' * 90}{RESET}")
    print(f"{GREEN}FINAL RESULTS:{RESET}")
    print(f"{GREEN}{'=' * 90}{RESET}")
    print(f"Best single tree accuracy: {final_accuracy:.2%}")
    print(f"Random Forest accuracy:    {rf_accuracy:.2%}")
    print(f"Improvement:               {(rf_accuracy - final_accuracy) * 100:+.2f} percentage points")
    print(f"{GREEN}{'=' * 90}{RESET}")

    times['evaluation'] = time.time() - t6

    # ==========================
    # TIMING SUMMARY
    # ==========================
    total_time = time.time() - start_time
    times['total'] = total_time
    results['times'] = times

    print(f"\n{'=' * 90}")
    print("EXECUTION TIME BREAKDOWN:")
    print(f"{'=' * 90}")
    print(f"  Preprocessing:        {times['preprocessing']:7.2f}s")
    print(f"  Correlation Analysis: {times['correlation']:7.2f}s")
    print(f"  Feature Selection:    {times['feature_selection']:7.2f}s")
    print(f"  Hyperparameter Opt:   {times['optimization']:7.2f}s")
    print(f"  Model Training:       {times['training']:7.2f}s")
    print(f"  Evaluation:           {times['evaluation']:7.2f}s")
    print(f"  {'-' * 86}")
    print(f"  TOTAL:                {total_time:7.2f}s")
    print(f"{'=' * 90}\n")

    return results


def feature_engineering(df):
    """
    Create additional features from existing ones.
    """
    print("Creating engineered features...")
    df = df.copy()

    # Family size
    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['FamilySize'] = df['FamilySize'].astype('int8')

    # Is alone
    df['IsAlone'] = (df['FamilySize'] == 1).astype('int8')

    # Fare per person (avoid division by zero)
    df['FarePerPerson'] = df['Fare'] / df['FamilySize']
    df['FarePerPerson'] = df['FarePerPerson'].astype('float32')

    # Age groups (only for valid ages)
    df['AgeGroup'] = -1
    valid_age_mask = df['Age'] != -1
    if valid_age_mask.any():
        df.loc[valid_age_mask, 'AgeGroup'] = pd.cut(
            df.loc[valid_age_mask, 'Age'],
            bins=[0, 12, 18, 35, 60, 100],
            labels=[0, 1, 2, 3, 4]
        ).astype('int8')
    df['AgeGroup'] = df['AgeGroup'].astype('int8')

    # Cabin deck (first letter)
    df['Deck'] = df['Cabin'].str[0]
    df['HasCabin'] = (df['Cabin'] != 'None').astype('int8')

    print(f"  Created features: FamilySize, IsAlone, FarePerPerson, AgeGroup, Deck, HasCabin")

    return df


def optimize_tree_depth_cv(df, features, depth_range=(3, 15), cv_folds=5):
    """
    Find optimal max_depth using cross-validation.
    """
    from sklearn.model_selection import KFold
    from main import build_decision_tree, predict_batch

    print(f"\nTesting depths from {depth_range[0]} to {depth_range[1]} with {cv_folds}-fold CV...")

    cv_scores = {}
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)

    for depth in range(depth_range[0], depth_range[1] + 1):
        fold_accuracies = []

        for fold, (train_idx, val_idx) in enumerate(kf.split(df)):
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            # Build tree on training fold
            tree = build_decision_tree(
                train_df, features,
                max_depth=depth,
                gini_threshold=0.01,
                min_group=1,
                current_depth=0
            )

            # Evaluate on validation fold
            features_all = features + ['PassengerId', 'Survived']
            predictions = predict_batch(tree, val_df, features_all)
            accuracy = predictions['accuracy'].mean()
            fold_accuracies.append(accuracy)

        cv_scores[depth] = {
            'mean': np.mean(fold_accuracies),
            'std': np.std(fold_accuracies),
            'folds': fold_accuracies
        }

        print(f"  Depth {depth:2d}: {cv_scores[depth]['mean']:.4f} "
              f"(+/- {cv_scores[depth]['std']:.4f})")

    # Find best depth
    best_depth = max(cv_scores, key=lambda d: cv_scores[d]['mean'])
    best_score = cv_scores[best_depth]['mean']

    print(f"\n{GREEN}Optimal depth: {best_depth} with CV accuracy {best_score:.4f}{RESET}")

    return best_depth, cv_scores


def save_results(results, output_dir='./results'):
    """
    Save all results to files.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Save predictions
    results['rf_predictions'].to_csv(f"{output_dir}/predictions.csv", index=False)

    # Save feature importance
    if 'feature_report' in results:
        results['feature_report']['feature_scores'].to_csv(
            f"{output_dir}/feature_importance.csv"
        )

    # Save model parameters
    with open(f"{output_dir}/model_summary.txt", 'w') as f:
        f.write("TITANIC ML MODEL SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Selected Features: {results['selected_features']}\n")
        f.write(f"Optimal Depth: {results.get('optimal_depth', 'N/A')}\n")
        f.write(f"Best Tree Accuracy: {results['best_tree_accuracy']:.4f}\n")
        f.write(f"Random Forest Accuracy: {results['rf_accuracy']:.4f}\n")

    print(f"\nResults saved to {output_dir}/")


# ==========================
# MAIN EXECUTION
# ==========================
if __name__ == "__main__":
    # Configuration
    FILE_PATH = r"C:\Users\Mateusz\Downloads\titanic\train.csv"

    # Run pipeline
    results = run_improved_pipeline(
        file_path=FILE_PATH,
        enable_visualization=False,  # Set to True to see plots
        feature_strategy='balanced',  # 'aggressive', 'balanced', or 'conservative'
        optimize_depth=True  # Set to False to skip depth optimization
    )

    # Optionally save results
    # save_results(results, output_dir='./titanic_results')

    print(f"\n{GREEN}Pipeline completed successfully!{RESET}\n")