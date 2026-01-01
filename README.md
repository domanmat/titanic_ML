# Titanic Survival Prediction - Random Forest from Scratch

A complete **Random Forest** implementation built from scratch in Python to predict passenger survival on the Titanic dataset. No scikit-learn for the core algorithm - coded from scratch to deep-dive into the algorithm to learn its rules and behavior.  

## What This Does

Implements a full Random Forest classifier including:
- **Decision Trees** with Gini impurity splitting
- **Bootstrap aggregating (Bagging)** with random sampling
- **Out-of-Bag (OOB)** error estimation
- **Feature randomization** at each split
- **Ensemble voting** for final predictions
- **Hyperparameter tuning** with grid search

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install pandas numpy matplotlib seaborn
   ```

2. **Download the Titanic dataset:**
   - Get `train.csv` from [Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic/data)
   - Update the path in `main.py`:
     ```python
     file_path = r"C:\Users\YourName\Downloads\titanic\train.csv"
     ```

3. **Run the model:**
   ```bash
   python main.py
   ```

## 📊 Results

- **Single Decision Tree:** ~68% accuracy
- **Random Forest (20 trees):** ~85% accuracy
- **OOB Score:** ~80% accuracy
- Proper train/test/validation split (60%/20%/20%)

## 🔧 Key Features

### Core Implementation
- **Custom Decision Tree** (`build_decision_tree`)
  - Gini impurity for splits
  - Configurable max depth, min samples, purity threshold
  - Prevents overfitting with multiple stopping criteria

- **Random Forest** (`train_trees`)
  - Bootstrap sampling (with replacement)
  - Random feature selection at each split
  - OOB validation without separate validation set
  - Ensemble predictions with voting

### Analysis Tools
- **Correlation Analysis** (`correlation_analysis.py`)
  - Pearson correlation for numerical features
  - Cramér's V for categorical associations
  - Automatic feature importance ranking

- **Data Processing** (`data_process.py`)
  - Smart missing value handling
  - Type optimization (int8, float32)
  - Categorical encoding

- **Visualization** (`visualize_survival_data.py`)
  - Scatter plots for all feature combinations
  - PDF export for offline analysis

## 📁 Project Structure

```
├── main.py                      # Main script with Random Forest
├── main-2025-14-12.py          # Alternative version with variations
├── data_inspection.py          # Data quality checks
├── data_process.py             # Data preprocessing
├── split_data.py               # Train/test/validation splitting
├── gini_Y_impurity.py          # Gini impurity calculation
├── correlation_analysis.py     # Feature correlation analysis
├── feature_selection.py        # Automatic feature selection
├── survival_counter.py         # Accuracy tracking
├── visualize_survival_data.py  # Data visualization
└── train.csv                   # Titanic dataset (download separately)
```

## ⚙️ Configuration

Edit hyperparameters in `main.py`:

```python
# Tree parameters
min_group = 3              # Minimum samples per leaf
max_depth = 6              # Maximum tree depth
gini_threshold = 0.01      # Purity threshold for stopping

# Forest parameters
n_estimators = 20          # Number of trees
rand_percent = 50          # % of data per tree (bootstrap)
death_threshold = 0.6      # Classification threshold

# Feature selection
max_features = "sqrt"      # Features per split: "sqrt", None, or 0.8
features_train = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']
```

## How It Works

### 1. Data Splitting
```python
df_train, df_test, df_validation = split_data.calc(processed_df, 
                                                     train_share=0.6, 
                                                     test_share=0.2)
```

### 2. Build Random Forest
- Each tree trains on a **bootstrap sample** (50% of training data)
- At each split, considers only a **random subset of features**
- Uses **OOB samples** for validation (samples not in bootstrap)

### 3. Make Predictions
- Each tree votes on the prediction
- **Majority voting** determines final class
- Provides confidence scores based on vote percentages

### 4. Evaluation
- **OOB Score:** Validates on out-of-bag samples
- **Test Set:** Evaluates on held-out 20% of data
- **Ensemble Analysis:** Shows agreement levels across trees

## 📈 Example Output

```
Building 20 trees...
Tree number 1
Average accuracy of tree  1:    75.82%
==========================================
...
OOB accuracy of the Random Forest = 80.12%

Generating ensemble predictions using 20 trees...
Unanimous decisions (>95%):        512 samples,  23 mistakes (4.5%)
High agreement (95-70%):           201 samples,  41 mistakes (20.4%)
Split decisions (50-70%):           89 samples,  31 mistakes (34.8%)

Accuracy of the Random Forest on test data = 82.02%
```

## Learning Outcomes

This project demonstrates:
- **Decision Trees:** Recursive binary splitting with Gini impurity
- **Random Forests:** Ensemble learning with bagging
- **Bootstrap Sampling:** Training on random subsets with replacement
- **OOB Validation:** Using unsampled data for validation
- **Feature Randomization:** Decorrelating trees by limiting features
- **Voting Systems:** Combining predictions from multiple models

## Advanced Features

### Hyperparameter Search
Run automated grid search:
```python
hyperparameter_search_1d(do=True, n_repeats=5)
```
Tests multiple values for each hyperparameter and finds optimal settings.

### Feature Importance
Automatic correlation analysis selects most predictive features:
```python
corr_results, features_train = correlation_analysis.comprehensive_correlation_analysis(
    processed_df,
    target='Survived',
    corr_threshold=0.01
)
```

### OOB Scoring
No need for validation set - uses out-of-bag samples:
```python
oob_score = oob_forest_score(trained_trees, df_train)
```

## Why From Scratch?

This implementation shows exactly how Random Forests work under the hood:
- Complete control over tree building logic
- Understanding of bootstrap aggregating
- Insight into feature randomization
- Clear visualization of ensemble voting

Perfect for learning, teaching, or understanding the algorithm deeply!

## 📝 License

MIT License - feel free to use for learning and experimentation.

## Acknowledgments

- Dataset: [Kaggle Titanic Competition](https://www.kaggle.com/c/titanic)
- Inspired by Breiman's Random Forest algorithm (2001)
- Built for educational purposes to understand ensemble methods

---

**Note:** For production use, consider scikit-learn's `RandomForestClassifier` which includes optimizations, parallel processing, and additional features. This implementation is designed for learning and transparency.
