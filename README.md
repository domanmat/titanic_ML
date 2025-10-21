# Titanic Machine Learning Project

A custom implementation of a decision tree classifier with ensemble learning for predicting survival on the Titanic dataset, built from scratch using Python and pandas.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [How It Works](#how-it-works)
- [Configuration](#configuration)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements a decision tree algorithm from scratch to predict passenger survival on the Titanic. Unlike using pre-built libraries like scikit-learn, this implementation provides full control over the tree-building process and includes a custom ensemble training method that builds multiple trees on random data subsets to find the best performing model.

## ✨ Features

- **Custom Decision Tree Implementation**: Built from scratch without using sklearn's DecisionTreeClassifier
- **Ensemble Training**: Trains multiple trees on random data subsets and selects the best performer
- **Gini Impurity Optimization**: Uses Gini impurity as the splitting criterion with optimized vectorized calculations
- **Modular Architecture**: Clean separation of concerns with dedicated modules for each functionality
- **Data Preprocessing**: Handles missing values and optimizes data types
- **Visualization Support**: Generate scatter plots showing survival patterns across feature combinations
- **Performance Metrics**: Tracks accuracy across training sets and provides detailed tree structure output
- **Configurable Parameters**:
  - Maximum tree depth
  - Gini impurity threshold
  - Minimum group size
  - Training data percentage
  - Number of training sessions
  - Random seed for reproducibility

## 📦 Requirements

```
pandas
numpy
matplotlib
```

Python version: 3.7+

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/domanmat/titanic_ML.git
cd titanic_ML
```

2. Install required packages:
```bash
pip install pandas numpy matplotlib
```

3. Download the Titanic dataset:
   - Download `train.csv` from [Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic/data)
   - Update the file path in `main.py`:
   ```python
   file_path = r"path/to/your/train.csv"
   ```

## 💻 Usage

### Basic Usage

Run the main script to train and evaluate the model:
```bash
python main.py
```

### Customization

Edit the configuration variables in `main.py`:

```python
# Feature selection
features = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']

# Tree parameters
max_depth = 10          # Maximum depth of the tree
gini_threshold = 0.01   # Minimum Gini impurity to continue splitting
min_group = 1           # Minimum samples per leaf node

# Ensemble training parameters
rand_percent = 70       # Percentage of data to use for each tree
rand_sessions = 10      # Number of trees to train and evaluate
```

### Enable Visualization

Uncomment the visualization line in `main.py`:

```python
visualize_survival_data.figure(processed_df, enable_visualization=True)
```

This creates a PDF with scatter plots showing relationships between features and survival status.

## 📁 Project Structure

```
titanic_ML/
│
├── main.py                         # Main execution script
├── data_inspection.py              # Data inspection and missing value analysis
├── data_process.py                 # Data preprocessing and type optimization
├── gini_Y_impurity.py             # Gini impurity calculation for target variable
├── survival_counter.py            # Accuracy tracking utility
├── visualize_survival_data.py     # Scatter plot generation
├── train.csv                      # Training dataset (download separately)
├── README.md                      # This file
└── Figure_1.pdf                   # Generated visualization (if enabled)
```

### Module Descriptions

- **main.py**: Orchestrates the entire workflow including data loading, preprocessing, tree building, ensemble training, and evaluation
- **data_inspection.py**: Provides the `check()` function to analyze dataset structure and missing values
- **data_process.py**: Contains the `calc()` function for handling missing values and optimizing data types
- **gini_Y_impurity.py**: Implements the `calc()` function to compute Gini impurity for the target variable
- **survival_counter.py**: Utility function to track correct predictions during tree building
- **visualize_survival_data.py**: Creates comprehensive scatter plot visualizations of all feature pairs

## 🔍 How It Works

### 1. Data Preprocessing

The `data_process.calc()` function:
- Fills missing values:
  - Age: `-1` for unknown ages
  - Cabin: `'None'` for missing cabin information
  - Embarked: `'None'` for missing embarkation port
- Optimizes data types (int8, float32, float64) for memory efficiency
- Rounds Age to 2 decimal places
- Encodes categorical variables (Sex: male=0, female=1)
- Truncates strings to reasonable lengths

### 2. Decision Tree Algorithm

The algorithm recursively splits the dataset to maximize survival prediction accuracy:

1. **Calculate Gini Impurity**: Measures heterogeneity of the target variable (Survived)
2. **Find Best Split**: Tests all possible split points for each feature using optimized vectorized operations
3. **Recursive Splitting**: Creates branches until reaching:
   - Maximum depth
   - Gini threshold (sufficiently pure nodes)
   - Minimum group size
   - Groups that would split into sizes smaller than `min_group`
4. **Leaf Nodes**: Final predictions based on majority class in each leaf

### 3. Ensemble Training Method

The `train_trees()` function implements a custom ensemble approach:

1. **Generate Random Subsets**: Creates multiple random subsets of the training data
2. **Train Multiple Trees**: Builds a decision tree on each subset
3. **Cross-Validation**: Evaluates each tree on all other subsets
4. **Select Best Model**: Chooses the tree with highest average accuracy across all validation sets

This approach helps prevent overfitting and improves generalization.

### 4. Key Functions

#### Data Processing
- `data_inspection.check()`: Analyzes missing values and data structure
- `data_process.calc()`: Cleans and prepares data for modeling

#### Tree Building
- `calc_tree_node()`: Finds optimal split for a given node using vectorized operations
- `build_decision_tree()`: Recursively constructs the tree
- `gini_Y_impurity.calc()`: Computes Gini impurity efficiently
- `gini_X_impurity()`: Calculates feature heterogeneity

#### Prediction & Evaluation
- `predict_single()`: Makes prediction for one sample
- `predict_batch()`: Makes predictions on multiple samples
- `train_trees()`: Ensemble training with cross-validation
- `print_tree()`: Visualizes tree structure with color-coded output

#### Utilities
- `df_random_slice()`: Samples random subset of data
- `survival_counter.calc()`: Tracks prediction accuracy

## ⚙️ Configuration

### Feature Selection

Available features:
- `Pclass`: Passenger class (1=1st, 2=2nd, 3=3rd)
- `Sex`: Gender (0=male, 1=female)
- `Age`: Age in years (-1 for unknown, rounded to 2 decimals)
- `SibSp`: Number of siblings/spouses aboard
- `Parch`: Number of parents/children aboard
- `Fare`: Ticket fare
- `Cabin`: Cabin number (optional, encoded)
- `Embarked`: Port of embarkation (optional, S/C/Q)

**Recommended**: Use `['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare']` for best balance of performance and speed.

### Tree Parameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `max_depth` | Maximum tree depth | 5-15 | 10 |
| `gini_threshold` | Stop splitting when Gini ≤ threshold | 0.0-0.1 | 0.01 |
| `min_group` | Minimum samples per leaf | 1-10 | 1 |

### Ensemble Parameters

| Parameter | Description | Typical Range | Default |
|-----------|-------------|---------------|---------|
| `rand_percent` | Percentage of data per tree | 50-90 | 70 |
| `rand_sessions` | Number of trees to train | 5-20 | 10 |
| `random_seed` | Seed for reproducibility | Any int or None | 21 |

## 📊 Results

### Output Format

The model provides comprehensive output including:

1. **Training Progress**: Detailed logging of tree construction at each depth level
2. **Tree Structure**: Visual representation with color-coded leaves (green) and decision rules
3. **Ensemble Results**: Accuracy of each tree on all validation sets
4. **Final Metrics**: 
   - Accuracy on training subset
   - Average accuracy across all validation sets
   - Accuracy on full dataset

### Example Output

```
Final score of the obtained tree No.3: 
	on its own set No.3: 424 out of 623, giving 68.06% accuracy,
	on all training data: 64.23% averaged.

Accuracy on full data = 79.91%

Time part 1:   0.234 seconds - loading modules and processing the data
Time part 2:   8.456 seconds - single tree build time
Time part 3:  85.123 seconds - full training
Time part 4:   0.892 seconds - final testing
Total execution time: 94.7050 seconds
```

### Performance Metrics

- **Training Accuracy**: 65-70% (on 70% random subsets)
- **Full Dataset Accuracy**: 78-82% (typical range)
- **Execution Time**: Varies based on parameters
  - Single tree: 5-15 seconds
  - Ensemble (10 trees): 60-120 seconds

## 🎨 Visualization

When enabled, the `visualize_survival_data.figure()` function generates:

- **28 scatter plots**: All 2-feature combinations from 8 parameters
- **Color coding**:
  - Green circles (○): Survived passengers
  - Red crosses (×): Deceased passengers
- **Layout**: 4 plots per row in a grid format
- **Output**: High-quality PDF file

This helps identify patterns and correlations between features and survival outcomes.

## 🚀 Performance Optimizations

The implementation includes several optimizations:

1. **Vectorized Gini Calculations**: Uses pandas vectorization instead of loops
2. **Sorted Split Finding**: Sorts data once per feature, then uses `np.searchsorted()`
3. **Lazy DataFrame Creation**: Only creates split DataFrames for the best split
4. **Optimized Data Types**: Uses int8 and float32 to reduce memory footprint
5. **Modular Design**: Separation of concerns improves maintainability

## 🤝 Contributing

Contributions are welcome! Here are some ways to improve the project:

### Potential Improvements

- **Pruning**: Add post-pruning to reduce overfitting
- **Feature Engineering**: Create interaction features or polynomial features
- **Parallel Processing**: Use multiprocessing for ensemble training
- **Visualization**: Add interactive tree visualizations
- **Test Set Support**: Add functionality to predict on Kaggle's test.csv
- **Model Export**: Save/load trained models
- **Hyperparameter Tuning**: Automated grid search for optimal parameters

### How to Contribute

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is open source and available under the MIT License.

## 🙏 Acknowledgments

- Dataset from [Kaggle's Titanic Competition](https://www.kaggle.com/c/titanic)
- Inspired by classical machine learning algorithms and ensemble methods
- Built as an educational project to understand decision tree mechanics

## 📧 Contact

For questions or suggestions, please open an issue on GitHub.

## 🔬 Educational Notes

This project demonstrates several key concepts in machine learning:

- **Decision Trees**: Recursive partitioning based on information gain
- **Gini Impurity**: Measure of node purity for classification
- **Ensemble Learning**: Training multiple models for better generalization
- **Cross-Validation**: Evaluating models on multiple data subsets
- **Overfitting Prevention**: Using depth limits, minimum group sizes, and purity thresholds

---

**Note**: This is an educational project demonstrating decision tree and ensemble concepts from scratch. For production use, consider established libraries like scikit-learn with more robust implementations and additional features.
