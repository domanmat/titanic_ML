import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from itertools import combinations


def visualize_survival_data(df):
    """
    Create scatter plots for all 2-element combinations of parameters,
    showing Survived status with color and marker coding.

    Parameters:
    - df: pandas DataFrame with processed data
    """
    print("\n### VISUALIZING SURVIVAL DATA ###")
    print("=" * 60)

    # Parameters to create combinations from
    parameters = ['Age', 'Pclass', 'Fare', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']

    # Generate all 2-element combinations
    param_pairs = list(combinations(parameters, 2))

    print(f"Creating {len(param_pairs)} scatter plots...")
    print("=" * 60 + "\n")

    # Calculate grid dimensions
    n_plots = len(param_pairs)
    n_cols = 4  # 4 plots per row for better visibility
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create figure with subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    # Prepare data - convert categorical variables to numeric for plotting
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

        # Separate data by survival status
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
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    plt.tight_layout()
    plt.savefig('titanic_survival_analysis.png', dpi=150, bbox_inches='tight')
    print("Plots saved as 'titanic_survival_analysis.png'")
    plt.show()

    print("\n" + "=" * 60)
    print(f"Generated {n_plots} scatter plots showing survival patterns")
    print("=" * 60 + "\n")


def process_data(df):
    """
    Process the Titanic dataset by handling missing values and optimizing data types.

    Parameters:
    - df: pandas DataFrame to process

    Returns:
    - Processed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    print("Processing data...")
    print("=" * 60)

    # Handle missing values
    print("\n1. Filling missing values...")
    processed_df['Age'] = processed_df['Age'].fillna(-1)
    processed_df['Cabin'] = processed_df['Cabin'].fillna('None')
    processed_df['Embarked'] = processed_df['Embarked'].fillna('None')

    # Fill remaining columns with 'None'
    for col in processed_df.columns:
        if col not in ['Age', 'Cabin', 'Embarked']:
            if processed_df[col].isnull().any():
                processed_df[col] = processed_df[col].fillna('None')

    print("   Missing values filled.")

    # Process and optimize data types
    print("\n2. Optimizing data types...")

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
    processed_df['Sex'] = processed_df['Sex'].astype(str).str[:10]

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

    print("   Data types optimized.")

    # Display data type summary
    print("\n3. Data type summary:")
    print("-" * 60)
    print(processed_df.dtypes)

    print("\n" + "=" * 60)
    print("Processing complete!")
    print("=" * 60 + "\n")

    return processed_df


def inspect_data(df, detailed=True):
    """
    Inspect the dataset for missing data.

    Parameters:
    - df: pandas DataFrame to inspect
    - detailed: bool, if True shows full analysis, if False shows only summary
    """
    print("\n" + "=" * 60 + "\n")

    if detailed:
        # Display basic information about the dataset
        print("Dataset Shape:", df.shape)
        print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
        print("\n" + "=" * 60 + "\n")

        # Display first 10 rows with all columns
        print("First 10 rows of the dataset:")
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(df.head(10))
        print("\n" + "=" * 60 + "\n")

        # Missing data analysis as a single row with column headers
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
        print("\n" + "=" * 60 + "\n")

        # Summary statistics
        total_cells = df.shape[0] * df.shape[1]
        total_missing = df.isnull().sum().sum()
        print(f"Total cells in dataset: {total_cells}")
        print(f"Total missing cells: {total_missing}")
        print(f"Overall missing percentage: {(total_missing / total_cells * 100):.2f}%")
        print("\n" + "=" * 60 + "\n")

    # Display columns with missing data (shown in both modes)
    print("Columns with missing data:")
    print("-" * 60)

    missing_count = df.isnull().sum()
    missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

    # Transpose so columns are headers
    missing_analysis = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_pct
    }).T

    # Filter to only columns with missing data
    cols_with_missing = missing_analysis.loc[:, (missing_analysis.loc['Missing_Count'] > 0)]

    if not cols_with_missing.empty:
        print(cols_with_missing)
    else:
        print("No missing data found in any column!")

    print("\n" + "=" * 60 + "\n")


# Load the CSV file
file_path = r"C:\Users\Mateusz\Downloads\titanic\train.csv"
df = pd.read_csv(file_path)

# Inspect original data
print("\n### ORIGINAL DATA INSPECTION ###")
inspect_data(df, detailed=True)  # Change to False for summary only

# Process the data
processed_df = process_data(df)

# Inspect processed data
print("\n### PROCESSED DATA INSPECTION ###")
inspect_data(processed_df, detailed=True)  # Change to False for summary only

# Display first 10 rows of processed data with all columns
print("\n### FIRST 10 ROWS OF PROCESSED DATA ###")
print("=" * 60)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
print(processed_df.head(10))
print("=" * 60)

# Visualize the survival data
visualize_survival_data(processed_df)