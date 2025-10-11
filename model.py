import pandas as pd


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

        # Display first few rows
        print("First 5 rows of the dataset:")
        print(df.head())
        print("\n" + "=" * 60 + "\n")

        # Missing data analysis as rows
        print("Missing Data Analysis:")
        print("-" * 60)

        missing_count = df.isnull().sum()
        missing_pct = (df.isnull().sum() / len(df) * 100).round(2)

        # Create DataFrame with columns as index (displayed as rows)
        missing_analysis = pd.DataFrame({
            'Missing_Count': missing_count,
            'Missing_Percentage': missing_pct
        })

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

    missing_analysis = pd.DataFrame({
        'Missing_Count': missing_count,
        'Missing_Percentage': missing_pct
    })

    cols_with_missing = missing_analysis[missing_analysis['Missing_Count'] > 0]

    if not cols_with_missing.empty:
        print(cols_with_missing)
    else:
        print("No missing data found in any column!")

    print("\n" + "=" * 60 + "\n")


# Load the CSV file
file_path = r"C:\Users\Mateusz\Downloads\titanic\train.csv"
df = pd.read_csv(file_path)

# Choose inspection mode:
# detailed=True for full analysis
# detailed=False for summary only

inspect_data(df, detailed=False)  # Change to False for summary only