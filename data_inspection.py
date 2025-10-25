import pandas as pd

def check(df, detailed=True):
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
