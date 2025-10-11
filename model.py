import pandas as pd

# Load the CSV file
file_path = r"C:\Users\Mateusz\Downloads\titanic\train.csv"
df = pd.read_csv(file_path)

# Display basic information about the dataset
print("Dataset Shape:", df.shape)
print(f"Total Rows: {df.shape[0]}, Total Columns: {df.shape[1]}")
print("\n" + "="*60 + "\n")

# Display first few rows
print("First 5 rows of the dataset:")
print(df.head())
print("\n" + "="*60 + "\n")

# Calculate missing data for each column
print("Missing Data Analysis:")
print("-" * 60)

missing_data = pd.DataFrame({
    'Column': df.columns,
    'Missing_Count': df.isnull().sum().values,
    'Missing_Percentage': (df.isnull().sum().values / len(df) * 100).round(2)
})

# Sort by missing count (descending)
missing_data = missing_data.sort_values('Missing_Count', ascending=False)

print(missing_data.to_string(index=False))
print("\n" + "="*60 + "\n")

# Summary statistics
total_cells = df.shape[0] * df.shape[1]
total_missing = df.isnull().sum().sum()
print(f"Total cells in dataset: {total_cells}")
print(f"Total missing cells: {total_missing}")
print(f"Overall missing percentage: {(total_missing / total_cells * 100):.2f}%")
print("\n" + "="*60 + "\n")

# Display columns with missing data only
cols_with_missing = missing_data[missing_data['Missing_Count'] > 0]
if not cols_with_missing.empty:
    print("Columns with missing data:")
    print(cols_with_missing.to_string(index=False))
else:
    print("No missing data found in any column!")