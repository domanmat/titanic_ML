def calc(df,detailed):
    """
    Process the Titanic dataset by handling missing values and optimizing df_processed types.

    Parameters:
    - df: pandas DataFrame to process
    - detailed: boolean, writes details of processing

    Returns:
    - Processed DataFrame
    """
    # Create a copy to avoid modifying the original
    processed_df = df.copy()

    print("\nProcessing data...")
    # print("=" * 60)

    # Handle missing values
    print("1. Filling missing values...")
    processed_df['Age'] = processed_df['Age'].fillna(-1)
    processed_df['Cabin'] = processed_df['Cabin'].fillna('None')
    processed_df['Embarked'] = processed_df['Embarked'].fillna('None')

    # Fill remaining columns with 'None'
    for col in processed_df.columns:
        if col not in ['Age', 'Cabin', 'Embarked']:
            if processed_df[col].isnull().any():
                processed_df[col] = processed_df[col].fillna('None')

    print("   Missing values filled.")

    # Process and optimize df_processed types
    print("2. Optimizing df_processed types...")

    # Data processing and types
    processed_df['Cabin'] = processed_df['Cabin'].astype(str).str[:10]
    processed_df['Age'] = (processed_df['Age'].round(2))
    processed_df['Age'] = (processed_df['Age'].astype('float32'))
    processed_df['Embarked'] = processed_df['Embarked'].astype(str).str[:5]
    processed_df['PassengerId'] = processed_df['PassengerId'].astype(int)
    processed_df['Name'] = processed_df['Name'].astype(str).str[:100]
    processed_df['Pclass'] = processed_df['Pclass'].astype('int8')
    processed_df['Survived'] = processed_df['Survived'].astype('int8')
    processed_df['Sex'] = processed_df['Sex'].map({'male': 0, 'female': 1})
    processed_df['Sex'] = processed_df['Sex'].astype('int8')
    processed_df['Parch'] = processed_df['Parch'].astype('int8')
    processed_df['SibSp'] = processed_df['SibSp'].astype('int8')
    processed_df['Fare'] = processed_df['Fare'].astype('float64')
    processed_df['Ticket'] = processed_df['Ticket'].astype(str).str[:100]

    if detailed:
        # Display first 10 rows of processed df_processed with all columns
        print("\n### FIRST 10 ROWS OF PROCESSED DATA ###")
        print("=" * 60)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', None)
        print(processed_df.head(10))
        print("=" * 60)

        print("   Data types optimized.")

        # Display df_processed type summary
        print("\n3. Data type summary:")
        print("-" * 60)
        print(processed_df.dtypes)

        print("=" * 60)
        print("Processing complete!")
        print("=" * 60)
    print()
    return processed_df
