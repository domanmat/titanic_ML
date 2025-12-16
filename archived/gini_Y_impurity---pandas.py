def calc(df):
    """
    Calculate Gini impurity for each df of survived
    """
    if len(df) == 0:
        return 0
    survived_ratio = df['Survived'].sum() / len(df)
    # survived_ratio = df['Survived'].mean() #the same speed
    return 2 * survived_ratio * (1 - survived_ratio)