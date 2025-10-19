def calc(group_size, survived_group):
    """
    Calculate Gini impurity for each df of survived
    """
    survived_ratio = survived_group / group_size if group_size > 0 else 0
    return 2 * survived_ratio * (1 - survived_ratio)