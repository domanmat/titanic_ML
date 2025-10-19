def calc(survived, total_score, len_df, survival_r):
    """
    Add a value to the global counter variable.

    Parameters:
    - total score = final number of survived
    """
    # global total_score  # Declare that we're using the global variable
    if survival_r >= 0.50:
        total_score += survived
    else:
        total_score += len_df-survived
    return total_score