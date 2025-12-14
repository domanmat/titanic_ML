import pandas as pd
import numpy as np

def calc(df, train_share, test_share, random_seed):
    """
    Rozdziela zbiór danych na podzbiory treningowy, testowy i walidacyjny.

    Parameters:
    - processed_df: pandas DataFrame - dane oryginalne po przetworzeniu
    - random_seed: int - ziarno losowości dla powtarzalności wyników

    Returns:
    - train_df: DataFrame - zbiór treningowy (60% danych)
    - test_df: DataFrame - zbiór testowy (20% danych)
    - validation_df: DataFrame - zbiór walidacyjny (20% danych)
    - indices_dict: dict - słownik z indeksami dla każdego zbioru
    """

    # Resetujemy indeksy dla wygody
    df = df.reset_index(drop=True)

    # Ustawiamy ziarno dla powtarzalności
    if random_seed is not None:
        np.random.seed(random_seed + 1)  # +1 aby nie powtarzać tego samego ziarna

    # Tasujemy indeksy
    n = len(df)
    shuffled_indices = np.random.permutation(n)

    # Obliczamy granice podziałów (60%, 20%, 20%)
    train_size = int(n * train_share)
    test_size = int(n * test_share)

    # Dzielimy indeksy
    train_indices = shuffled_indices[:train_size]
    test_indices = shuffled_indices[train_size:train_size + test_size]
    validation_indices = shuffled_indices[train_size + test_size:]

    # Tworzymy podzbiory
    df_train = df.iloc[train_indices].copy()
    df_test = df.iloc[test_indices].copy()
    df_validation = df.iloc[validation_indices].copy()

    # Słownik z indeksami (opcjonalnie, jeśli potrzebne)
    dict_indices = {
        'train': train_indices,
        'test': test_indices,
        'validation': validation_indices
    }

    return df_train, df_test, df_validation, dict_indices