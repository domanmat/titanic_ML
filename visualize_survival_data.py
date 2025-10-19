import matplotlib.pyplot as plt
from itertools import combinations
def figure(df, enable_visualization=True):
    """
    Create scatter plots for all 2-element combinations of parameters,
    showing Survived status with color and marker coding.

    Parameters:
    - df: pandas DataFrame with processed df_processed
    - enable_visualization: bool, if False the function returns without creating plots
    """
    if not enable_visualization:
        print("\nVisualization is disabled.")
        return

    print("\n### VISUALIZING SURVIVAL DATA ###")
    print("=" * 60)

    # Parameters to create combinations from
    parameters = ['Age', 'Pclass', 'Fare', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']

    # Generate all 2-element combinations
    param_pairs = list(combinations(parameters, 2))

    print(f"Creating {len(param_pairs)} scatter plots...")
    print("=" * 60)

    # Calculate grid dimensions
    n_plots = len(param_pairs)
    n_cols = 4  # 4 plots per row for better visibility
    n_rows = (n_plots + n_cols - 1) // n_cols

    # Create figure with subplots (normal size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5 * n_rows))
    axes = axes.flatten() if n_plots > 1 else [axes]

    # Prepare df_processed - convert categorical variables to numeric for plotting
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

        # Separate df_processed by survival status
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
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(n_plots, len(axes)):
        axes[idx].axis('off')

    # Increase spacing between plots by 1.5x
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=3.0)

    # Save as PDF in the specified directory
    output_path = r"C:\Users\Mateusz\Downloads\titanic\Figure_1.pdf"
    plt.savefig(output_path, format='pdf', bbox_inches='tight')
    print(f"Plot saved as '{output_path}'")
    plt.close(fig)  # Close the figure instead of showing it

    print("=" * 60)
    print(f"Generated {n_plots} scatter plots showing survival patterns")
    print("=" * 60)
