import numpy as np  # Numerical and matrix operations.
import seaborn as sns  # Statistical data visualization.
import matplotlib.pyplot as plt  # Plot and figure creation.
import pandas as pd
# Matplotlib is a versatile tool for building plots from scratch,
# while Seaborn simplifies the creation of statistical charts.

def plot_explained_variance(explained_var, n_components):
    """
    Plots the explained variance.
    Args:
      explained_var (array): Array with the percentage of variance explained
        by each principal component. Usually computed as
        explained_var = fit.explained_variance_ratio_ * 100.
      n_components (int): Total number of principal components.
        Usually computed as fit.n_components.
    """
    # Create a range of principal component numbers from 1 to n_components
    component_range = np.arange(1, n_components + 1)

    # Create an 8x6 figure
    plt.figure(figsize=(8, 6))

    # Plot the explained variance against the number of principal components
    plt.plot(component_range, explained_var, marker='o')

    # Axis labels
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Explained Variance')

    # Plot title
    plt.title('Explained Variance per Principal Component')

    # Set x-axis ticks to match the number of components
    plt.xticks(component_range)

    # Show a grid on the plot
    plt.grid(True)

    # Add bars under each point to represent the percentage of explained variance
    # - 'width': bar width, set to 0.2 units here.
    # - 'align': bar alignment relative to the x-axis points.
    #   'center' means the bars are centered under the points.
    # - 'alpha': bar transparency. 0.7 means the bars are 70% transparent.
    plt.bar(component_range, explained_var, width=0.2, align='center', alpha=0.7)

    # Show the plot
    plt.show()


#####################################################################################################
def plot_cos2_heatmap(cos2):
    """
    Generates a heatmap of the squared loadings on the Principal Components (squared cosines).

    Args:
        cos2 (pd.DataFrame): DataFrame of squared cosines, where rows are variables and columns are Principal Components.

    """
    # Create an 8x8 inch figure for the plot
    plt.figure(figsize=(8, 8))

    # Use a heatmap to visualize 'cos2' with a single color
    sns.heatmap(cos2, cmap='Blues', linewidths=0.5, annot=False)

    # Label the axes (row/column names can be customized if needed)
    plt.xlabel('Principal Components')
    plt.ylabel('Variables')

    # Set the plot title
    plt.title('Squared Loadings on the Principal Components')

    # Show the plot
    plt.show()

#######################################################################################################
def plot_corr_cos(n_components, correlations_with_pcs):
    """
    Generates plots with one vector per variable, using the principal components as axes.
    The direction and length of each vector represent the correlation between the variable and two components.
    The color represents the sum of the squared cosines.

    Args:
        n_components (int): Number of selected principal components.
        correlations_with_pcs (DataFrame): DataFrame with the correlation matrix between variables and components.
    """
    # Define color map
    cmap = plt.get_cmap('coolwarm')

    for i in range(n_components):
        for j in range(i + 1, n_components):
            # Compute the sum of squared cosines
            sum_cos2 = correlations_with_pcs.iloc[:, i] ** 2 + correlations_with_pcs.iloc[:, j] ** 2

            # Create figure and axes
            fig, ax = plt.subplots(figsize=(10, 10))

            # Draw a circle of radius 1
            circle = plt.Circle((0, 0), 1, fill=False, color='b', linestyle='dotted')
            ax.add_patch(circle)

            # Draw a vector for each variable
            for k, var_name in enumerate(correlations_with_pcs.index):
                x = correlations_with_pcs.iloc[k, i]
                y = correlations_with_pcs.iloc[k, j]

                # Color based on the sum of squared cosines
                color = cmap(sum_cos2.iloc[k])

                # Draw vector and label
                ax.quiver(0, 0, x, y, angles='xy', scale_units='xy', scale=1, color=color)
                ax.text(x, y, var_name, color=color, fontsize=12, ha='right', va='bottom')

            # Draw axes
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8)

            # Labels and limits
            ax.set_xlabel(f'Principal Component {i + 1}')
            ax.set_ylabel(f'Principal Component {j + 1}')
            ax.set_xlim(-1.1, 1.1)
            ax.set_ylim(-1.1, 1.1)

            # Add colorbar
            norm = plt.Normalize(vmin=sum_cos2.min(), vmax=sum_cos2.max())
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
            fig.colorbar(sm, ax=ax, orientation='vertical', label='cos²')

            # Grid and show
            ax.grid(True)
            plt.show()

##################################################################################################

def plot_cos2_bars(cos2):
    """
    Generates a bar chart of the variance of each variable explained by the components, using the squared loadings (cos^2).

    Args:
        cos2 (pd.DataFrame): DataFrame with the squared loadings of the variables on the principal components.

    Returns:
        None
    """
    # Create an 8x6 inch figure for the plot
    plt.figure(figsize=(8, 6))

    # Bar chart of the variance explained for each variable
    sns.barplot(x=cos2.sum(axis=1), y=cos2.index, color="blue")

    # Label the axes
    plt.xlabel('Sum of $cos^2$')
    plt.ylabel('Variables')

    # Set the plot title
    plt.title('Variance of Each Variable Explained by the Principal Components')

    # Show the plot
    plt.show()



#########################################################################################################

#######################################################################################

def plot_proportional_contributions(cos2, eigenvalues, n_components):
    """
    Computes the contribution of each variable to the principal components and
    generates a heatmap with the results.
    Args:
        cos2 (DataFrame): DataFrame of squared loadings (cos^2).
        eigenvalues (array): Array of eigenvalues associated with the principal components.
        n_components (int): Number of selected principal components.
    """
    # Compute the contributions by multiplying cos2 by the square root of the eigenvalues
    contributions = cos2 * np.sqrt(eigenvalues)

    # Initialize a list for the contribution sums
    contribution_sums = []

    # Compute the sum of contributions for each principal component
    for i in range(n_components):
        component_name = f'Component {i + 1}'
        contribution_sum = np.sum(contributions[component_name])
        contribution_sums.append(contribution_sum)

    # Compute the proportional contributions by dividing by the contribution sums
    proportional_contributions = contributions.div(contribution_sums, axis=1) * 100

    # Create an 8x8 inch figure for the plot
    plt.figure(figsize=(8, 8))

    # Use a heatmap to visualize the proportional contributions
    sns.heatmap(proportional_contributions, cmap='Blues', linewidths=0.5, annot=False)

    # Label the axes (row/column names can be customized if needed)
    plt.xlabel('Principal Components')
    plt.ylabel('Variables')

    # Set the plot title
    plt.title('Proportional Contributions of the Variables to the Principal Components')

    # Show the plot
    plt.show()

    # Return the DataFrame of proportional contributions
    return proportional_contributions

######################################################################################################
def plot_pca_scatter(pca, standardized_data, n_components):
    """
    Generates scatter plots of the observations on pairs of selected principal components.

    Args:
        pca (PCA): Previously fitted PCA object.
        standardized_data (pd.DataFrame): DataFrame of standardized data.
        n_components (int): Number of selected principal components.
    """
    # Plot the observations on each pair of selected components
    principal_components = pca.transform(standardized_data)

    for i in range(n_components):
        for j in range(i + 1, n_components):  # Avoid duplicate pairs
            # Scatter plot of the observations on the two principal components
            plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
            plt.scatter(principal_components[:, i], principal_components[:, j])

            # Add labels to the observations
            observation_labels = list(standardized_data.index)

            for k, label in enumerate(observation_labels):
                plt.annotate(label, (principal_components[k, i], principal_components[k, j]))

            # Draw dashed lines for the axes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

            # Label the axes
            plt.xlabel(f'Principal Component {i + 1}')
            plt.ylabel(f'Principal Component {j + 1}')

            # Set the plot title
            plt.title('Scatter Plot of Observations in PCA Space')

            plt.show()

################################################################################




def plot_pca_scatter_with_vectors(pca, standardized_data, n_components, components_):
    """
    Generates scatter plots of the observations on pairs of selected principal components
    with vectors of the scaled correlations between variables and components.

    Args:
        pca (PCA): Previously fitted PCA object.
        standardized_data (pd.DataFrame): DataFrame of standardized data.
        n_components (int): Number of selected principal components.
        components_: Array with the components.
    """
    # Plot the observations on each pair of selected components
    principal_components = pca.transform(standardized_data)

    for i in range(n_components):
        for j in range(i + 1, n_components):  # Avoid duplicate pairs
            # Scatter plot of the observations on the two principal components
            plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
            plt.scatter(principal_components[:, i], principal_components[:, j])

            # Add labels to the observations
            observation_labels = list(standardized_data.index)

            for k, label in enumerate(observation_labels):
                plt.annotate(label, (principal_components[k, i], principal_components[k, j]))

            # Draw dashed lines for the axes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

            # Label the axes
            plt.xlabel(f'Principal Component {i + 1}')
            plt.ylabel(f'Principal Component {j + 1}')

            # Set the plot title
            plt.title('Scatter Plot of Observations and Variables in PCA Space')


            # Add vectors representing the scaled correlations between variables and components
            fit = pca.fit(standardized_data)
            coeff = np.transpose(fit.components_)
            scaled_coeff = 8 * coeff  # 8 = scaling factor used, adjust per example
            for var_idx in range(scaled_coeff.shape[0]):
                plt.arrow(0, 0, scaled_coeff[var_idx, i], scaled_coeff[var_idx, j], color='red', alpha=0.5)
                plt.text(scaled_coeff[var_idx, i], scaled_coeff[var_idx, j],
                     standardized_data.columns[var_idx], color='red', ha='center', va='center')

            plt.show()

#####################################################################################################

def plot_pca_scatter_with_categories(categorical_data, principal_components, n_components, cat_var):
    """
    Generates scatter plots of the observations on pairs of selected principal components, with categories.

    Args:
        categorical_data (pd.DataFrame): DataFrame containing the categories.
        principal_components (np.ndarray): Matrix of principal components.
        n_components (int): Number of selected principal components.
        cat_var (str): Name of the categorical variable.
    """
    # Convert to numpy array if it is a DataFrame
    if isinstance(principal_components, pd.DataFrame):
        principal_components = principal_components.values

    # Get the unique categories
    categories = categorical_data[cat_var].unique()

    for i in range(n_components):
        for j in range(i + 1, n_components):  # Avoid duplicate pairs
            # Scatter plot of the observations on the two principal components
            plt.figure(figsize=(8, 6))  # Adjust the figure size if needed
            plt.scatter(principal_components[:, i], principal_components[:, j])

            for category in categories:
                # Filter the observations by category
                category_observations = principal_components[categorical_data[cat_var] == category]
                # Compute the category centroid
                centroid = np.mean(category_observations, axis=0)
                plt.scatter(centroid[i], centroid[j], label=category, s=100, marker='o')

            # Add labels to the observations
            observation_labels = list(categorical_data.index)

            for k, label in enumerate(observation_labels):
                plt.annotate(label, (principal_components[k, i], principal_components[k, j]))

            # Draw dashed lines for the axes
            plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
            plt.axvline(0, color='black', linestyle='--', linewidth=0.8)

            # Label the axes
            plt.xlabel(f'Principal Component {i + 1}')
            plt.ylabel(f'Principal Component {j + 1}')

            # Set the plot title
            plt.title('Scatter Plot of Observations in PCA Space')

            # Show the legend for the categories
            plt.legend()
            plt.show()


def plot_pca_scatter_with_categories2(categorical_data, principal_components, n_components, cat_var):
    """
    Generates scatter plots of the observations on pairs of selected principal components, with categories.

    Args:
        categorical_data (pd.DataFrame): DataFrame containing the categories.
        principal_components (pd.DataFrame/np.ndarray): Matrix of principal components.
        n_components (int): Number of selected principal components.
        cat_var (str): Name of the categorical variable.
    """
    # Convert to numpy array if it is a DataFrame
    if isinstance(principal_components, pd.DataFrame):
        principal_components = principal_components.values

    # Get the unique categories
    categories = categorical_data[cat_var].unique()

    # Define colors for the categories
    colors = plt.cm.Set3(np.linspace(0, 1, len(categories)))

    for i in range(n_components):
        for j in range(i + 1, n_components):
            fig, ax = plt.subplots(figsize=(10, 8))

            # Plot the points of each category
            for idx, category in enumerate(categories):
                mask = categorical_data[cat_var] == category
                # Points of each category
                ax.scatter(
                    principal_components[mask, i],
                    principal_components[mask, j],
                    c=[colors[idx]],
                    label=category,
                    alpha=0.6
                )

                # Compute and plot the centroid
                category_observations = principal_components[mask]
                centroid = np.mean(category_observations, axis=0)
                ax.scatter(
                    centroid[i],
                    centroid[j],
                    c=[colors[idx]],
                    s=200,
                    marker='*',
                    edgecolor='black',
                    linewidth=1.5,
                    label=f'Centroid {category}'
                )

            # Add labels to the observations
            for k, label in enumerate(categorical_data.index):
                ax.annotate(
                    label,
                    (principal_components[k, i], principal_components[k, j]),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8
                )

            # Draw axes
            ax.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)
            ax.axvline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.3)

            # Labels and title
            ax.set_xlabel(f'Principal Component {i + 1}')
            ax.set_ylabel(f'Principal Component {j + 1}')
            ax.set_title('Distribution of Observations by Species in PCA Space')

            # Legend
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

            # Grid and layout adjustment
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
