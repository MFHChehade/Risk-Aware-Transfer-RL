import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap_with_arrows(values, arrows, locations, 
                             title="Heat Map of Rewards", 
                             x_label="X", y_label="Y"):
    """
    Plot a heatmap with arrows on specific cells.

    Parameters:
    values (2D array): Values to be plotted as a heatmap.
    arrows (list): List of arrow directions for each cell in 'locations'.
                   Arrows can be '→', '←', '↑', or '↓'.
    locations (list of tuples): Coordinates (x, y) of cells corresponding to arrows.
    title (str): Title of the plot (default is 'Heat Map of Rewards').
    x_label (str): Label for the x-axis (default is 'X').
    y_label (str): Label for the y-axis (default is 'Y').
    """
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(values, annot=True, fmt=".2f", cmap='viridis', cbar=True)
    ax.invert_yaxis()
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    for i in range(len(arrows)):
        arrow_list = arrows[i]
        x, y = locations[i]
        for arrow in arrow_list:
            if arrow in ('→', '←', '↑', '↓'):
                x_coord = x + 0.5  # Center X-coordinate of the cell
                y_coord = y + 0.5  # Center Y-coordinate of the cell
                
                arrow_x, arrow_y = x_coord, y_coord  # Default coordinates for arrow
                
                # Adjust arrow coordinates based on direction
                if arrow == '→':
                    arrow_x += 0.3
                elif arrow == '←':
                    arrow_x -= 0.3
                elif arrow == '↑':
                    arrow = '↓'
                    arrow_y -= 0.3
                else:
                    arrow = '↑'
                    arrow_y += 0.3

                # Annotate arrows on the heatmap
                ax.annotate(arrow, (arrow_x, arrow_y), color='red', fontsize=20,
                            ha='center', va='center')

    plt.show()

