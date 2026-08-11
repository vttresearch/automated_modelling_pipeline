import matplotlib.pyplot as plt
from matplotlib.widgets import Button

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import pandas as pd

def visualize_results_plt(f_name, targets, df_measured, forecast_len, lead_time, update_rate):
    # Calculate the number of columns and their indexes so that there are no empty spaces in the plots.
    column_cnt = forecast_len // update_rate
    new_cols = {}
    for i in range(lead_time, column_cnt):
        index_start = i * update_rate
        index_end = (forecast_len // column_cnt - 1) + i * update_rate
        col_name = f'forecast_{index_start}_{index_end + 1}'
        new_cols[col_name] = [f'forecast_{j}' for j in range(index_start, index_end + 1)]

    for target in targets:
        col_name = target['column']
        target[f_name][col_name] = df_measured[col_name]

        pred_to_plot = target[f_name].copy()

        # Efficiently create new columns using a dictionary and add them in one operation
        new_cols_dict = {
            col: pred_to_plot[val].max(axis=1) for col, val in new_cols.items()
        }
        new_cols_df = pd.DataFrame(new_cols_dict)
        pred_to_plot = pd.concat([pred_to_plot, new_cols_df], axis=1)

        pred_to_plot = pred_to_plot[[col_name] + list(new_cols.keys())]
        styles = ['--' if i == col_name else '-' for i in list(pred_to_plot.columns)]
        ax = pred_to_plot.plot(style=styles)

        fig = ax.get_figure()
        fig.set_size_inches(8, 6)  # Adjust figure size
        fig.subplots_adjust(bottom=0.25)  # Leave space for buttons
        fig.canvas.manager.set_window_title(f'{f_name}')
        leg = ax.get_legend()
        lines = ax.get_lines()
        lined = {}

        # Map legend lines to corresponding plot lines
        for leg_line, orig_line in zip(leg.get_lines(), lines):
            leg_line.set_picker(True)  # Enable picking for legend lines
            lined[leg_line] = orig_line

        # Also enable picking for legend labels (text)
        for leg_text in leg.get_texts():
            leg_text.set_picker(True)

        # Initially hide all lines except measured value, first, and last forecast
        for i, line in enumerate(lines):
            if i != 0 and i != 1 and i != len(lines) - 1:  # Keep only measured, first, and last forecasts visible
                line.set_visible(False)
                leg.get_lines()[i].set_alpha(0.2)  # Fade legend entries for hidden lines

        # Function to toggle line visibility when legend is clicked (for both lines and labels)
        def on_pick(event):
            # If the click is on a legend line
            if event.artist in lined:
                leg_line = event.artist
                orig_line = lined[leg_line]
                visible = not orig_line.get_visible()
                orig_line.set_visible(visible)
                leg_line.set_alpha(1.0 if visible else 0.2)
                fig.canvas.draw()

            # If the click is on a legend label (text)
            elif isinstance(event.artist, plt.Text):
                text = event.artist
                for i, leg_line in enumerate(leg.get_lines()):
                    if leg_line.get_label() == text.get_text():
                        orig_line = lines[i]
                        visible = not orig_line.get_visible()
                        orig_line.set_visible(visible)
                        leg_line.set_alpha(1.0 if visible else 0.2)
                        fig.canvas.draw()

        fig.canvas.mpl_connect('pick_event', on_pick)

        # Button to toggle visibility of all lines
        ax_button_all = plt.axes([0.2, 0.05, 0.2, 0.075])  # Button for all lines
        button_all = Button(ax_button_all, 'Toggle All')

        def toggle_all(event):
            all_visible = any(line.get_visible() for line in lines)
            for line, leg_line in zip(lines, leg.get_lines()):
                visible = not all_visible
                line.set_visible(visible)
                leg_line.set_alpha(1.0 if visible else 0.2)
            button_all.label.set_text("Hide All" if not all_visible else "Show All")
            fig.canvas.draw()

        button_all.on_clicked(toggle_all)

        plt.show()
def visualize_distribution(f_name, targets, df_measured, baseline=None, baseline_name=None, mlflow_active_run=False):

    for target in targets:
        col_name = target['column']
        target[f_name][col_name] = df_measured[col_name]

        pred_to_plot = target[f_name].copy()

        # Identify forecast columns
        forecast_columns = [col for col in pred_to_plot.columns if col.startswith('forecast')]
        forecast_mean = pred_to_plot[forecast_columns].mean(axis=1)
        forecast_min = pred_to_plot[forecast_columns].min(axis=1)
        forecast_max = pred_to_plot[forecast_columns].max(axis=1)

        # First 10% of forecast columns
        first_10_cols = forecast_columns[:max(1, int(len(forecast_columns) * 0.1))]

        # Calculate stats for the first 10% of forecasts
        first10_min = pred_to_plot[first_10_cols].min(axis=1)
        first10_max = pred_to_plot[first_10_cols].max(axis=1)
        first10_mean = pred_to_plot[first_10_cols].mean(axis=1)

        plt.figure(figsize=(10, 6))

        # Plot real values
        plt.plot(pred_to_plot.index, pred_to_plot[col_name],
                 label=target['plot_label'],
                 color='#FF4500', linewidth=1.5)

        # Plot the forecast distribution as a shaded area
        plt.fill_between(pred_to_plot.index, forecast_min, forecast_max,
                         color='#ADD8E6', alpha=0.3, label='Forecast distribution')

        # Plot first 10% forecast distribution
        plt.fill_between(pred_to_plot.index, first10_min, first10_max,
                         color="#98FB98", alpha=0.5, label="First 10% Forecast distribution")



        # Plot the forecast mean as a dashed line
        plt.plot(pred_to_plot.index, forecast_mean,
                 color='#1E90FF', linestyle='--', linewidth=1.8, label='Forecast mean')

        if (baseline is not None) and (col_name in baseline.keys()):
            # Plot the forecast mean as a dashed line
            plt.plot(baseline[col_name].index, baseline[col_name],
                     color='#555555', linestyle='--', linewidth=1.8, label=baseline_name)


        # Add title, labels, and legend
        plt.title(f'{f_name} - Forecast Distribution vs. {target["plot_label"]}', fontsize=16, color='#333333')
        plt.xlabel('Time', fontsize=14, color='#333333')
        plt.ylabel(target['output'], fontsize=14, color='#333333')
        plt.legend(loc='upper left', fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.5)

        # Log the plot to MLflow if an experiment is running
        if mlflow_active_run:
            import mlflow
            fig = plt.gcf()  # Get current figure
            mlflow.log_figure(fig, f"plots/{f_name}_{col_name}_forecast_distribution.png")

        plt.show()
        pass


def visualize_rmses(mse_periods, model_name):
    for target, periods in mse_periods.items():
        plt.figure(figsize=(10, 5))
        period_names = list(periods.keys())
        mse_values = list(periods.values())
        # Convert to RMSE
        rmse_values = [np.sqrt(mse) for mse in mse_values]
        plt.plot(period_names, rmse_values, label=f'Target: {target}', marker='o', color='blue')

        plt.title(f'RMSE by forecast period for target: {target}, model {model_name}')
        plt.xlabel('Forecast period')
        plt.ylabel('RMSE')

        # Calculate nth label to ensure max 20 ticks
        nth_label = max(1, len(period_names) // 20)
        plt.xticks(ticks=range(0, len(period_names), nth_label), labels=period_names[::nth_label], rotation=45)

        plt.tight_layout()
        plt.show()


def plot_pairwise(data, title, feature_names, color='blue', alpha=0.5, figsize=(20, 20)):
    """Create pairwise scatter plots.
    
    Args:
        data: numpy array of shape (n_samples, n_features) containing the data
        title: str, title for the overall figure
        feature_names: list of str, names of features corresponding to data columns
        color: str or color specification, color for all data points (default: 'blue')
        alpha: float, transparency level (0-1, default: 0.5)
    
    Returns:
        matplotlib Figure object
    
    Example:
        >>> data = np.random.randn(100, 3)
        >>> fig = plot_pairwise(data, "My Data", 
        ...                     ["feature1", "feature2", "feature3"])
        >>> plt.savefig("pairwise_plot.png")
    """
    
    n_features = len(feature_names)
    fig, axes = plt.subplots(n_features, n_features, figsize=figsize)
    
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            
            if i == j:
                # Diagonal: histogram
                ax.hist(data[:, i], bins=30, alpha=alpha, color=color,
                       edgecolor='black', linewidth=0.5)
            else:
                # Off-diagonal: scatter plot
                ax.scatter(data[:, j], data[:, i],
                         c=color, s=20, alpha=alpha, 
                         edgecolors='black', linewidth=0.3)
            
            # Set labels
            if i == n_features - 1:
                ax.set_xlabel(feature_names[j], fontsize=10)
            else:
                ax.set_xticklabels([])
            
            if j == 0:
                ax.set_ylabel(feature_names[i], fontsize=10)
            else:
                ax.set_yticklabels([])
            
            ax.grid(True, alpha=0.3)
    
    fig.suptitle(title, y=0.995, fontsize=14, fontweight='bold')
    plt.subplots_adjust(hspace=0.05, wspace=0.05)
    
    return fig
