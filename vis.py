import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Set folder path
data_folder = 'data'
output_folder = 'plots'
os.makedirs(output_folder, exist_ok=True)

# Loop through all CSV files in the folder
for filename in os.listdir(data_folder):
    if filename.endswith('.csv'):
        filepath = os.path.join(data_folder, filename)
        print(f'Processing {filename}...')

        # Load CSV assuming no header row
        df = pd.read_csv(filepath, header=None)
        df.columns = [f'Actor_{i+1}' for i in range(df.shape[1])]
        df['Generation'] = range(1, df.shape[0] + 1)

        # Reshape
        df_long = df.melt(id_vars='Generation', var_name='Actor', value_name='Score')

        # Plot
        plt.figure(figsize=(14, 6))
        sns.boxplot(x='Generation', y='Score', data=df_long, palette='Set3')
        plt.title(f'Score Distribution - {filename[:-4]}')
        plt.xlabel('Generation')
        plt.ylabel('Score')
        plt.grid(True)
        plt.tight_layout()

        # Save plot
        plot_filename = os.path.splitext(filename)[0] + '_boxplot.png'
        plot_path = os.path.join(output_folder, plot_filename)
        plt.savefig(plot_path)
        plt.close()

        print(f'Plot saved to {plot_path}')

