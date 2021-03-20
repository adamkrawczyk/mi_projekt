import matplotlib.pyplot as plt
from pandas_ods_reader import read_ods
import pandas as pd


def draw_single_data_plot(time_col, column_name, actual_column):
    time_col = time_col[2:]
    column_name = column_name + " " + actual_column[1]
    actual_column = actual_column[2:]
    side_df = pd.DataFrame({
        'Time': time_col,
        column_name: actual_column
    })

    return side_df, column_name

df = read_ods('data/MI-walczak.ods', 7)
column_names = df.columns

time_col = df['Time']
drum_pr = df[column_names[3]]
time_col = time_col[2:]
drum_pr = drum_pr[2:]

# time_col = pd.to_datetime(time_col, '%D-%H-%M-%S')
for i in range(2, len(column_names)):
    name = column_names[i]
    first_drum_pr, name = draw_single_data_plot(time_col, name, df[name])
    first_drum_pr.plot(x='Time', y=name)

# first_drum_pr = draw_single_data_plot(time_col, column_names[4], df[column_names[4]])
# first_drum_pr.plot(x='Time', y=column_names[4])


plt.show()

