# Code to import and plot lookup table data
import pandas as pd
import matplotlib.pyplot as plt
from lookup_table_data import table1_data, table2_data

# Make dataframes
col_names = [
    "-DeltaY_min / DeltaY",
    "t_min / T1", 
    "T0i / T1"
]
table1 = pd.DataFrame(table1_data, columns=col_names)
col_names = [
    "DeltaY_max / DeltaY",
    "t_max / T1",
    "T0s / T1"
]
table2 = pd.DataFrame(table2_data, columns=col_names)

# Plot data

fig, ax = plt.subplots()
table1.set_index("-DeltaY_min / DeltaY").plot(ax=ax, 
                                              secondary_y=["T0i / T1"])
ax.set_title("Types VII and VIII systems")
ax.grid()

fig, ax = plt.subplots()
table2.set_index("DeltaY_max / DeltaY").plot(ax=ax, 
                                             secondary_y=["T0s / T1"])
ax.set_title("Types IX and X systems")
ax.grid()

plt.show()