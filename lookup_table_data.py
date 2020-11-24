import numpy as np
from scipy import interpolate

# Data from table 10.3 of notes from
# GEL-2005 course on Linear Systems and Control
# by André Desbiens, Summer 2020.
# Note that this table was taken from the following
# publication:
# Tiré de Manipulated variable based PI tuning and 
# detection of poor settings : An industrial 
# experience, A. Pomerleau et É. Poulin, ISA 
# Transactions, Vol. 43, No. 3, pp. 445-457, 2004.


# Table 1 - Types VII and VIII systems
# (left 3 columns of original table)
# Columns:
#  1. -DeltaY_min / DeltaY
#  2. t_min / T1
#  3. T0i / T1

table1_data = np.array([
    [0.01, 0.14, 0.16],
    [0.02, 0.19, 0.23],
    [0.03, 0.22, 0.29],
    [0.04, 0.25, 0.34],
    [0.05, 0.28, 0.39],
    [0.06, 0.31, 0.44],
    [0.07, 0.32, 0.48],
    [0.08, 0.34, 0.52],
    [0.09, 0.36, 0.56],
    [0.1, 0.38, 0.6],
    [0.2, 0.49, 0.96],
    [0.3, 0.56, 1.28],
    [0.4, 0.61, 1.58],
    [0.5, 0.65, 1.88],
    [0.6, 0.68, 2.17],
    [0.7, 0.71, 2.46],
    [0.8, 0.73, 2.75],
    [0.9, 0.75, 3.03],
    [1.0, 0.77, 3.32],
    [1.1, 0.78, 3.6],
    [1.2, 0.79, 3.87],
    [1.3, 0.81, 4.15],
    [1.4, 0.82, 4.43],
    [1.5, 0.82, 4.7],
    [1.6, 0.83, 4.98],
    [1.7, 0.84, 5.26],
    [1.8, 0.85, 5.53],
    [1.9, 0.85, 5.81],
    [2.0, 0.86, 6.09],
    [2.2, 0.87, 6.63],
    [2.4, 0.88, 7.18],
    [2.6, 0.89, 7.72],
    [2.8, 0.89, 8.27],
    [3.0, 0.9, 8.82],
    [3.2, 0.9, 9.37],
    [3.4, 0.91, 9.91],
    [3.6, 0.91, 10.46],
    [3.8, 0.92, 11.01],  # was 11.28 in original doc
    [4.0, 0.92, 11.56],
    [4.5, 0.93, 12.91],
    [5.0, 0.93, 14.28]
])


# Table 2 - Types IX and X systems
# (right-most 3 columns of original table)
# Columns:
#  1. -DeltaY_max / DeltaY
#  2. t_max / T1
#  3. T0s / T1


table2_data = np.array([
    [1.02,  3.13,  1.47],
    [1.04,  2.69,  1.59],
    [1.06,  2.45,  1.69],
    [1.08,  2.28,  1.78],
    [1.1 ,  2.16,  1.86],
    [1.15,  1.95,  2.05],
    [1.2 ,  1.81,  2.23],
    [1.25,  1.72,  2.39],
    [1.3 ,  1.65,  2.55],
    [1.35,  1.58,  2.71],
    [1.4 ,  1.54,  2.86],
    [1.45,  1.5 ,  3.01],
    [1.5 ,  1.46,  3.16],
    [1.55,  1.43,  3.31],
    [1.6 ,  1.41,  3.45],
    [1.65,  1.38,  3.6 ],
    [1.7 ,  1.36,  3.74],
    [1.75,  1.35,  3.88],
    [1.8 ,  1.33,  4.03],
    [1.85,  1.32,  4.17],
    [1.9 ,  1.3 ,  4.31],
    [1.95,  1.29,  4.45],
    [2.  ,  1.28,  4.6 ],
    [2.1 ,  1.26,  4.87],
    [2.2 ,  1.24,  5.16],
    [2.3 ,  1.23,  5.43],
    [2.4 ,  1.21,  5.71],
    [2.5 ,  1.2 ,  5.98],
    [2.6 ,  1.19,  6.26],
    [2.7 ,  1.18,  6.54],
    [2.8 ,  1.17,  6.81],
    [2.9 ,  1.16,  7.09],
    [3.  ,  1.16,  7.36],
    [3.5 ,  1.13,  8.73],
    [4.  ,  1.11, 10.1 ],
    [4.5 ,  1.1 , 11.47],
    [5.  ,  1.08, 12.84], 
    [6.  ,  1.07, 15.56], 
    [7.  ,  1.06, 18.28],
    [8.  ,  1.05, 21.  ],
    [9.  ,  1.04, 23.72]
])


# Define interpolation functions for value look-up
table1_interp = interpolate.interp1d(table1_data[:,0], 
                                     table1_data[:,1:], axis=0)
table2_interp = interpolate.interp1d(table2_data[:,0], 
                                     table2_data[:,1:], axis=0)
