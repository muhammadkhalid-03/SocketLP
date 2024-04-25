from scipy.optimize import linprog
import numpy as np
from numpy import random




#objective function
c = np.array([2000] * 12 + [320] * 12 + [400] * 12 + [8] * 12 + [180] * 12 +[0] * 12)
c=c.transpose()
"""
x_i - 20w_i - o_i = 0

w_1 - h_i + f_i= 30 (at i = 1)
w_i - w_{i-1} - h_i + f_i = 0 (i = 2, 3, ..., 12)

s_i - s_{i-1} - x_i = - d_i     [1]*12+[-1]*12+[-1]*12
o_i - 6w_i <= 0
w_i, x_i, o_i, h_i, f_i, s_i >= 0   bnd
"""

"""
Helper function for filling subdiagonals.

Inputs:
- matrix: A matrix with rows and columns both in multiples of 12
- index: Index of first entry of sub-matrix to start at
- val: Value to change sub-diagonal entries to

Outputs:
- matrix: Updated matrix
"""
def fillSubDiag(matrix, r, c, val):
    for i in range(r+1, r+12):
        matrix[i][c] = val #start at 2nd row of sub-matrix with 
        c+=1
    return matrix

#LHS EQUALITIES

#initialize matrix w/ 36 rows, 72 columns
matrix_eq = np.zeros((36, 72))

#first equality
np.fill_diagonal(matrix_eq[0:12, 0:12], -20) #w
np.fill_diagonal(matrix_eq[0:12, 48:60], -1) #o
np.fill_diagonal(matrix_eq[0:12, 60:72], 1) #x

#2nd equality
np.fill_diagonal(matrix_eq[12:24, 0:12], 1) #w
matrix_eq = fillSubDiag(matrix_eq, 12, 0, -1) #w sub-diag
np.fill_diagonal(matrix_eq[12:24, 12:24], -1) #h
np.fill_diagonal(matrix_eq[12:24, 24:36], 1) #f

#3rd equality
np.fill_diagonal(matrix_eq[24:36, 36:48], 1) #s
matrix_eq = fillSubDiag(matrix_eq, 24, 36, -1) #s sub-diag
np.fill_diagonal(matrix_eq[24:36, 60:72], -1) #x


#RHS EQUALITIES

v1 = np.array([0]*12)
v2 = np.array([30] + [0]*11)
v3 = random.randint(low=-920, high=-440, size=12) #d

# Transpose each array
v1_transposed = v1.transpose()
v2_transposed = v2.transpose()
v3_transposed = v3.transpose()

print("\nValues of d:\n", v3_transposed)

# Stack the transposed arrays
final_eq_vector = np.hstack((v1_transposed, v2_transposed, v3_transposed))


#LHS INEQUALITIES

#initialize matrix w/ 12 rows, 72 columns
matrix_ineq = np.zeros((12, 72))

#first and only inequality
np.fill_diagonal(matrix_ineq[0:12, 0:12], 6) #w
np.fill_diagonal(matrix_ineq[0:12, 48:60], -1) #o


#RHS INEQUALITIES

v_ineq = np.array([0]*12).transpose()

# Define the bounds for each variable for each month
bounds_w = [(0, float("inf"))] * 12
bounds_h = [(0, float("inf"))] * 12
# bounds_f = [(0, 30)] + [(0, float("inf"))] * 11  # Limit f_1 to (0, 30)
bounds_f = [(0, float("inf"))] *12
bounds_s = [(0, float("inf"))] * 12
bounds_o = [(0, float("inf"))] * 12
bounds_x = [(0, float("inf"))] * 12

# Concatenate the bounds for all variables
bounds = np.array(bounds_w + bounds_h + bounds_f + bounds_s + bounds_o + bounds_x)


opt = linprog(c=c, A_ub=matrix_ineq, b_ub=v_ineq, A_eq=matrix_eq, b_eq=final_eq_vector, bounds=bounds, method='simplex')

print(opt)

print(opt.success)

print(opt.x)