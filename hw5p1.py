from scipy.optimize import linprog
import numpy as np
from numpy import random

#objective function
c = [2000] * 12 + [320] * 12 + [400] * 12 + [8] * 12 + [180] * 12 +[0] * 12

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
- matrix: 36x72 matrix containing entries for lhs_eq
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
np.fill_diagonal(matrix_eq[0:12, 0:12], -20) #1st row 1st col
np.fill_diagonal(matrix_eq[0:12, 48:60], -1) #1st row 2nd col

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
v3 = random.randint(low=-920, high=-440, size=12)

# Transpose each array
v1_transposed = v1.transpose()
v2_transposed = v2.transpose()
v3_transposed = v3.transpose()

# Stack the transposed arrays
final_eq_vector = np.hstack((v1_transposed, v2_transposed, v3_transposed))


#LHS INEQUALITIES

#initialize matrix w/ 12 rows, 72 columns
matrix_ineq = np.zeros((84, 72))

#first and only inequality
np.fill_diagonal(matrix_ineq[0:12, 0:12], 6) #w
np.fill_diagonal(matrix_ineq[0:12, 48:60], -1) #o

#all the other inequalities
np.fill_diagonal(matrix_ineq[12:24, 0:12], 1) #w
np.fill_diagonal(matrix_ineq[24:36, 12:24], 1) #h
np.fill_diagonal(matrix_ineq[36:48, 24:36], 1) #f
np.fill_diagonal(matrix_ineq[48:60, 36:48], 1) #s
np.fill_diagonal(matrix_ineq[60:72, 48:60], 1) #o
np.fill_diagonal(matrix_ineq[72:84, 60:72], 1) #x


#RHS INEQUALITIES

v_ineq = np.array([0]*84)
final_ineq_vector = v_ineq.transpose()

opt = linprog(c=c, A_ub=matrix_ineq, b_ub=final_ineq_vector, A_eq=matrix_eq, b_eq=final_eq_vector)

print(opt)