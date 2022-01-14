import cv2
import numpy as np
import matplotlib.pyplot as plt

'''returns homogeneous coordinated 'num' points in 'x_y_range' X 'x_y_range' '''
def pick_random_points(num, x_y_range):
    Points = np.ones((3, num))

    picked_points = []
    for i in range(num):
        x = np.random.randint(0, x_y_range)
        Points[0, i] = x
        y = np.random.randint(0, x_y_range)
        while (x, y) in picked_points:
            #print("picked again!")
            y = np.random.randint(0, x_y_range)
        Points[1, i] = y
        picked_points.append((x, y))
    return Points


inlier_number = 15
outlier_number = 30
gaussian_std = 2 #gaussian noise sigma

# pick random 100 points for Q inliers (homogeneous coordinate)

x_y_range = 256 * np.sqrt((inlier_number / 10))
#print(f'I will pick {inlier_number} points from {x_y_range} X {x_y_range}')
Q_inliers = pick_random_points(inlier_number, x_y_range)

#print(Q_inliers)

#gaussian noise

noised_Q_inliers = np.ones((3, inlier_number))
for n in range(inlier_number):
    x_noise = np.random.normal(0, gaussian_std)
    noised_Q_inliers[0, n] = Q_inliers[0, n] + x_noise
    y_noise = np.random.normal(0, gaussian_std)
    noised_Q_inliers[1, n] = Q_inliers[1, n] + y_noise

#print("noised Q points:")
#print(noised_Q_inliers)

'''    
Make random rotation and translation matrix
rotation 'x' angle and do translation (h, k)
[[cosx -sinx    h]
 [sinx  cosx    k]
 [   0     0    1]]
'''
angle = np.random.rand(1) * np.pi * 2
translation = np.random.rand(2) * 20

rt = [[np.cos(angle)[0], -np.sin(angle)[0], translation[0]],
      [np.sin(angle)[0],  np.cos(angle)[0], translation[1]],
      [               0,                 0,              1]]
    
#print(rt)

P_inliers = np.around(rt @ noised_Q_inliers)
#print('P inliers:')
#print(P_inliers)

'''ground_truth_match[0]: Q_inliers & ground_truth_match[1]: P_inliers'''
ground_truth_match = np.concatenate((np.expand_dims(Q_inliers, axis=0), np.expand_dims(P_inliers, axis=0)), axis=0)

#print('ground truth match: ')
#print(ground_truth_match)

Q_outliers = pick_random_points(outlier_number, x_y_range)
P_outliers = pick_random_points(outlier_number, x_y_range)
#print(Q_outliers)

Q_points = np.concatenate((Q_inliers, Q_outliers), axis=1)
P_points = np.concatenate((P_inliers, P_outliers), axis=1) 

#print(Q_points)

point_number = inlier_number + outlier_number
corrM = np.zeros((point_number ** 2, point_number ** 2))

'''candidate assignments[i] is ith [Q_point, P_point]'''
candidate_assignments = []
for i in range(point_number):
    for j in range(point_number):
        candidate_assignments.append([i, j])
candidate_assignments = np.array(candidate_assignments)

#print('candidate assignments: ')
#print(candidate_assignments)

'''Algorithm start'''

'''
A1. 
Build the symmetric non-negative n x n matrix M as described n Section 2.
Set correspondece Map corrM
'''
sigma_d = 3

for m in range(candidate_assignments.shape[0]):
    for n in range(candidate_assignments.shape[0]):

        if m == n:
            corrM[m, n] = 0
            continue

        ass_a = candidate_assignments[m]
        ass_b = candidate_assignments[n]

        point_i = Q_points[:, ass_a[0]].T
        point_ii = P_points[:, ass_a[1]].T
        point_j = Q_points[:, ass_b[0]].T
        point_jj = P_points[:, ass_b[1]].T
        #print('point_i shape: ', point_i.shape)

        d_ij = np.linalg.norm(point_i - point_j)
        #print('distance ij: ', d_ij)
        d_iijj = np.linalg.norm(point_ii - point_jj)

        if np.abs(d_ij - d_iijj) < 3 * sigma_d:
            corrM[m, n] = 4.5 - ((d_ij - d_iijj) ** 2) / (2 * sigma_d ** 2)
        else:
            corrM[m, n] = 0

'''
A2.
Let x* be the principal eigenvector of M. 
Initialize the solution vector x with n X 1 ero vector. 
Initialize L with the set of all candidate assignemnts.
'''
eig_val, eig_vec = np.linalg.eig(corrM)
# eig_vec : column vectors
#print("eigen values: ", eig_val)
print("largest eigen value index: ", np.argmax(eig_val))
#print("eigen vectors: ")
#print(eig_vec)

principal_eig_vec = eig_vec[:, 0].T
print("principal eigen vector: ")
print(principal_eig_vec)

indicator_vec = np.zeros(candidate_assignments.shape[0])
L = candidate_assignments.tolist()

'''
A3.
Find a* = argmax(x*(a)). 
If x*(a*) = 0 stop and retur the solution x. 
Otherwise set x(a*) = 1 and remove a* from L.
'''
while len(L) != 0:
    most_proper_assignment_idx = np.argmax(principal_eig_vec)

    if principal_eig_vec[most_proper_assignment_idx] == 0:
        break

    indicator_vec[most_proper_assignment_idx] = 1

    most_proper_assignment = candidate_assignments[most_proper_assignment_idx].tolist()
    L.remove(most_proper_assignment)
    principal_eig_vec[most_proper_assignment_idx] = 0

    '''
    A4. 
    Remove from L all potential assignments in conflict with a* = (i, i'). 
    These are assignments of the form (i, k) and (q, i') for one-to-one correspondence constraints 
    (they will be of the form (i,k) for one-to-many constraints).
    '''
    L_temp = L.copy()
    for assignment in L_temp:
        if assignment[0] == most_proper_assignment[0] or assignment[1] == most_proper_assignment[1]:
            L.remove(assignment)
            principal_eig_vec[candidate_assignments.tolist().index(assignment)] = 0
    
    '''
    A5.
    If L is empty return the solution x. Otherwise go back to A3.
    '''
    
#print("indicator vector x: ")
#print(indicator_vec)

matched_assignments = candidate_assignments[indicator_vec == 1, :]
print(matched_assignments)

correct_match_num = 0
for i in range(inlier_number):
    if matched_assignments[i, 0] == matched_assignments[i, 1]:
        correct_match_num += 1

print("correct match number: ", correct_match_num)
matched_assignments = matched_assignments.tolist()


'''
TODO 
1. Cut the matches that have low accuracy.
2. Compare the time with the linear matching
'''