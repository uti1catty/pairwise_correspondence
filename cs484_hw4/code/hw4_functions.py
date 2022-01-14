import cv2
import numpy as np
import time


def get_interest_points(image, descriptor_window_image_width):
    # Local Feature Stencil Code
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of interest points for the input image

    # 'image' can be grayscale or color, your choice.
    # 'descriptor_window_image_width', in pixels.
    #   This is the local feature descriptor width. It might be useful in this function to
    #   (a) suppress boundary interest points (where a feature wouldn't fit entirely in the image, anyway), or
    #   (b) scale the image filters being used.
    # Or you can ignore it.

    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.

    # Implement the Harris corner detector (See Szeliski 4.1.1) to start with.

    # If you're finding spurious interest point detections near the boundaries,
    # it is safe to simply suppress the gradients / corners near the edges of
    # the image.

    Gauss_kernel_size = 3
    Gauss_sigma = 1.5
    harris_detector_param = 0.06
    C_threshold = 0.01
    local_maxima_window = 2


    x_derivative = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
    y_derivative = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])

    Ix = cv2.filter2D(image, -1, x_derivative)
    Iy = cv2.filter2D(image, -1, y_derivative)
    
    Ixx = Ix**2
    IxIy = Ix * Iy
    Iyy = Iy**2
    '''
    sum_filter = np.ones((3, 3))

    kerneled_Ixx = cv2.filter2D(Ixx, -1, sum_filter)
    kerneled_IxIy = cv2.filter2D(IxIy, -1, sum_filter)
    kerneled_Iyy = cv2.filter2D(Iyy, -1, sum_filter)
    '''
    Gaussian1d = cv2.getGaussianKernel(Gauss_kernel_size, Gauss_sigma) 
    Gaussian2d = np.outer(Gaussian1d, Gaussian1d.T)

    kerneled_Ixx = cv2.filter2D(Ixx, -1, Gaussian2d)
    kerneled_IxIy = cv2.filter2D(IxIy, -1, Gaussian2d)
    kerneled_Iyy = cv2.filter2D(Iyy, -1, Gaussian2d)

    #kerneled_Ixx = cv2.GaussianBlur(Ixx, Gauss_kernel_size, Gauss_sigma)
    #kerneled_IxIy = cv2.GaussianBlur(IxIy, Gauss_kernel_size, Gauss_sigma)
    #kerneled_Iyy = cv2.GaussianBlur(Iyy, Gauss_kernel_size, Gauss_sigma)
    
    #time_start = time.time()
    '''
    #ground truth version
    Cornerness = np.zeros((image.shape[0], image.shape[1]))
    #harris_detector_param = 0.04    #alpha
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            M_i_j = np.array([[kerneled_Ixx[i][j], kerneled_IxIy[i][j]],
                              [kerneled_IxIy[i][j], kerneled_Iyy[i][j]]])
            determinant = np.linalg.det(M_i_j)
            trace = np.trace(M_i_j)
            Corner = determinant - (harris_detector_param * (trace**2))
            Cornerness[i][j] = Corner
    '''
    #Fast calcuate version
    #Ms: [[IxIx, IxIy],
    #     [IyIx, IyIy]] in x,y
    Ms = np.zeros((image.shape[0], image.shape[1], 2, 2))
    Ms[:, :, 0, 0] = kerneled_Ixx
    Ms[:, :, 0, 1] = kerneled_IxIy
    Ms[:, :, 1, 0] = kerneled_IxIy
    Ms[:, :, 1, 1] = kerneled_Iyy

    determinants = np.linalg.det(Ms)
    traces = np.trace(Ms, axis1=2, axis2=3)
    #harris_detector_param = 0.04    #alpha
    Cornerness = determinants - harris_detector_param * traces**2
    
    #time_end = time.time()

    #print("Elpased time: %.2fs"%(time_end-time_start))

    half_descriptor_window = descriptor_window_image_width // 2

    #suppress edge
    Cornerness[0 : half_descriptor_window - 1, :] = -1
    Cornerness[:, 0 : half_descriptor_window - 1] = -1
    Cornerness[Cornerness.shape[0] - half_descriptor_window : Cornerness.shape[0], :] = -1
    Cornerness[:, Cornerness.shape[1] - half_descriptor_window : Cornerness.shape[1]] = -1

    #Cornerness = cv2.cornerHarris(image, 2, 3, 0.05)
    
    x = np.array([])
    y = np.array([])

    #find local maxima
    for m in range(Cornerness.shape[0]):
        for n in range(Cornerness.shape[1]):
            if Cornerness[m][n] > C_threshold:
                current_value = Cornerness[m][n]

                local_maxima = True
                for a in range(-local_maxima_window, local_maxima_window+1):
                    breaker = False
                    for b in range(-local_maxima_window, local_maxima_window+1):
                        if m+a >= 0 and m+a < image.shape[0] and n+b >= 0 and n+b < image.shape[1] and Cornerness[m+a][n+b] > current_value:
                            local_maxima = False
                            breaker = True
                            break
                    if breaker:
                        break
                
                if local_maxima:
                    x = np.append(x, [n])
                    y = np.append(y, [m])

    #x, y = np.where(Cornerness > 0.5)  #return tuple (x array, y array)
    print("x length: ", len(x))
    print("y length: ", len(y))

    # Placeholder that you can delete -- random points
    #x = np.floor(np.random.rand(500) * np.float32(image.shape[1]))
    #y = np.floor(np.random.rand(500) * np.float32(image.shape[0]))
    return x,y

    # After computing interest points, here's roughly how many we return
    # For each of the three image pairs
    # - Notre Dame: ~1300 and ~1700
    # - Mount Rushmore: ~3500 and ~4500
    # - Episcopal Gaudi: ~1000 and ~9000



def get_descriptors(image, x, y, descriptor_window_image_width):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Returns a set of feature descriptors for a given set of interest points.

    # 'image' can be grayscale or color, your choice.
    # 'x' and 'y' are nx1 vectors of x and y coordinates of interest points.
    #   The local features should be centered at x and y.
    # 'descriptor_window_image_width', in pixels, is the local feature descriptor width.
    #   You can assume that descriptor_window_image_width will be a multiple of 4
    #   (i.e., every cell of your local SIFT-like feature will have an integer width and height).
    # If you want to detect and describe features at multiple scales or
    # particular orientations, then you can add input arguments.

    # 'features' is the array of computed features. It should have the
    #   following size: [length(x) x feature dimensionality] (e.g. 128 for
    #   standard SIFT)

    x_derivative = np.array([[-1, 0, 1],
                             [-2, 0, 2],
                             [-1, 0, 1]])
    y_derivative = np.array([[-1, -2, -1],
                             [0, 0, 0],
                             [1, 2, 1]])


    x_derivatived = cv2.filter2D(image, -1, x_derivative)
    y_derivatived = cv2.filter2D(image, -1, y_derivative)
    gradient_direction = np.arctan2(y_derivatived, x_derivatived)
    gradient_magnitude = np.sqrt(x_derivatived**2 + y_derivatived**2)

    x_derivatived = x_derivatived.reshape(1, x_derivatived.shape[0], x_derivatived.shape[1])
    y_derivatived = y_derivatived.reshape(1, y_derivatived.shape[0], y_derivatived.shape[1])
    gradient_direction = gradient_direction.reshape(1, gradient_direction.shape[0], gradient_direction.shape[1])
    gradient_magnitude = gradient_magnitude.reshape(1, gradient_magnitude.shape[0], gradient_magnitude.shape[1])

    data = np.concatenate((x_derivatived, y_derivatived), axis=0)
    data = np.concatenate((data, gradient_direction), axis=0)
    data = np.concatenate((data, gradient_magnitude), axis=0)

    # data: x_derivative / y_derivative / gradient_direction (-pi/2 ~ pi/2) / gradient_magnitude stack to axis 0 direction

    half_window = descriptor_window_image_width // 2
    quarter_window = descriptor_window_image_width // 4

    features = np.zeros((x.shape[0], quarter_window * quarter_window * 8))

    for i in range(len(x)):
        point_x = int(x[i])
        point_y = int(y[i])
        point_data = data[:, point_y - half_window + 1:point_y + half_window , point_x - half_window + 1:point_x + half_window]

        histogram = np.zeros((quarter_window, quarter_window, 8))

        total_gradient = np.zeros((8))
        for m in range(quarter_window):
            for n in range(quarter_window):
                part_data = point_data[:, m*4:m*4 + 3, n*4:n*4+3]
                part_histogram = np.zeros((8))
                
                part_histogram[0] = np.sum(part_data[3, (part_data[2] >= 0) & (part_data[2] < np.pi/4)])
                part_histogram[1] = np.sum(part_data[3, (part_data[2] >= np.pi/4) & (part_data[2] < np.pi/2)])
                part_histogram[2] = np.sum(part_data[3, (part_data[2] >= np.pi/2) & (part_data[2] < 3*np.pi/4)])
                part_histogram[3] = np.sum(part_data[3, (part_data[2] >= 3*np.pi/4) & (part_data[2] <= np.pi)])
                part_histogram[4] = np.sum(part_data[3, (part_data[2] >= -np.pi) & (part_data[2] < -3*np.pi/4)])
                part_histogram[5] = np.sum(part_data[3, (part_data[2] >= -3*np.pi/4) & (part_data[2] < -np.pi/2)])
                part_histogram[6] = np.sum(part_data[3, (part_data[2] >= -np.pi/2) & (part_data[2] < -np.pi/4)])
                part_histogram[7] = np.sum(part_data[3, (part_data[2] >= -np.pi/4) & (part_data[2] < 0)])

                total_gradient += part_histogram
                histogram[m, n, :] = part_histogram

        #orentation normalized
        dominant_orientation = np.argmax(total_gradient)
        for m in range(quarter_window):
            for n in range(quarter_window):
                part_histogram = histogram[m, n, :]
                orentation_normalized_histogram = np.zeros((8))
                orentation_normalized_histogram[0:(8 - dominant_orientation)] = part_histogram[dominant_orientation:8]
                orentation_normalized_histogram[(8 - dominant_orientation):8] = part_histogram[0:dominant_orientation]

                histogram[m, n, :] = orentation_normalized_histogram

        feature = histogram.reshape(histogram.shape[0] * histogram.shape[1] * histogram.shape[2])

        #if norm is 0, it is [0,0,...,0] feature
        if np.linalg.norm(feature) != 0:
            feature /= np.linalg.norm(feature)  #Normalize feature vector
            feature[feature > 0.2] = 0.2        #Clamp to 0.2
            feature /= np.linalg.norm(feature)  #Renormalize

        features[i] = feature

    # Placeholder that you can delete. Empty features.
    #features = np.zeros((x.shape[0], 128))
    return features

def match_features(features1, features2):
    # Written by James Hays for CS 143 @ Brown / CS 4476/6476 @ Georgia Tech
    # Revised Python codes are written by Inseung Hwang at KAIST.

    # Please implement the "nearest neighbor distance ratio test",
    # Equation 4.18 in Section 4.1.3 of Szeliski.
    # For extra credit you can implement spatial verification of matches.

    #
    # Please assign a confidence, else the evaluation function will not work.
    #

    # This function does not need to be symmetric (e.g., it can produce
    # different numbers of matches depending on the order of the arguments).

    # Input:
    # 'features1' and 'features2' are the n x feature dimensionality matrices.
    #
    # Output:
    # 'matches' is a k x 2 matrix, where k is the number of matches. The first
    #   column is an index in features1, the second column is an index in features2.
    #
    # 'confidences' is a k x 1 matrix with a real valued confidence for every match.
    
    next = 0

    distances = np.zeros((features1.shape[0], features2.shape[0]))

    for i in range(features1.shape[0]):
        f1 = features1[i]
        f1 = f1.reshape(1, f1.shape[0])
        minus = np.repeat(f1, features2.shape[0], axis=0) - features2
        euclidean_distance = np.linalg.norm(minus, axis=1) # 1 feature of f1 & every features of f2

        distances[i] = euclidean_distance

    #print("distances:\n", distances)

    distances_temp = np.copy(distances)
    first_closest_distance_idxs = np.argmin(distances_temp, axis=1)
    #print("first_closest_distance_idxs:\n", first_closest_distance_idxs)
    max_distances = np.max(distances_temp, axis=1)

    i = 0
    for row_closest in first_closest_distance_idxs:
        distances_temp[i, row_closest] = max_distances[i]       # make it to its row's largest distance value to find second_closest
        i += 1
    
    second_closest_distance_idxs = np.argmin(distances_temp, axis=1)
    #print("second_closest_distance_idxs:\n", second_closest_distance_idxs)

    matches = np.zeros((1,2))
    confidences = np.array([])
    for i in range(len(first_closest_distance_idxs)):
        row_first_idx = first_closest_distance_idxs[i]
        first_closest = distances[i, row_first_idx]
        row_second_idx = second_closest_distance_idxs[i]
        second_closest = distances[i, row_second_idx]
        #print("first_closest: ", first_closest, end="  ")
        #print("second_closest: ", second_closest)

        # if second_closest is 0, it first, second are all [0,0...,0] feature. Ignore it.
        if second_closest == 0:
            continue
        else:
            NNDR = first_closest / second_closest
        #print("NNDR: ", NNDR)

        matches = np.append(matches, [[i, row_first_idx]], axis=0)
        confidences = np.append(confidences, [1 - NNDR], axis=0)

    matches = matches[1:, :]
    
    #print("matches shape: ", matches.shape, '\n', "confidences shape: ", confidences.shape)


    # Placeholder random matches and confidences.
    #num_features = min(features1.shape[0], features2.shape[0])
    #matches = np.zeros((num_features, 2))
    #matches[:,0] = np.random.permutation(num_features)
    #matches[:,1] = np.random.permutation(num_features)
    #confidences = np.random.rand(num_features)
    return matches, confidences

