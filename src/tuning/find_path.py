import numpy as np
import mlrose

def find_best_path(points_data, random_state = 0, use_coords = True):
    '''Find the shortest closed path (in euclidean distance)
    using Elastic Net Algorithm in mlrose package.'''
    # tuning curve size: (numNeruo, numBin) (might be high dimensional: numNeuro>3)
    # numBin = num_pts
    points_data = np.array(points_data)
    if len(points_data.shape) == 1:
        points_data = points_data.reshape((1,points_data.size))
    num_pts = points_data.shape[1] # number of points
    
    def euclidean_distance(x,y):
        return np.sqrt(np.sum((x-y)**2))

    if use_coords:
        coords_list = []
        for i in range(num_pts):
            coords_list += [tuple(points_data[:,i])]
        # Initialize fitness function object using coords_list
        fitness = mlrose.TravellingSales(coords = coords_list)
    else:
        # use euclidean distances computed

        dist_list = []
        for i in range(num_pts):
            for j in range(num_pts):
                if i!=j:
                    dist_list.append((i, j, euclidean_distance(points_data[:,i], points_data[:,j])))
        # Initialize fitness function object using dist_list
        fitness = mlrose.TravellingSales(distances = dist_list)

    problem_fit = mlrose.TSPOpt(length = num_pts, fitness_fn = fitness,
                                maximize=False)

    # Solve problem using the genetic algorithm
    best_path, best_path_len = mlrose.genetic_alg(problem_fit, random_state = random_state)

    return best_path, best_path_len # length of closed curve


def compute_path_len(points_data, path_vec, path_close = True):
    def euclidean_distance(x,y):
        return np.sqrt(np.sum((x-y)**2))

    path_len = 0
    for i in range(points_data.shape[1]-1):
        path_len += euclidean_distance(points_data[:,path_vec[i]], points_data[:, path_vec[i+1]])
    if path_close:
        path_len += euclidean_distance(points_data[:,path_vec[-1]], points_data[:, path_vec[0]])

    return path_len