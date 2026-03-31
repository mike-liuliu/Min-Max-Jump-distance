
import pickle
import random
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
import numpy as np

def Algorithm_3_Sample_one_path_by_stochastic_greedy(X, start_id, end_id, distance_matrix, k = 3, show_path = False):
    
    if start_id == end_id:
        return 0
    
    
    path_list = []
    remaining_list = list(range(len(X)))

    next_id = start_id
 
    remaining_list.remove(next_id)
    path_list.append(next_id)
 
    while next_id != end_id:  
 
  
        next_dis_to_others = distance_matrix[next_id][remaining_list]
 
        temp_id = np.argsort(next_dis_to_others)[:k]
    
        near_id  = np.array(remaining_list)[temp_id].tolist()
        
        nearest_id = near_id[0]
 
        if end_id  == nearest_id:
            next_id = end_id
        else:
            next_id = random.choice(near_id)  

        remaining_list.remove(next_id)
        path_list.append(next_id)
    
    n = len(path_list)
    
    jump_list = [distance_matrix[path_list[i]][path_list[i+1]] for i in range(n-1)]
    
    if show_path:
        plt.scatter(X[:,0],X[:,1])
        plt.plot(X[path_list][:,0], X[path_list][:,1], c ="r")
        plt.scatter(X[start_id][0], X[start_id][1], c ="w",marker = '*')
        plt.scatter(X[end_id][0], X[end_id][1], c ="w",marker = '*')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.show()
    
 
    return max(jump_list)