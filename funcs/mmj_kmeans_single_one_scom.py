def KMeans_several_times_ambi_points_single_one_scom(X, k, times, distance_matrix):
    loss = np.inf
    for kk in range(times):  
        temp_cluster_idx, temp_centers, temp_loss, temp_strong_ambi_p_idx , temp_weak_ambi_p_idx= K_means_ambi_points_single_one_scom()(X, k, distance_matrix)
        if temp_loss < loss:
            print("Got a better one! Loss is: ", np.round(temp_loss, 8))
            cluster_idx, centers, loss, strong_ambi_p_idx, weak_ambi_p_idx = temp_cluster_idx, temp_centers, temp_loss, temp_strong_ambi_p_idx, temp_weak_ambi_p_idx
    return cluster_idx, centers, loss, strong_ambi_p_idx, weak_ambi_p_idx 

def test_mmj_kmeans_single_one_scom(data_id, datasets, datasets_true_K, attempts = 20):
    
    global mmj_matrix
    
    X = datasets[data_id] 
    num_clusters = datasets_true_K[data_id]

 
    
#     mmj_matrix = np.sqrt(mmj_matrix)

    label, centers_idx, loss, strong_ambi_p_idx, weak_ambi_p_idx = KMeans_several_times_ambi_points_single_one_scom(X, num_clusters, attempts, mmj_matrix)
    print("The single One-SCOM  of each cluster: ", centers_idx)
 
    if strong_ambi_p_idx or weak_ambi_p_idx:
        plot_2D_or_3D_data_empty_circles_weak_strong_multi_one_scom(X, label,centers_idx, strong_ambi_p_idx, weak_ambi_p_idx)        
    else:
        plot_2D_or_3D_data(X, label, centers_idx = centers_idx, plot_center = 1)


class K_means_ambi_points_single_one_scom(): 
    def X_to_centers_dist(self, X, centers_idx): 
        m, n = len(X), len(centers_idx)
        dists = np.zeros((m,n))        
        for i in range(m):
            for j in range(n):
                ttt = centers_idx[j]

                ppp = [self.distance_matrix[i,iki]  for iki in ttt]
   
                dists[i,j] = np.min(ppp)
        return dists

    def init_centers(self, X, K):  
        row, col = X.shape
        while 1:
            centers_idx = []
            for number in range(K):
                randIndex = np.random.randint(row)
                centers_idx.append([randIndex])
            if len_centers_idx(centers_idx) == K:
                break        
        return centers_idx
    
    
    def update_assignment(self, centers_idx, X):  
        row, col = X.shape        
        label = np.empty([row])
 
        distances = self.X_to_centers_dist(X, centers_idx)
        distances = np.round(distances,15)        
        n_clusters = len(centers_idx)  
        strong_ambi_p_idx = []
        weak_ambi_p_idx = []
 
        for kk in range(row):
            tt_min = min(distances[kk])
            kikk = sum(distances[kk] == tt_min)
            if  kikk > 1:    
                belong_n = random_select_from_equal(distances[kk])
                label[kk] = belong_n
                
                if kikk < n_clusters:
                    weak_ambi_p_idx.append(kk)
                else:
                    strong_ambi_p_idx.append(kk)           
            else:
                belong_n = np.argmin(distances[kk])
                label[kk] = belong_n
    
        label_n = [int(ii) for ii in label]  
        label = np.array(label_n)
 
        return label, strong_ambi_p_idx, weak_ambi_p_idx

   
    def update_centers(self, X, label, K):

        centers_idx = []
        N, D = X.shape
 
        for kkk in range(K):
            clu_index = [ii for ii in range(N) if label[ii] == kkk] 
            square_dis_list = [sum(self.distance_matrix_square[pp,clu_index]) for pp in range(N)] 
            
            tt_min = np.min(square_dis_list)
            multi_one_scom = [ii for ii in range(N) if square_dis_list[ii] == tt_min]
 
            multi_one_scom = [multi_one_scom[0]]
                
#             mmm = np.argmin(square_dis_list)         
            centers_idx.append(multi_one_scom)
        if len_centers_idx(centers_idx) != K:
            centers_idx = self.init_centers(X, K)
        return centers_idx

    def get_loss(self, centers_idx, label, X):  
        avg_loss = 0.0
        N, D = X.shape
        for i in range(N):             
            ttt = centers_idx[label[i]]
            ppp = [self.distance_matrix_square[i,iki] for iki in ttt]
            avg_loss += np.min(ppp)

        avg_loss /= N

        return avg_loss

    def __call__(self, X, K, distance_matrix, max_iters=100, abs_tol=1e-16, rel_tol=1e-16, verbose=0):
        
        self.distance_matrix = distance_matrix
        self.distance_matrix_square = distance_matrix**2
        
        centers_idx =  self.init_centers(X, K)
 
        for it in range(max_iters):
            label,strong_ambi_p_idx, weak_ambi_p_idx = self.update_assignment(centers_idx, X)
            centers_idx = self.update_centers(X, label, K)
            loss = self.get_loss(centers_idx, label, X)

            if it:
                diff = np.abs(prev_loss - loss)
                if diff < abs_tol and diff / prev_loss < rel_tol:
                    break
            prev_loss = loss
            if verbose:
                pass
        return label, centers_idx, loss, strong_ambi_p_idx, weak_ambi_p_idx