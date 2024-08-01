def cal_new_label_with_mmj_classifier(X, label, strong_ambi_p_idx, weak_ambi_p_idx):
    N = len(X)
    K = len(set(label))
    border_p = strong_ambi_p_idx + weak_ambi_p_idx
    
    if not border_p:
        return label
    
    
    ordi_p = [ii for ii in range(N) if ii not in border_p]  
    
    X_divide_idx_no_border_p = []
    for p in range(K):
        ttt = [ii for ii in ordi_p if label[ii] == p]
        X_divide_idx_no_border_p.append(ttt)
        
    Euc_distance_matrix = pairwise_distances(X)
    Euc_distance_matrix = np.round(Euc_distance_matrix,15)        

    new_label = label.copy()

    each_clu_mmj_matrix_list = []
    centroid_idx_inner = []
    centroid_idx_outter = []

    ordi_p_label = [label[ii] for ii in ordi_p]
    X_ordi_p = X[ordi_p]    
    
    X_divide = got_X_divide_from_labels(X_ordi_p, ordi_p_label)

    for j, each_divide in enumerate(X_divide):
        mmj_matri =  cal_mmj_matrix_by_algo_4_Calculation_and_Copy(each_divide) 
        each_clu_mmj_matrix_list.append(mmj_matri)

        mmj_matri_S = mmj_matri**2

        ttt = np.sum(mmj_matri_S, axis = 1)

        ppp = np.argmin(ttt)

        centroid_idx_inner.append(ppp)
        centroid_idx_outter.append(X_divide_idx_no_border_p[j][ppp])    
    
    
    for each_bor_p in border_p:

        mmj_dis_to_each_cluster_cen_list = []

        for i, _ in enumerate(centroid_idx_outter):
            centroid_i_inner = centroid_idx_inner[i]
            each_mmj_matri = each_clu_mmj_matrix_list[i]        
            dis_row = each_mmj_matri[centroid_i_inner]
            each_divide_idx_no_border_p = X_divide_idx_no_border_p[i]

            N_i = len(dis_row)

            temp_list = []
            for w in range(N_i):
                outter_idx = each_divide_idx_no_border_p[w]

                euc = Euc_distance_matrix[outter_idx,each_bor_p]   
                temp_p_to_cen_mmj_dis = np.max([euc, dis_row[w]])
                temp_list.append(temp_p_to_cen_mmj_dis)

            p_to_cen_mmj_dis = np.min(temp_list)
            mmj_dis_to_each_cluster_cen_list.append(p_to_cen_mmj_dis)
        label_each_bor_p = np.argmin(mmj_dis_to_each_cluster_cen_list)
        new_label[each_bor_p] = label_each_bor_p
        
 
    return new_label

def cal_label_for_new_points_with_mmj_classifier(X, label, strong_ambi_p_idx, weak_ambi_p_idx, X_new):
    N = len(X)
    K = len(set(label))
    border_p = strong_ambi_p_idx + weak_ambi_p_idx
 
    
    ordi_p = [ii for ii in range(N) if ii not in border_p]  
    
    X_divide_idx_no_border_p = []
    for p in range(K):
        ttt = [ii for ii in ordi_p if label[ii] == p]
        X_divide_idx_no_border_p.append(ttt)
 
    each_clu_mmj_matrix_list = []
    centroid_idx_inner = []
    centroid_idx_outter = []

    ordi_p_label = [label[ii] for ii in ordi_p]
    X_ordi_p = X[ordi_p]    
    
    X_divide = got_X_divide_from_labels(X_ordi_p, ordi_p_label)

    for j, each_divide in enumerate(X_divide):
        mmj_matri =  cal_mmj_matrix_by_algo_4_Calculation_and_Copy(each_divide) 
        each_clu_mmj_matrix_list.append(mmj_matri)

        mmj_matri_S = mmj_matri**2

        ttt = np.sum(mmj_matri_S, axis = 1)

        ppp = np.argmin(ttt)

        centroid_idx_inner.append(ppp)
        centroid_idx_outter.append(X_divide_idx_no_border_p[j][ppp])    
    
    X_new_label = np.zeros(len(X_new))
    for q, X_new_p in enumerate(X_new):

        mmj_dis_to_each_cluster_cen_list = []

        for i, _ in enumerate(centroid_idx_outter):
            centroid_i_inner = centroid_idx_inner[i]
            each_mmj_matri = each_clu_mmj_matrix_list[i]        
            dis_row = each_mmj_matri[centroid_i_inner]
            each_divide_idx_no_border_p = X_divide_idx_no_border_p[i]

            N_i = len(dis_row)

            temp_list = []
            for w in range(N_i):
                outter_idx = each_divide_idx_no_border_p[w]
                
                euc = np.linalg.norm(X[outter_idx] - X_new_p).round(15)
 
                temp_p_to_cen_mmj_dis = np.max([euc, dis_row[w]])
                temp_list.append(temp_p_to_cen_mmj_dis)

            p_to_cen_mmj_dis = np.min(temp_list)
            mmj_dis_to_each_cluster_cen_list.append(p_to_cen_mmj_dis)
        label_X_new_p = np.argmin(mmj_dis_to_each_cluster_cen_list)
        X_new_label[q] = label_X_new_p
        
 
    return X_new_label