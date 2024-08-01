import matplotlib.pyplot as plt
import numpy as np

def generate_even_box_X_extend(X, N_num, extend_how_much):
    min_x, max_x = np.min(X[:,0]) - extend_how_much, np.max(X[:,0]) + extend_how_much
    min_y, max_y = np.min(X[:,1]) - extend_how_much, np.max(X[:,1]) + extend_how_much
    aa = max_x - min_x
    bb = max_y - min_y
    
    k = np.sqrt(N_num/(aa*bb))

    a = int(k*aa)
    b = int(k*bb)
    # print(a,b)
    x = np.linspace(min_x, max_x, a)
    y = np.linspace(min_y, max_y, b)

    xx, yy = np.meshgrid(x, y)
    xx = xx.reshape(-1,1)
    yy = yy.reshape(-1,1)

    tt = np.array([[xx[i],yy[i]] for i in range(len(xx))]).reshape(-1,2)
#     plt.scatter(tt[:,0],tt[:,1])
    return tt

def plot_even_box_and_data(X_even_box, labels, X):
    a,b = X_even_box.shape
    markers = ["." , "+", "s" , "x", "v" , "1" , "p", "P", "*", "o" , "d"]
    X_divide = got_X_divide_from_labels(X_even_box, labels)
    for tt in range(len(X_divide)):
#         plt.scatter(X_divide[tt][:,0],X_divide[tt][:,1], marker=markers[tt%len(markers)], s = 10)
        plt.scatter(X_divide[tt][:,0],X_divide[tt][:,1],  s = 10)
    plt.scatter(X[:, 0], X[:, 1], c = "w", s = 2)
    plt.show()

def got_X_divide_from_labels(X, labels):
    X_divide = []
    for jj, ii in enumerate(list(set(labels))):
        assert jj == ii, "jj == ii"
        ppp = X[labels == ii]
        X_divide.append(ppp)
    return X_divide