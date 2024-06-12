0. This is the source code of the paper "Min-Max-Jump distance and its applications."

1. Implementation of MMJ-SC, MMJ-CH, and MMJ-DB are based on the source code of the scikit-learn project.
Implementation of the Kmeans Class is based on the source code provided by Avi Arora in a tutorial artical.
See: https://analyticsarora.com/k-means-for-beginners-how-to-build-from-scratch-in-python/

2. In function index_plot_first_n_label_one_data, if the index's score is "smaller is better", then the "smaller_better" hyper-parameter should be set to True. Otherwise, if the index's score is "larger is better", then the "smaller_better" hyper-parameter should be set to False.

3. Readers can test their own index function, the API is:

    def index_function(X, label):
   
    some codes to compute the index value ...
    
    return the_index_value

then call the index_plot_first_n_label_one_data function.

4. To use precomputed mmj distance matrix, readers should download and unzip the "mmj_distance_matrix_precomputed.zip" file firstly.

5. License.
   
License of the source code : Apache License, Version 2.0
License of new data: Creative Commons Attribution 4.0 International

6. Citation:

@article{liu2023min,
  title={Min-Max-Jump distance and its applications},
  author={Liu, Gangli},
  journal={arXiv preprint arXiv:2301.05994},
  year={2023}
}

7. The "multiple_label_145.p" and "mmj_distance_matrix_precomputed.zip" files are larger than 100MB, so they are stored on Git Large File Storage (LFS), readers may need to download it manually.

8. The whole repository (about 1.2G in total) can be downloaded from google drive.
URL:
https://drive.google.com/file/d/12h5Yu2Us1CO--uwPQYIUUnfsGo8i2H8g/view?usp=drive_link



 