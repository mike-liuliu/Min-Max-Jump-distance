0. This is the source code of the paper "Min-Max-Jump distance and its applications."

1. Implementation of MMJ-SC, MMJ-CH, and MMJ-DB are based on the source code of the scikit-learn project.
Implementation of the K_means_ambi_points_multi_one_scom Class is based on the source code provided by Avi Arora in a tutorial artical.
See: https://analyticsarora.com/k-means-for-beginners-how-to-build-from-scratch-in-python/

2. In function index_plot_first_n_label_one_data, if the index's score is "smaller is better", then the "smaller_better" hyper-parameter should be set to True. Otherwise, if the index's score is "larger is better", then the "smaller_better" hyper-parameter should be set to False.

3. Readers can test their own index function, the API is:

    def index_function(X, label):
   
    some codes to compute the index value ...
    
    return the_index_value

then call the index_plot_first_n_label_one_data function. Note the "smaller_better" hyper-parameter.

4. License.
   
License of the source code : Apache License, Version 2.0
License of new data: Creative Commons Attribution 4.0 International

5. Citation:

@article{liu2023min,
  title={Min-Max-Jump distance and its applications},
  author={Liu, Gangli},
  journal={arXiv preprint arXiv:2301.05994},
  year={2023}
}

6. The "multiple_label_145.p" file is larger than 100MB, so it was compressed to make it smaller. Readers need to unzip the "multiple_label_145.p.zip" file in the "data" folder.

7. Performance of other clustering evaluation indices can be found at:
https://github.com/mike-liuliu/gl_index

8. Sketch proof of the theorems and corollary in Section 3.3 ( Other properties of MMJ distance),  can be found in another paper, see: 
https://openreview.net/forum?id=2BOb4SvDFr


 
