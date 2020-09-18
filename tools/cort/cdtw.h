int get_idx(int var_idx, int time_idx, int nbvar);
double compute_diff(double *x, double *y, int t_x, int t_y, int nbvar);
void compute_slope(double *x, int start_x, int end_x, int nb_var, double *res);
void swap_pointers(double **x, double **y);
void print_multivariate(double *x, int nbvar, int timelen);
void twed(double *x, double *y, int size_x, int size_y, double *distances);
void dtw(double *x, double *y, int size_x, int size_y, int nbvar, double *path, double* distances, int window);
void dtw_weighted(double *x, double *y, int size_x, int size_y, int nbvar, double* distances, double* weight, int window);
void dtw_given_path(double* temporal_distances, int* path, double* weight, int path_length, double* distances);
double cort_simple(double *x, double *y, const unsigned time_start, const unsigned time_end,  const unsigned nb_var);
double cort(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k);
void cort_window_path(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k, double* path, double* distances, int window);
double cort_dtw(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k);
void cort_dtw_window_path(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k, double* path, double* distances, int window);
