

#include <stdlib.h>
#include <stdio.h>
#include <float.h>
#include <math.h>
#include <float.h> /* for max_float */
#include <string.h> /* for memset*/
#include "cdtw.h"

double max2(const int x, const int y){
  return (x<y) ? y : x;
}

double min2(const int x, const int y){
  return (x<y) ? x : y;
}

double min3(const double x, const double y, const double z){
  if (x < y)
    if (x < z) return x;
  if (y < z) return y;
  return z;
}


int get_idx(int var_idx, int time_idx, int nbvar){
  return  (nbvar*time_idx)+var_idx;
}


void swap_pointers(double **x, double **y){
  double *temp = *x;
  *x = *y;
  *y = temp;
}

/*
"""
Compute the difference between two points:
ret = \sum_{var \in variable} (x[t_x,var] - x[t_y,var])

:param x: the input vector x. x is an array representing a matrix compatible with get_idx function.
:type x: floating point array.
:param y: the input vector x. x is an array representing a matrix compatible with get_idx function.
:type y: floating point array.
:param t_x: the time considered in x
:param t_y: the time considered in y
:param nbvar: the number of variables (1 for univariate, 2 for bivariate, ...)
:return: the sum at the variable level: \sum_{v\in var} |x[t_x, var] - x[t_y, var]|
"""
 */
double compute_diff(double *x, double *y, int t_x, int t_y, int nbvar){
  int var;
  double res = 0.0;
  double *posx = x+get_idx(0, t_x, nbvar);
  double *posy = y+get_idx(0, t_y, nbvar);
  for (var=0; var < nbvar; ++var, ++posx, ++posy){
    /* double temp = (*posx) - (*posy); */
    res += fabs(*posx - *posy);
  }
  return res;
}

double compute_diff_d(double *x, double *y, int t_x, int t_y, int nbvar){
  int var;
  double res = 0.0;
  printf("compute_diff_partial res=");
  for (var=0; var < nbvar; ++var){
    res += fabs(x[get_idx(var, t_x, nbvar)]-y[get_idx(var, t_y, nbvar)]);
    printf("%lf ", res);
  }
  printf("\n");
  return res;
}

/* assign each element of x the double_max value */
void set_max(double *x, const unsigned size_x){
  unsigned i;
  for (i=0; i < size_x; ++i){
    x[i] = DBL_MAX;
  }
}

double compute_diff_l2(double *x, double *y, int t_x, int t_y, int nbvar){
  int var;
  double res = 0.0;
  double *posx = x+get_idx(0, t_x, nbvar);
  double *posy = y+get_idx(0, t_y, nbvar);
  for (var=0; var < nbvar; ++var, ++posx, ++posy){
    double temp = (*posx) - (*posy);
    res += temp*temp;
  }
  return res;
}


// compute the min and update pos with the element that is the smaller.
// in the context of dtw l stands for left, b for botom, d for diagonal
double min3_direction(const double b, const double l, const double d,  char *pos){
  if (l < b){  if (l < d){  *pos = 'l';      return l; }  }
  if (b < d){    *pos = 'b';    return b;  }
  *pos = 'd';
  return d;
}


void show_mat(char **path_mat, int size_i, int size_j){
  int i,j;
  printf("path matrix\n");
  for (j=size_j-1; j>=0; j--){
    for (i=0; i< size_i; i++){
      printf("%c ",path_mat[i][j] );
    }
    printf("\n");
  }
}


// computes the c_t and A_t as define in eq 7 and 8 of paper
// res is of size 2*size_x  and will contains the following :
// denom (output) denom(t) = N(t,x) = normalized number of time timestamps t is aligned
// num = ct = normalized aligned values at time t (same shape as x)
void compute_ct(double *x, double *y, int size_x, int size_y, int nbvar,
		int *alignment, int alignment_size,
		double* num, double *denom){
  char debug = 0;
  if (debug){printf("NB var=%d\n,size_x=%d, size_y=%d, path_size=%d", nbvar, size_x, size_y, alignment_size);}
  // set to 0 (check if needed)
  memset(num, 0, size_x*nbvar*sizeof(double));
  memset(denom, 0, size_x*sizeof(double));

  // compute the statistics from the alignment
  int i, t, tx, v;
  for (i=0, t=0; i < alignment_size; i+=2, t++){
    tx = alignment[i];
    denom[tx] += 1; // computes N(t,x)
    for (v=0; v < nbvar; v++){
      // compute numerator
      num[get_idx(v, tx, nbvar)] += y[get_idx(v, alignment[i+1], nbvar)];
    }
  }
  // normalizing ie dividing by alignment size
  alignment_size /= 2;
  for (i=0; i < size_x*nbvar; i++){
    num[i] /= alignment_size;
  }
  for (i=0; i < size_x; i++){
    denom[i] /= alignment_size;
    }
  if (debug){
    printf("\nx = ");
    for (i=0; i< size_x; i++){ printf("%lf ", x[i]);}
    printf("\ny = ");
    for (i=0; i< size_y; i++){ printf("%lf ", y[i]);}
    printf("\ndenom= ");
    for (i=0; i< size_x; i++){ printf("%lf ", denom[i]); }
    printf("\nnum= ");
    for (i=0; i< size_x*nbvar; i++){ printf("%lf ", num[i]); }
    printf("\n");
  }
}


// computes the c_t and A_t as define in eq 7 and 8 of paper
// res is of size 2*size_x  and will contains the following :
// denom (output) denom(t) = N(t,x) = normalized number of time timestamps t is aligned
// num = ct = normalized aligned values at time t (same shape as x)
void compute_At(double *x, double *y, int size_x, int size_y, int nbvar, int *alignment,
	       int alignment_size, double* ret){
  char debug = 0;
  if (debug)     printf("\nsize_x=%d, size_y=%d, nb_var=%d, align_size=%d\n",
			size_x, size_y, nbvar, alignment_size);
  // set to 0
  // TODO: check if faster done in C or use np.zeros in python
  memset(ret, 0, size_x*sizeof(double));

  // compute the statistics from the alignment
  int i;
  for (i=0; i < alignment_size; i+=2){
    if (debug) printf("adding diff between %d and %d", alignment[i], alignment[i+1]);
    ret[alignment[i]] += compute_diff_l2(x, y, alignment[i], alignment[i+1], nbvar);
    if (debug) printf("=%lf done\n", ret[alignment[i]]);
  }
  // normalizing ie dividing by alignment size
  alignment_size /= 2;
  for (i=0; i < size_x; i++){
    ret[i] /= alignment_size;
  }

  if (debug){
    printf("align_size=%d\n", alignment_size);
    printf("\nsize_x=%d, size_y=%d \nx = ", size_x, size_y);
    for (i=0; i< size_x*nbvar; i++){ printf("%lf ", x[i]);}
    printf("\ny = ");
    for (i=0; i< size_y*nbvar; i++){ printf("%lf ", y[i]);}
    printf("\nret= ");
    for (i=0; i< size_x; i++){ printf("%lf ", ret[i]); }
    printf("\n");
  }
}


/*
Computes the multivariates cort between x and y. X and Y should have the same shape
(ie share the same timelen and number of var.
 computes cort(A, B) = \\frac{\\sum_i^{T-2} \sum_v^{#var}(A_{iv} - A_{i+1v}) \\times (B_{iv} - B_{i+1v})}
    {\\sqrt{\\sum_i^{T-2} \sum_v^{#var}(A_{iv} - A_{i+1v})^2} \\times
    \\sqrt{\\sum_i^{T-2} \sum_v^{#var} (B_{iv} - B_{i+1v})^2}}
*/


// Computes the dissimilarity of a cell
// numerator : the numerator of the cort distance
// denom1 : the first part of the denominator of the cort distance
// denom1 : the second part of the denominator of the cort distance
// dtw :  the dtw part
 double diss_cort(const double numerator,  double denom1, double denom2, double value, int k){
	if (denom1 < 0.000001) denom1=1.0;
	if (denom2 < 0.000001) denom2=1.0;
	int sign = 1;
	if (numerator < 0) sign = -1;
	//const double tmp = numerator / sqrt(denom1 * denom2);
	//return 2 / ( 1 + exp(k*tmp) ) * value;
	const double tmp = (numerator * numerator * sign) / (denom1 * denom2);
	return tmp;
	//return (1 - tmp) * value;
}


// Computes the dissimilarity of a cell
// numerator : the numerator of the cort distance
// denom1 : the first part of the denominator of the cort distance
// denom1 : the second part of the denominator of the cort distance
// dtw :  the dtw part
 double diss_cort_times_dtw(const double numerator,  double denom1, double denom2, double value, int k){
	if (denom1 < 0.000001) denom1=1.0;
	if (denom2 < 0.000001) denom2=1.0;
	int sign = 1;
	if (numerator < 0) sign = -1;
	//const double tmp = numerator / sqrt(denom1 * denom2);
	//return 2 / ( 1 + exp(k*tmp) ) * value;
	const double tmp = (numerator * numerator * sign) / (denom1 * denom2);
	//return tmp;
	return (1 - tmp) * value;
}


/*
"""
Compute the slope values of the curve, namely:
res[t] = \sum_{var \in variable} (x[t+1,var] - x[t,var])

:param x: the input vector x. x is an array representing a matrix compatible with get_idx function.
:type x: floating point array.
:param nbvar: the number of variables (1 for univariate, 2 for bivariate, ...)
:param time_len: the time_len of x
:param res: the target array. Will be filled by the procedure. Should be of size end_x - startx_x. \
No allocation done, no bound check.
"""
 */
void compute_slope(double *x, int start_x, int end_x, int nb_var, double *res){
  int t, var;
  for (t = 0; t < (end_x-start_x -1); t++){
    res[t]=0.0;
    for (var=0; var<nb_var; var++){
      res[t] +=	x[get_idx(var, start_x+t+1, nb_var)] - x[get_idx(var, start_x+t, nb_var)];
    }
  }
}


void print_multivariate(double *x, int nbvar, int timelen){
  int var, t;
  printf("\n");
  for (var=0; var <nbvar; var++){
    for (t =0; t<timelen; t++){
      printf("%f ", x[get_idx(var, t, nbvar)]);
    }
    printf("\n");
  }
}

void twed(double *x, double *y, int size_x, int size_y, double *distances)
{

  double **D = (double **)calloc(size_x+1, sizeof(double*));
  double *Di1 = (double *)calloc(size_x+1, sizeof(double));
  double *Dj1 = (double *)calloc(size_y+1, sizeof(double));
  int i, j;
  for(i=0; i<size_x+1; i++)
  {
    D[i]=(double *)calloc(size_y+1, sizeof(double));
  }

  double lambda = 1;
  double nu = 1;
  double ta[size_x];
  double tb[size_y];
  double tsa[size_x];
  double tsb[size_y];

  for(i=0; i<size_x; i++)
  {
	  tsa[i] = i+1;
	  tsb[i] = i+1;
	  ta[i] = x[i];
	  tb[i] = y[i];
  }

  double dist = 0;
  double disti1 = 0;
  double distj1 = 0;
  double temp;
  for(j=1; j<size_y+1; j++)
  {
	  distj1 = 0;
	  if (j > 1)
	  {
		  temp = tb[j-2] - tb[j-1];
		  distj1 = distj1 + temp*temp;
	  }
	  else
	  {
		  distj1 = distj1 + tb[j-1]*tb[j-1];
	  }
	  Dj1[j] = distj1;
  }
  for(i=1; i<size_x+1; i++)
  {
	  disti1 = 0;
	  if (i > 1)
	  {
		  temp = ta[i-2] - ta[i-1];
		  disti1 = disti1 + temp*temp;
	  }
	  else
	  {
		  disti1 = disti1 + ta[i-1]*ta[i-1];
	  }
	  Di1[i] = disti1;

	  for(j=1; j<size_y+1; j++)
	  {
		  dist = 0;
		  temp = ta[i-1] - tb[j-1];
		  temp = temp * temp;
		  dist = dist + temp;
		  if (i>1 && j>1)
		  {
			  temp = ta[i-2] - tb[j-2];
			  temp = temp * temp;
			  dist = dist + temp;
		  }
		  D[i][j] = dist;
	  }
  }

  D[0][0] = 0;
  for(i=1; i<size_x+1; i++)
	  D[i][0] = D[i-1][0] + Di1[i];
  for(j=1; j<size_y+1; j++)
	  D[0][j] = D[0][j-1] + Dj1[j];

  double htrans = 0;
  double dmin = 0;
  double dist0 = 0;
  for(i=1; i<size_x+1; i++)
  {
	  for(j=1; j<size_y+1; j++)
	  {
		  htrans = fabs(tsa[i-1] - tsb[j-1]);
		  if (i>1 && j>1)
		  {
			  htrans = htrans + fabs(tsa[i-2] - tsb[j-2]);
		  }
		  dist0 = D[i-1][j-1] + nu*htrans + D[i][j];
		  dmin = dist0;

		  if(i>1)
		  {
			  htrans = tsa[i-1] - tsa[i-2];
		  }
		  else
		  {
			  htrans = tsa[i-1];
		  }
		  dist = Di1[i] + D[i-1][j] + lambda + nu*htrans;
		  if (dmin > dist)
		  {
			  dmin = dist;
		  }

		  if(j>1)
		  {
			  htrans = tsb[j-1] - tsb[j-2];
		  }
		  else
		  {
			  htrans = tsb[j-1];
		  }
		  dist = Dj1[j] + D[i][j-1] + lambda + nu*htrans;
		  if (dmin > dist)
		  {
			  dmin = dist;
		  }
		  D[i][j] = dmin;
	  }
  }
  distances[0] = D[size_x][size_y];
  if(D)
  {
    for(i=0; i<size_x+1; i++)
    {
       free(D[i]);
    }
    free(D);
  }
  if (Di1) free(Di1);
  if (Dj1) free(Dj1);
}

void dtw(double *x, double *y, int size_x, int size_y, int nbvar, double *path, double* distances, int window){
  double *cost_mem = malloc(sizeof(double) * size_x * size_y);
  char *path_mem = calloc(sizeof(char), (size_x * size_y));

  double **cost = malloc(sizeof(double*) * size_x);
  char **path_mat = malloc(sizeof(char*) * (size_x));

  // not enough memory : clean and return -1
  if (! (cost && path_mat && cost_mem && path_mem))
  {
    goto clean;
  }

  int i, j;
  // linking 2D matrices to the 1D ones
  cost[0] = cost_mem;
  path_mat[0] = path_mem;
  for (i=1; i < size_x; i++)
  {
    cost[i] = cost[i-1] + size_y;
    path_mat[i] = path_mat[i-1] + size_y;
  }

  for (i=0; i<size_x; i++)
	  for (j=0; j<size_y; j++)
		  cost[i][j] = INFINITY;
  int k=0;
  int m=0;

  cost[0][0] = compute_diff(x, y, 0, 0, nbvar);
  double temp_val = cost[0][0];
  for(i=1; i<window+1; i++)
  {
	 cost[i][0] = compute_diff(x, y, i, 0, nbvar) + temp_val;
	 temp_val = cost[i][0];
	 path_mat[i][0] = 'b';
  }
  temp_val = cost[0][0];

  for(j=1; j<window+1; j++)
  {
	 cost[0][j] = compute_diff(x, y, 0, j, nbvar) + temp_val;
	 temp_val = cost[0][j];
	 path_mat[0][j] = 'l';
  }
  path_mat[0][0] = 's';

  // doing the dynamic programming
  for (j=1; j<size_y; j++)
  {
	k = max2(1, j-window);
	m = min2(size_x, j+window+1);
    for (i=k; i<m; i++)
    {
      cost[i][j] = min3_direction(cost[i-1][j], cost[i][j-1],  cost[i-1][j-1], &(path_mat[i][j]))
      + compute_diff(x, y, i, j, nbvar);
    }
  }

  // backtracking the path
  i = size_x - 1;
  j = size_y - 1;
  distances[0] = cost[i][j];
  double distance_1 = 0.0;
  double distance_2 = 0.0;
  double count_1 = 0.0;
	double count_2 = 0.0;
  // if (debug)  show_mat(path_mat, size_x, size_y);

  int pos_path = 0;
  path[0] = 0;
  pos_path++;
  char temp_char;
  while (path_mat[i][j] != 's')
  {
    path[pos_path] = i;
    pos_path++;
    path[pos_path] = j;
    pos_path++;
    if (path_mat[i][j] == 'b')
    {
	  temp_char = 'b';
      path[pos_path] = cost[i][j] - cost[i-1][j];
      i--;
    }
    else
    {
      if (path_mat[i][j] == 'l')
      {
		temp_char = 'l';
        path[pos_path] = cost[i][j] - cost[i][j-1];
	    j--;
      }
      else
      {
        path[pos_path] = cost[i][j] - cost[i-1][j-1];
		temp_char = 'd';
	    i--;
	    j--;
      }
    }
	distance_1 += path[pos_path];
	distance_2 += path[pos_path];
	count_1 += 1;
	count_2 += 1;
	if (temp_char == 'b')
    {
      distances[1] += (distance_1 / count_1);
	  distance_1 = 0.0;
	  count_1 = 0.0;
    }
    else
    {
      if (temp_char == 'l')
      {
        distances[2] += (distance_2 / count_2);
	    distance_2 = 0.0;
	    count_2 = 0.0;
      }
      else
      {
        distances[1] += (distance_1 / count_1);
	    distance_1 = 0.0;
	    count_1 = 0.0;
		distances[2] += (distance_2 / count_2);
	    distance_2 = 0.0;
	    count_2 = 0.0;
      }
    }
    pos_path++;
  }

  // adding the first points
  path[pos_path] = 0;
  pos_path++;
  path[pos_path] = 0;
  pos_path++;
  path[pos_path] = cost[0][0];
  pos_path++;
  path[0] = pos_path;
  distances[1] += ((distance_1 + cost[0][0]) / (count_1 + 1.0));
  distances[2] += ((distance_2 + cost[0][0]) / (count_2 + 1.0));
clean:
  free(cost);
  free(cost_mem);
  free(path_mat);
  free(path_mem);
}

void dtw_given_path(double* temporal_distances, int* path, double* weight, int path_length, double* distances){

  int i;
  int curr_i = path[0];
  double curr_distance = temporal_distances[0];
  distances[0] = temporal_distances[0] * weight[curr_i];
  double curr_count = 1.0;
  for (i=1; i < path_length; i++)
  {
    distances[0] += temporal_distances[i] * weight[path[i]];
    if (path[i] != curr_i)
    {
        distances[1] += ((curr_distance / curr_count) * weight[path[curr_i]]);
        curr_distance = temporal_distances[i];
        curr_count = 1.0;
        curr_i = path[i];
    }
    else
    {
        curr_distance += temporal_distances[i];
        curr_count += 1;
    }
  }
  distances[1] += ((curr_distance / curr_count) * weight[path[curr_i]]);

}

void dtw_weighted(double *x, double *y, int size_x, int size_y, int nbvar, double* distances, double* weight, int window){
  double *cost_mem = malloc(sizeof(double) * size_x * size_y);
  char *path_mem = calloc(sizeof(char), (size_x * size_y));

  double **cost = malloc(sizeof(double*) * size_x);
  char **path_mat = malloc(sizeof(char*) * (size_x));

  // not enough memory : clean and return -1
  if (! (cost && path_mat && cost_mem && path_mem))
  {
    goto clean;
  }

  int i, j;
  // linking 2D matrices to the 1D ones
  cost[0] = cost_mem;
  path_mat[0] = path_mem;
  for (i=1; i < size_x; i++)
  {
    cost[i] = cost[i-1] + size_y;
    path_mat[i] = path_mat[i-1] + size_y;
  }

  for (i=0; i<size_x; i++)
	  for (j=0; j<size_y; j++)
		  cost[i][j] = INFINITY;
  int k=0;
  int m=0;

  cost[0][0] = compute_diff(x, y, 0, 0, nbvar) * weight[0];
  path_mat[0][0] = 's';
  double temp_val = cost[0][0];

  for(i=1; i<window+1; i++)
  {
	 cost[i][0] = compute_diff(x, y, i, 0, nbvar) * weight[i] + temp_val;
	 temp_val = cost[i][0];
	 path_mat[i][0] = 'b';
  }
  temp_val = cost[0][0];

  for(j=1; j<window+1; j++)
  {
	 cost[0][j] = compute_diff(x, y, 0, j, nbvar) * weight[0] + temp_val;
	 temp_val = cost[0][j];
	 path_mat[0][j] = 'l';
  }
  path_mat[0][0] = 's';

  // doing the dynamic programming
  for (j=1; j<size_y; j++)
  {
	k = max2(1, j-window);
	m = min2(size_x, j+window+1);
    for (i=k; i<m; i++)
    {
      cost[i][j] = min3_direction(cost[i-1][j], cost[i][j-1],  cost[i-1][j-1], &(path_mat[i][j]))
      + compute_diff(x, y, i, j, nbvar) * weight[i];
    }
  }

  // backtracking the path
  i = size_x - 1;
  j = size_y - 1;
  distances[0] = cost[i][j];
  double distance_1 = 0.0;
  double count_1 = 0.0;
  char temp_char;
  double path;
  while (path_mat[i][j] != 's')
  {
    if (path_mat[i][j] == 'b')
    {
	  temp_char = 'b';
      path = cost[i][j] - cost[i-1][j];
      i--;
    }
    else
    {
      if (path_mat[i][j] == 'l')
      {
		temp_char = 'l';
        path = cost[i][j] - cost[i][j-1];
	    j--;
      }
      else
      {
        path = cost[i][j] - cost[i-1][j-1];
		temp_char = 'd';
	    i--;
	    j--;
      }
    }
	distance_1 += path;
	count_1 += 1;
	if ((temp_char == 'b') || (temp_char == 'd'))
    {
      distances[1] += (distance_1 / count_1);
	  distance_1 = 0.0;
	  count_1 = 0.0;
    }
  }
  distances[1] += ((distance_1 + cost[0][0]) / (count_1 + 1.0));
clean:
  free(cost);
  free(cost_mem);
  free(path_mat);
  free(path_mem);
}

int get_idX(int var_idx, int time_idx, int nbvar){
  return  (nbvar*time_idx)+var_idx;
}

double cort_simple(double *x, double *y, const unsigned time_start, const unsigned time_end,  const unsigned nb_var){
  unsigned int t, var;
  double slopex, slopey;
  double num;
  double sumSquareXslope, sumSquareYslope;
  const char debug = 0;
  num = sumSquareXslope = sumSquareYslope = 0.0;
  if (debug) printf("time_start=%d, time_end=%d, nb_var=%d\n", time_start, time_end, nb_var);
  for (t = time_start; t < time_end-1; t++){
    for (var=0; var<nb_var; var++){
      slopex = x[get_idX(var, t+1, nb_var)] -  x[get_idX(var, t, nb_var)];
      slopey = y[get_idX(var, t+1, nb_var)] -  y[get_idX(var, t, nb_var)];
      if (debug) printf("\tvar %d: slopex = %f, slopey = %f\n", var, slopex, slopey);
      num += slopex * slopey;
      sumSquareXslope += slopex * slopex;
      sumSquareYslope += slopey * slopey;
    }
  }
  if (debug) printf("num=%f, %f, %f\n", num, sumSquareXslope, sumSquareYslope );
  return num/sqrt(sumSquareXslope*sumSquareYslope);
}


double cort(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k){
  const char debug = 0;

  /* cort is computed incrementally by computing incrementally each of its parts */

  int i, j;
  double ret = -1.0;
  double cort_num_temp,  cort_denomX_temp, cort_denomY_temp, value_temp, dissimilarity_min, dissimilarity_temp;
  double *valueP, *valueC; // the "value" part in the behavior/value balance
  double *diffx, *diffy;
  double *cortNumP, *cortNumC; // for computing the numerator part of the cort
  double *cortDenomXC, *cortDenomXP, *cortDenomYC, *cortDenomYP; // for computing the denominator part of the cort
  const int timelen_x = end_x - start_x;
  const int timelen_y = end_y - start_y;
  double local_diff = 0.0;
  valueP = valueC = diffx = diffy = cortNumP = cortNumC = cortDenomXC = cortDenomXP = cortDenomYC = cortDenomYP = NULL;
  valueP = (double*) malloc(timelen_x * sizeof(double));
  valueC = (double*) malloc(timelen_x * sizeof(double));
  diffx = (double*) malloc( sizeof(double)* (timelen_x-1)); /* -1 since they are less diff than values */
  diffy = (double*) malloc(sizeof(double) * (timelen_y-1));
  cortNumP = (double*) malloc(timelen_x * sizeof(double));
  cortNumC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomXC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomXP = (double*) malloc(timelen_x * sizeof(double));
  cortDenomYC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomYP = (double*) malloc(timelen_x * sizeof(double));

  if (debug){printf("K=%f\n", k); }

  if ( ! (valueC && valueP && diffx && diffy  && cortNumP && cortNumC &&
	  cortDenomXC && cortDenomXP && cortDenomYC && cortDenomYP)){
    goto free_dtw_no_path_multivariate;
  }

  compute_slope(x, start_x, end_x, nbvar, diffx);
  compute_slope(y, start_y, end_y, nbvar, diffy);
  valueP[0] = compute_diff(x, y, start_x, start_y, nbvar);
  cortNumP[0] = diffx[0] * diffy[0];
  cortDenomXP[0] = (diffx[0]*diffx[0]);
  cortDenomYP[0] = (diffy[0]*diffy[0]);
  dissimilarity_min = diss_cort(cortNumP[0],   cortDenomXP[0],  cortDenomYP[0], valueP[0], k);
  for (i=1; i < timelen_x-1; i++){
    local_diff = compute_diff(x, y, i+start_x, start_y, nbvar);
    valueP[i] = valueP[i-1] + local_diff;
    cortNumP[i] = cortNumP[i-1] + (diffx[i]*diffy[0]);
    cortDenomXP[i] = cortDenomXP[i-1] + (diffx[i]*diffx[i]);
    cortDenomYP[i] = cortDenomYP[i-1] + (diffy[0]*diffy[0]);
  }
  for (j=1; j< timelen_y-1; j++)
  {
    if (debug){
      printf("j=%d, valueP: ", j) ;
      for (i=0; i< 20; i++){printf(" %f", valueP[i]);}
      printf("...");
      for (i=timelen_x-20; i< timelen_x; i++){printf(" %f", valueP[i]);}
      printf("\n");
    }
    valueC[0] = valueP[0] + compute_diff(x, y, start_x, start_y+j, nbvar);
    cortNumC[0] = cortNumP[0] + (diffx[0]*diffy[j]);
    cortDenomXC[0] = cortDenomXP[0] + (diffx[0]*diffx[0]);
    cortDenomYC[0] = cortDenomYP[0] + (diffy[j]*diffy[j]);
    for (i=1; i < timelen_x-1; i++){
      local_diff = compute_diff(x, y, i+start_x, j+start_y, nbvar);
      if (debug) {
	printf("i=%d, j=%d, local_diff =%lf\n", i, j, local_diff);
	/*if ((i==1) && (j==1107)){

	  printf("printing a: ");   print_multivariate(x, nbvar, timelen_x);
	  printf("\nprinting b:");  print_multivariate(y, nbvar, timelen_y);

	  printf("computediff (%d,%d)= %f\n", i,j, local_diff);
	  compute_diff_d(x, y, i+start_x, j+start_y, nbvar);
	  exit(1);
	}
	*/
      }
      /* i-1, j-1 */
      value_temp = valueP[i-1] + local_diff;

      cort_num_temp = cortNumP[i-1] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXP[i-1] + (diffx[i]*diffx[i]);
      cort_denomY_temp = cortDenomYP[i-1] + (diffy[j]*diffy[j]);
      dissimilarity_min = diss_cort(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      valueC[i] = value_temp;
      cortNumC[i] = cort_num_temp;
      cortDenomXC[i] = cort_denomX_temp;
      cortDenomYC[i] = cort_denomY_temp;
      if (debug) {printf("\ti=%d, j=%d, (i-1, j-1): valueC[%d]=%lf, \n", i, j, i, value_temp); }

      /* i-1, j */
      value_temp = valueC[i-1] + local_diff;
      cort_num_temp = cortNumC[i-1] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXC[i-1] + (diffx[i] * diffx[i]);
      cort_denomY_temp = cortDenomYC[i-1] + (diffy[j]*diffy[j]);
      dissimilarity_temp = diss_cort(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      if (debug) {printf("\ti=%d, j=%d, (i-1, j):  valueC[%d]=%lf\n", i, j, i, value_temp); }

      if (dissimilarity_temp > dissimilarity_min){
	dissimilarity_min = dissimilarity_temp;
	valueC[i] = value_temp;
	cortNumC[i] = cort_num_temp;
	cortDenomXC[i] = cort_denomX_temp;
	cortDenomYC[i] = cort_denomY_temp;
      }

      /* i, j-1 */
      value_temp = valueP[i] + local_diff;
      cort_num_temp = cortNumP[i] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXP[i] + (diffx[i] * diffx[i]);
      cort_denomY_temp = cortDenomYP[i] + (diffy[j] * diffy[j]);
      dissimilarity_temp = diss_cort(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      if (debug) {printf("\ti=%d, j=%d, (i, j-1):   valueC[%d]=%lf\n", i, j, i, value_temp); }
      if (dissimilarity_temp > dissimilarity_min){
	dissimilarity_min = dissimilarity_temp;
	valueC[i] = value_temp;
	cortNumC[i] = cort_num_temp;
	cortDenomXC[i] = cort_denomX_temp;
	cortDenomYC[i] = cort_denomY_temp;
      }
      if (debug) {printf("FINAL i=%d, j=%d, (i-1, j):  valueC[%d]=%lf\n\n", i, j, i, valueC[i]); }
    }
    /* switching current and predecessor */
    swap_pointers(&valueP, &valueC);
    swap_pointers(&cortNumP, &cortNumC);
    swap_pointers(&cortDenomXC, &cortDenomXP);
    swap_pointers(&cortDenomYC, &cortDenomYP);
  }
  /* result is read in valueP because switching between valueP and valueC is done */
  //ret = diss_cort(cortNumP[timelen_x-2],   cortDenomXP[timelen_x-2],
		      //cortDenomYP[timelen_x-2], valueP[timelen_x-2], k);
  ret = 1 - (cortNumP[timelen_x-2]/sqrt(cortDenomXP[timelen_x-2] * cortDenomYP[timelen_x-2]));

  free_dtw_no_path_multivariate:
  if (valueP) free(valueP);
  if (valueC) free(valueC);
  if (cortNumC) free(cortNumC);
  if (cortNumP) free(cortNumP);
  if (cortDenomXC) free(cortDenomXC);
  if (cortDenomXP) free(cortDenomXP);
  if (cortDenomYC) free(cortDenomYC);
  if (cortDenomYP) free(cortDenomYP);
  if (diffx) free(diffx);
  if (diffy) free(diffy);
  return ret;

}

void averaged_cort_distances(double * diffy, double* diffx, double* path, double* distances){
    int i, j;
    int curr_i, curr_j;
    int count_i, count_j;
    curr_i = path[1];
    curr_j = path[2];
    int path_i, path_j;

    double nomX, curr_val_X, denomXI, denomXJ;
    double nomY, curr_val_Y, denomYI, denomYJ;
    nomX=0, nomY=0;
    count_i=0;
    count_j=0;
    curr_val_X=0, curr_val_Y=0;
    denomXI=0, denomXJ=0;
    denomYI=0, denomYJ=0;

    for (i=1, j=2; i<path[0]; i=i+3, j=j+3)
    {
        path_i = path[i];
        path_j = path[j];
        if (curr_i != path_i)
        {
            curr_val_Y = curr_val_Y / count_j;
            nomX += diffx[curr_i] * curr_val_Y;
            denomXI += diffx[curr_i] * diffx[curr_i];
            denomXJ += curr_val_Y * curr_val_Y;
            curr_i = path_i;
            curr_val_Y = diffy[path_j];
            count_j = 1;
        }
        else
        {
            curr_val_Y += diffy[path_j];
            count_j += 1;
        }
        if (curr_j != path_j)
        {
            curr_val_X = curr_val_X / count_i;
            nomY += diffy[curr_j] * curr_val_X;
            denomYI += diffy[curr_j] * diffy[curr_j];
            denomYJ += curr_val_X * curr_val_X;
            curr_j = path_j;
            curr_val_X = diffx[path_i];
            count_i = 1;
        }
        else
        {
            curr_val_X += diffx[path_i];
            count_i += 1;
        }
    }

	curr_val_Y = curr_val_Y / count_j;
    nomX += diffx[curr_i] * curr_val_Y;
    denomXI += diffx[curr_i] * diffx[curr_i];
    denomXJ += curr_val_Y * curr_val_Y;

	curr_val_X = curr_val_X / count_i;
    nomY += diffy[curr_j] * curr_val_X;
    denomYI += diffy[curr_j] * diffy[curr_j];
    denomYJ += curr_val_X * curr_val_X;
    distances[1] = 1 - (nomX / sqrt(denomXI * denomXJ));
    distances[2] = 1 - (nomY / sqrt(denomYI * denomYJ));
}

void cort_window_path(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k, double* path, double* distances, int window){
  // const char debug = 0;
  int i, j;
  /* cort is computed incrementally by computing incrementally each of its parts */

  const int timelen_x = end_x - start_x;
  const int timelen_y = end_y - start_y;
  double *cost_mem = malloc(sizeof(double) * timelen_x * timelen_y);
  char *path_mem = calloc(sizeof(char), (timelen_x * timelen_y));
  double **cost = malloc(sizeof(double*) * timelen_x);
  char **path_mat = malloc(sizeof(char*) * (timelen_x));

  // linking 2D matrices to the 1D ones
  cost[0] = cost_mem;
  path_mat[0] = path_mem;
  for (i=1; i < timelen_x; i++)
  {
    cost[i] = cost[i-1] + timelen_y;
    path_mat[i] = path_mat[i-1] + timelen_y;
  }
  int kk=0;
  int m=0;


  double cort_num_temp,  cort_denomX_temp, cort_denomY_temp, value_temp, dissimilarity_min, dissimilarity_temp;
  double *diffx, *diffy;
  double *cortNumP, *cortNumC; // for computing the numerator part of the cort
  double *cortDenomXC, *cortDenomXP, *cortDenomYC, *cortDenomYP; // for computing the denominator part of the cort
  double local_diff = 0.0;
  diffx = diffy = cortNumP = cortNumC = cortDenomXC = cortDenomXP = cortDenomYC = cortDenomYP = NULL;
  diffx = (double*) malloc( sizeof(double)* (timelen_x-1)); /* -1 since they are less diff than values */
  diffy = (double*) malloc(sizeof(double) * (timelen_y-1));
  cortNumP = (double*) malloc(timelen_x * sizeof(double));
  cortNumC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomXC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomXP = (double*) malloc(timelen_x * sizeof(double));
  cortDenomYC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomYP = (double*) malloc(timelen_x * sizeof(double));

  if ( ! (diffx && diffy  && cortNumP && cortNumC &&
	  cortDenomXC && cortDenomXP && cortDenomYC && cortDenomYP)){
    goto free_dtw_no_path_multivariate;
  }

  compute_slope(x, start_x, end_x, nbvar, diffx);
  compute_slope(y, start_y, end_y, nbvar, diffy);
  cost[0][0] = compute_diff(x, y, start_x, start_y, nbvar);
  for (i=0; i<timelen_x; i++)
  {
	  cortNumP[i] = -INFINITY;
	  cortNumC[i] = -INFINITY;
  }
  path_mat[0][0] = 's';
  cortNumP[0] = diffx[0] * diffy[0];
  cortDenomXP[0] = (diffx[0]*diffx[0]);
  cortDenomYP[0] = (diffy[0]*diffy[0]);

  if (window != 0)
  {
	  cortNumC[0] = cortNumP[0] + (diffx[0]*diffy[1]);
	  cortDenomXC[0] = cortDenomXP[0] + (diffx[0]*diffx[0]);
	  cortDenomYC[0] = cortDenomYP[0] + (diffy[1]*diffy[1]);
  }
  else
	  cortNumC[0] = -INFINITY;
  m = min2(timelen_x-1, window+1);
  for (i=1; i<m; i++)
  {
    local_diff = compute_diff(x, y, i+start_x, start_y, nbvar);
    cost[0][i] = cost[0][i-1] + local_diff;
	path_mat[0][i] = 'l';
    cortNumP[i] = cortNumP[i-1] + (diffx[i]*diffy[0]);
    cortDenomXP[i] = cortDenomXP[i-1] + (diffx[i]*diffx[i]);
    cortDenomYP[i] = cortDenomYP[i-1] + (diffy[0]*diffy[0]);
  }

  for (j=1; j< timelen_y-1; j++)
  {
	kk = max2(0, j-window);
	m = min2(timelen_x-1, j+window+1);
	if (window != 0)
	{
		cost[j][kk] = cost[j-1][kk] + compute_diff(x, y, start_x, start_y+j, nbvar);
		path_mat[j][kk] = 'b';
		cortNumC[kk] = cortNumP[kk] + (diffx[kk]*diffy[j]);
		cortDenomXC[kk] = cortDenomXP[kk] + (diffx[kk]*diffx[kk]);
		cortDenomYC[kk] = cortDenomYP[kk] + (diffy[j]*diffy[j]);
		kk++;
	}
    for (i=kk; i<m; i++)
	{
      local_diff = compute_diff(x, y, i+start_x, j+start_y, nbvar);
      /* i-1, j-1 */
      value_temp = cost[j-1][i-1] + local_diff;
	  path_mat[j][i] = 'd';
      cort_num_temp = cortNumP[i-1] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXP[i-1] + (diffx[i]*diffx[i]);
      cort_denomY_temp = cortDenomYP[i-1] + (diffy[j]*diffy[j]);
      dissimilarity_min = diss_cort(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      cost[j][i] = value_temp;
      cortNumC[i] = cort_num_temp;
      cortDenomXC[i] = cort_denomX_temp;
      cortDenomYC[i] = cort_denomY_temp;

      /* i-1, j */
      value_temp = cost[j][i-1] + local_diff;
      cort_num_temp = cortNumC[i-1] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXC[i-1] + (diffx[i] * diffx[i]);
      cort_denomY_temp = cortDenomYC[i-1] + (diffy[j]*diffy[j]);
      dissimilarity_temp = diss_cort(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);

      if (dissimilarity_temp > dissimilarity_min)
	  {
		dissimilarity_min = dissimilarity_temp;
		cost[j][i] = value_temp;
		path_mat[j][i] = 'l';
		cortNumC[i] = cort_num_temp;
		cortDenomXC[i] = cort_denomX_temp;
		cortDenomYC[i] = cort_denomY_temp;
      }

      /* i, j-1 */
      value_temp = cost[j-1][i] + local_diff;
      cort_num_temp = cortNumP[i] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXP[i] + (diffx[i] * diffx[i]);
      cort_denomY_temp = cortDenomYP[i] + (diffy[j] * diffy[j]);
      dissimilarity_temp = diss_cort(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      if (dissimilarity_temp > dissimilarity_min)
	  {
		dissimilarity_min = dissimilarity_temp;
		cost[j][i] = value_temp;
		path_mat[j][i] = 'b';
		cortNumC[i] = cort_num_temp;
		cortDenomXC[i] = cort_denomX_temp;
		cortDenomYC[i] = cort_denomY_temp;
      }
    }
    /* switching current and predecessor */
    swap_pointers(&cortNumP, &cortNumC);
    swap_pointers(&cortDenomXC, &cortDenomXP);
    swap_pointers(&cortDenomYC, &cortDenomYP);
  }
  distances[0] = 1 - (cortNumP[timelen_x-2]/sqrt(cortDenomXP[timelen_x-2] * cortDenomYP[timelen_x-2]));
  i = timelen_x - 2;
  j = timelen_y - 2;

  // if (debug)  show_mat(path_mat, size_x, size_y);

  int pos_path = 0;
  path[0] = 0;
  pos_path++;
  while (path_mat[j][i] != 's')
  {
    path[pos_path] = j;
    pos_path++;
    path[pos_path] = i;
    pos_path++;
    if (path_mat[j][i] == 'l')
    {
      path[pos_path] = cost[j][i] - cost[j][i-1];
      i--;
    }
    else
    {
      if (path_mat[j][i] == 'b')
      {
        path[pos_path] = cost[j][i] - cost[j-1][i];
	    j--;
      }
      else
      {
        path[pos_path] = cost[j][i] - cost[j-1][i-1];
	    i--;
	    j--;
      }
    }
    pos_path++;
  }

  // adding the first points
  path[pos_path] = 0;
  pos_path++;
  path[pos_path] = 0;
  pos_path++;
  path[pos_path] = cost[0][0];
  pos_path++;
  path[0] = pos_path;
  averaged_cort_distances(diffx, diffy, path, distances);
	/*
  for (i=0; i<end_x; i++)
  {
	  for(j=0; j<end_y; j++)
		  printf("%f ", cost[i][j]);
  	  printf("\n");
  }
  for (i=0; i<end_x-1; i++)
  {
	  for(j=0; j<end_y-1; j++)
		  printf("[%d%d]:%c ", i, j, path_mat[i][j]);
  	  printf("\n");
  }
  */
  free_dtw_no_path_multivariate:
  if (path_mem) free(path_mem);
  if (path_mat) free(path_mat);
  if (cost) free(cost);
  if (cost_mem) free(cost_mem);
  if (cortNumC) free(cortNumC);
  if (cortNumP) free(cortNumP);
  if (cortDenomXC) free(cortDenomXC);
  if (cortDenomXP) free(cortDenomXP);
  if (cortDenomYC) free(cortDenomYC);
  if (cortDenomYP) free(cortDenomYP);
  if (diffx) free(diffx);
  if (diffy) free(diffy);

}


double cort_dtw(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k){
  const char debug = 0;

  /* cort is computed incrementally by computing incrementally each of its parts */

  int i, j;
  double ret = -1.0;
  double cort_num_temp,  cort_denomX_temp, cort_denomY_temp, value_temp, dissimilarity_min, dissimilarity_temp;
  double *valueP, *valueC; // the "value" part in the behavior/value balance
  double *diffx, *diffy;
  double *cortNumP, *cortNumC; // for computing the numerator part of the cort
  double *cortDenomXC, *cortDenomXP, *cortDenomYC, *cortDenomYP; // for computing the denominator part of the cort
  const int timelen_x = end_x - start_x;
  const int timelen_y = end_y - start_y;
  double local_diff = 0.0;
  valueP = valueC = diffx = diffy = cortNumP = cortNumC = cortDenomXC = cortDenomXP = cortDenomYC = cortDenomYP = NULL;
  valueP = (double*) malloc(timelen_x * sizeof(double));
  valueC = (double*) malloc(timelen_x * sizeof(double));
  diffx = (double*) malloc( sizeof(double)* (timelen_x-1)); /* -1 since they are less diff than values */
  diffy = (double*) malloc(sizeof(double) * (timelen_y-1));
  cortNumP = (double*) malloc(timelen_x * sizeof(double));
  cortNumC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomXC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomXP = (double*) malloc(timelen_x * sizeof(double));
  cortDenomYC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomYP = (double*) malloc(timelen_x * sizeof(double));

  if (debug){printf("K=%f\n", k); }

  if ( ! (valueC && valueP && diffx && diffy  && cortNumP && cortNumC &&
	  cortDenomXC && cortDenomXP && cortDenomYC && cortDenomYP)){
    goto free_dtw_no_path_multivariate;
  }

  compute_slope(x, start_x, end_x, nbvar, diffx);
  compute_slope(y, start_y, end_y, nbvar, diffy);
  valueP[0] = compute_diff(x, y, start_x, start_y, nbvar);
  cortNumP[0] = diffx[0] * diffy[0];
  cortDenomXP[0] = (diffx[0]*diffx[0]);
  cortDenomYP[0] = (diffy[0]*diffy[0]);
  dissimilarity_min = diss_cort_times_dtw(cortNumP[0],   cortDenomXP[0],  cortDenomYP[0], valueP[0], k);

  for (i=1; i < timelen_x-1; i++){
    local_diff = compute_diff(x, y, i+start_x, start_y, nbvar);
    valueP[i] = valueP[i-1] + local_diff;
    cortNumP[i] = cortNumP[i-1] + (diffx[i]*diffy[0]);
    cortDenomXP[i] = cortDenomXP[i-1] + (diffx[i]*diffx[i]);
    cortDenomYP[i] = cortDenomYP[i-1] + (diffy[0]*diffy[0]);
  }
  for (j=1; j< timelen_y-1; j++){
    if (debug){
      printf("j=%d, valueP: ", j) ;
      for (i=0; i< 20; i++){printf(" %f", valueP[i]);}
      printf("...");
      for (i=timelen_x-20; i< timelen_x; i++){printf(" %f", valueP[i]);}
      printf("\n");
    }
    valueC[0] = valueP[0] + compute_diff(x, y, start_x, start_y+j, nbvar);
    cortNumC[0] = cortNumP[0] + (diffx[0]*diffy[j]);
    cortDenomXC[0] = cortDenomXP[0] + (diffx[0]*diffx[0]);
    cortDenomYC[0] = cortDenomYP[0] + (diffy[j]*diffy[j]);
    for (i=1; i < timelen_x-1; i++){
      local_diff = compute_diff(x, y, i+start_x, j+start_y, nbvar);
      if (debug) {
	printf("i=%d, j=%d, local_diff =%lf\n", i, j, local_diff);
	/*if ((i==1) && (j==1107)){

	  printf("printing a: ");   print_multivariate(x, nbvar, timelen_x);
	  printf("\nprinting b:");  print_multivariate(y, nbvar, timelen_y);

	  printf("computediff (%d,%d)= %f\n", i,j, local_diff);
	  compute_diff_d(x, y, i+start_x, j+start_y, nbvar);
	  exit(1);
	}
	*/
      }
      /* i-1, j-1 */
      value_temp = valueP[i-1] + local_diff;

      cort_num_temp = cortNumP[i-1] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXP[i-1] + (diffx[i]*diffx[i]);
      cort_denomY_temp = cortDenomYP[i-1] + (diffy[j]*diffy[j]);
      dissimilarity_min = diss_cort_times_dtw(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      valueC[i] = value_temp;
      cortNumC[i] = cort_num_temp;
      cortDenomXC[i] = cort_denomX_temp;
      cortDenomYC[i] = cort_denomY_temp;
      if (debug) {printf("\ti=%d, j=%d, (i-1, j-1): valueC[%d]=%lf, \n", i, j, i, value_temp); }

      /* i-1, j */
      value_temp = valueC[i-1] + local_diff;
      cort_num_temp = cortNumC[i-1] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXC[i-1] + (diffx[i] * diffx[i]);
      cort_denomY_temp = cortDenomYC[i-1] + (diffy[j]*diffy[j]);
      dissimilarity_temp = diss_cort_times_dtw(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      if (debug) {printf("\ti=%d, j=%d, (i-1, j):  valueC[%d]=%lf\n", i, j, i, value_temp); }

      if (dissimilarity_temp < dissimilarity_min){
	dissimilarity_min = dissimilarity_temp;
	valueC[i] = value_temp;
	cortNumC[i] = cort_num_temp;
	cortDenomXC[i] = cort_denomX_temp;
	cortDenomYC[i] = cort_denomY_temp;
      }

      /* i, j-1 */
      value_temp = valueP[i] + local_diff;
      cort_num_temp = cortNumP[i] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXP[i] + (diffx[i] * diffx[i]);
      cort_denomY_temp = cortDenomYP[i] + (diffy[j] * diffy[j]);
      dissimilarity_temp = diss_cort_times_dtw(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      if (debug) {printf("\ti=%d, j=%d, (i, j-1):   valueC[%d]=%lf\n", i, j, i, value_temp); }
      if (dissimilarity_temp < dissimilarity_min){
	dissimilarity_min = dissimilarity_temp;
	valueC[i] = value_temp;
	cortNumC[i] = cort_num_temp;
	cortDenomXC[i] = cort_denomX_temp;
	cortDenomYC[i] = cort_denomY_temp;
      }
      if (debug) {printf("FINAL i=%d, j=%d, (i-1, j):  valueC[%d]=%lf\n\n", i, j, i, valueC[i]); }
    }
    /* switching current and predecessor */
    swap_pointers(&valueP, &valueC);
    swap_pointers(&cortNumP, &cortNumC);
    swap_pointers(&cortDenomXC, &cortDenomXP);
    swap_pointers(&cortDenomYC, &cortDenomYP);
  }
  /* result is read in valueP because switching between valueP and valueC is done */
  ret = (cortNumP[timelen_x-2]/sqrt(cortDenomXP[timelen_x-2] * cortDenomYP[timelen_x-2]));
  ret = (1 - ret) * valueP[timelen_x-2];

  free_dtw_no_path_multivariate:
  if (valueP) free(valueP);
  if (valueC) free(valueC);
  if (cortNumC) free(cortNumC);
  if (cortNumP) free(cortNumP);
  if (cortDenomXC) free(cortDenomXC);
  if (cortDenomXP) free(cortDenomXP);
  if (cortDenomYC) free(cortDenomYC);
  if (cortDenomYP) free(cortDenomYP);
  if (diffx) free(diffx);
  if (diffy) free(diffy);
  return ret;

}

void averaged_cort_dtw_distances(double * diffy, double* diffx, double* path, double* distances){
    int i, j, k;
    int curr_i, curr_j;
    int count_i, count_j;
    curr_i = path[1];
    curr_j = path[2];
    int path_i, path_j;
    double nomX, curr_val_X, denomXI, denomXJ, dtwX, curr_dtwX;
    double nomY, curr_val_Y, denomYI, denomYJ, dtwY, curr_dtwY;
    nomX=0, nomY=0;
    count_i=0;
    count_j=0;
    curr_val_X=0, curr_val_Y=0;
    denomXI=0, denomXJ=0;
    denomYI=0, denomYJ=0;
	dtwX=0, dtwY=0;
	curr_dtwX=0, curr_dtwY=0;

    for (i=1, j=2, k=3; i<path[0]; i=i+3, j=j+3, k=k+3)
    {
        path_i = path[i];
        path_j = path[j];
        if (curr_i != path_i)
        {
            curr_val_Y = curr_val_Y / count_j;
            nomX += diffx[curr_i] * curr_val_Y;
			dtwX += curr_dtwX / count_j;
            denomXI += diffx[curr_i] * diffx[curr_i];
            denomXJ += curr_val_Y * curr_val_Y;
            curr_i = path_i;
            curr_val_Y = diffy[path_j];
			curr_dtwX = path[k];
            count_j = 1;
        }
        else
        {
            curr_val_Y += diffy[path_j];
			curr_dtwX += path[k];
            count_j += 1;
        }
        if (curr_j != path_j)
        {
            curr_val_X = curr_val_X / count_i;
            nomY += diffy[curr_j] * curr_val_X;
			dtwY += curr_dtwY / count_i;
            denomYI += diffy[curr_j] * diffy[curr_j];
            denomYJ += curr_val_X * curr_val_X;
            curr_j = path_j;
            curr_val_X = diffx[path_i];
			curr_dtwY = path[k];
            count_i = 1;
        }
        else
        {
            curr_val_X += diffx[path_i];
			curr_dtwY += path[k];
            count_i += 1;
        }
    }

	curr_val_Y = curr_val_Y / count_j;
    nomX += diffx[curr_i] * curr_val_Y;
    denomXI += diffx[curr_i] * diffx[curr_i];
    denomXJ += curr_val_Y * curr_val_Y;

	curr_val_X = curr_val_X / count_i;
    nomY += diffy[curr_j] * curr_val_X;
    denomYI += diffy[curr_j] * diffy[curr_j];
    denomYJ += curr_val_X * curr_val_X;

	dtwX += curr_dtwX / count_j;
	dtwY += curr_dtwY / count_i;
    distances[1] = 1 - (nomX / sqrt(denomXI * denomXJ));
    distances[2] = 1 - (nomY / sqrt(denomYI * denomYJ));
	distances[1] *= dtwX;
	distances[2] *= dtwY;
}

void cort_dtw_window_path(double* x, double* y, int start_x, int start_y, int end_x, int end_y, int nbvar, double k, double* path, double* distances, int window){
  // const char debug = 0;
  int i, j;
  /* cort is computed incrementally by computing incrementally each of its parts */

  const int timelen_x = end_x - start_x;
  const int timelen_y = end_y - start_y;
  double *cost_mem = malloc(sizeof(double) * timelen_x * timelen_y);
  char *path_mem = calloc(sizeof(char), (timelen_x * timelen_y));
  double **cost = malloc(sizeof(double*) * timelen_x);
  char **path_mat = malloc(sizeof(char*) * (timelen_x));

  // linking 2D matrices to the 1D ones
  cost[0] = cost_mem;
  path_mat[0] = path_mem;
  for (i=1; i < timelen_x; i++)
  {
    cost[i] = cost[i-1] + timelen_y;
    path_mat[i] = path_mat[i-1] + timelen_y;
  }

  double cort_num_temp,  cort_denomX_temp, cort_denomY_temp, value_temp, dissimilarity_min, dissimilarity_temp;
  double *diffx, *diffy;
  double *cortNumP, *cortNumC; // for computing the numerator part of the cort
  double *cortDenomXC, *cortDenomXP, *cortDenomYC, *cortDenomYP; // for computing the denominator part of the cort
  double local_diff = 0.0;
  diffx = diffy = cortNumP = cortNumC = cortDenomXC = cortDenomXP = cortDenomYC = cortDenomYP = NULL;
  diffx = (double*) malloc( sizeof(double)* (timelen_x-1)); /* -1 since they are less diff than values */
  diffy = (double*) malloc(sizeof(double) * (timelen_y-1));
  cortNumP = (double*) malloc(timelen_x * sizeof(double));
  cortNumC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomXC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomXP = (double*) malloc(timelen_x * sizeof(double));
  cortDenomYC = (double*) malloc(timelen_x * sizeof(double));
  cortDenomYP = (double*) malloc(timelen_x * sizeof(double));
  int kk=0;
  int m=0;

  if ( ! (diffx && diffy  && cortNumP && cortNumC &&
	  cortDenomXC && cortDenomXP && cortDenomYC && cortDenomYP)){
    goto free_dtw_no_path_multivariate;
  }
  for (i=0; i<end_x; i++)
	  for (j=0; j<end_y; j++)
		  cost[i][j] = INFINITY;
  compute_slope(x, start_x, end_x, nbvar, diffx);
  compute_slope(y, start_y, end_y, nbvar, diffy);
  cost[0][0] = compute_diff(x, y, start_x, start_y, nbvar);
  for (i=0; i<timelen_x; i++)
  {
	  cortNumP[i] = -INFINITY;
	  cortNumC[i] = -INFINITY;
  }
  path_mat[0][0] = 's';
  cortNumP[0] = diffx[0] * diffy[0];
  cortDenomXP[0] = (diffx[0]*diffx[0]);
  cortDenomYP[0] = (diffy[0]*diffy[0]);

  if (window != 0)
  {
	  cortNumC[0] = cortNumP[0] + (diffx[0]*diffy[1]);
	  cortDenomXC[0] = cortDenomXP[0] + (diffx[0]*diffx[0]);
	  cortDenomYC[0] = cortDenomYP[0] + (diffy[1]*diffy[1]);
  }
  else
	  cortNumC[0] = -INFINITY;

  m = min2(timelen_x-1, window+1);
  for (i=1; i<m; i++)
  {
    local_diff = compute_diff(x, y, i+start_x, start_y, nbvar);
    cost[0][i] = cost[0][i-1] + local_diff;
	path_mat[0][i] = 'l';
    cortNumP[i] = cortNumP[i-1] + (diffx[i]*diffy[0]);
    cortDenomXP[i] = cortDenomXP[i-1] + (diffx[i]*diffx[i]);
    cortDenomYP[i] = cortDenomYP[i-1] + (diffy[0]*diffy[0]);
  }

  for (j=1; j< timelen_y-1; j++)
  {

	kk = max2(0, j-window);
	m = min2(timelen_x-1, j+window+1);
	if (window != 0)
	{
		cost[j][kk] = cost[j-1][kk] + compute_diff(x, y, start_x, start_y+j, nbvar);
		path_mat[j][kk] = 'b';
		cortNumC[kk] = cortNumP[kk] + (diffx[kk]*diffy[j]);
		cortDenomXC[kk] = cortDenomXP[kk] + (diffx[kk]*diffx[kk]);
		cortDenomYC[kk] = cortDenomYP[kk] + (diffy[j]*diffy[j]);
		kk++;
	}
    for (i=kk; i<m; i++)
	{
      local_diff = compute_diff(x, y, i+start_x, j+start_y, nbvar);
      /* i-1, j-1 */
      value_temp = cost[j-1][i-1] + local_diff;
	  path_mat[j][i] = 'd';
      cort_num_temp = cortNumP[i-1] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXP[i-1] + (diffx[i]*diffx[i]);
      cort_denomY_temp = cortDenomYP[i-1] + (diffy[j]*diffy[j]);
      dissimilarity_min = diss_cort_times_dtw(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      cost[j][i] = value_temp;
      cortNumC[i] = cort_num_temp;
      cortDenomXC[i] = cort_denomX_temp;
      cortDenomYC[i] = cort_denomY_temp;

      /* i-1, j */
      value_temp = cost[j][i-1] + local_diff;
      cort_num_temp = cortNumC[i-1] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXC[i-1] + (diffx[i] * diffx[i]);
      cort_denomY_temp = cortDenomYC[i-1] + (diffy[j]*diffy[j]);
      dissimilarity_temp = diss_cort_times_dtw(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      if (dissimilarity_temp < dissimilarity_min)
	  {
		dissimilarity_min = dissimilarity_temp;
		cost[j][i] = value_temp;
		path_mat[j][i] = 'l';
		cortNumC[i] = cort_num_temp;
		cortDenomXC[i] = cort_denomX_temp;
		cortDenomYC[i] = cort_denomY_temp;
      }

      /* i, j-1 */
      value_temp = cost[j-1][i] + local_diff;
      cort_num_temp = cortNumP[i] + (diffx[i]*diffy[j]);
      cort_denomX_temp = cortDenomXP[i] + (diffx[i] * diffx[i]);
      cort_denomY_temp = cortDenomYP[i] + (diffy[j] * diffy[j]);
      dissimilarity_temp = diss_cort_times_dtw(cort_num_temp,   cort_denomX_temp,  cort_denomY_temp, value_temp, k);
      if (dissimilarity_temp < dissimilarity_min)
	  {
		dissimilarity_min = dissimilarity_temp;
		cost[j][i] = value_temp;
		path_mat[j][i] = 'b';
		cortNumC[i] = cort_num_temp;
		cortDenomXC[i] = cort_denomX_temp;
		cortDenomYC[i] = cort_denomY_temp;
      }
    }
    /* switching current and predecessor */
    swap_pointers(&cortNumP, &cortNumC);
    swap_pointers(&cortDenomXC, &cortDenomXP);
    swap_pointers(&cortDenomYC, &cortDenomYP);
  }
  distances[0] = 1 - (cortNumP[timelen_x-2]/sqrt(cortDenomXP[timelen_x-2] * cortDenomYP[timelen_x-2]));
  distances[0] = distances[0] * cost[timelen_y-2][timelen_x-2];
  i = timelen_x - 2;
  j = timelen_y - 2;

  // if (debug)  show_mat(path_mat, size_x, size_y);
  int pos_path = 0;
  path[0] = 0;
  pos_path++;
  while (path_mat[j][i] != 's')
  {
    path[pos_path] = j;
    pos_path++;
    path[pos_path] = i;
    pos_path++;
	//printf("\npos_path[%d][%d]=%f", i, j, cost[i][j]);
    if (path_mat[j][i] == 'l')
    {
      path[pos_path] = cost[j][i] - cost[j][i-1];
      i--;
    }
    else
    {
      if (path_mat[j][i] == 'b')
      {
        path[pos_path] = cost[j][i] - cost[j-1][i];
	    j--;
      }
      else
      {
        path[pos_path] = cost[j][i] - cost[j-1][i-1];
	    i--;
	    j--;
      }
    }
    pos_path++;
  }

  // adding the first points
  path[pos_path] = 0;
  pos_path++;
  path[pos_path] = 0;
  pos_path++;
  path[pos_path] = cost[0][0];
  pos_path++;
  path[0] = pos_path;
  averaged_cort_dtw_distances(diffx, diffy, path, distances);
  free_dtw_no_path_multivariate:
  if (path_mem) free(path_mem);
  if (path_mat) free(path_mat);
  if (cost) free(cost);
  if (cost_mem) free(cost_mem);
  if (cortNumC) free(cortNumC);
  if (cortNumP) free(cortNumP);
  if (cortDenomXC) free(cortDenomXC);
  if (cortDenomXP) free(cortDenomXP);
  if (cortDenomYC) free(cortDenomYC);
  if (cortDenomYP) free(cortDenomYP);
  if (diffx) free(diffx);
  if (diffy) free(diffy);

}

//Not used for now
double cort_given_path_weighted(double* x, double* y, int end_x, int end_y, double* weight, int* path, int path_length){
	double *diffx, *diffy;
	double ret = 0.0;
	diffx = (double*) malloc( sizeof(double)* (end_x-1));
    diffy = (double*) malloc(sizeof(double) * (end_y-1));
	compute_slope(x, 0, end_x, 1, diffx);
    compute_slope(y, 0, end_y, 1, diffy);
	int i;
    int curr_i;
    int count_j;
    curr_i = path[0];
    int path_i;

    double nomX, curr_val, denomXI, denomXJ;
    nomX=0;
    count_j=0;
    curr_val=0;
    denomXI=0, denomXJ=0;

    for (i=0; i<path_length; i=i+1)
    {
        path_i = path[i];
        if (curr_i != path_i)
        {
            curr_val = curr_val / count_j;
            nomX += diffx[curr_i] * curr_val * weight[curr_i];
            denomXI += diffx[curr_i] * diffx[curr_i];
            denomXJ += curr_val * curr_val;
            curr_i = path_i;
            curr_val = diffy[i];
            count_j = 1;
        }
        else
        {
            curr_val += diffy[i];
            count_j += 1;
        }
    }
	curr_val = curr_val / count_j;
    nomX += diffx[curr_i] * curr_val * weight[curr_i];
    denomXI += diffx[curr_i] * diffx[curr_i];
    denomXJ += curr_val * curr_val;
    ret = nomX / sqrt(denomXI * denomXJ);
	return ret;
}

double cort_dtw_given_path_weighted(double* x, double* y, int end_x, int end_y, double* weight, int* path, int path_length){
	double *diffx, *diffy;
	double ret = 0.0;
	double dtw = 0.0;
	diffx = (double*) malloc( sizeof(double)* (end_x-1));
    diffy = (double*) malloc(sizeof(double) * (end_y-1));
	compute_slope(x, 0, end_x, 1, diffx);
    compute_slope(y, 0, end_y, 1, diffy);
	int i;
    int curr_i;
    int count_j;
    curr_i = path[0];
    int path_i;

    double nomX, curr_val, denomXI, denomXJ;
    nomX=0;
    count_j=0;
    curr_val=0;
    denomXI=0, denomXJ=0;

    for (i=0; i<path_length; i=i+1)
    {
        path_i = path[i];
		dtw += compute_diff(x, y, path[i], i, 1)*weight[path[i]];
        if (curr_i != path_i)
        {
            curr_val = curr_val / count_j;
            nomX += diffx[curr_i] * curr_val * weight[curr_i];
            denomXI += diffx[curr_i] * diffx[curr_i];
            denomXJ += curr_val * curr_val;
            curr_i = path_i;
            curr_val = diffy[i];
            count_j = 1;
        }
        else
        {
            curr_val += diffy[i];
            count_j += 1;
        }
    }
	dtw += compute_diff(x, y, path[i], i, 1)*weight[path[i]];
	curr_val = curr_val / count_j;
    nomX += diffx[curr_i] * curr_val * weight[curr_i];
    denomXI += diffx[curr_i] * diffx[curr_i];
    denomXJ += curr_val * curr_val;
    ret = dtw * (nomX / sqrt(denomXI * denomXJ));
	return ret;
}

