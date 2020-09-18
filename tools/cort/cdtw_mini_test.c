

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

int main()
{
        double series_a[] = {-0.4372,-0.22005,1.2297,-0.33123,0.14221,-1.536,-1.5676,0.018222,0.93458,-1.3208,0.64249,-0.59441,0.14686,-0.99587,0.25908,-1.3274,-0.42101,-0.31574,-0.52498,0.98952,1.0196,0.85924,-0.032788,-0.29606,-0.19108,1.404,-0.83699,-1.5498,0.20429,-1.1532,-1.3681,0.18443,1.1182,-1.5917,1.4581,-1.3203,-0.14477,1.5169,-0.84683,0.57612,0.31723,-0.5132,0.56943,-1.6662,1.6567,1.1969,0.2863,-1.509,0.82179,-1.4828,-0.58952,0.97631,-1.5423,0.44828,-0.20616,-1.7411,0.85776,-0.54208,-1.4072,-0.98026,-1.1258,1.5532,-0.99184,1.043,-1.1068,0.7843,1.0168,1.446,0.87415,1.3391,-1.4993,-0.12036,0.16029,1.1242,-0.28276,-0.98326,0.62503,-0.054796,-0.38173,1.4229,-1.3775,-1.1716,0.45338,1.5441,-0.70787,-1.5756,-0.146,0.24379,-0.50875,0.5586,0.032611,-0.82677,-1.005,-1.0301,-1.4583,-1.6647,-0.39536,-0.46997,1.243,1.2623,0.73828,1.6758,-0.88728,0.85856,-0.58347,-1.0096,0.93856,-0.37483,0.71181,-0.48123,-1.5242,-0.46048,0.20732,-1.6428,-0.68728,1.0799,-0.49711,-1.685,-0.11284,0.93546,-0.23511,-1.4167,-0.2555,-1.4603,1.6022,-0.61861,0.11425,-1.3747,0.64766,1.2132,0.58232,-0.33915,0.73265,1.04,-0.63934,0.12795,1.0634,0.65786,-1.6212,1.2981,-0.74749,-0.088544,0.31257,-1.0407,1.4901,1.1939,-1.2452,-0.60294,0.12051,1.4625,1.5599,0.90457,-1.1983,1.1717,0.82164,0.35979,0.6596,-1.7168,-1.576,0.10217,-0.7047,-1.5902,1.4669,0.78347,0.93531,-0.11285,1.6311,-1.5426,-0.088161,-0.98356,0.99063,0.13794,-1.057,1.6388,-1.2945,0.28362,-1.5666,-1.0826,-0.64449,1.1821,0.977,-0.92661,1.6224,1.3602,0.96187,1.5724,0.83537,-1.2729,0.36115,-0.83085,0.15117,-0.62405,-0.69434,1.402,0.77038,-0.29639,-0.41402,-1.7183,0.72892,-1.7238,-1.0632,1.5942,1.599,0.81323,-0.35419,1.0284,-0.31871,0.63939,-0.72692,0.99719,-0.96916,-0.43661,-1.222,-0.29691,-0.2418,0.35842,-1.695,-0.24608,1.0891,1.2657,-1.2732,0.61061,0.044918,-0.86169,1.1075,1.5988,-1.5858,-0.73026,0.23325,1.3433,0.80168,-1.4146,-1.1857,-1.4849,-1.3133,-0.08935,1.4318,0.2029,-0.036293,0.042795,-0.421,0.37366,0.34634,-1.4278,-0.13201,-1.1659,0.38332,0.24662,0.31684,0.1795,-0.54931,0.81489,-0.28084,1.4863,-0.63963,-0.59861,-0.23536,1.5048,-0.077043,0.156,-0.99667,1.3563,-1.4429,1.1059,-1.3052,0.46328,-1.6975,1.2661,1.5721,0.64198,0.65734,-1.7437,-0.33421,-1.3255,1.4049,-1.5601,-0.7119,1.175,1.4817,-1.3136,-1.1852,0.07549,1.3352,-0.16535,-1.4322,0.16193,0.95986,-0.027495,1.0897,0.61229,-0.8984,-1.3087,0.39005,-1.2271,-1.5414,0.0047497,1.2977,1.4545,-0.45358,1.0457,-0.69129,1.0653,1.0522,-0.57608,-1.3324,1.1809,1.4276,-0.9551,-1.273,-1.1188,1.5901,1.2837,-0.5971,-0.69372,0.43718,1.1048,-0.92848,1.4481,-1.6358,0.33598,0.59458,-0.55823,-0.39447,-0.61861,-0.66211,-0.11987,1.129,0.14307,1.5845,1.2437,1.6638,1.3481,1.4892,-0.36257,0.20737,0.28538,-0.15116,-1.1707,0.28425,0.76682,1.1361,-0.96009,0.73387,0.51155,-1.5228,0.74231,-1.6439,-0.71189,0.65121,0.22306,0.2395,0.39928,1.3328,0.99077,-0.65369,0.63792,-1.2394,1.5027,-0.46869,1.6169,0.49623,0.93934,1.6308,0.47099,1.4083,0.49336,-0.6017,-0.0029044,-0.50671,-0.5222,-1.1116,-0.70935,-0.70558,0.16218,-1.2315,0.44028,-0.87593,1.121,0.88599,1.0145,0.095126,0.49075,0.43449,-1.3537,1.4925,-0.78225,-0.038399,-1.3423,0.26695,0.76995,0.21535,-1.009,-1.188,-0.51815,-0.48077,1.3282,0.00024853,0.38648,-1.4851,0.44617,0.85766,1.2652,0.45679,-1.4391,-0.19052,0.65366,1.501,0.84605,0.22505,0.057481,1.3409,0.88718,1.083,0.73183,0.73852,1.5262,0.13906,-0.49529,-0.1362,-0.91951,-0.79181,-1.1667,0.48464,-1.3182,-1.213,1.3584,1.2684,-0.012809,0.23743,0.40006,0.23743,0.51444,1.3322,1.4055,-1.1957,0.4,-0.41746,1.2353,0.017987,-0.53047,0.25543,-0.45155,-0.68426,-0.16122,0.37439,0.8619,-0.066941,-0.86457,0.36528,-1.5015,-1.4757,-1.6036,1.3322,-1.4671,0.99958,-0.82228,-0.26065,0.98151,1.5782,-0.72803,-1.0633,1.2376,1.2488,1.5781,1.2085,-1.0667,-1.3826,-1.5517,1.1664,1.3137,-0.72231,-1.4041,-1.5107,0.8829,-0.43089,0.15122,-0.9389,-0.31751,0.99288,0.5353,-0.5233,-0.61247,1.6221,0.78982,-1.1836,-0.5276,-1.498,-0.22096,1.4164,-0.95381,-1.219,-0.97063,0.17276,-0.62248,1.5965,0.043086,-0.51715,-1.6879,-1.4363,0.62526};
       

double series_b[] = {1.3055,-0.3104,-0.80506,-1.2891,0.52102,-1.0066,1.3718,1.3153,0.84243,0.11568,-1.5446,0.77905,0.77302,0.38443,-0.57817,1.2517,-0.94582,0.85752,0.15058,-0.15612,1.2861,-0.95919,0.20896,1.0698,-1.5128,1.3516,1.0708,1.459,0.88141,0.08552,-0.51407,0.59093,1.2106,-0.48771,0.61386,-0.29102,0.47822,1.639,-0.72301,0.36116,0.85886,1.4167,0.93031,0.72061,-1.3525,1.3428,1.1746,0.80035,-1.1624,-1.5847,0.65944,-1.1367,-0.23218,0.95678,0.74178,1.2341,0.70542,0.99835,0.026812,1.079,-0.3486,-0.70309,-0.47257,-1.4424,0.17592,0.38383,-1.1792,0.028779,-1.3414,-1.6572,1.4565,-0.38527,-0.50608,-0.57129,-0.039166,-0.073844,-1.2626,0.97897,-1.7009,1.5129,1.1228,-0.9953,-1.5117,0.80408,-1.0035,-0.093279,0.3506,-0.51454,0.94935,-1.1834,0.73079,-0.44158,-1.4803,-1.4208,0.67694,-1.5562,0.56648,-1.7092,1.199,-0.49045,-0.8888,1.308,1.5644,-1.5407,-0.29376,0.8671,0.74833,0.83029,1.5019,1.5886,-0.039857,-0.54567,1.5591,-1.1103,-1.3764,-1.009,-0.48897,-1.4633,1.5244,0.40001,0.48315,1.449,-0.23504,-1.0188,1.1031,1.3468,-1.1361,-0.48323,0.01456,-0.25525,-1.3424,-0.049603,1.5825,-0.60775,-1.2567,-0.55784,-1.3543,1.4575,-0.34846,0.18437,0.25615,1.2076,-1.5669,-1.4325,-0.2867,0.032933,-0.37572,-1.0209,0.98118,0.24991,1.0511,1.174,1.41,-1.4375,1.3441,-0.19946,-1.4394,-0.65417,-1.2805,-1.0952,1.3428,-0.30674,-1.4182,-0.27444,1.0063,0.62216,1.5921,-1.6418,0.31663,-1.3751,-0.35748,1.5877,0.26323,0.084069,-0.58058,-1.5623,0.98291,0.23161,-1.6722,-1.5857,0.43251,-0.13002,-0.53369,0.67739,1.2082,1.2527,1.155,-1.3841,0.91034,1.4685,-1.214,-1.668,-0.27679,0.74366,1.2952,-0.63986,1.3255,1.3571,-0.74584,-1.2165,-1.2682,1.6538,-0.37388,-1.4896,1.5724,1.2476,-0.75368,-0.23662,-0.88148,0.95934,-1.2281,1.4623,1.4316,-1.2458,1.2276,-0.76671,1.6004,0.35244,1.3905,0.28274,-1.6315,-0.66632,1.1761,1.5787,1.0314,-0.49475,0.97182,1.204,-0.19361,0.13798,0.79561,0.47301,-0.99306,-1.15,1.4649,-1.3721,-0.43743,-1.567,0.0068073,0.55221,0.057771,-0.81144,-1.6474,0.35559,0.50746,0.51559,-1.1295,-0.75975,1.4038,-0.22215,-0.35196,0.51334,1.0974,1.1592,-1.5309,-0.70363,0.57425,0.75414,0.8816,-0.16823,-0.82585,-1.1107,1.2346,-1.2819,-1.1714,-0.26132,0.98461,-1.5873,-0.18184,-1.4298,-1.4371,-0.094979,-0.61116,0.29069,-1.568,1.6249,1.3954,0.8428,-1.5242,-0.10695,0.34006,0.13018,-1.0954,-0.32685,-1.1423,-0.97074,-1.4543,-1.494,-0.2418,-1.283,-0.87362,0.26469,0.3579,1.5919,-1.0007,1.4097,1.1173,0.53965,0.058102,-0.96199,0.12457,1.2767,-0.00085149,-0.72929,1.126,1.4277,-1.0723,-0.78428,-0.22789,-1.6019,-1.2333,-0.50861,0.31714,0.0034956,1.474,0.55752,-0.92937,1.0804,-0.31971,-1.0187,-0.0072355,-0.33306,-1.1736,-0.060206,-0.5492,0.30918,0.73944,-0.26966,0.55436,1.0726,-0.70315,-0.48233,0.28555,-0.57249,1.5673,1.1717,0.23151,-0.60658,0.48905,-0.96631,-0.81401,-0.18644,1.4006,-1.4084,0.80814,1.5076,0.4338,0.26698,1.6464,0.45479,0.20075,0.82678,0.81027,1.521,-1.0844,-1.4991,-0.99696,-0.84688,0.0019881,0.39332,1.3035,-1.1022,-1.6689,0.80646,-0.70576,0.80203,1.0813,-1.3448,-0.29473,-0.70141,0.77299,0.49181,0.85068,-1.4496,-1.5031,0.62903,-1.5536,-0.22804,-1.1248,-0.21228,1.6352,-0.94433,-1.1265,-0.59055,-1.1717,1.26,-0.73122,-1.6843,-0.076552,-1.3759,0.59078,1.3933,-1.521,1.163,0.30531,0.62613,0.092072,-0.011333,-0.58973,-0.68824,1.5376,0.49127,-1.6932,-0.8647,1.027,-1.5562,1.4775,-0.10586,0.95336,0.27241,-0.3557,0.92543,-0.53327,-1.579,-0.49576,1.3676,1.0453,-0.18033,-0.071618,-0.90876,0.96899,1.0284,0.30066,-1.0479,0.92601,0.19542,0.35461,-0.74809,-1.6913,-1.6542,0.44218,-0.44905,-1.4945,0.18305,0.56681,0.22675,-0.63495,0.99361,-0.071429,0.70302,1.4399,-0.096163,-0.70134,0.70982,-1.4576,1.5582,-0.47977,0.99434,-1.4904,1.079,1.1732,0.076825,-0.92258,-0.12262,-0.89851,-1.3198,0.88532,0.52455,-1.4687,-0.6028,-1.1754,1.1991,-1.5987,0.18859,0.53645,1.2905,-0.82387,-0.54979,1.3445,0.24303,-1.1275,0.31475,0.7657,-0.75184,-0.23088,0.132,0.29869,-1.2406,-0.37063,1.3049,1.5358,1.5752,1.0583,-0.26162,0.66025,0.48019,1.4205,0.60129,-1.5403,-0.46679,0.084825,0.1966,-0.5481,-0.97996,-0.26761,-0.72931,1.2109,0.88543,-0.096156,1.5551};
    double res = cort_simple(series_a, series_b, 0, 200, 1);
    printf("cort simple %f\n", res);
    double res2 = cort(series_a, series_b, 0, 0, 200, 200, 1, 1);
    printf("cort %f\n", res2);
    return 0;
}
