/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/


/* potential error codes */
#define potential_err_ok                    0
#define potential_err_null                  -1
#define potential_err_malloc                -2
#define potential_err_bounds                -3


/* some constants */
#define potential_degree                    5
#define potential_ivalsmin                  2
#define potential_ivalsmax                  10
#define potential_N                         100


/* potential flags */
#define potential_flag_none                  0
#define potential_flag_LJ126                 1
#define potential_flag_Ewald                 2
#define potential_flag_Coulomb               4
#define potential_flag_single                6


/* the last error */
extern int potential_err;


/* the potential structure */
struct potential {

    /* interval edges */
    double a, b;
    
    /* flags */
    unsigned int flags;
    
    /* coefficients for the interval transform */
    double alpha[3];
    
    /* nr of intervals */
    int n;
    
    /* the coefficients */
    double *c;
    
    /* the interval centres and widths */
    double *mi, *hi;
    
    };
    

/* associated functions */
int potential_init ( struct potential *p , double (*f)( double ) , double (*fp)( double ) , double (*f6p)( double ) , double a , double b , double tol );
int potential_getcoeffs ( double (*f)( double ) , double (*fp)( double ) , double *xi , int n , double *c , double *err );
double potential_getalpha ( double (*f6p)( double ) , double a , double b );
struct potential *potential_createLJ126 ( double a , double b , double A , double B , double tol );
struct potential *potential_createLJ126_Ewald ( double a , double b , double A , double B , double q , double kappa , double tol );
struct potential *potential_create_Ewald ( double a , double b , double q , double kappa , double tol );
#ifdef USE_SINGLE
/* inline */ void potential_eval ( struct potential *p , float r2 , float *e , float *f );
/* inline */ void potential_eval_expl ( struct potential *p , float r2 , float *e , float *f );
#else
/* inline */ void potential_eval ( struct potential *p , double r2 , double *e , double *f );
/* inline */ void potential_eval_expl ( struct potential *p , double r2 , double *e , double *f );
#endif

/* helper functions */
double potential_LJ126 ( double r , double A , double B );
double potential_LJ126_p ( double r , double A , double B );
double potential_LJ126_6p ( double r , double A , double B );
double potential_LJ126_r2 ( double r2 , double A , double B );
double potential_LJ126_r2_p ( double r2 , double A , double B );
double potential_LJ126_r2_6p ( double r2 , double A , double B );
double potential_LJn6_r2 ( double r2 , int n , double eps , double sigma );
double potential_LJn6_r2_p ( double r2 , int n , double eps , double sigma );
double potential_LJn6_r2_6p ( double r2 , int n , double eps , double sigma );
double potential_Coulomb_r2 ( double r2 );
double potential_Coulomb_r2_p ( double r2 );
double potential_Coulomb_r2_6p ( double r2 );
double potential_Ewald ( double r , double kappa );
double potential_Ewald_p ( double r , double kappa );
double potential_Ewald_6p ( double r , double kappa );
double potential_Ewald_r2 ( double r2 , double kappa );
double potential_Ewald_r2_p ( double r2 , double kappa );
double potential_Ewald_r2_6p ( double r2 , double kappa );
