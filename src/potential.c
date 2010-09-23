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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

/* include local headers */
#include "potential.h"


/* the last error */
int potential_err = potential_err_ok;


/*////////////////////////////////////////////////////////////////////////////// */
/* void potential_eval_expl */
//
/* evaluates the given potential at the given point. */
/*////////////////////////////////////////////////////////////////////////////// */

#ifdef USE_SINGLE
/* inline */ void potential_eval_expl ( struct potential *p , float r2 , float *e , float *f ) {

    const float isqrtpi = 0.56418958354775628695;
    const float kappa = 3.0;
    double r = sqrt(r2), ir = 1.0 / r, ir2 = ir * ir, ir4, ir6, ir12, t1, t2;

    /* init f and e */
    *e = 0.0; *f = 0.0;

    /* do we have a Lennard-Jones interaction? */
    if ( p->flags & potential_flag_LJ126 ) {
    
        /* init some variables */
        ir4 = ir2 * ir2; ir6 = ir4 * ir2; ir12 = ir6 * ir6;
        
        /* compute the energy and the force */
        *e = ( p->alpha[0] * ir12 - p->alpha[1] * ir6 );
        *f = 6.0 * ir * ( -2.0 * p->alpha[0] * ir12 + p->alpha[1] * ir6 );
    
        }
        
    /* do we have an Ewald short-range part? */
    if ( p->flags & potential_flag_Ewald ) {
    
        /* get some values we will re-use */
        t2 = r * kappa;
        t1 = erfc( t2 );
    
        /* compute the energy and the force */
        *e += p->alpha[2] * t1 * ir;
        *f += p->alpha[2] * ( -2.0 * isqrtpi * exp( -t2 * t2 ) * kappa * ir - t1 * ir2 );
    
        }
    
    /* do we have a Coulomb interaction? */
    if ( p->flags & potential_flag_Coulomb ) {
    
        /* get some values we will re-use */
        t2 = r * kappa;
        t1 = erfc( t2 );
    
        /* compute the energy and the force */
        *e += p->alpha[2] * ir;
        *f += -p->alpha[2] * ir2;
    
        }
    
    }
#else
/* inline */ void potential_eval_expl ( struct potential *p , double r2 , double *e , double *f ) {

    const double isqrtpi = 0.56418958354775628695;
    const double kappa = 3.0;
    double r = sqrt(r2), ir = 1.0 / r, ir2 = ir * ir, ir4, ir6, ir12, t1, t2;

    /* init f and e */
    *e = 0.0; *f = 0.0;

    /* do we have a Lennard-Jones interaction? */
    if ( p->flags & potential_flag_LJ126 ) {
    
        /* init some variables */
        ir4 = ir2 * ir2; ir6 = ir4 * ir2; ir12 = ir6 * ir6;
        
        /* compute the energy and the force */
        *e = ( p->alpha[0] * ir12 - p->alpha[1] * ir6 );
        *f = 6.0 * ir * ( -2.0 * p->alpha[0] * ir12 + p->alpha[1] * ir6 );
    
        }
        
    /* do we have an Ewald short-range part? */
    if ( p->flags & potential_flag_Ewald ) {
    
        /* get some values we will re-use */
        t2 = r * kappa;
        t1 = erfc( t2 );
    
        /* compute the energy and the force */
        *e += p->alpha[2] * t1 * ir;
        *f += p->alpha[2] * ( -2.0 * isqrtpi * exp( -t2 * t2 ) * kappa * ir - t1 * ir2 );
    
        }
    
    /* do we have a Coulomb interaction? */
    if ( p->flags & potential_flag_Coulomb ) {
    
        /* get some values we will re-use */
        t2 = r * kappa;
        t1 = erfc( t2 );
    
        /* compute the energy and the force */
        *e += p->alpha[2] * ir;
        *f += -p->alpha[2] * ir2;
    
        }
    
    }
#endif


/*////////////////////////////////////////////////////////////////////////////// */
/* double potential_LJ126 */
//
/* provide some basic functions over r^2 and r to construct potentials */
/*////////////////////////////////////////////////////////////////////////////// */

inline double potential_LJ126 ( double r , double A , double B ) {

    double ir = 1.0/r, ir2 = ir * ir, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

    return ( A * ir12 - B * ir6 );

    }
    
inline double potential_LJ126_p ( double r , double A , double B ) {

    double ir = 1.0/r, ir2 = ir * ir, ir4 = ir2*ir2, ir12 = ir4 * ir4 * ir4;

    return 6 * ir * ( -2 * A * ir12 + B * ir4 * ir2 );

    }
    
inline double potential_LJ126_6p ( double r , double A , double B ) {

    double r2 = r * r, ir2 = 1.0 / r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

    return 10080 * ir12 * ( 884 * A * ir6 - 33 * B );
        
    }
    
double potential_LJ126_r2 ( double r2 , double A , double B ) {

    double ir2 = 1.0/r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

    return ( A * ir12 - B * ir6 );

    }
    
double potential_LJ126_r2_p ( double r2 , double A , double B ) {

    double ir2 = 1.0/r2, ir4 = ir2*ir2, ir12 = ir4 * ir4 * ir4;

    return 3 * ( -2 * A * ir12 * ir2 + B * ir4 * ir4 );

    }
    
double potential_LJ126_r2_6p ( double r2 , double A , double B ) {

    double ir2 = 1.0 / r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

    return 10080 * ( 33 * A * ir12 * ir12 - 2 * B * ir12 * ir6 );
        
    }
    
double potential_LJn6_r2 ( double r2 , int n , double eps , double sigma ) {

    double ir = 1.0 / sqrt(r2), ir2 = 1.0/r2, ir6 = ir2*ir2*ir2;
    double sigma2 = sigma * sigma, sigma6 = sigma2 * sigma2 * sigma2;

    return eps * (n / (n - 6)) * pow( n/6 , 6 / (n-6) ) *
        ( pow( sigma * ir , n ) - sigma6 * ir6 );

    }
    
double potential_LJn6_r2_p ( double r2 , int n , double eps , double sigma ) {

    double ir = 1.0 / sqrt(r2), ir2 = 1.0/r2, ir4 = ir2*ir2;
    double sigma2 = sigma * sigma, sigma6 = sigma2 * sigma2 * sigma2;

    return eps * (n / (n - 6)) * pow( n/6 , 6 / (n-6) ) *
        ( -0.5 * n * pow( sigma * ir , n ) * ir2 - 3 * sigma6 * ir4 * ir4 );

    }
    
double potential_LJn6_r2_6p ( double r2 , int n , double eps , double sigma ) {

    double t38, t50, t49, t48, t40, t2, t4, t5, t6, t12, t16;
    double t21, t25, t27, t32, t36, t41, t47, t51;

    t38 = 1/(n-6.0);
    t50 = 6.0*t38;
    t49 = 2.0*t38;
    t48 = n*t38;
    t40 = r2*r2;
    t2 = pow(46656.0,-1.0*t38);
    t4 = pow(n,1.0*t50);
    t5 = sigma*sigma;
    t6 = t5*t5;
    t12 = pow(n,1.0*(-5.0+n)*t50);
    t16 = pow(n,1.0*(-24.0+5.0*n)*t38);
    t21 = pow(n,3.0*(-4.0+n)*t38);
    t25 = pow(n,1.0*(-3.0+n)*t49);
    t27 = pow(n,1.0*t48);
    t32 = pow(n,1.0*(-9.0+2.0*n)*t49);
    t36 = sqrt(r2);
    t41 = pow(sigma/t36,1.0*n);
    t47 = t40*t40;
    t51 = t47*t47;
    
    return eps*t2*(-1290240.0*t4*t6*t5+(t12+30.0*t16+1800.0*t21+4384.0*t25+
        3840.0*t27+340.0*t32)*r2*t40*t41)/r2/t51*t48/64.0;
        
    }
    
double potential_Coulomb_r2 ( double r2 ) {

    return 1.0 / sqrt(r2);

    }
    
double potential_Coulomb_r2_p ( double r2 ) {

    double ir2 = 1.0 / r2;

    return 0.5 * ir2 * sqrt(ir2);

    }
    
double potential_Coulomb_r2_6p ( double r2 ) {

    double ir2 = 1.0 / r2, ir6 = ir2 * ir2 * ir2;

    return 162.421875 * ir6 * sqrt(ir2);

    }
    
inline double potential_Ewald ( double r , double kappa ) {

    return erfc( kappa * r ) / r;

    }
    
inline double potential_Ewald_p ( double r , double kappa ) {

    double r2 = r * r, ir = 1.0 / r, ir2 = ir * ir;
    const double isqrtpi = 0.56418958354775628695;

    return -2 * exp( -kappa*kappa * r2 ) * kappa * ir * isqrtpi -
        erfc( kappa * r ) * ir2;

    }
    
inline double potential_Ewald_6p ( double r , double kappa ) {

    double r2 = r*r, ir2 = 1.0 / r2, r4 = r2*r2, ir4 = ir2*ir2, ir6 = ir2*ir4;
    double kappa2 = kappa*kappa;
    double t6, t23;
    const double isqrtpi = 0.56418958354775628695;

    t6 = erfc(kappa*r);
    t23 = exp(-kappa2*r2);
    return 720.0*t6/r*ir6+(1440.0*ir6+(960.0*ir4+(384.0*ir2+(144.0+(-128.0*r2+64.0*kappa2*r4)*kappa2)*kappa2)*kappa2)*kappa2)*kappa*isqrtpi*t23;
    
    }
    
double potential_Ewald_r2 ( double r2 , double kappa ) {

    double r = sqrt(r2);

    return erfc( kappa * r ) / r;

    }
    
double potential_Ewald_r2_p ( double r2 , double kappa ) {

    double r = sqrt(r2), ir2 = 1.0 / r2;
    const double isqrtpi = 0.56418958354775628695;

    return -exp( -kappa*kappa * r2 ) * kappa * ir2 * isqrtpi -
        0.5 * erfc( kappa * r ) * ir2 / r;

    }
    
double potential_Ewald_r2_6p ( double r2 , double kappa ) {

    double r = sqrt(r2), ir2 = 1.0 / r2, r4 = r2*r2, r6 = r2*r4, r8 = r4*r4, r12 = r6*r6;
    double kappa2 = kappa*kappa, kappa3 = kappa*kappa2, kappa5 = kappa2*kappa3;
    double t2, t15, t40;
    const double isqrtpi = 0.56418958354775628695;

    t2 = erfc(kappa*r);
    t15 = 1.0/r8;
    t40 = exp(-kappa2*r2);
    return 10395.0/64.0*t2/r4/r*t15+(10395.0/32.0*kappa/r12+(693.0/8.0*kappa5+3465.0/16.0*kappa3*ir2)*t15+(99.0/4.0/r6+(kappa2*ir2+11.0/2.0/r4)*kappa2)*kappa2*kappa5)*t40*isqrtpi;

    }
    
    

/*////////////////////////////////////////////////////////////////////////////// */
/* void potential_eval */
//
/* evaluates the given potential at the given point. */
/*////////////////////////////////////////////////////////////////////////////// */

#ifdef USE_SINGLE
/* inline */ void potential_eval ( struct potential *p , float r2 , float *e , float *f ) {

    int ind, k;
    float x, ee, eff, c[potential_degree+1], r = sqrt(r2);
    
    /* is r in the house? */
    /* if ( r < p->a || r > p->b ) */
    /*     printf("potential_eval: requested potential at r=%e, not in [%e,%e].\n",r,p->a,p->b); */
    
    /* compute the index */
    ind = p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]);
    
    /* adjust x to the interval */
    x = (r - p->mi[ind]) * p->hi[ind];
    
    /* get the table offset */
    for ( k = 0 ; k <= potential_degree ; k++ )
        c[k] = p->c[ind * (potential_degree + 1) + k];
    
    /* compute the potential and its derivative */
    ee = c[0] * x + c[1];
    eff = c[0];
    for ( k = 2 ; k <= potential_degree ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        }

    /* store the result */
    *e = ee; *f = eff * p->hi[ind];    
    }
#else    
/* inline */ void potential_eval ( struct potential *p , double r2 , double *e , double *f ) {

    int ind, k;
    double x, ee, eff, *c, r = sqrt(r2);
    
    /* is r in the house? */
    /* if ( r < p->a || r > p->b ) */
    /*     printf("potential_eval: requested potential at r=%e, not in [%e,%e].\n",r,p->a,p->b); */
    
    /* compute the index */
    ind = p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]);
    
    /* adjust x to the interval */
    x = (r - p->mi[ind]) * p->hi[ind];
    
    /* get the table offset */
    c = &(p->c[ind * (potential_degree + 1)]);
    
    /* compute the potential and its derivative */
    ee = c[0] * x + c[1];
    eff = c[0];
    for ( k = 2 ; k <= potential_degree ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        }

    /* store the result */
    *e = ee; *f = eff * p->hi[ind];    
    }
#endif

/*////////////////////////////////////////////////////////////////////////////// */
/* struct potential *potential_create_Ewald */
//
/* creates a potential structure representing an Ewald real-space potential with the */
/* charge product q and the screening distance kappa in the interval [a,b] */
/* to the specified tolerance tol. */
/*////////////////////////////////////////////////////////////////////////////// */

struct potential *potential_create_Ewald ( double a , double b , double q , double kappa , double tol ) {

    struct potential *p;
    
    /* the potential functions */
    double f ( double r ) {
        return q * potential_Ewald( r , kappa );
        }
        
    double dfdr ( double r ) {
        return q * potential_Ewald_p( r , kappa );
        }
        
    double d6fdr6 ( double r ) {
        return q * potential_Ewald_6p( r , kappa );
        }
        
    /* allocate the potential */
    if ( ( p = (struct potential *)malloc( sizeof( struct potential ) ) ) == NULL ) {
        potential_err = potential_err_malloc;
        return NULL;
        }
        
    /* fill this potential */
    if ( potential_init( p , &f , &dfdr , &d6fdr6 , a , b , tol ) < 0 ) {
        free(p);
        return NULL;
        }
    
    /* return it */
    return p;

    }
    

/*////////////////////////////////////////////////////////////////////////////// */
/* struct potential *potential_createLJ126_Ewald */
//
/* creates a potential structure representing a Lennard-Jones 12-6 interaction */
/* with the parameters A and B and an Ewald real-space potential with the */
/* charge product q and the screening distance kappa in the interval [a,b] */
/* to the specified tolerance tol. */
/*////////////////////////////////////////////////////////////////////////////// */

struct potential *potential_createLJ126_Ewald ( double a , double b , double A , double B , double q , double kappa , double tol ) {

    struct potential *p;
    
    /* the potential functions */
    double f ( double r ) {
        return potential_LJ126 ( r , A , B ) +
            q * potential_Ewald( r , kappa );
        }
        
    double dfdr ( double r ) {
        return potential_LJ126_p ( r , A , B ) +
            q * potential_Ewald_p( r , kappa );
        }
        
    double d6fdr6 ( double r ) {
        return potential_LJ126_6p ( r , A , B ) +
            q * potential_Ewald_6p( r , kappa );
        }
        
    /* allocate the potential */
    if ( ( p = (struct potential *)malloc( sizeof( struct potential ) ) ) == NULL ) {
        potential_err = potential_err_malloc;
        return NULL;
        }
        
    /* fill this potential */
    if ( potential_init( p , &f , &dfdr , &d6fdr6 , a , b , tol ) < 0 ) {
        free(p);
        return NULL;
        }
    
    /* return it */
    return p;

    }
    

/*////////////////////////////////////////////////////////////////////////////// */
/* struct potential *potential_createLJ126 */
//
/* creates a potential structure representing a Lennard-Jones 12-6 interaction */
/* with the parameters A and B in the interval [a,b] to the specified tolerance */
/* tol. */
/*////////////////////////////////////////////////////////////////////////////// */

struct potential *potential_createLJ126 ( double a , double b , double A , double B , double tol ) {

    struct potential *p;
    
    double f ( double r2 ) {
    
        double ir2 = 1.0/r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;
    
        return ( A * ir12 - B * ir6 );
        
        }

    double fp ( double r2 ) {
    
        double ir2 = 1.0/r2, ir4 = ir2*ir2, ir12 = ir4 * ir4 * ir4;
    
        return 3 * ( -2 * A * ir12 * ir2 + B * ir4 * ir4 );
        
        }

    double f6p ( double r2 ) {
    
        double ir2 = 1.0 / r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;
    
        return 10080 * ( 33 * A * ir12 * ir12 - 2 * B * ir12 * ir6 );
        
        }
        
    /* allocate the potential */
    if ( ( p = (struct potential *)malloc( sizeof( struct potential ) ) ) == NULL ) {
        potential_err = potential_err_malloc;
        return NULL;
        }
        
    /* fill this potential */
    if ( potential_init( p , &f , &fp , &f6p , a , b , tol ) < 0 ) {
        free(p);
        return NULL;
        }
    
    /* return it */
    return p;

    }
    

/*////////////////////////////////////////////////////////////////////////////// */
/* int potential_init */
//
/* initialize the potential with the given function 'f', its derivative 'fp' */
/* and it's 6th derivative 'f6p' to the absolute tolerance 'tol' in the */
/* interval [a,b]. the optimal number of intervals and mapping is chosen */
/* automatically. */
/*////////////////////////////////////////////////////////////////////////////// */

int potential_init ( struct potential *p , double (*f)( double ) , double (*fp)( double ) , double (*f6p)( double ) , double a , double b , double tol ) {

    double alpha, w;
    int l = potential_ivalsmin, r = potential_ivalsmax, m;
    double err_l, err_r, err_m;
    double *xi_l, *xi_r, *xi_m;
    double *c_l, *c_r, *c_m;
    int i, k;
    double e;

    /* check inputs */
    if ( p == NULL || f == NULL || fp == NULL || f6p == NULL )
        return potential_err = potential_err_null;
        
    /* set the boundaries */
    p->a = a; p->b = b;
    
    /* compute the optimal alpha for this potential */
    alpha = potential_getalpha(f6p,a,b);
    /* printf("potential_init: alpha is %e\n",alpha); fflush(stdout); */
    
    /* compute the interval transform */
    w = 1.0 / (a - b); w *= w;
    p->alpha[0] = -( a * ( alpha * b - a ) ) * w;
    p->alpha[1] = ( alpha * ( b + a ) - 2 * a ) * w;
    p->alpha[2] = -(alpha - 1) * w;
    
    /* compute the smallest interpolation... */
    /* printf("potential_init: trying l=%i...\n",l); fflush(stdout); */
    xi_l = (double *)malloc( sizeof(double) * (l + 1) );
    c_l = (double *)malloc( sizeof(double) * l * (potential_degree+1) );
    for ( i = 0 ; i <= l ; i++ ) {
        xi_l[i] = a + (b - a) * i / l;
        while ( fabs( (e = i - l * (p->alpha[0] + xi_l[i]*(p->alpha[1] + xi_l[i]*p->alpha[2]))) ) > 3 * l * DBL_EPSILON )
            xi_l[i] += e / (l * (p->alpha[1] + 2*xi_l[i]*p->alpha[2]));
        }
    if ( potential_getcoeffs(f,fp,xi_l,l,c_l,&err_l) < 0 )
        return potential_err;
    /* fflush(stderr); printf("potential_init: err_l=%e.\n",err_l); */
        
    /* if this interpolation is good enough, stop here! */
    if ( err_l < tol ) {
        p->n = l;
        p->c = c_l;
        p->alpha[0] *= p->n; p->alpha[1] *= p->n; p->alpha[2] *= p->n;
        p->mi = (double *)malloc( sizeof(double) * l );
        p->hi = (double *)malloc( sizeof(double) * l );
        for ( k = 0 ; k < l ; k++ ) {
            p->mi[k] = 0.5 * (xi_l[k] + xi_l[k+1]);
            p->hi[k] = 2.0 / (xi_l[k+1] - xi_l[k]);
            }
        free(xi_l);
        return potential_err_ok;
        }
        
    /* loop until we have an upper bound on the right... */
    while ( 1 ) {
    
        /* compute the larger interpolation... */
        /* printf("potential_init: trying r=%i...\n",r); fflush(stdout); */
        xi_r = (double *)malloc( sizeof(double) * (r + 1) );
        c_r = (double *)malloc( sizeof(double) * r * (potential_degree+1) );
        for ( i = 0 ; i <= r ; i++ ) {
            xi_r[i] = a + (b - a) * i / r;
            while ( fabs( (e = i - r*(p->alpha[0] + xi_r[i]*(p->alpha[1] + xi_r[i]*p->alpha[2]))) ) > 3 * r * DBL_EPSILON )
                xi_r[i] += e / (r * (p->alpha[1] + 2*xi_r[i]*p->alpha[2]));
            }
        if ( potential_getcoeffs(f,fp,xi_r,r,c_r,&err_r) < 0 )
            return potential_err;
        /* printf("potential_init: err_r=%e.\n",err_r); fflush(stdout); */
            
        /* if this is better than tolerance, break... */
        if ( err_r < tol )
            break;
            
        /* otherwise, l=r and r = 2*r */
        else {
            l = r; err_l = err_r;
            free(xi_l); xi_l = xi_r;
            free(c_l); c_l = c_r;
            r *= 2;
            }

        } /* loop until we have a good right estimate */
        
    /* we now have a left and right estimate -- binary search! */
    while ( r - l > 1 ) {
    
        /* find the middle */
        m = 0.5 * ( r + l );
        
        /* construct that interpolation */
        /* printf("potential_init: trying m=%i...\n",m); fflush(stdout); */
        xi_m = (double *)malloc( sizeof(double) * (m + 1) );
        c_m = (double *)malloc( sizeof(double) * m * (potential_degree+1) );
        for ( i = 0 ; i <= m ; i++ ) {
            xi_m[i] = a + (b - a) * i / m;
            while ( fabs( (e = i - m*(p->alpha[0] + xi_m[i]*(p->alpha[1] + xi_m[i]*p->alpha[2]))) ) > 3 * m * DBL_EPSILON )
                xi_m[i] += e / (m * (p->alpha[1] + 2*xi_m[i]*p->alpha[2]));
            }
        if ( potential_getcoeffs(f,fp,xi_m,m,c_m,&err_m) < 0 )
            return potential_err;
        /* printf("potential_init: err_m=%e.\n",err_m); fflush(stdout); */
            
        /* go left? */
        if ( err_m > tol ) {
            l = m; err_l = err_m;
            free(xi_l); xi_l = xi_m;
            free(c_l); c_l = c_m;
            }
            
        /* otherwise, go right... */
        else {
            r = m; err_r = err_m;
            free(xi_r); xi_r = xi_m;
            free(c_r); c_r = c_m;
            }
                
        } /* binary search */
        
    /* as of here, the right estimate is the smallest interpolation below */
    /* the requested tolerance */
    p->n = r;
    p->c = c_r;
    p->alpha[0] *= p->n; p->alpha[1] *= p->n; p->alpha[2] *= p->n;
    p->mi = (double *)malloc( sizeof(double) * r );
    p->hi = (double *)malloc( sizeof(double) * r );
    for ( k = 0 ; k < r ; k++ ) {
        p->mi[k] = 0.5 * (xi_r[k] + xi_r[k+1]);
        p->hi[k] = 2.0 / (xi_r[k+1] - xi_r[k]);
        }
    free(xi_r);
    free(xi_l); free(c_l);
        
    /* all is well that ends well... */
    return potential_err_ok;
    
    }
    
    
/*////////////////////////////////////////////////////////////////////////////// */
/* int potential_getcoeffs */
//
/* compute the coefficients of the function 'f' with derivative 'fp' */
/* over the 'n' intervals between the 'xi' and store an estimate of the */
/* maximum absolute interpolation error in 'err'. */
/*////////////////////////////////////////////////////////////////////////////// */

int potential_getcoeffs ( double (*f)( double ) , double (*fp)( double ) , double *xi , int n , double *c , double *err ) {

    int i, j, k, ind;
    double phi[7], cee[6], fa, fb, dfa, dfb;
    double h, m, w, e, x;
    double fx[potential_N];

    /* check input sanity */
    if ( f == NULL || fp == NULL || xi == NULL || err == NULL )
        return potential_err = potential_err_null;
        
    /* init the maximum interpolation error */
    *err = 0.0;
    
    /* loop over all intervals... */
    for ( i = 0 ; i < n ; i++ ) {
    
        /* set the initial index */
        ind = i * (potential_degree + 1);
        
        /* get the interval centre and width */
        m = (xi[i] + xi[i+1]) / 2;
        h = (xi[i+1] - xi[i]) / 2;
        
        /* evaluate f and fp at the edges */
        fa = f(xi[i]); fb = f(xi[i+1]);
        dfa = fp(xi[i]) * h; dfb = fp(xi[i+1]) * h;
        
        /* compute the coefficients phi of f */
        for ( k = 0 ; k < potential_N ; k++ )
            fx[k] = f( m + h * cos( k * M_PI / potential_N ) );
        for ( j = 0 ; j < 7 ; j++ ) {
            phi[j] = (fa + (1-2*(j%2))*fb) / 2;
            for ( k = 1 ; k < potential_N ; k++ )
                phi[j] += fx[k] * cos( j * k * M_PI / potential_N );
            phi[j] *= 2.0 / potential_N;
            }
        
        /* compute the first four coefficients */
        cee[0] = (4*(fa + fb) + dfa - dfb) / 4;
        cee[1] = -(9*(fa - fb) + dfa + dfb) / 16;
        cee[2] = (dfb - dfa) / 8;
        cee[3] = (fa - fb + dfa + dfb) / 16;
        cee[4] = 0.0;
        cee[5] = 0.0;
        
        /* add the 4th correction... */
        w = ( 6 * ( cee[0] - phi[0]) - 4 * ( cee[2] - phi[2] ) - phi[4] ) / ( 36 + 16 + 1 );
        cee[0] += -6 * w;
        cee[2] += 4 * w;
        cee[4] = -w;

        /* add the 5th correction... */
        w = ( 2 * ( cee[1] - phi[1]) - 3 * ( cee[3] - phi[3] ) - phi[5] ) / ( 4 + 9 + 1 );
        cee[1] += -2 * w;
        cee[3] += 3 * w;
        cee[5] = -w;
        
        /* convert to monomials on the interval [-1,1] */
        c[ind+5] = cee[0]/2 - cee[2] + cee[4];
        c[ind+4] = cee[1] - 3*cee[3] + 5*cee[5];
        c[ind+3] = 2*cee[2] - 8*cee[4];
        c[ind+2] = 4*cee[3] - 20*cee[5];
        c[ind+1] = 8*cee[4];
        c[ind] = 16*cee[5];

        /* compute a local error estimate (klutzy) */
        for ( k = 0 ; k < potential_N ; k++ ) {
            x = cos( k * M_PI / potential_N );
            e = fabs( fx[k] - c[ind+5]
                -x * ( c[ind+4] + 
                x * ( c[ind+3] + 
                x * ( c[ind+2] + 
                x * ( c[ind+1] + 
                x * c[ind] )))) );
            if ( e > *err )
                *err = e;
            }
        
        }
        
    /* all is well that ends well... */
    return potential_err_ok;

    }
    
    
/*////////////////////////////////////////////////////////////////////////////// */
/* double potential_getastar */
//
/* compute the optimal mapping i = floor( n*(alpha x + (1-alpha )x^2 )) for */
/* the given function (i.e. its sixth derivative). */
/*////////////////////////////////////////////////////////////////////////////// */

double potential_getalpha ( double (*f6p)( double ) , double a , double b ) {

    double xi[potential_N], fx[potential_N];
    int i, j;
    double temp;
    double alpha[4], fa[4];
    const double golden = 2.0 / (1 + sqrt(5));
    
    /* start by evaluating f6p at the N nodes between 'a' and 'b' */
    for ( i = 0 ; i < potential_N ; i++ ) {
        xi[i] = a + (b-a) * i / (potential_N - 1);
        fx[i] = f6p( xi[i] );
        }
        
    /* set the initial values for alpha */
    alpha[0] = 0; alpha[3] = 2;
    alpha[1] = alpha[3] - 2 * golden; alpha[2] = alpha[0] + 2 * golden;
    for ( i = 0 ; i < 4 ; i++ ) {
        fa[i] = 0.0;
        for ( j = 0 ; j < potential_N ; j++ ) {
            temp = fabs( pow( alpha[i] + 2 * (1 - alpha[i]) * xi[j] , -6 ) * fx[j] );
            if ( temp > fa[i] )
                fa[i] = temp;
            }
        }
        
    /* main loop (brent's algorithm) */
    while ( alpha[3] - alpha[0] > 1.0e-4 ) {
    
        /* go west? */
        if ( fa[1] < fa[2] ) {
            alpha[3] = alpha[2]; fa[3] = fa[2];
            alpha[2] = alpha[1]; fa[2] = fa[1];
            alpha[1] = alpha[3] - (alpha[3] - alpha[0]) * golden;
            i = 1;
            }
            
        /* nope, go east... */
        else {
            alpha[0] = alpha[1]; fa[0] = fa[1];
            alpha[1] = alpha[2]; fa[1] = fa[2];
            alpha[2] = alpha[0] + (alpha[3] - alpha[0]) * golden;
            i = 2;
            }
            
        /* compute the new value */
        fa[i] = 0.0;
        for ( j = 0 ; j < potential_N ; j++ ) {
            temp = fabs( pow( alpha[i] + 2 * (1 - alpha[i]) * xi[j] , -6 ) * fx[j] );
            if ( temp > fa[i] )
                fa[i] = temp;
            }
    
        } /* main loop */
        
    /* return the average */
    return (alpha[0] + alpha[3]) / 2;

    }

