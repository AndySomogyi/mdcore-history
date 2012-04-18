/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU Lesser General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/

/* Include configuration header */
#include "../config.h"

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <limits.h>

/* Include some conditional headers. */
#include "../config.h"
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef HAVE_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "potential.h"
#include "cell.h"
#include "fifo.h"
#include "space.h"
#include "engine.h"
#include "rigid.h"


/* Global variables. */
/** The ID of the last error. */
int rigid_err = rigid_err_ok;
unsigned int rigid_rcount = 0;

/* the error macro. */
#define error(id)				( rigid_err = errs_register( id , rigid_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *rigid_err_msg[3] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory."
	};
    

/**
 * @brief Evaluate (SHAKE) a list of rigid constraints
 *
 * @param rs Pointer to an array of #rigid.
 * @param N Nr of rigids in @c r.
 * @param e Pointer to the #engine in which these rigids are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #rigid_err_ok or <0 on error (see #rigid_err)
 */
 
int rigid_eval_shake ( struct rigid *rs , int N , struct engine *e ) {

    int iter, rid, k, j, pid, pjd, nr_parts, nr_constr, shift;
    struct part *p[rigid_maxparts], **partlist;
    struct cell *c[rigid_maxparts], **celllist;
    struct rigid *r;
    FPTYPE dt, idt, xp[3*rigid_maxparts];
    FPTYPE m[rigid_maxparts], tol, lambda, w;
    FPTYPE vc[3*rigid_maxconstr], res[rigid_maxconstr], max_res, h[3];
    FPTYPE wvc[3*rigid_maxconstr];
    // FPTYPE vcom[3];

    /* Check for bad input. */
    partlist = e->s.partlist;
    celllist = e->s.celllist;
    tol = e->tol_rigid;
    dt = e->dt;
    idt = 1.0/dt;
    if ( rs == NULL || e == NULL )
        return error(rigid_err_null);
        
    /* Get some local values. */
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = e->s.h[k];
        
    /* Loop over the rigid constraints. */
    for ( rid = 0 ; rid < N ; rid++ ) {
    
        /* Get some local values we'll be re-usnig quite a bit. */
        r = &rs[rid];
        nr_parts = r->nr_parts;
        nr_constr = r->nr_constr;
    
        /* Check if the particles are local, if not bail. */
        for ( k = 0 ; k < nr_parts ; k++ ) {
            if ( ( p[k] = partlist[ r->parts[k] ] ) == NULL )
                break;
            c[k] = celllist[ r->parts[k] ];
            m[k] = e->types[ p[k]->type ].mass;
            }
        if ( k < nr_parts )
            continue;
            
        /* Are all the parts ghosts? */
        for ( k = 0 ; k < nr_parts && (p[k]->flags & part_flag_ghost) ; k++ );
        if ( k == nr_parts )
            continue;
            
        /* Load the particle positions relative to the first particle's cell. */
        for ( k = 0 ; k < nr_parts ; k++ )
            if ( c[k] != c[0] )
                for ( j = 0 ; j < 3 ; j++ ) {
                    shift = c[k]->loc[j] - c[0]->loc[j];
                    if ( shift > 1 )
                        shift = -1;
                    else if ( shift < -1 )
                        shift = 1;
                    xp[3*k+j] = p[k]->x[j] + h[j]*shift;
                    }
            else
                for ( j = 0 ; j < 3 ; j++ )
                    xp[3*k+j] = p[k]->x[j];
                    
        /* Create the gradient vectors. */
        for ( k = 0 ; k < nr_constr ; k++ ) {
            pid = r->constr[k].i;
            pjd = r->constr[k].j;
            for ( j = 0 ; j < 3 ; j++ ) {
                vc[k*3+j] = (xp[3*pid+j] - dt*p[pid]->v[j]) - (xp[3*pjd+j] - dt*p[pjd]->v[j]);
                wvc[k*3+j] = FPTYPE_ONE / ( m[pid] + m[pjd] ) * vc[3*k+j];
                }
            }
            
        /* for ( k = 0 ; k < nr_parts ; k++ )
            printf( "rigid_eval_shake: part %i (%i) at [ %e , %e , %e ].\n" , k , r->parts[k] , xp[3*k] , xp[3*k+1] , xp[3*k+2] );
        for ( k = 0 ; k < nr_constr ; k++ ) {
            pid = r->constr[k].i;
            pjd = r->constr[k].j;
            printf( "rigid_eval_shake: constr %i between parts %i and %i, d=%e.\n" , k , r->constr[k].i , r->constr[k].j , sqrt(r->constr[k].d2) );
            printf( "rigid_eval_shake: vc is [ %e , %e , %e ].\n" , vc[3*k] , vc[3*k+1] , vc[3*k+2] );
            printf( "rigid_eval_shake: dx is [ %e , %e , %e ].\n" , xp[3*pid]-xp[3*pjd] , xp[3*pid+1]-xp[3*pjd+1] , xp[3*pid+2]-xp[3*pjd+2] );
            } */
                    
        /* Main SHAKE loop. */
        for ( iter = 0 ; iter < rigid_maxiter ; iter++ ) {
        
            /* Compute the residues (squared). */
            for ( max_res = 0.0 , k = 0 ; k < nr_constr ; k++ ) {
                pid = r->constr[k].i;
                pjd = r->constr[k].j;
                res[k] = r->constr[k].d2;
                for ( j = 0 ; j < 3 ; j++ )
                    res[k] -= ( xp[3*pid+j] - xp[3*pjd+j] ) * ( xp[3*pid+j] - xp[3*pjd+j] );
                if ( fabs(res[k]) > max_res )
                    max_res = fabs(res[k]);
                }
                
            /* for ( k = 0 ; k < nr_constr ; k++ )
                printf( "rigid_eval_shake: res[%i]=%e.\n" , k , res[k] ); */
            
            /* Are we done? */
            if ( max_res < tol )
                break;
                
            /* Adjust each constraint. */
            for ( k = 0 ; k < nr_constr ; k++ ) {
                pid = r->constr[k].i;
                pjd = r->constr[k].j;
                lambda = 0.5 * res[k] / ( (xp[3*pid] - xp[3*pjd])*vc[3*k] + (xp[3*pid+1] - xp[3*pjd+1])*vc[3*k+1] + (xp[3*pid+2] - xp[3*pjd+2])*vc[3*k+2] );
                for ( j = 0 ; j < 3 ; j++ ) {
                    w = lambda * wvc[3*k+j];
                    xp[3*pid+j] += w * m[pjd];
                    xp[3*pjd+j] -= w * m[pid];
                    }
                }
        
            } /* Main SHAKE loop. */
        if ( iter == rigid_maxiter ) {
            printf( "rigid_eval_shake: rigid %i failed to converge in less than %i iterations.\n" , rid , rigid_maxiter );
            for ( k = 0 ; k < nr_constr ; k++ ) {
                pid = r->constr[k].i;
                pjd = r->constr[k].j;
                printf( "rigid_eval_shake: constr %i between parts %i and %i, d=%e.\n" , k , r->parts[pid] , r->parts[pjd] , sqrt(r->constr[k].d2) );
                printf( "rigid_eval_shake: res[%i]=%e.\n" , k , res[k] );
                }
            }
            
            
        /* for ( k = 0 ; k < nr_parts ; k++ )
            printf( "rigid_eval_shake: part %i at [ %e , %e , %e ].\n" , k , xp[3*k] , xp[3*k+1] , xp[3*k+2] );
        fflush(stdout); getchar(); */
            
        /* Set the new (corrected) particle positions and velocities. */
        /* for ( k = 0 ; k < nr_parts ; k++ )
            printf( "rigid_eval_shake: part %i has v=[ %e , %e , %e ].\n" , k , p[k]->v[0] , p[k]->v[1] , p[k]->v[2] ); */
        /* vcom[0] = 0.0; vcom[1] = 0.0; vcom[2] = 0.0;
        for ( k = 0 ; k < nr_parts ; k++ )
            for ( j = 0 ; j < 3 ; j++ )
                vcom[j] += p[k]->v[j] * e->types[p[k]->type].mass;
        printf( "rigid_eval_shake: old vcom is [ %e , %e , %e ].\n" , vcom[0] , vcom[1] , vcom[2] ); */
        for ( k = 0 ; k < nr_parts ; k++ ) {
            if ( c[k] != c[0] )
                for ( j = 0 ; j < 3 ; j++ ) {
                    shift = c[k]->loc[j] - c[0]->loc[j];
                    if ( shift > 1 )
                        shift = -1;
                    else if ( shift < -1 )
                        shift = 1;
                    p[k]->v[j] += idt * ( xp[3*k+j] - h[j]*shift - p[k]->x[j] );
                    p[k]->x[j] = xp[3*k+j] - h[j]*shift;
                    }
            else
                for ( j = 0 ; j < 3 ; j++ ) {
                    p[k]->v[j] += idt * ( xp[3*k+j] - p[k]->x[j] );
                    p[k]->x[j] = xp[3*k+j];
                    }
            }
        /* for ( k = 0 ; k < nr_parts ; k++ )
            printf( "rigid_eval_shake: part %i has ||v||=%e.\n" , k ,
                sqrt( p[k]->v[0]*p[k]->v[0] + p[k]->v[1]*p[k]->v[1] + p[k]->v[2]*p[k]->v[2] ) ); */
        /* fflush(stdout); getchar(); */
        /* for ( k = 0 ; k < nr_parts ; k++ ) {
            for ( j = 0 ; j < 3 ; j++ )
                vcom[j] -= p[k]->v[j] * m[k];
            }
        printf( "rigid_eval_shake: shift in vcom is %e.\n" , sqrt( vcom[0]*vcom[0] + vcom[1]*vcom[1] + vcom[2]*vcom[2] ) );
        for ( k = 0 ; k < nr_parts ; k++ )
            printf( "rigid_eval_shake: part %i (%s) has id=%i, mass=%e (%e).\n" , k , e->types[p[k]->type].name , p[k]->id , m[k] , e->types[p[k]->type].mass );
        fflush(stdout); getchar(); */
    
        } /* Loop over the constraints. */
        
    /* Bail quitely. */
    return rigid_err_ok;
        
    }
    
    
    
    
    
    
    
    
