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
#include "angle.h"


/* Global variables. */
/** The ID of the last error. */
int angle_err = angle_err_ok;
unsigned int angle_rcount = 0;

/* the error macro. */
#define error(id)				( angle_err = errs_register( id , angle_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *angle_err_msg[2] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered."
	};
    

/**
 * @brief Evaluate a list of angleed interactions
 *
 * @param b Pointer to an array of #angle.
 * @param N Nr of angles in @c b.
 * @param nr_threads Number of computational threads.
 * @param cid_div #cell id modulus.
 * @param e Pointer to the #engine in which these angles are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #angle_err_ok or <0 on error (see #angle_err)
 *
 * Computes only the interactions on particles inside cells @c c where
 * @c c->id % nr_threads == cid_div.
 */
 
int angle_eval_div ( struct angle *a , int N , int nr_threads , int cid_div , struct engine *e , double *epot_out ) {

    int aid, pid, pjd, pkd, cid, cjd = 0, ckd = 0, k, *loci, *locj, *lock, shift;
    double h[3], epot = 0.0;
    struct space *s;
    struct part *pi, *pj, *pk, **partlist;
    struct cell **celllist, *ci, *cj, *ck;
    struct potential *pot;
    FPTYPE xi[3], xj[3], xk[3], dxi[3] , dxk[3], ctheta, wi, wk, incr;
    register FPTYPE t1, t10, t11, t12, t13, t21, t22, t23, t24, t25, t26, t27, t3,
        t5, t6, t7, t8, t9, t4, t14, t2;
    struct potential **pots;
#if defined(VECTORIZE)
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE dummy = 0.0;
    FPTYPE *effi[4], *effj[4], *effk[4];
    FPTYPE cthetaq[4] __attribute__ ((aligned (16)));
    FPTYPE ee[4] __attribute__ ((aligned (16)));
    FPTYPE eff[4] __attribute__ ((aligned (16)));
    FPTYPE diq[12], dkq[12];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( a == NULL || e == NULL )
        return error(angle_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    incr = ((double)nr_threads) / s->nr_real;
    pots = e->p_angle;
    partlist = s->partlist;
    celllist = s->celllist;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the angles. */
    for ( aid = 0 ; aid < N ; aid++ ) {
    
        /* Do we own this bond? */
        pid = a[aid].i; pjd = a[aid].j; pkd = a[aid].k;
        if ( ( ci = celllist[ pid ] ) == NULL ||
             ( cj = celllist[ pjd ] ) == NULL ||
             ( ck = celllist[ pkd ] ) == NULL )
            continue;

        /* Skip if both parts are in the wrong cell. */
        cid = ci->id * incr;
        cjd = cj->id * incr;
        ckd = ck->id * incr;
        if ( cid != cid_div && cjd != cid_div && ckd != cid_div )
            continue;
            
        /* Get the potential. */
        if ( ( pot = pots[ a[aid].pid ] ) == NULL )
            continue;
    
        /* Get the particles involved. */
        pi = partlist[ pid ];
        pj = partlist[ pjd ];
        pk = partlist[ pkd ];
        
        /* get the particle positions relative to pj's cell. */
        loci = ci->loc;
        locj = cj->loc;
        lock = ck->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            xj[k] = pj->x[k];
            shift = loci[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xi[k] = pi->x[k] + shift*h[k];
            shift = lock[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xk[k] = pk->x[k] + shift*h[k];
            }
            
        /* This is Maple-generated code, see "angles.maple" for details. */
        t2 = xj[2]*xj[2];
        t4 = xj[1]*xj[1];
        t14 = xj[0]*xj[0];
        t21 = t2+t4+t14;
        t24 = -FPTYPE_TWO*xj[2];
        t25 = -FPTYPE_TWO*xj[1];
        t26 = -FPTYPE_TWO*xj[0];
        t6 = (t24+xi[2])*xi[2]+(t25+xi[1])*xi[1]+(t26+xi[0])*xi[0]+t21;
        t3 = FPTYPE_ONE/sqrt(t6);
        t10 = xk[0]-xj[0];
        t11 = xi[2]-xj[2];
        t12 = xi[1]-xj[1];
        t13 = xi[0]-xj[0];
        t8 = xk[2]-xj[2];
        t9 = xk[1]-xj[1];
        t7 = t13*t10+t12*t9+t11*t8;
        t27 = t3*t7;
        t5 = (t24+xk[2])*xk[2]+(t25+xk[1])*xk[1]+(t26+xk[0])*xk[0]+t21;
        t1 = FPTYPE_ONE/sqrt(t5);
        t23 = t1/t5*t7;
        t22 = FPTYPE_ONE/t6*t27;
        dxi[0] = (t10*t3-t13*t22)*t1;
        dxi[1] = (t9*t3-t12*t22)*t1;
        dxi[2] = (t8*t3-t11*t22)*t1;
        dxk[0] = (t13*t1-t10*t23)*t3;
        dxk[1] = (t12*t1-t9*t23)*t3;
        dxk[2] = (t11*t1-t8*t23)*t3;
        ctheta = FPTYPE_FMAX( -FPTYPE_ONE , FPTYPE_FMIN( FPTYPE_ONE , t1*t27 ) );
        
        /* printf( "angle_eval: cos of angle %i (%s-%s-%s) is %e.\n" , aid ,
            e->types[pi->type].name , e->types[pj->type].name , e->types[pk->type].name , ctheta ); */
        /* printf( "angle_eval: ids are ( %i , %i , %i ).\n" , pi->id , pj->id , pk->id );
        if ( e->s.celllist[pid] != e->s.celllist[pjd] )
            printf( "angle_eval: pi and pj are in different cells!\n" );
        if ( e->s.celllist[pkd] != e->s.celllist[pjd] )
            printf( "angle_eval: pk and pj are in different cells!\n" );
        printf( "angle_eval: xi-xj is [ %e , %e , %e ], ||xi-xj||=%e.\n" ,
            xi[0]-xj[0] , xi[1]-xj[1] , xi[2]-xj[2] , sqrt( (xi[0]-xj[0])*(xi[0]-xj[0]) + (xi[1]-xj[1])*(xi[1]-xj[1]) + (xi[2]-xj[2])*(xi[2]-xj[2]) ) );
        printf( "angle_eval: xk-xj is [ %e , %e , %e ], ||xk-xj||=%e.\n" ,
            xk[0]-xj[0] , xk[1]-xj[1] , xk[2]-xj[2] , sqrt( (xk[0]-xj[0])*(xk[0]-xj[0]) + (xk[1]-xj[1])*(xk[1]-xj[1]) + (xk[2]-xj[2])*(xk[2]-xj[2]) ) ); */
        /* printf( "angle_eval: dxi is [ %e , %e , %e ], ||dxi||=%e.\n" ,
            dxi[0] , dxi[1] , dxi[2] , sqrt( dxi[0]*dxi[0] + dxi[1]*dxi[1] + dxi[2]*dxi[2] ) );
        printf( "angle_eval: dxk is [ %e , %e , %e ], ||dxk||=%e.\n" ,
            dxk[0] , dxk[1] , dxk[2] , sqrt( dxk[0]*dxk[0] + dxk[1]*dxk[1] + dxk[2]*dxk[2] ) ); */
        if ( ctheta < pot->a || ctheta > pot->b ) {
            printf( "angle_eval[%i]: angle %i (%s-%s-%s) out of range [%e,%e], ctheta=%e.\n" ,
                e->nodeID , aid , e->types[pi->type].name , e->types[pj->type].name , e->types[pk->type].name , pot->a , pot->b , ctheta );
            ctheta = fmax( pot->a , fmin( pot->b , ctheta ) );
            }

        #ifdef VECTORIZE
            /* add this angle to the interaction queue. */
            cthetaq[icount] = ctheta;
            diq[icount*3] = dxi[0];
            diq[icount*3+1] = dxi[1];
            diq[icount*3+2] = dxi[2];
            dkq[icount*3] = dxk[0];
            dkq[icount*3+1] = dxk[1];
            dkq[icount*3+2] = dxk[2];
            effi[icount] = ( cid == cid_div ? pi->f : &dummy );
            effj[icount] = ( cjd == cid_div ? pj->f : &dummy );
            effk[icount] = ( ckd == cid_div ? pk->f : &dummy );
            potq[icount] = pot;
            icount += 1;

            #if defined(FPTYPE_SINGLE)
                /* evaluate the angles if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4single_r( potq , cthetaq , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                            effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                            effj[l][k] += wi + wk;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
            #elif defined(FPTYPE_DOUBLE)
                /* evaluate the angles if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4double_r( potq , cthetaq , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                            effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                            effj[l][k] += wi + wk;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
            #endif
        #else
            /* evaluate the angle */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , ctheta , &ee , &eff );
            #else
                potential_eval_r( pot , ctheta , &ee , &eff );
            #endif
            
            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                if ( cid == cid_div )
                    pi->f[k] -= ( wi = eff * dxi[k] );
                if ( ckd == cid_div )
                    pk->f[k] -= ( wk = eff * dxk[k] );
                if ( cjd == cid_div )
                    pj->f[k] += eff * ( dxi[k] + dxk[k] );
                }

            /* tabulate the energy */
            epot += ee;
        #endif
        
        } /* loop over angles. */
        
        
    #if defined(VEC_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single_r( potq , cthetaq , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                    effj[l][k] += wi + wk;
                    }
                }

            }
    #elif defined(VEC_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4double_r( potq , cthetaq , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                    effj[l][k] += wi + wk;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return angle_err_ok;
    
    }


/**
 * @brief Evaluate a list of angleed interactions
 *
 * @param b Pointer to an array of #angle.
 * @param N Nr of angles in @c b.
 * @param e Pointer to the #engine in which these angles are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #angle_err_ok or <0 on error (see #angle_err)
 */
 
int angle_eval ( struct angle *a , int N , struct engine *e , double *epot_out ) {

    int aid, pid, pjd, pkd, k, *loci, *locj, *lock, shift;
    double h[3], epot = 0.0;
    struct space *s;
    struct part *pi, *pj, *pk, **partlist;
    struct cell **celllist;
    struct potential *pot;
    FPTYPE xi[3], xj[3], xk[3], dxi[3] , dxk[3], ctheta, wi, wk;
    register FPTYPE t1, t10, t11, t12, t13, t21, t22, t23, t24, t25, t26, t27, t3,
        t5, t6, t7, t8, t9, t4, t14, t2;
    struct potential **pots;
#if defined(VECTORIZE)
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE *effi[4], *effj[4], *effk[4];
    FPTYPE cthetaq[4] __attribute__ ((aligned (16)));
    FPTYPE ee[4] __attribute__ ((aligned (16)));
    FPTYPE eff[4] __attribute__ ((aligned (16)));
    FPTYPE diq[12], dkq[12];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( a == NULL || e == NULL )
        return error(angle_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p_angle;
    partlist = s->partlist;
    celllist = s->celllist;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the angles. */
    for ( aid = 0 ; aid < N ; aid++ ) {
    
        /* Get the particles involved. */
        pid = a[aid].i; pjd = a[aid].j; pkd = a[aid].k;
        if ( ( pi = partlist[ pid] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        if ( ( pk = partlist[ pkd ] ) == NULL )
            continue;
        
        /* Skip if all three are ghosts. */
        if ( ( pi->flags & part_flag_ghost ) && ( pj->flags & part_flag_ghost ) && ( pk->flags & part_flag_ghost ) )
            continue;
            
        /* Get the potential. */
        if ( ( pot = pots[ a[aid].pid ] ) == NULL )
            continue;
    
        /* get the particle positions relative to pj's cell. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            xj[k] = pj->x[k];
            shift = loci[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xi[k] = pi->x[k] + shift*h[k];
            shift = lock[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xk[k] = pk->x[k] + shift*h[k];
            }
            
        /* This is Maple-generated code, see "angles.maple" for details. */
        t2 = xj[2]*xj[2];
        t4 = xj[1]*xj[1];
        t14 = xj[0]*xj[0];
        t21 = t2+t4+t14;
        t24 = -FPTYPE_TWO*xj[2];
        t25 = -FPTYPE_TWO*xj[1];
        t26 = -FPTYPE_TWO*xj[0];
        t6 = (t24+xi[2])*xi[2]+(t25+xi[1])*xi[1]+(t26+xi[0])*xi[0]+t21;
        t3 = FPTYPE_ONE/sqrt(t6);
        t10 = xk[0]-xj[0];
        t11 = xi[2]-xj[2];
        t12 = xi[1]-xj[1];
        t13 = xi[0]-xj[0];
        t8 = xk[2]-xj[2];
        t9 = xk[1]-xj[1];
        t7 = t13*t10+t12*t9+t11*t8;
        t27 = t3*t7;
        t5 = (t24+xk[2])*xk[2]+(t25+xk[1])*xk[1]+(t26+xk[0])*xk[0]+t21;
        t1 = FPTYPE_ONE/sqrt(t5);
        t23 = t1/t5*t7;
        t22 = FPTYPE_ONE/t6*t27;
        dxi[0] = (t10*t3-t13*t22)*t1;
        dxi[1] = (t9*t3-t12*t22)*t1;
        dxi[2] = (t8*t3-t11*t22)*t1;
        dxk[0] = (t13*t1-t10*t23)*t3;
        dxk[1] = (t12*t1-t9*t23)*t3;
        dxk[2] = (t11*t1-t8*t23)*t3;
        ctheta = FPTYPE_FMAX( -FPTYPE_ONE , FPTYPE_FMIN( FPTYPE_ONE , t1*t27 ) );
        
        /* printf( "angle_eval: cos of angle %i (%s-%s-%s) is %e.\n" , aid ,
            e->types[pi->type].name , e->types[pj->type].name , e->types[pk->type].name , ctheta ); */
        /* printf( "angle_eval: ids are ( %i , %i , %i ).\n" , pi->id , pj->id , pk->id );
        if ( e->s.celllist[pid] != e->s.celllist[pjd] )
            printf( "angle_eval: pi and pj are in different cells!\n" );
        if ( e->s.celllist[pkd] != e->s.celllist[pjd] )
            printf( "angle_eval: pk and pj are in different cells!\n" );
        printf( "angle_eval: xi-xj is [ %e , %e , %e ], ||xi-xj||=%e.\n" ,
            xi[0]-xj[0] , xi[1]-xj[1] , xi[2]-xj[2] , sqrt( (xi[0]-xj[0])*(xi[0]-xj[0]) + (xi[1]-xj[1])*(xi[1]-xj[1]) + (xi[2]-xj[2])*(xi[2]-xj[2]) ) );
        printf( "angle_eval: xk-xj is [ %e , %e , %e ], ||xk-xj||=%e.\n" ,
            xk[0]-xj[0] , xk[1]-xj[1] , xk[2]-xj[2] , sqrt( (xk[0]-xj[0])*(xk[0]-xj[0]) + (xk[1]-xj[1])*(xk[1]-xj[1]) + (xk[2]-xj[2])*(xk[2]-xj[2]) ) ); */
        /* printf( "angle_eval: dxi is [ %e , %e , %e ], ||dxi||=%e.\n" ,
            dxi[0] , dxi[1] , dxi[2] , sqrt( dxi[0]*dxi[0] + dxi[1]*dxi[1] + dxi[2]*dxi[2] ) );
        printf( "angle_eval: dxk is [ %e , %e , %e ], ||dxk||=%e.\n" ,
            dxk[0] , dxk[1] , dxk[2] , sqrt( dxk[0]*dxk[0] + dxk[1]*dxk[1] + dxk[2]*dxk[2] ) ); */
        if ( ctheta < pot->a || ctheta > pot->b ) {
            printf( "angle_eval[%i]: angle %i (%s-%s-%s) out of range [%e,%e], ctheta=%e.\n" ,
                e->nodeID , aid , e->types[pi->type].name , e->types[pj->type].name , e->types[pk->type].name , pot->a , pot->b , ctheta );
            ctheta = fmax( pot->a , fmin( pot->b , ctheta ) );
            }

        #ifdef VECTORIZE
            /* add this angle to the interaction queue. */
            cthetaq[icount] = ctheta;
            diq[icount*3] = dxi[0];
            diq[icount*3+1] = dxi[1];
            diq[icount*3+2] = dxi[2];
            dkq[icount*3] = dxk[0];
            dkq[icount*3+1] = dxk[1];
            dkq[icount*3+2] = dxk[2];
            effi[icount] = pi->f;
            effj[icount] = pj->f;
            effk[icount] = pk->f;
            potq[icount] = pot;
            icount += 1;

            #if defined(FPTYPE_SINGLE)
                /* evaluate the angles if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4single_r( potq , cthetaq , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                            effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                            effj[l][k] += wi + wk;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
            #elif defined(FPTYPE_DOUBLE)
                /* evaluate the angles if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4double_r( potq , cthetaq , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                            effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                            effj[l][k] += wi + wk;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
            #endif
        #else
            /* evaluate the angle */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , ctheta , &ee , &eff );
            #else
                potential_eval_r( pot , ctheta , &ee , &eff );
            #endif
            
            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                pi->f[k] -= ( wi = eff[l] * dxi[k] );
                pk->f[k] -= ( wk = eff[l] * dxk[k] );
                pj->f[k] += wi + wk;
                }

            /* tabulate the energy */
            epot += ee;
        #endif
        
        } /* loop over angles. */
        
        
    #if defined(VEC_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single_r( potq , cthetaq , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                    effj[l][k] += wi + wk;
                    }
                }

            }
    #elif defined(VEC_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4double_r( potq , cthetaq , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                    effj[l][k] += wi + wk;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return angle_err_ok;
    
    }


/**
 * @brief Evaluate a list of angleed interactions
 *
 * @param b Pointer to an array of #angle.
 * @param N Nr of angles in @c b.
 * @param e Pointer to the #engine in which these angles are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 *
 * This function differs from #angle_eval in that the forces are added to
 * the array @c f instead of directly in the particle data.
 * 
 * @return #angle_err_ok or <0 on error (see #angle_err)
 */
 
int angle_evalf ( struct angle *a , int N , struct engine *e , FPTYPE *f , double *epot_out ) {

    int aid, pid, pjd, pkd, k, *loci, *locj, *lock, shift;
    double h[3], epot = 0.0;
    struct space *s;
    struct part *pi, *pj, *pk, **partlist;
    struct cell **celllist;
    struct potential *pot;
    FPTYPE xi[3], xj[3], xk[3], dxi[3] , dxk[3], ctheta, wi, wk;
    register FPTYPE t1, t10, t11, t12, t13, t21, t22, t23, t24, t25, t26, t27, t3,
        t5, t6, t7, t8, t9, t4, t14, t2;
    struct potential **pots;
#if defined(VECTORIZE)
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE *effi[4], *effj[4], *effk[4];
    FPTYPE cthetaq[4] __attribute__ ((aligned (16)));
    FPTYPE ee[4] __attribute__ ((aligned (16)));
    FPTYPE eff[4] __attribute__ ((aligned (16)));
    FPTYPE diq[12], dkq[12];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( a == NULL || e == NULL )
        return error(angle_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p_angle;
    partlist = s->partlist;
    celllist = s->celllist;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the angles. */
    for ( aid = 0 ; aid < N ; aid++ ) {
    
        /* Get the particles involved. */
        pid = a[aid].i; pjd = a[aid].j; pkd = a[aid].k;
        if ( ( pi = partlist[ pid] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        if ( ( pk = partlist[ pkd ] ) == NULL )
            continue;
        
        /* Skip if all three are ghosts. */
        if ( ( pi->flags & part_flag_ghost ) && ( pj->flags & part_flag_ghost ) && ( pk->flags & part_flag_ghost ) )
            continue;
            
        /* Get the potential. */
        if ( ( pot = pots[ a[aid].pid ] ) == NULL )
            continue;
    
        /* get the particle positions relative to pj's cell. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            xj[k] = pj->x[k];
            shift = loci[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xi[k] = pi->x[k] + h[k]*shift;
            shift = lock[k] - locj[k];
            if ( shift > 1 )
                shift = -1;
            else if ( shift < -1 )
                shift = 1;
            xk[k] = pk->x[k] + h[k]*shift;
            }
            
        /* This is Maple-generated code, see "angles.maple" for details. */
        t2 = xj[2]*xj[2];
        t4 = xj[1]*xj[1];
        t14 = xj[0]*xj[0];
        t21 = t2+t4+t14;
        t24 = -FPTYPE_TWO*xj[2];
        t25 = -FPTYPE_TWO*xj[1];
        t26 = -FPTYPE_TWO*xj[0];
        t6 = (t24+xi[2])*xi[2]+(t25+xi[1])*xi[1]+(t26+xi[0])*xi[0]+t21;
        t3 = FPTYPE_ONE/sqrt(t6);
        t10 = xk[0]-xj[0];
        t11 = xi[2]-xj[2];
        t12 = xi[1]-xj[1];
        t13 = xi[0]-xj[0];
        t8 = xk[2]-xj[2];
        t9 = xk[1]-xj[1];
        t7 = t13*t10+t12*t9+t11*t8;
        t27 = t3*t7;
        t5 = (t24+xk[2])*xk[2]+(t25+xk[1])*xk[1]+(t26+xk[0])*xk[0]+t21;
        t1 = FPTYPE_ONE/sqrt(t5);
        t23 = t1/t5*t7;
        t22 = FPTYPE_ONE/t6*t27;
        dxi[0] = (t10*t3-t13*t22)*t1;
        dxi[1] = (t9*t3-t12*t22)*t1;
        dxi[2] = (t8*t3-t11*t22)*t1;
        dxk[0] = (t13*t1-t10*t23)*t3;
        dxk[1] = (t12*t1-t9*t23)*t3;
        dxk[2] = (t11*t1-t8*t23)*t3;
        ctheta = FPTYPE_FMAX( -FPTYPE_ONE , FPTYPE_FMIN( FPTYPE_ONE , t1*t27 ) );
        
        /* printf( "angle_eval: angle %i is %e rad.\n" , aid , ctheta ); */
        if ( ctheta < pot->a || ctheta > pot->b ) {
            printf( "angle_evalf: angle %i (%s-%s-%s) out of range [%e,%e], ctheta=%e.\n" ,
                aid , e->types[pi->type].name , e->types[pj->type].name , e->types[pk->type].name , pot->a , pot->b , ctheta );
            ctheta = fmax( pot->a , fmin( pot->b , ctheta ) );
            }

        #ifdef VECTORIZE
            /* add this angle to the interaction queue. */
            cthetaq[icount] = ctheta;
            diq[icount*3] = dxi[0];
            diq[icount*3+1] = dxi[1];
            diq[icount*3+2] = dxi[2];
            dkq[icount*3] = dxk[0];
            dkq[icount*3+1] = dxk[1];
            dkq[icount*3+2] = dxk[2];
            effi[icount] = &f[ 4*pid ];
            effj[icount] = &f[ 4*pjd ];
            effk[icount] = &f[ 4*pkd ];
            potq[icount] = pot;
            icount += 1;

            #if defined(FPTYPE_SINGLE)
                /* evaluate the angles if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4single_r( potq , cthetaq , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                            effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                            effj[l][k] += wi + wk;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
            #elif defined(FPTYPE_DOUBLE)
                /* evaluate the angles if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4double_r( potq , cthetaq , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                            effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                            effj[l][k] += wi + wk;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
            #endif
        #else
            /* evaluate the angle */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , ctheta , &ee , &eff );
            #else
                potential_eval_r( pot , ctheta , &ee , &eff );
            #endif
            
            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                f[4*pid+k] -= ( wi = eff * dxi[k] );
                f[4*pkd+k] -= ( wk = eff * dxk[k] );
                f[4*pjd+k] += wi + wk;
                }

            /* tabulate the energy */
            epot += ee;
        #endif
        
        } /* loop over angles. */
        
    #if defined(VEC_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single_r( potq , cthetaq , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                    effj[l][k] += wi + wk;
                    }
                }

            }
    #elif defined(VEC_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                cthetaq[k] = cthetaq[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4double_r( potq , cthetaq , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    effi[l][k] -= ( wi = eff[l] * diq[3*l+k] );
                    effk[l][k] -= ( wk = eff[l] * dkq[3*l+k] );
                    effj[l][k] += wi + wk;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return angle_err_ok;
    
    }



