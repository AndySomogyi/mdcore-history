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
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "potential.h"
#include "cell.h"
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
 * @brief Evaluate a list of angleed interactoins
 *
 * @param b Pointer to an array of #angle.
 * @param N Nr of angles in @c b.
 * @param e Pointer to the #engine in which these angles are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #angle_err_ok or <0 on error (see #angle_err)
 */
 
int angle_eval ( struct angle *a , int N , struct engine *e , double *epot_out ) {

    int aid, pid, pjd, pkd, k, *loci, *locj, *lock, shifti[3], shiftk[3], emt;
    double h[3], epot = 0.0;
    struct space *s;
    struct part *pi, *pj, *pk, **partlist;
    struct cell **celllist;
    struct potential *pot;
    FPTYPE vij[3], vkj[3], dx[3], dprod, vij2, vkj2, invij, invkj, invijk, ctheta, r2;
    FPTYPE w, wi, wk;
    struct potential **pots, **pots_nb;
#if defined(VECTORIZE)
    struct potential *potq[4], *potq_nb[4];
    int icount = 0, icount_nb = 0, l;
    FPTYPE *effi[4], *effj[4], *effk[4];
    FPTYPE *effi_nb[4], *effk_nb[4];
    FPTYPE cthetaq[4] __attribute__ ((aligned (16)));
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE ee[4] __attribute__ ((aligned (16)));
    FPTYPE eff[4] __attribute__ ((aligned (16)));
    FPTYPE dijq[12], dkjq[12], dxq[12];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( a == NULL || e == NULL )
        return error(angle_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p_angle;
    pots_nb = e->p;
    emt = e->max_type;
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
    
        /* get the angle rays vij and vkj. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            shifti[k] = loci[k] - locj[k];
            if ( shifti[k] > 1 )
                shifti[k] = -1;
            else if ( shifti[k] < -1 )
                shifti[k] = 1;
            shiftk[k] = lock[k] - locj[k];
            if ( shiftk[k] > 1 )
                shiftk[k] = -1;
            else if ( shiftk[k] < -1 )
                shiftk[k] = 1;
            }
        vij2 = 0.0; vkj2 = 0.0; dprod = 0.0; r2 = 0.0;
        for ( k = 0 ; k < 3 ; k++ ) {
            vij[k] = pi->x[k] - pj->x[k] + shifti[k]*h[k];
            vkj[k] = pk->x[k] - pj->x[k] + shiftk[k]*h[k];
            vij2 += vij[k] * vij[k];
            vkj2 += vkj[k] * vkj[k];
            dprod += vij[k] * vkj[k];
            dx[k] = vij[k] - vkj[k];
            r2 += dx[k] * dx[k];
            }
        invij = 1.0 / sqrt(vij2);
        invkj = 1.0 / sqrt(vkj2);
        invijk = invij * invkj;
        wi = dprod * invij * invij;
        wk = dprod * invkj * invkj;
        ctheta = dprod * invijk;

        /* printf( "angle_eval: angle %i is %e rad.\n" , aid , ctheta );
        if ( ctheta < pot->a || ctheta > pot->b )
            printf( "angle_eval: angle %i out of range, ctheta=%e.\n" , aid , ctheta ); */

        #ifdef VECTORIZE
            /* add this angle to the interaction queue. */
            cthetaq[icount] = dprod * invijk;
            dijq[icount*3] = invijk * vkj[0] - wi * vij[0];
            dijq[icount*3+1] = invijk * vkj[1] - wi * vij[1];
            dijq[icount*3+2] = invijk * vkj[2] - wi * vij[2];
            dkjq[icount*3] = invijk * vij[0] - wk * vkj[0];
            dkjq[icount*3+1] = invijk * vij[1] - wk * vkj[1];
            dkjq[icount*3+2] = invijk * vij[2] - wk * vkj[2];
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
                            wi = eff[l] * dijq[l*3+k]; wk = eff[l] * dkjq[l*3+k];
                            effi[l][k] += wi;
                            effk[l][k] += wk;
                            effj[l][k] -= wi + wk;
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
                            wi = eff[l] * dijq[l*3+k]; wk = eff[l] * dkjq[l*3+k];
                            effi[l][k] += wi;
                            effk[l][k] += wk;
                            effj[l][k] -= wi + wk;
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
                wi = eff * ( invijk * vkj[k] - wi * vij[k] );
                wk = eff * ( invijk * vij[k] - wk * vkj[k] );
                pi->f[k] += wi;
                pk->f[k] += wk;
                pj->f[k] -= wi + wk;
                }

            /* tabulate the energy */
            epot += ee;
        #endif
        
        
        /* Do we have a non-bonded interaction between i and k? */
        if ( ( pot = pots_nb[ pi->type * emt + pk->type ] ) == NULL )
            continue;
        
        #ifdef VECTORIZE
            /* add this bond to the interaction queue. */
            r2q[icount_nb] = r2;
            dxq[icount_nb*3] = dx[0];
            dxq[icount_nb*3+1] = dx[1];
            dxq[icount_nb*3+2] = dx[2];
            effi_nb[icount_nb] = pi->f;
            effk_nb[icount_nb] = pk->f;
            potq_nb[icount_nb] = pot;
            icount_nb += 1;
            
            #if defined(FPTYPE_SINGLE)
                /* evaluate the bonds if the queue is full. */
                if ( icount_nb == 4 ) {
                    
                    potential_eval_vec_4single( potq_nb , r2q , ee , eff );
                    
                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot -= ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = eff[l] * dxq[l*3+k];
                            effi_nb[l][k] += w;
                            effk_nb[l][k] -= w;
                            }
                        }
                    
                    /* re-set the counter. */
                    icount_nb = 0;
                    
                    }
            #elif defined(FPTYPE_DOUBLE)
                /* evaluate the bonds if the queue is full. */
                if ( icount_nb == 4 ) {
                    
                    potential_eval_vec_4double( potq_nb , r2q , ee , eff );
                    
                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot -= ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = eff[l] * dxq[l*3+k];
                            effi_nb[l][k] += w;
                            effk_nb[l][k] -= w;
                            }
                        }
                    
                    /* re-set the counter. */
                    icount_nb = 0;
                    
                    }
            #endif
        #else
            /* evaluate the bond */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , r2 , &ee , &eff );
            #else
                potential_eval( pot , r2 , &ee , &eff );
            #endif
            
            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = eff * dx[k];
                pi->f[k] += w;
                pk->f[k] -= w;
                }

            /* tabulate the energy */
            epot -= ee;
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
                    wi = eff[l] * dijq[l*3+k]; wk = eff[l] * dkjq[l*3+k];
                    effi[l][k] += wi;
                    effk[l][k] += wk;
                    effj[l][k] -= wi + wk;
                    }
                }

            }
        if ( icount_nb > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount_nb ; k < 4 ; k++ ) {
                potq_nb[k] = potq_nb[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single( potq_nb , r2q , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount_nb ; l++ ) {
                epot -= ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi_nb[l][k] += w;
                    effk_nb[l][k] -= w;
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
                    wi = eff[l] * dijq[l*3+k]; wk = eff[l] * dkjq[l*3+k];
                    effi[l][k] += wi;
                    effk[l][k] += wk;
                    effj[l][k] -= wi + wk;
                    }
                }

            }
        if ( icount_nb > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount_nb ; k < 4 ; k++ ) {
                potq_nb[k] = potq_nb[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4double( potq_nb , r2q , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot -= ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi_nb[l][k] += w;
                    effk_nb[l][k] -= w;
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
 * @brief Evaluate a list of angleed interactoins
 *
 * @param b Pointer to an array of #angle.
 * @param N Nr of angles in @c b.
 * @param e Pointer to the #engine in which these angles are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #angle_err_ok or <0 on error (see #angle_err)
 */
 
int angle_evalf ( struct angle *a , int N , struct engine *e , FPTYPE *f , double *epot_out ) {

    int aid, pid, pjd, pkd, k, *loci, *locj, *lock, shifti[3], shiftk[3], emt;
    double h[3], epot = 0.0;
    struct space *s;
    struct part *pi, *pj, *pk, **partlist;
    struct cell **celllist;
    struct potential *pot;
    FPTYPE vij[3], vkj[3], dx[3], dprod, vij2, vkj2, invij, invkj, invijk, ctheta, r2;
    FPTYPE w, wi, wk;
    struct potential **pots, **pots_nb;
#if defined(VECTORIZE)
    struct potential *potq[4], *potq_nb[4];
    int icount = 0, icount_nb = 0, l;
    FPTYPE *effi[4], *effj[4], *effk[4];
    FPTYPE *effi_nb[4], *effk_nb[4];
    FPTYPE cthetaq[4] __attribute__ ((aligned (16)));
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE ee[4] __attribute__ ((aligned (16)));
    FPTYPE eff[4] __attribute__ ((aligned (16)));
    FPTYPE dijq[12], dkjq[12], dxq[12];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( a == NULL || e == NULL )
        return error(angle_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p_angle;
    pots_nb = e->p;
    emt = e->max_type;
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
    
        /* get the angle rays vij and vkj. */
        loci = celllist[ pid ]->loc;
        locj = celllist[ pjd ]->loc;
        lock = celllist[ pkd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            shifti[k] = loci[k] - locj[k];
            if ( shifti[k] > 1 )
                shifti[k] = -1;
            else if ( shifti[k] < -1 )
                shifti[k] = 1;
            shiftk[k] = lock[k] - locj[k];
            if ( shiftk[k] > 1 )
                shiftk[k] = -1;
            else if ( shiftk[k] < -1 )
                shiftk[k] = 1;
            }
        vij2 = 0.0; vkj2 = 0.0; dprod = 0.0; r2 = 0.0;
        for ( k = 0 ; k < 3 ; k++ ) {
            vij[k] = pi->x[k] - pj->x[k] + shifti[k]*h[k];
            vkj[k] = pk->x[k] - pj->x[k] + shiftk[k]*h[k];
            vij2 += vij[k] * vij[k];
            vkj2 += vkj[k] * vkj[k];
            dprod += vij[k] * vkj[k];
            dx[k] = vij[k] - vkj[k];
            r2 += dx[k] * dx[k];
            }
        invij = 1.0 / sqrt(vij2);
        invkj = 1.0 / sqrt(vkj2);
        invijk = invij * invkj;
        wi = dprod * invij * invij;
        wk = dprod * invkj * invkj;
        ctheta = dprod * invijk;

        /* printf( "angle_eval: angle %i is %e rad.\n" , aid , ctheta );
        if ( ctheta < pot->a || ctheta > pot->b )
            printf( "angle_eval: angle %i out of range, ctheta=%e.\n" , aid , ctheta ); */

        #ifdef VECTORIZE
            /* add this angle to the interaction queue. */
            cthetaq[icount] = dprod * invijk;
            dijq[icount*3] = invijk * vkj[0] - wi * vij[0];
            dijq[icount*3+1] = invijk * vkj[1] - wi * vij[1];
            dijq[icount*3+2] = invijk * vkj[2] - wi * vij[2];
            dkjq[icount*3] = invijk * vij[0] - wk * vkj[0];
            dkjq[icount*3+1] = invijk * vij[1] - wk * vkj[1];
            dkjq[icount*3+2] = invijk * vij[2] - wk * vkj[2];
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
                            wi = eff[l] * dijq[l*3+k]; wk = eff[l] * dkjq[l*3+k];
                            effi[l][k] += wi;
                            effk[l][k] += wk;
                            effj[l][k] -= wi + wk;
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
                            wi = eff[l] * dijq[l*3+k]; wk = eff[l] * dkjq[l*3+k];
                            effi[l][k] += wi;
                            effk[l][k] += wk;
                            effj[l][k] -= wi + wk;
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
                wi = eff * ( invijk * vkj[k] - wi * vij[k] );
                wk = eff * ( invijk * vij[k] - wk * vkj[k] );
                pi->f[k] += wi;
                pk->f[k] += wk;
                pj->f[k] -= wi + wk;
                }

            /* tabulate the energy */
            epot += ee;
        #endif
        
        
        /* Do we have a non-bonded interaction between i and k? */
        if ( ( pot = pots_nb[ pi->type * emt + pk->type ] ) == NULL )
            continue;
        
        #ifdef VECTORIZE
            /* add this bond to the interaction queue. */
            r2q[icount_nb] = r2;
            dxq[icount_nb*3] = dx[0];
            dxq[icount_nb*3+1] = dx[1];
            dxq[icount_nb*3+2] = dx[2];
            effi_nb[icount_nb] = &f[ 4*pid ];
            effk_nb[icount_nb] = &f[ 4*pkd ];
            potq_nb[icount_nb] = pot;
            icount_nb += 1;
            
            #if defined(FPTYPE_SINGLE)
                /* evaluate the bonds if the queue is full. */
                if ( icount_nb == 4 ) {
                    
                    potential_eval_vec_4single( potq_nb , r2q , ee , eff );
                    
                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot -= ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = eff[l] * dxq[l*3+k];
                            effi_nb[l][k] += w;
                            effk_nb[l][k] -= w;
                            }
                        }
                    
                    /* re-set the counter. */
                    icount_nb = 0;
                    
                    }
            #elif defined(FPTYPE_DOUBLE)
                /* evaluate the bonds if the queue is full. */
                if ( icount_nb == 4 ) {
                    
                    potential_eval_vec_4double( potq_nb , r2q , ee , eff );
                    
                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot -= ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = eff[l] * dxq[l*3+k];
                            effi_nb[l][k] += w;
                            effk_nb[l][k] -= w;
                            }
                        }
                    
                    /* re-set the counter. */
                    icount_nb = 0;
                    
                    }
            #endif
        #else
            /* evaluate the bond */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , r2 , &ee , &eff );
            #else
                potential_eval( pot , r2 , &ee , &eff );
            #endif
            
            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = eff * dx[k];
                pi->f[k] += w;
                pk->f[k] -= w;
                }

            /* tabulate the energy */
            epot -= ee;
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
                    wi = eff[l] * dijq[l*3+k]; wk = eff[l] * dkjq[l*3+k];
                    effi[l][k] += wi;
                    effk[l][k] += wk;
                    effj[l][k] -= wi + wk;
                    }
                }

            }
        if ( icount_nb > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount_nb ; k < 4 ; k++ ) {
                potq_nb[k] = potq_nb[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single( potq_nb , r2q , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount_nb ; l++ ) {
                epot -= ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi_nb[l][k] += w;
                    effk_nb[l][k] -= w;
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
                    wi = eff[l] * dijq[l*3+k]; wk = eff[l] * dkjq[l*3+k];
                    effi[l][k] += wi;
                    effk[l][k] += wk;
                    effj[l][k] -= wi + wk;
                    }
                }

            }
        if ( icount_nb > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount_nb ; k < 4 ; k++ ) {
                potq_nb[k] = potq_nb[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4double( potq_nb , r2q , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot -= ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi_nb[l][k] += w;
                    effk_nb[l][k] -= w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    *epot_out += epot;
    
    /* We're done here. */
    return angle_err_ok;
    
    }



