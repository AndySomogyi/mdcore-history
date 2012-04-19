/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2011 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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
#include "exclusion.h"


/* Global variables. */
/** The ID of the last error. */
int exclusion_err = exclusion_err_ok;
unsigned int exclusion_rcount = 0;

/* the error macro. */
#define error(id)				( exclusion_err = errs_register( id , exclusion_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *exclusion_err_msg[2] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered."
	};
    

/**
 * @brief Evaluate a list of exclusioned interactoins
 *
 * @param b Pointer to an array of #exclusion.
 * @param N Nr of exclusions in @c b.
 * @param nr_threads Number of computational threads.
 * @param cid_div #cell id modulus.
 * @param e Pointer to the #engine in which these exclusions are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #exclusion_err_ok or <0 on error (see #exclusion_err)
 *
 * Computes only the interactions on particles inside cells @c c where
 * @c c->id % nr_threads == cid_div.
 *
 */
 
int exclusion_eval_div ( struct exclusion *b , int N , int nr_threads , int cid_div , struct engine *e , double *epot_out ) {

    int bid, pid, pjd, cid, cjd = 0, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0, incr;
    struct space *s;
    struct part *pi, *pj, **partlist;
    struct cell **celllist, *ci, *cj;
    struct potential *pot, **pots;
    FPTYPE pix[4], dx[4], r2, w;
#if defined(VECTORIZE)
    struct potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dummy = 0.0;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE dxq[VEC_SIZE*3];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( b == NULL || e == NULL )
        return error(exclusion_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    incr = ((double)nr_threads) / s->nr_real;
    pots = e->p;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
    pix[3] = FPTYPE_ZERO;
        
    /* Loop over the exclusions. */
    for ( bid = 0 ; bid < N ; bid++ ) {
    
        /* Do we own this bond? */
        pid = b[bid].i; pjd = b[bid].j;
        if ( ( ci = celllist[ pid ] ) == NULL ||
             ( cj = celllist[ pjd ] ) == NULL )
            continue;

        /* Skip if both parts are in the wrong cell. */
        cid = ci->id * incr;
        cjd = cj->id * incr;
        if ( cid != cid_div && cjd != cid_div )
            continue;
            
        /* Get the particles involved. */
        pi = partlist[ pid ];
        pj = partlist[ pjd ];
        
        /* Get the potential. */
        if ( ( pot = pots[ pj->type*ld_pots + pi->type ] ) == NULL )
            continue;
    
        /* get the distance between both particles */
        loci = ci->loc; locj = cj->loc;
        for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            pix[k] = pi->x[k] + h[k]*shift[k];
            }
        r2 = fptype_r2( pix , pj->x , dx );
        
        /* Out of range? */
        if ( r2 > pot->b*pot->b )
            continue;

        #ifdef VECTORIZE
            /* add this exclusion to the interaction queue. */
            r2q[icount] = r2;
            dxq[icount*3] = dx[0];
            dxq[icount*3+1] = dx[1];
            dxq[icount*3+2] = dx[2];
            effi[icount] = ( cid == cid_div ? pi->f : &dummy );
            effj[icount] = ( cjd == cid_div ? pj->f : &dummy );
            potq[icount] = pot;
            icount += 1;

            /* evaluate the interactions if the queue is full. */
            if ( icount == VEC_SIZE ) {

                #if defined(FPTYPE_SINGLE)
                    #if VEC_SIZE==8
                    potential_eval_vec_8single( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_4single( potq , r2q , ee , eff );
                    #endif
                #elif defined(FPTYPE_DOUBLE)
                    #if VEC_SIZE==4
                    potential_eval_vec_4double( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_2double( potq , r2q , ee , eff );
                    #endif
                #endif

                /* update the forces and the energy */
                for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                    epot += ee[l];
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff[l] * dxq[l*3+k];
                        effi[l][k] += w;
                        effj[l][k] -= w;
                        }
                    }

                /* re-set the counter. */
                icount = 0;

                }
        #else
            /* evaluate the exclusion */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , r2 , &ee , &eff );
            #else
                potential_eval( pot , r2 , &ee , &eff );
            #endif

            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = eff * dx[k];
                if ( cid == cid_div )
                    pi->f[k] += w;
                if ( cjd == cid_div )
                    pj->f[k] -= w;
                }

            /* tabulate the energy */
            epot -= ee;
        #endif

        } /* loop over exclusions. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            #if defined(VEC_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single( potq , r2q , ee , eff );
                #else
                potential_eval_vec_4single( potq , r2q , ee , eff );
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double( potq , r2q , ee , eff );
                #else
                potential_eval_vec_2double( potq , r2q , ee , eff );
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] += w;
                    effj[l][k] -= w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return exclusion_err_ok;
    
    }


/**
 * @brief Evaluate a list of exclusioned interactoins
 *
 * @param b Pointer to an array of #exclusion.
 * @param N Nr of exclusions in @c b.
 * @param nr_threads Number of computational threads.
 * @param cid_mod #cell id modulus.
 * @param e Pointer to the #engine in which these exclusions are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #exclusion_err_ok or <0 on error (see #exclusion_err)
 *
 * Computes only the interactions on particles inside cells @c c where
 * @c c->id % nr_threads == cid_mod.
 *
 */
 
int exclusion_eval_mod ( struct exclusion *b , int N , int nr_threads , int cid_mod , struct engine *e , double *epot_out ) {

    int bid, pid, pjd, cid, cjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct part *pi, *pj, **partlist;
    struct cell **celllist;
    struct potential *pot, **pots;
    FPTYPE dx[3], r2, w;
#if defined(VECTORIZE)
    struct potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE dummy = 0.0;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE dxq[VEC_SIZE*3];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( b == NULL || e == NULL )
        return error(exclusion_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the exclusions. */
    for ( bid = 0 ; bid < N ; bid++ ) {
    
        /* Get the particles involved. */
        pid = b[bid].i; pjd = b[bid].j;
        if ( ( pi = partlist[ pid ] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        
        /* Skip if both ghosts. */
        if ( ( pi->flags & part_flag_ghost ) && 
             ( pj->flags & part_flag_ghost ) )
            continue;
            
        /* Skip if both in the wrong cell. */
        cid = celllist[ pid ]->id % nr_threads;
        cjd = celllist[ pjd ]->id % nr_threads;
        if ( ( cid != cid_mod ) && ( cjd != cid_mod ) )
            continue;
            
        /* Get the potential. */
        if ( ( pot = pots[ pj->type*ld_pots + pi->type ] ) == NULL )
            continue;
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            dx[k] = pi->x[k] - pj->x[k] + h[k]*shift[k];
            r2 += dx[k] * dx[k];
            }
        
        /* Out of range? */
        if ( r2 > pot->b*pot->b )
            continue;

        #ifdef VECTORIZE
            /* add this exclusion to the interaction queue. */
            r2q[icount] = r2;
            dxq[icount*3] = dx[0];
            dxq[icount*3+1] = dx[1];
            dxq[icount*3+2] = dx[2];
            effi[icount] = ( cid == cid_mod ? pi->f : &dummy );
            effj[icount] = ( cjd == cid_mod ? pj->f : &dummy );
            potq[icount] = pot;
            icount += 1;

            /* evaluate the interactions if the queue is full. */
            if ( icount == VEC_SIZE ) {

                #if defined(FPTYPE_SINGLE)
                    #if VEC_SIZE==8
                    potential_eval_vec_8single( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_4single( potq , r2q , ee , eff );
                    #endif
                #elif defined(FPTYPE_DOUBLE)
                    #if VEC_SIZE==4
                    potential_eval_vec_4double( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_2double( potq , r2q , ee , eff );
                    #endif
                #endif

                /* update the forces and the energy */
                for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                    epot += ee[l];
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff[l] * dxq[l*3+k];
                        effi[l][k] += w;
                        effj[l][k] -= w;
                        }
                    }

                /* re-set the counter. */
                icount = 0;

                }
        #else
            /* evaluate the exclusion */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , r2 , &ee , &eff );
            #else
                potential_eval( pot , r2 , &ee , &eff );
            #endif

            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = eff * dx[k];
                if ( cid == cid_mod )
                    pi->f[k] += w;
                if ( cjd == cid_mod )
                    pj->f[k] -= w;
                }

            /* tabulate the energy */
            epot -= ee;
        #endif

        } /* loop over exclusions. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            #if defined(VEC_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single( potq , r2q , ee , eff );
                #else
                potential_eval_vec_4single( potq , r2q , ee , eff );
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double( potq , r2q , ee , eff );
                #else
                potential_eval_vec_2double( potq , r2q , ee , eff );
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] += w;
                    effj[l][k] -= w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return exclusion_err_ok;
    
    }



/**
 * @brief Evaluate a list of exclusioned interactoins
 *
 * @param b Pointer to an array of #exclusion.
 * @param N Nr of exclusions in @c b.
 * @param e Pointer to the #engine in which these exclusions are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #exclusion_err_ok or <0 on error (see #exclusion_err)
 */
 
int exclusion_eval ( struct exclusion *b , int N , struct engine *e , double *epot_out ) {

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct part *pi, *pj, **partlist;
    struct cell **celllist;
    struct potential *pot, **pots;
    FPTYPE dx[3], r2, w;
#if defined(VECTORIZE)
    struct potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE ee[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE eff[VEC_SIZE] __attribute__ ((aligned (16)));
    FPTYPE dxq[VEC_SIZE*3];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( b == NULL || e == NULL )
        return error(exclusion_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the exclusions. */
    for ( bid = 0 ; bid < N ; bid++ ) {
    
        /* Get the particles involved. */
        pid = b[bid].i; pjd = b[bid].j;
        if ( ( pi = partlist[ pid ] ) == NULL ||
             ( pj = partlist[ pjd ] ) == NULL )
            continue;
        
        /* Skip if both ghosts. */
        if ( ( pi->flags & part_flag_ghost ) && 
             ( pj->flags & part_flag_ghost ) )
            continue;
            
        /* Get the potential. */
        if ( ( pot = pots[ pj->type*ld_pots + pi->type ] ) == NULL )
            continue;
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            dx[k] = pi->x[k] - pj->x[k] + h[k]*shift[k];
            r2 += dx[k] * dx[k];
            }
        
        /* Out of range? */
        if ( r2 > pot->b*pot->b )
            continue;

        #ifdef VECTORIZE
            /* add this exclusion to the interaction queue. */
            r2q[icount] = r2;
            dxq[icount*3] = dx[0];
            dxq[icount*3+1] = dx[1];
            dxq[icount*3+2] = dx[2];
            effi[icount] = pi->f;
            effj[icount] = pj->f;
            potq[icount] = pot;
            icount += 1;

            /* evaluate the interactions if the queue is full. */
            if ( icount == VEC_SIZE ) {

                #if defined(FPTYPE_SINGLE)
                    #if VEC_SIZE==8
                    potential_eval_vec_8single( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_4single( potq , r2q , ee , eff );
                    #endif
                #elif defined(FPTYPE_DOUBLE)
                    #if VEC_SIZE==4
                    potential_eval_vec_4double( potq , r2q , ee , eff );
                    #else
                    potential_eval_vec_2double( potq , r2q , ee , eff );
                    #endif
                #endif

                /* update the forces and the energy */
                for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                    epot += ee[l];
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff[l] * dxq[l*3+k];
                        effi[l][k] += w;
                        effj[l][k] -= w;
                        }
                    }

                /* re-set the counter. */
                icount = 0;

                }
        #else
            /* evaluate the exclusion */
            #ifdef EXPLICIT_POTENTIALS
                potential_eval_expl( pot , r2 , &ee , &eff );
            #else
                potential_eval( pot , r2 , &ee , &eff );
            #endif

            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = eff * dx[k];
                pi->f[k] += w;
                pj->f[k] -= w;
                }

            /* tabulate the energy */
            epot -= ee;
        #endif

        } /* loop over exclusions. */
        
    #if defined(VECTORIZE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < VEC_SIZE ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            #if defined(VEC_SINGLE)
                #if VEC_SIZE==8
                potential_eval_vec_8single( potq , r2q , ee , eff );
                #else
                potential_eval_vec_4single( potq , r2q , ee , eff );
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double( potq , r2q , ee , eff );
                #else
                potential_eval_vec_2double( potq , r2q , ee , eff );
                #endif
            #endif

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] += w;
                    effj[l][k] -= w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return exclusion_err_ok;
    
    }



