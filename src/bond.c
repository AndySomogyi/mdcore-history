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
#include "bond.h"


/* Global variables. */
/** The ID of the last error. */
int bond_err = bond_err_ok;
unsigned int bond_rcount = 0;

/* the error macro. */
#define error(id)				( bond_err = errs_register( id , bond_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *bond_err_msg[2] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered."
	};
    

/**
 * @brief Evaluate a list of bonded interactoins
 *
 * @param b Pointer to an array of #bond.
 * @param N Nr of bonds in @c b.
 * @param e Pointer to the #engine in which these bonds are evaluated.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * @return #bond_err_ok or <0 on error (see #bond_err)
 */
 
int bond_eval ( struct bond *b , int N , struct engine *e , double *epot_out ) {

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct part *pi, *pj, **partlist;
    struct cell **celllist;
    struct potential *pot, **pots;
    FPTYPE dx[3], r2, w;
#if defined(VECTORIZE)
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE *effi[4], *effj[4];
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE ee[4] __attribute__ ((aligned (16)));
    FPTYPE eff[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( b == NULL || e == NULL )
        return error(bond_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p_bond;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the bonds. */
    for ( bid = 0 ; bid < N ; bid++ ) {
    
        /* Get the particles involved. */
        pid = b[bid].i; pjd = b[bid].j;
        if ( ( pi = partlist[ pid ] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        
        /* Skip if both ghosts. */
        if ( pi->flags & part_flag_ghost && pj->flags & part_flag_ghost )
            continue;
            
        /* Get the potential. */
        if ( ( pot = pots[ pj->type*ld_pots + pi->type ] ) == NULL )
            continue;
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            }
        for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
            dx[k] = pi->x[k] - pj->x[k] + shift[k]*h[k];
            r2 += dx[k] * dx[k];
            }

        if ( r2 < pot->a*pot->a || r2 > pot->b*pot->b ) {
            printf( "bond_evalf: bond %i (%s-%s) out of range [%e,%e], r=%e.\n" ,
                bid , e->types[pi->type].name , e->types[pj->type].name , pot->a , pot->b , sqrt(r2) );
            r2 = fmax( pot->a*pot->a , fmin( pot->b*pot->b , r2 ) );
            }

        #ifdef VECTORIZE
            /* add this bond to the interaction queue. */
            r2q[icount] = r2;
            dxq[icount*3] = dx[0];
            dxq[icount*3+1] = dx[1];
            dxq[icount*3+2] = dx[2];
            effi[icount] = pi->f;
            effj[icount] = pj->f;
            potq[icount] = pot;
            icount += 1;

            #if defined(FPTYPE_SINGLE)
                /* evaluate the bonds if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4single( potq , r2q , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = eff[l] * dxq[l*3+k];
                            effi[l][k] -= w;
                            effj[l][k] += w;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
            #elif defined(FPTYPE_DOUBLE)
                /* evaluate the bonds if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4double( potq , r2q , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = eff[l] * dxq[l*3+k];
                            effi[l][k] -= w;
                            effj[l][k] += w;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

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
                pi->f[k] -= w;
                pj->f[k] += w;
                }

            /* tabulate the energy */
            epot += ee;
        #endif

        } /* loop over bonds. */
        
    #if defined(VEC_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single( potq , r2q , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #elif defined(VEC_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4double( potq , r2q , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
    
    }



/**
 * @brief Evaluate a list of bonded interactoins
 *
 * @param b Pointer to an array of #bond.
 * @param N Nr of bonds in @c b.
 * @param e Pointer to the #engine in which these bonds are evaluated.
 * @param f An array of @c FPTYPE in which to aggregate the resulting forces.
 * @param epot_out Pointer to a double in which to aggregate the potential energy.
 * 
 * This function differs from #bond_eval in that the forces are added to
 * the array @c f instead of directly in the particle data.
 * 
 * @return #bond_err_ok or <0 on error (see #bond_err)
 */
 
int bond_evalf ( struct bond *b , int N , struct engine *e , FPTYPE *f , double *epot_out ) {

    int bid, pid, pjd, k, *loci, *locj, shift[3], ld_pots;
    double h[3], epot = 0.0;
    struct space *s;
    struct part *pi, *pj, **partlist;
    struct cell **celllist;
    struct potential *pot, **pots;
    FPTYPE dx[3], r2, w;
#if defined(VECTORIZE)
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE *effi[4], *effj[4];
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE ee[4] __attribute__ ((aligned (16)));
    FPTYPE eff[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12];
#else
    FPTYPE ee, eff;
#endif
    
    /* Check inputs. */
    if ( b == NULL || e == NULL || f == NULL )
        return error(bond_err_null);
        
    /* Get local copies of some variables. */
    s = &e->s;
    pots = e->p_bond;
    partlist = s->partlist;
    celllist = s->celllist;
    ld_pots = e->max_type;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Loop over the bonds. */
    for ( bid = 0 ; bid < N ; bid++ ) {
    
        /* Get the particles involved. */
        pid = b[bid].i; pjd = b[bid].j;
        if ( ( pi = partlist[ pid ] ) == NULL )
            continue;
        if ( ( pj = partlist[ pjd ] ) == NULL )
            continue;
        
        /* Skip if both ghosts. */
        if ( pi->flags & part_flag_ghost && pj->flags & part_flag_ghost )
            continue;
            
        /* Get the potential. */
        if ( ( pot = pots[ pj->type*ld_pots + pi->type ] ) == NULL )
            continue;
    
        /* get the distance between both particles */
        loci = celllist[ pid ]->loc; locj = celllist[ pjd ]->loc;
        for ( k = 0 ; k < 3 ; k++ ) {
            shift[k] = loci[k] - locj[k];
            if ( shift[k] > 1 )
                shift[k] = -1;
            else if ( shift[k] < -1 )
                shift[k] = 1;
            }
        for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
            dx[k] = pi->x[k] - pj->x[k] + shift[k]*h[k];
            r2 += dx[k] * dx[k];
            }

        if ( r2 < pot->a*pot->a || r2 > pot->b*pot->b ) {
            printf( "bond_evalf: bond %i (%s-%s) out of range [%e,%e], r=%e.\n" ,
                bid , e->types[pi->type].name , e->types[pj->type].name , pot->a , pot->b , sqrt(r2) );
            r2 = fmax( pot->a*pot->a , fmin( pot->b*pot->b , r2 ) );
            }

        #ifdef VECTORIZE
            /* add this bond to the interaction queue. */
            r2q[icount] = r2;
            dxq[icount*3] = dx[0];
            dxq[icount*3+1] = dx[1];
            dxq[icount*3+2] = dx[2];
            effi[icount] = &( f[ 4*pid ] );
            effj[icount] = &( f[ 4*pjd ] );
            potq[icount] = pot;
            icount += 1;

            #if defined(FPTYPE_SINGLE)
                /* evaluate the bonds if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4single( potq , r2q , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = eff[l] * dxq[l*3+k];
                            effi[l][k] -= w;
                            effj[l][k] += w;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
            #elif defined(FPTYPE_DOUBLE)
                /* evaluate the bonds if the queue is full. */
                if ( icount == 4 ) {

                    potential_eval_vec_4double( potq , r2q , ee , eff );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        epot += ee[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = eff[l] * dxq[l*3+k];
                            effi[l][k] -= w;
                            effj[l][k] += w;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

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
                f[ 4*pid + k ] -= w;
                f[ 4*pjd + k ] += w;
                }

            /* tabulate the energy */
            epot += ee;
        #endif

        } /* loop over bonds. */
        
    #if defined(VEC_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single( potq , r2q , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #elif defined(VEC_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4double( potq , r2q , ee , eff );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += ee[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif
    
    /* Store the potential energy. */
    if ( epot_out != NULL )
        *epot_out += epot;
    
    /* We're done here. */
    return bond_err_ok;
    
    }



