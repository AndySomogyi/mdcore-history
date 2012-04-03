/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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
#ifdef CELL
    #include <libspe2.h>
    #include <libmisc.h>
    #define ceil128(v) (((v) + 127) & ~127)
#endif
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef HAVE_SETAFFINITY
    #include <sched.h>
#endif
#ifdef HAVE_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "engine.h"
#include "runner.h"



#ifdef CELL
    /* the SPU executeable */
    extern spe_program_handle_t runner_spu;
#endif


/* the error macro. */
#define error(id)				( runner_err = errs_register( id , runner_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
extern char *runner_err_msg[];
    

/**
 * @brief Fill in the pairwise Verlet list entries for the given cell pair
 *        if needed and compute the interactions.
 * 
 * @param r The #runner computing the pair.
 * @param cell_i The first cell.
 * @param cell_j The second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err)
 *
 */
 
int runner_dopair_verlet ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift , struct cellpair *cp ) {

    struct part *part_i, *part_j;
    struct space *s;
    int count = 0;
    int i, j, k, imax, qpos, lo, hi;
    struct {
        short int lo, hi;
        } *qstack;
    struct part *parts_i, *parts_j;
    struct potential *pot, **pots;
    struct engine *eng;
    int emt, pioff;
    FPTYPE cutoff, cutoff2, skin, skin2, r2, dx[3], w;
    unsigned int *parts, temp, dmaxdist;
    short int pivot;
    FPTYPE dscale;
    FPTYPE shift[3], inshift, nshift;
    FPTYPE pix[3], *pif;
    int pind, pid, nr_pairs, count_i, count_j;
    double epot = 0.0;
    short int *pairs;
#if defined(VECTORIZE)
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE *effi[4], *effj[4];
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE e[4] __attribute__ ((aligned (16)));
    FPTYPE f[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12];
#else
    FPTYPE e, f;
#endif
    
    /* break early if one of the cells is empty */
    count_i = cell_i->count;
    count_j = cell_j->count;
    if ( count_i == 0 || count_j == 0 || ( cell_i == cell_j && count_i < 2 ) )
        return runner_err_ok;
    
    /* get the space and cutoff */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    pots = eng->p;
    skin = fmin( s->h[0] , fmin( s->h[1] , s->h[2] ) );
    skin2 = skin * skin;
    cutoff = s->cutoff;
    cutoff2 = cutoff*cutoff;
    dscale = (FPTYPE)SHRT_MAX / ( 3 * sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] ) );
    dmaxdist = 2 + dscale*skin;
    
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * count_i );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * count_i );
        if ( cell_i != cell_j ) {
            parts_j = (struct part *)alloca( sizeof(struct part) * count_j );
            memcpy( parts_j , cell_j->parts , sizeof(struct part) * count_j );
            }
        else
            parts_j = parts_i;
        }
        
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
    /* Is this a self interaction? */
    if ( cell_i == cell_j ) {

        /* loop over all particles */
        for ( i = 1 ; i < count_i ; i++ ) {

            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            pioff = part_i->type * emt;
            pif = &( part_i->f[0] );

            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {

                /* get the other particle */
                part_j = &(parts_j[j]);

                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }

                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                // runner_rcount += 1;

                /* fetch the potential, if any */
                pot = pots[ pioff + part_j->type ];
                if ( pot == NULL )
                    continue;

                #if defined(VECTORIZE)
                    /* add this interaction to the interaction queue. */
                    r2q[icount] = r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = pif;
                    effj[icount] = part_j->f;
                    potq[icount] = pot;
                    icount += 1;

                    #if defined(FPTYPE_SINGLE)
                        /* evaluate the interactions if the queue is full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4single( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f[l] * dxq[l*3+k];
                                    effi[l][k] -= w;
                                    effj[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount = 0;

                            }
                    #elif defined(FPTYPE_DOUBLE)
                        /* evaluate the interactions if the queue is full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4double( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f[l] * dxq[l*3+k];
                                    effi[l][k] -= w;
                                    effj[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount = 0;

                            }
                    #endif
                #else
                    /* evaluate the interaction */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl( pot , r2 , &e , &f );
                    #else
                        potential_eval( pot , r2 , &e , &f );
                    #endif

                    /* update the forces */
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = f * dx[k];
                        pif[k] -= w;
                        part_j->f[k] += w;
                        }

                    /* tabulate the energy */
                    epot += e;
                #endif

                } /* loop over all other particles */

            } /* loop over all particles */

        }

    /* No, genuine pair. */
    else {

        /* Do we need to re-compute the pairwise Verlet list? */
        if ( s->verlet_rebuild || cp->pairs == NULL ) {

            /* Has sufficient memory for the Verlet list been allocated? */
            if ( cp->pairs == NULL || cp->size < count_i * count_j ) {

                /* Clear lists if needed. */
                if ( cp->pairs != NULL )
                    free( cp->pairs );

                /* Set the size and width. */
                cp->size = (count_i + 10) * (count_j + 10);

                /* Allocate the list data. */
                if ( ( cp->pairs = (short int *)malloc( sizeof(short int) * cp->size ) ) == NULL )
                    return error(runner_err_malloc);
                if ( cp->nr_pairs == NULL && ( cp->nr_pairs = (short int *)malloc( sizeof(short int) * ( count_i + count_j + 20 ) ) ) == NULL )
                    return error(runner_err_malloc);

                }
    
            /* Allocate work arrays on stack. */
            if ( ( parts = alloca( sizeof(unsigned int) * (count_i + count_j) ) ) == NULL ||
                 ( qstack = alloca( sizeof(short int) * 2 * (count_i + count_j) ) ) == NULL )
                return error(runner_err_malloc);
        
            /* start by filling the particle ids of both cells into ind and d */
            nshift = sqrt( pshift[0]*pshift[0] + pshift[1]*pshift[1] + pshift[2]*pshift[2] );
            inshift = 1.0 / nshift;
            shift[0] = pshift[0]*inshift; shift[1] = pshift[1]*inshift; shift[2] = pshift[2]*inshift;
            for ( i = 0 ; i < count_i ; i++ ) {
                part_i = &( parts_i[i] );
                parts[count] = (i << 16) |
                    (unsigned int)( dscale * ( nshift + part_i->x[0]*shift[0] + part_i->x[1]*shift[1] + part_i->x[2]*shift[2] ) );
                count += 1;
                }
            for ( i = 0 ; i < count_j ; i++ ) {
                part_i = &( parts_j[i] );
                parts[count] = (i << 16) |
                    (unsigned int)( dscale * ( nshift + (part_i->x[0]+pshift[0])*shift[0] + (part_i->x[1]+pshift[1])*shift[1] + (part_i->x[2]+pshift[2])*shift[2] ) );
                count += 1;
                }

            /* Sort parts in cell_i in decreasing order with quicksort */
            qstack[0].lo = 0; qstack[0].hi = count_i - 1; qpos = 0;
            while ( qpos >= 0 ) {
                lo = qstack[qpos].lo; hi = qstack[qpos].hi;
                qpos -= 1;
                if ( hi - lo < 15 ) {
                    for ( i = lo ; i < hi ; i++ ) {
                        imax = i;
                        for ( j = i+1 ; j <= hi ; j++ )
                            if ( (parts[j] & 0xffff) > (parts[imax] & 0xffff) )
                                imax = j;
                        if ( imax != i ) {
                            temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                            }
                        }
                    }
                else {
                    pivot = parts[ ( lo + hi ) / 2 ] & 0xffff;
                    i = lo; j = hi;
                    while ( i <= j ) {
                        while ( (parts[i] & 0xffff) > pivot ) i++;
                        while ( (parts[j] & 0xffff) < pivot ) j--;
                        if ( i <= j ) {
                            if ( i < j ) {
                                temp = parts[i]; parts[i] = parts[j]; parts[j] = temp;
                                }
                            i += 1; j -= 1;
                            }
                        }
                    if ( lo < j ) {
                        qpos += 1;
                        qstack[qpos].lo = lo;
                        qstack[qpos].hi = j;
                        }
                    if ( i < hi ) {
                        qpos += 1;
                        qstack[qpos].lo = i;
                        qstack[qpos].hi = hi;
                        }
                    }
                }
                
            /* Sort parts in cell_j in increasing order with quicksort */
            qstack[0].lo = count_i; qstack[0].hi = count - 1; qpos = 0;
            while ( qpos >= 0 ) {
                lo = qstack[qpos].lo; hi = qstack[qpos].hi;
                qpos -= 1;
                if ( hi - lo < 15 ) {
                    for ( i = lo ; i < hi ; i++ ) {
                        imax = i;
                        for ( j = i+1 ; j <= hi ; j++ )
                            if ( (parts[j] & 0xffff) < (parts[imax] & 0xffff) )
                                imax = j;
                        if ( imax != i ) {
                            temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                            }
                        }
                    }
                else {
                    pivot = parts[ ( lo + hi ) / 2 ] & 0xffff;
                    i = lo; j = hi;
                    while ( i <= j ) {
                        while ( (parts[i] & 0xffff) < pivot ) i++;
                        while ( (parts[j] & 0xffff) > pivot ) j--;
                        if ( i <= j ) {
                            if ( i < j ) {
                                temp = parts[i]; parts[i] = parts[j]; parts[j] = temp;
                                }
                            i += 1; j -= 1;
                            }
                        }
                    if ( lo < j ) {
                        qpos += 1;
                        qstack[qpos].lo = lo;
                        qstack[qpos].hi = j;
                        }
                    if ( i < hi ) {
                        qpos += 1;
                        qstack[qpos].lo = i;
                        qstack[qpos].hi = hi;
                        }
                    }
                }

            /* loop over the sorted list of particles in i */
            for ( i = 0 ; i < count_i ; i++ ) {

                /* Quit early? */
                if ( (parts[count_i] & 0xffff) - (parts[i] & 0xffff) > dmaxdist ) {
                    while ( i < count_i )
                        cp->nr_pairs[ parts[i++] >> 16 ] = 0;
                    break;
                    }

                /* get a handle on this particle */
                pid = parts[i] >> 16;
                part_i = &( parts_i[pid] );
                pix[0] = part_i->x[0] - pshift[0];
                pix[1] = part_i->x[1] - pshift[1];
                pix[2] = part_i->x[2] - pshift[2];
                pioff = part_i->type * emt;
                pif = &( part_i->f[0] );
                pind = 0;
                pairs = &( cp->pairs[ pid * count_j ] );

                /* loop over the left particles */
                for ( j = 0 ; j < count_j && (parts[count_i+j] & 0xffff) - (parts[i] & 0xffff) < dmaxdist ; j++ ) {

                    /* get a handle on the second particle */
                    part_j = &( parts_j[ parts[count_i+j] >> 16 ] );
                    
                    /* get the distance between both particles */
                    r2 = 0.0;
                    for ( k = 0 ; k < 3 ; k++ ) {
                        dx[k] = pix[k] - part_j->x[k];
                        r2 += dx[k] * dx[k];
                        }

                    /* is this within cutoff? */
                    if ( r2 > skin2 )
                        continue;
                    /* runner_rcount += 1; */

                    /* fetch the potential, if any */
                    pot = pots[ pioff + part_j->type ];
                    if ( pot == NULL )
                        continue;

                    /* Add this pair to the verlet list. */
                    pairs[pind] = parts[count_i+j] >> 16;
                    pind += 1;

                    /* is this within cutoff? */
                    if ( r2 > cutoff2 )
                        continue;
                    // runner_rcount += 1;

                    #if defined(VECTORIZE)
                        /* add this interaction to the interaction queue. */
                        r2q[icount] = r2;
                        dxq[icount*3] = dx[0];
                        dxq[icount*3+1] = dx[1];
                        dxq[icount*3+2] = dx[2];
                        effi[icount] = pif;
                        effj[icount] = part_j->f;
                        potq[icount] = pot;
                        icount += 1;

                        #if defined(FPTYPE_SINGLE)
                            /* evaluate the interactions if the queue is full. */
                            if ( icount == 4 ) {

                                potential_eval_vec_4single( potq , r2q , e , f );

                                /* update the forces and the energy */
                                for ( l = 0 ; l < 4 ; l++ ) {
                                    epot += e[l];
                                    for ( k = 0 ; k < 3 ; k++ ) {
                                        w = f[l] * dxq[l*3+k];
                                        effi[l][k] -= w;
                                        effj[l][k] += w;
                                        }
                                    }

                                /* re-set the counter. */
                                icount = 0;

                                }
                        #elif defined(FPTYPE_DOUBLE)
                            /* evaluate the interactions if the queue is full. */
                            if ( icount == 4 ) {

                                potential_eval_vec_4double( potq , r2q , e , f );

                                /* update the forces and the energy */
                                for ( l = 0 ; l < 4 ; l++ ) {
                                    epot += e[l];
                                    for ( k = 0 ; k < 3 ; k++ ) {
                                        w = f[l] * dxq[l*3+k];
                                        effi[l][k] -= w;
                                        effj[l][k] += w;
                                        }
                                    }

                                /* re-set the counter. */
                                icount = 0;

                                }
                        #endif
                    #else
                        /* evaluate the interaction */
                        #ifdef EXPLICIT_POTENTIALS
                            potential_eval_expl( pot , r2 , &e , &f );
                        #else
                            potential_eval( pot , r2 , &e , &f );
                        #endif

                        /* update the forces */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f * dx[k];
                            pif[k] -= w;
                            part_j->f[k] += w;
                            }

                        /* tabulate the energy */
                        epot += e;
                    #endif

                    } /* loop over the right particles. */

                /* Store the number of pairs for pid. */
                cp->nr_pairs[pid] = pind;

                } /* loop over the left particles */
            
            } /* do we need to re-build the pairwise verlet list? */
        
        /* Otherwise, just evaluate using the list. */
        else {

            /* Loop over the particles in cell_i. */
            for ( i = 0 ; i < count_i ; i++ ) {

                /* Skip this particle before we get too involved. */
                if ( ( nr_pairs = cp->nr_pairs[i] ) == 0 )
                    continue;

                /* Get the particle data. */
                part_i = &(parts_i[i]);
                pix[0] = part_i->x[0] - pshift[0];
                pix[1] = part_i->x[1] - pshift[1];
                pix[2] = part_i->x[2] - pshift[2];
                pif = &( part_i->f[0] );
                pairs = &( cp->pairs[ i * count_j ] );
                pioff = part_i->type * emt;

                /* Loop over the entries in the Verlet list. */
                for ( j = 0 ; j < nr_pairs ; j++ ) {

                    /* Get the other particle */
                    part_j = &( parts_j[ pairs[j] ] );

                    /* get the distance between both particles */
                    for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                        dx[k] = pix[k] - part_j->x[k];
                        r2 += dx[k] * dx[k];
                        }

                    /* is this within cutoff? */
                    if ( r2 > cutoff2 )
                        continue;
                    // runner_rcount += 1;

                    /* fetch the potential (non-NULL by design). */
                    pot = pots[ pioff + part_j->type ];

                    #if defined(VECTORIZE)
                        /* add this interaction to the interaction queue. */
                        r2q[icount] = r2;
                        dxq[icount*3] = dx[0];
                        dxq[icount*3+1] = dx[1];
                        dxq[icount*3+2] = dx[2];
                        effi[icount] = pif;
                        effj[icount] = part_j->f;
                        potq[icount] = pot;
                        icount += 1;

                        #if defined(FPTYPE_SINGLE)
                            /* evaluate the interactions if the queue is full. */
                            if ( icount == 4 ) {

                                potential_eval_vec_4single( potq , r2q , e , f );

                                /* update the forces and the energy */
                                for ( l = 0 ; l < 4 ; l++ ) {
                                    epot += e[l];
                                    for ( k = 0 ; k < 3 ; k++ ) {
                                        w = f[l] * dxq[l*3+k];
                                        effi[l][k] -= w;
                                        effj[l][k] += w;
                                        }
                                    }

                                /* re-set the counter. */
                                icount = 0;

                                }
                        #elif defined(FPTYPE_DOUBLE)
                            /* evaluate the interactions if the queue is full. */
                            if ( icount == 4 ) {

                                potential_eval_vec_4double( potq , r2q , e , f );

                                /* update the forces and the energy */
                                for ( l = 0 ; l < 4 ; l++ ) {
                                    epot += e[l];
                                    for ( k = 0 ; k < 3 ; k++ ) {
                                        w = f[l] * dxq[l*3+k];
                                        effi[l][k] -= w;
                                        effj[l][k] += w;
                                        }
                                    }

                                /* re-set the counter. */
                                icount = 0;

                                }
                        #endif
                    #else
                        /* evaluate the interaction */
                        #ifdef EXPLICIT_POTENTIALS
                            potential_eval_expl( pot , r2 , &e , &f );
                        #else
                            potential_eval( pot , r2 , &e , &f );
                        #endif

                        /* update the forces */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f * dx[k];
                            piff[k] -= w;
                            part_j->f[k] += w;
                            }

                        /* tabulate the energy */
                        epot += e;
                    #endif

                    } /* loop over pairs. */

                } /* loop over particles in cell_i. */

            }
            
        }
        
    #if defined(VEC_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single( potq , r2q , e , f );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += e[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * dxq[l*3+k];
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
            potential_eval_vec_4double( potq , r2q , e , f );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += e[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif
        
    /* Store the potential energy to cell_i. */
    if ( cell_j->flags & cell_flag_ghost || cell_i->flags & cell_flag_ghost )
        cell_i->epot += 0.5 * epot;
    else
        cell_i->epot += epot;
        
    /* Write local data back if needed. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* copy the particle data back */
        for ( i = 0 ; i < count_i ; i++ ) {
            cell_i->parts[i].f[0] = parts_i[i].f[0];
            cell_i->parts[i].f[1] = parts_i[i].f[1];
            cell_i->parts[i].f[2] = parts_i[i].f[2];
            }
        if ( cell_i != cell_j )
            for ( i = 0 ; i < count_j ; i++ ) {
                cell_j->parts[i].f[0] = parts_j[i].f[0];
                cell_j->parts[i].f[1] = parts_j[i].f[1];
                cell_j->parts[i].f[2] = parts_j[i].f[2];
                }
        }
        
    /* since nothing bad happened to us... */
    return runner_err_ok;

    }
    
    
/**
 * @brief Fill in the pairwise Verlet list entries for the given cell pair
 *        if needed and compute the interactions.
 * 
 * @param r The #runner computing the pair.
 * @param cell_i The first cell.
 * @param cell_j The second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err)
 *
 * This routine differs from #runner_dopair_verlet in that instead of
 * storing a Verlet table, the sorted particle ids are stored. This
 * requires only (size_i + size_j) entries as opposed to size_i*size_j
 * for the Verlet table, yet may be less efficient since particles
 * within the skin along the cell-pair axis are inspected, as opposed
 * to particles simply within the skin of each other.
 *
 */
 
int runner_dopair_verlet2 ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift , struct cellpair *cp ) {

    struct part *part_i, *part_j;
    struct space *s;
    int count = 0;
    int i, j, k, imax, qpos, lo, hi;
    struct {
        short int lo, hi;
        } *qstack;
    struct part *parts_i, *parts_j;
    struct potential *pot, **pots;
    struct engine *eng;
    int emt, pioff, dmaxdist;
    FPTYPE cutoff, cutoff2, r2, dx[3], w;
    unsigned int *parts, temp;
    short int pivot;
    FPTYPE dscale;
    FPTYPE shift[3], inshift, nshift;
    FPTYPE pix[3], *pif;
    int pid, count_i, count_j;
    double epot = 0.0;
#if defined(VECTORIZE)
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE *effi[4], *effj[4];
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE e[4] __attribute__ ((aligned (16)));
    FPTYPE f[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12];
#else
    FPTYPE e, f;
#endif
    
    /* break early if one of the cells is empty */
    count_i = cell_i->count;
    count_j = cell_j->count;
    if ( count_i == 0 || count_j == 0 || ( cell_i == cell_j && count_i < 2 ) )
        return runner_err_ok;
    
    /* get the space and cutoff */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    pots = eng->p;
    cutoff = s->cutoff;
    cutoff2 = cutoff*cutoff;
    dscale = (FPTYPE)SHRT_MAX / (3 * sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] ) );
    dmaxdist = 2 + dscale * (cutoff + 2*eng->s.verlet_maxdx);
    
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * count_i );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * count_i );
        if ( cell_i != cell_j ) {
            parts_j = (struct part *)alloca( sizeof(struct part) * count_j );
            memcpy( parts_j , cell_j->parts , sizeof(struct part) * count_j );
            }
        else
            parts_j = parts_i;
        }
        
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
    /* Is this a self interaction? */
    if ( cell_i == cell_j ) {

        /* loop over all particles */
        for ( i = 1 ; i < count_i ; i++ ) {

            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            pioff = part_i->type * emt;
            pif = &( part_i->f[0] );

            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {

                /* get the other particle */
                part_j = &(parts_j[j]);

                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }

                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */

                /* fetch the potential, if any */
                pot = pots[ pioff + part_j->type ];
                if ( pot == NULL )
                    continue;

                #if defined(VECTORIZE)
                    /* add this interaction to the interaction queue. */
                    r2q[icount] = r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = pif;
                    effj[icount] = part_j->f;
                    potq[icount] = pot;
                    icount += 1;

                    #if defined(FPTYPE_SINGLE)
                        /* evaluate the interactions if the queue is full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4single( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f[l] * dxq[l*3+k];
                                    effi[l][k] -= w;
                                    effj[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount = 0;

                            }
                    #elif defined(FPTYPE_DOUBLE)
                        /* evaluate the interactions if the queue is full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4double( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f[l] * dxq[l*3+k];
                                    effi[l][k] -= w;
                                    effj[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount = 0;

                            }
                    #endif
                #else
                    /* evaluate the interaction */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl( pot , r2 , &e , &f );
                    #else
                        potential_eval( pot , r2 , &e , &f );
                    #endif

                    /* update the forces */
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = f * dx[k];
                        pif[k] -= w;
                        part_j->f[k] += w;
                        }

                    /* tabulate the energy */
                    epot += e;
                #endif

                } /* loop over all other particles */

            } /* loop over all particles */

        }

    /* No, genuine pair. */
    else {
    
        /* Do we need to re-compute the pairwise Verlet list? */
        if ( s->verlet_rebuild || cp->pairs == NULL ) {

            /* Has sufficient memory for the Verlet list been allocated? */
            if ( cp->pairs == NULL || cp->size < count_i + count_j ) {

                /* Clear lists if needed. */
                if ( cp->pairs != NULL )
                    free( cp->pairs );

                /* Set the size and width. */
                cp->size = count_i + count_j + 20;

                /* Allocate the list data. */
                if ( ( cp->pairs = (short int *)malloc( sizeof(short int) * 2 * cp->size ) ) == NULL )
                    return error(runner_err_malloc);

                }
    
            /* Allocate work arrays on stack. */
            parts = (void *)cp->pairs;
            if ( ( qstack = alloca( sizeof(short int) * 2 * (count_i + count_j) ) ) == NULL )
                return error(runner_err_malloc);
        
            /* start by filling the particle ids of both cells into ind and d */
            nshift = sqrt( pshift[0]*pshift[0] + pshift[1]*pshift[1] + pshift[2]*pshift[2] );
            inshift = 1.0 / nshift;
            shift[0] = pshift[0]*inshift; shift[1] = pshift[1]*inshift; shift[2] = pshift[2]*inshift;
            for ( i = 0 ; i < count_i ; i++ ) {
                part_i = &( parts_i[i] );
                parts[count] = (i << 16) |
                    (unsigned int)( dscale * ( nshift + part_i->x[0]*shift[0] + part_i->x[1]*shift[1] + part_i->x[2]*shift[2] ) );
                count += 1;
                }
            for ( i = 0 ; i < count_j ; i++ ) {
                part_i = &( parts_j[i] );
                parts[count] = (i << 16) |
                    (unsigned int )( dscale * ( nshift + (part_i->x[0]+pshift[0])*shift[0] + (part_i->x[1]+pshift[1])*shift[1] + (part_i->x[2]+pshift[2])*shift[2] ) );
                count += 1;
                }

            /* Sort parts in cell_i in decreasing order with quicksort */
            qstack[0].lo = 0; qstack[0].hi = count_i - 1; qpos = 0;
            while ( qpos >= 0 ) {
                lo = qstack[qpos].lo; hi = qstack[qpos].hi;
                qpos -= 1;
                if ( hi - lo < 15 ) {
                    for ( i = lo ; i < hi ; i++ ) {
                        imax = i;
                        for ( j = i+1 ; j <= hi ; j++ )
                            if ( (parts[j] & 0xffff) > (parts[imax] & 0xffff) )
                                imax = j;
                        if ( imax != i ) {
                            temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                            }
                        }
                    }
                else {
                    pivot = parts[ ( lo + hi ) / 2 ] & 0xffff;
                    i = lo; j = hi;
                    while ( i <= j ) {
                        while ( (parts[i] & 0xffff) > pivot ) i++;
                        while ( (parts[j] & 0xffff) < pivot ) j--;
                        if ( i <= j ) {
                            if ( i < j ) {
                                temp = parts[i]; parts[i] = parts[j]; parts[j] = temp;
                                }
                            i += 1; j -= 1;
                            }
                        }
                    if ( lo < j ) {
                        qpos += 1;
                        qstack[qpos].lo = lo;
                        qstack[qpos].hi = j;
                        }
                    if ( i < hi ) {
                        qpos += 1;
                        qstack[qpos].lo = i;
                        qstack[qpos].hi = hi;
                        }
                    }
                }
                
            /* Sort parts in cell_j in increasing order with quicksort */
            qstack[0].lo = count_i; qstack[0].hi = count - 1; qpos = 0;
            while ( qpos >= 0 ) {
                lo = qstack[qpos].lo; hi = qstack[qpos].hi;
                qpos -= 1;
                if ( hi - lo < 15 ) {
                    for ( i = lo ; i < hi ; i++ ) {
                        imax = i;
                        for ( j = i+1 ; j <= hi ; j++ )
                            if ( (parts[j] & 0xffff) < (parts[imax] & 0xffff) )
                                imax = j;
                        if ( imax != i ) {
                            temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                            }
                        }
                    }
                else {
                    pivot = parts[ ( lo + hi ) / 2 ] & 0xffff;
                    i = lo; j = hi;
                    while ( i <= j ) {
                        while ( (parts[i] & 0xffff) < pivot ) i++;
                        while ( (parts[j] & 0xffff) > pivot ) j--;
                        if ( i <= j ) {
                            if ( i < j ) {
                                temp = parts[i]; parts[i] = parts[j]; parts[j] = temp;
                                }
                            i += 1; j -= 1;
                            }
                        }
                    if ( lo < j ) {
                        qpos += 1;
                        qstack[qpos].lo = lo;
                        qstack[qpos].hi = j;
                        }
                    if ( i < hi ) {
                        qpos += 1;
                        qstack[qpos].lo = i;
                        qstack[qpos].hi = hi;
                        }
                    }
                }
                
            /* Copy the sorted array to the pairwise list. */
            cp->count = count;
                
            }
            
        else {
            
            /* Get local copies of the list data. */
            parts = (void *)cp->pairs;
            
            }

        /* loop over the sorted list of particles in i */
        for ( i = 0 ; i < count_i ; i++ ) {
        
            /* Quit early? */
            if ( (parts[count_i] & 0xffff) - (parts[i] & 0xffff) > dmaxdist )
                break;

            /* get a handle on this particle */
            pid = parts[i] >> 16;
            part_i = &( parts_i[pid] );
            pix[0] = part_i->x[0] - pshift[0];
            pix[1] = part_i->x[1] - pshift[1];
            pix[2] = part_i->x[2] - pshift[2];
            pioff = part_i->type * emt;
            pif = &( part_i->f[0] );

            /* loop over the left particles */
            for ( j = 0 ; j < count_j && (parts[count_i+j] & 0xffff) - (parts[i] & 0xffff) < dmaxdist ; j++ ) {

                /* get a handle on the second particle */
                part_j = &( parts_j[ parts[count_i+j] >> 16 ] );

                /* fetch the potential, if any */
                pot = pots[ pioff + part_j->type ];
                if ( pot == NULL )
                    continue;

                /* get the distance between both particles */
                r2 = 0.0;
                for ( k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }

                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */

                #if defined(VECTORIZE)
                    /* add this interaction to the interaction queue. */
                    r2q[icount] = r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = pif;
                    effj[icount] = part_j->f;
                    potq[icount] = pot;
                    icount += 1;

                    #if defined(FPTYPE_SINGLE)
                        /* evaluate the interactions if the queue is full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4single( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f[l] * dxq[l*3+k];
                                    effi[l][k] -= w;
                                    effj[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount = 0;

                            }
                    #elif defined(FPTYPE_DOUBLE)
                        /* evaluate the interactions if the queue is full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4double( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f[l] * dxq[l*3+k];
                                    effi[l][k] -= w;
                                    effj[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount = 0;

                            }
                    #endif
                #else
                    /* evaluate the interaction */
                    #ifdef EXPLICIT_POTENTIALS
                        potential_eval_expl( pot , r2 , &e , &f );
                    #else
                        potential_eval( pot , r2 , &e , &f );
                    #endif

                    /* update the forces */
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = f * dx[k];
                        part_j->f[k] -= w;
                        pif[k] += w;
                        }

                    /* tabulate the energy */
                    epot += e;
                #endif

                }

            } /* loop over all particles */
            
        }
        
    #if defined(VEC_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ ) {
                potq[k] = potq[0];
                r2q[k] = r2q[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single( potq , r2q , e , f );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += e[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * dxq[l*3+k];
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
            potential_eval_vec_4double( potq , r2q , e , f );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount ; l++ ) {
                epot += e[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
    #endif
        
    /* Store the potential energy to cell_i. */
    if ( cell_j->flags & cell_flag_ghost || cell_i->flags & cell_flag_ghost )
        cell_i->epot += 0.5 * epot;
    else
        cell_i->epot += epot;
        
    /* Write local data back if needed. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* copy the particle data back */
        for ( i = 0 ; i < count_i ; i++ ) {
            cell_i->parts[i].f[0] = parts_i[i].f[0];
            cell_i->parts[i].f[1] = parts_i[i].f[1];
            cell_i->parts[i].f[2] = parts_i[i].f[2];
            }
        if ( cell_i != cell_j )
            for ( i = 0 ; i < count_j ; i++ ) {
                cell_j->parts[i].f[0] = parts_j[i].f[0];
                cell_j->parts[i].f[1] = parts_j[i].f[1];
                cell_j->parts[i].f[2] = parts_j[i].f[2];
                }
        }
        
    /* since nothing bad happened to us... */
    return runner_err_ok;

    }
    
    