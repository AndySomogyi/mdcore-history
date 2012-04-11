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
#include "fifo.h"
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
extern unsigned int runner_rcount;
    

/**
 * @brief Compute the pairwise interactions for the given pair using the sorted
 * interactions algorithm.
 *
 * @param r The #runner computing the pair.
 * @param cell_i The first cell.
 * @param cell_j The second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err)
 *
 * Sorts the particles from @c cell_i and @c cell_j along the normalized axis
 * @c pshift and tries to interact only those particles that are within
 * the cutoff distance along that axis.
 * 
 * It is assumed that @c cell_i != @c cell_j.
 *
 * @sa #runner_dopair.
 */
 
int runner_dopair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift ) {

    struct part *part_i, *part_j;
    struct space *s;
    int count = 0;
    int pid, i, j, k;
    struct part *parts_i, *parts_j;
    double epot = 0.0;
    struct potential *pot, **pots;
    struct engine *eng;
    int emt, pioff, count_i, count_j;
    FPTYPE cutoff, cutoff2, r2, dx[3], w;
    unsigned int *parts, dcutoff;
    FPTYPE dscale;
    FPTYPE shift[3], nshift, inshift;
    FPTYPE pix[3], *pif;
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
    
    /* get some useful data */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    pots = eng->p;
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    dscale = (FPTYPE)SHRT_MAX / ( 3 * sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] ) );
    dcutoff = 2 + dscale * cutoff;
    
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
        
    /* Are cell_i and cell_j the same? */
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
                part_j = &(parts_i[j]);
                
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                /* fetch the potential, if any */
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL )
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
                    
                } /* loop over all other particles */
        
            } /* loop over all particles */
    
        }
        
    /* Otherwise, sorted interaction. */
    else {
    
        /* Allocate work arrays on stack. */
        if ( ( parts = alloca( sizeof(unsigned int) * (count_i + count_j) ) ) == NULL )
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
        runner_sort_descending( parts , count_i );

        /* Sort parts in cell_j in increasing order with quicksort */
        runner_sort_ascending( &parts[count_i] , count_j );
                

        /* loop over the sorted list of particles in i */
        for ( i = 0 ; i < count_i ; i++ ) {
        
            /* Quit early? */
            if ( (parts[count_i] & 0xffff) - (parts[i] & 0xffff) > dcutoff )
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
            for ( j = 0 ; j < count_j && (parts[count_i+j] & 0xffff) - (parts[i] & 0xffff) < dcutoff ; j++ ) {

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
                        part_j->f[k] -= w;
                        pif[k] += w;
                        }

                    /* tabulate the energy */
                    epot += e;
                #endif

                }

            } /* loop over all particles */
            
        } /* pair or self interaction. */

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
        
    /* Store the potential energy to cell_i. */
    if ( cell_j->flags & cell_flag_ghost || cell_i->flags & cell_flag_ghost )
        cell_i->epot += 0.5 * epot;
    else
        cell_i->epot += epot;
        
    /* since nothing bad happened to us... */
    return runner_err_ok;

    }


/**
 * @brief Compute the pairwise interactions for the given pair using the sorted
 * interactions algorithm with explicit electrostatics.
 *
 * @param r The #runner computing the pair.
 * @param cell_i The first cell.
 * @param cell_j The second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err)
 *
 * Sorts the particles from @c cell_i and @c cell_j along the normalized axis
 * @c pshift and tries to interact only those particles that are within
 * the cutoff distance along that axis.
 * 
 * It is assumed that @c cell_i != @c cell_j.
 *
 * @sa #runner_dopair.
 */
 
int runner_dopair_ee ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift ) {

    struct part *part_i, *part_j;
    struct space *s;
    int *left, count = 0, lcount = 0;
    int i, j, k, imax, qpos, lo, hi;
    struct {
        short int lo, hi;
        } *qstack;
    struct part *parts_i, *parts_j;
    double epot = 0.0;
    struct potential *pot, *ep;
    struct engine *eng;
    int emt, pjoff, pioff, count_i, count_j;
    FPTYPE cutoff, cutoff2, r2, dx[3], dscale, w;
    struct {
        short int d, ind;
        } *parts, temp;
    short int pivot;
    FPTYPE shift[3], inshift;
    FPTYPE pjx[3], pix[3], piq, pjq, pijq;
#if defined(VECTORIZE)
    struct potential *potq[4], *potq_2[4];
    int icount = 0, icount_2 = 0, l;
    FPTYPE *effi[4], *effj[4], *pjf;
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE e[4] __attribute__ ((aligned (16)));
    FPTYPE f[4] __attribute__ ((aligned (16)));
    FPTYPE q[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12];
    FPTYPE *effi_2[4], *effj_2[4];
    FPTYPE r2q_2[4] __attribute__ ((aligned (16)));
    FPTYPE e_2[4] __attribute__ ((aligned (16)));
    FPTYPE f_2[4] __attribute__ ((aligned (16)));
    FPTYPE q_2[4] __attribute__ ((aligned (16)));
    FPTYPE dxq_2[12];
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
    ep = eng->ep;
    s = &(eng->s);
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    dscale = (FPTYPE)SHRT_MAX / sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] );
    
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
        
    /* Self interaction? */
    if ( cell_i == cell_j ) {
    
        /* loop over all particles */
        for ( i = 1 ; i < count_i ; i++ ) {
        
            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            pioff = part_i->type * emt;
            piq = part_i->q;
        
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
            
                /* get the other particle */
                part_j = &(parts_i[j]);
                pijq = piq * part_j->q;
                
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                /* fetch the potential, if any */
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL && pijq == 0.0f )
                    continue;
                    
                #if defined(VECTORIZE)
                    
                    /* both potential and charge. */
                    if ( pot != 0 && pijq != 0.0 ) {
                    
                        /* add this interaction to the ee interaction queue. */
                        r2q_2[icount] = r2;
                        dxq_2[icount*3] = dx[0];
                        dxq_2[icount*3+1] = dx[1];
                        dxq_2[icount*3+2] = dx[2];
                        effi_2[icount] = part_i->f;
                        effj_2[icount] = part_j->f;
                        potq_2[icount] = pot;
                        icount_2 += 1;
                        
                        }
                        
                    /* only charge or potential. */
                    else {
                    
                        /* add this interaction to the plain interaction queue. */
                        r2q[icount] = r2;
                        dxq[icount*3] = dx[0];
                        dxq[icount*3+1] = dx[1];
                        dxq[icount*3+2] = dx[2];
                        effi[icount] = part_i->f;
                        effj[icount] = part_j->f;
                        if ( pot == NULL ) {
                            potq[icount] = ep;
                            q[icount] = pijq;
                            }
                        else {
                            potq[icount] = pot;
                            q[icount] = 1.0;
                            }
                        icount += 1;
                        
                        }

                    #if defined(FPTYPE_SINGLE)
                        /* evaluate the interactions if the queues are full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4single( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e[l] * q[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f[l] * q[l] * dxq[l*3+k];
                                    effi[l][k] -= w;
                                    effj[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount = 0;

                            }
                        else if ( icount_2 == 4 ) {

                            potential_eval_vec_4single_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e_2[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f_2[l] * dxq_2[l*3+k];
                                    effi_2[l][k] -= w;
                                    effj_2[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount_2 = 0;

                            }
                    #elif defined(FPTYPE_DOUBLE)
                        /* evaluate the interactions if the queues are full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4double( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 2 ; l++ ) {
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
                        /* evaluate the interactions if the queues are full. */
                        if ( icount_2 == 2 ) {

                            potential_eval_vec_2double_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 2 ; l++ ) {
                                epot += e_2[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f_2[l] * dxq_2[l*3+k];
                                    effi_2[l][k] -= w;
                                    effj_2[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount_2 = 0;

                            }
                    #endif
                #else
                    if ( pot != NULL ) {
                    
                        /* evaluate the interaction */
                        if ( pijq != 0.0 )
                            potential_eval_ee( pot , ep , r2 , pijq , &e , &f );
                        else
                            potential_eval( pot , r2 , &e , &f );

                        /* update the forces */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f * dx[k];
                            part_i->f[k] -= w;
                            part_j->f[k] += w;
                            }

                        /* tabulate the energy */
                        epot += e;
                    
                        }
                        
                    else {
                    
                        /* evaluate the interaction */
                        potential_eval( ep , r2 , &e , &f );
                    
                        /* update the forces */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f * pijq * dx[k];
                            part_i->f[k] -= w;
                            part_j->f[k] += w;
                            }

                        /* tabulate the energy */
                        epot += e * pijq;
                    
                        }
                #endif
                    
                } /* loop over all other particles */
        
            } /* loop over all particles */
    
        }
        
    /* Otherwise, genuine pair. */
    else {
        
        /* Allocate work arrays on stack. */
        if ( ( left = (int *)alloca( sizeof(int) * count_i ) ) == NULL ||
             ( parts = alloca( sizeof(short int) * 2 * (count_i + count_j) ) ) == NULL ||
             ( qstack = alloca( sizeof(short int) * 2 * (count_i + count_j) ) ) == NULL )
            return error(runner_err_malloc);
        
        /* start by filling the particle ids of both cells into ind and d */
        inshift = 1.0 / sqrt( pshift[0]*pshift[0] + pshift[1]*pshift[1] + pshift[2]*pshift[2] );
        shift[0] = pshift[0]*inshift; shift[1] = pshift[1]*inshift; shift[2] = pshift[2]*inshift;
        for ( i = 0 ; i < count_i ; i++ ) {
            part_i = &( parts_i[i] );
            parts[count].ind = -i - 1;
            parts[count].d = dscale * ( part_i->x[0]*shift[0] + part_i->x[1]*shift[1] + part_i->x[2]*shift[2] );
            count += 1;
            }
        for ( i = 0 ; i < count_j ; i++ ) {
            part_i = &( parts_j[i] );
            parts[count].ind = i;
            parts[count].d = 1.0 + dscale * ( (part_i->x[0]+pshift[0])*shift[0] + (part_i->x[1]+pshift[1])*shift[1] + (part_i->x[2]+pshift[2])*shift[2] - cutoff );
            count += 1;
            }

        /* sort with quicksort */
        qstack[0].lo = 0; qstack[0].hi = count - 1; qpos = 0;
        while ( qpos >= 0 ) {
            lo = qstack[qpos].lo; hi = qstack[qpos].hi;
            qpos -= 1;
            if ( hi - lo < 15 ) {
                for ( i = lo ; i < hi ; i++ ) {
                    imax = i;
                    for ( j = i+1 ; j <= hi ; j++ )
                        if ( parts[j].d > parts[imax].d )
                            imax = j;
                    if ( imax != i ) {
                        temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                        }
                    }
                }
            else {
                pivot = parts[ ( lo + hi ) / 2 ].d;
                i = lo; j = hi;
                while ( i <= j ) {
                    while ( parts[i].d > pivot ) i++;
                    while ( parts[j].d < pivot ) j--;
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

        /* sort with selection sort */
        /* for ( i = 0 ; i < count-1 ; i++ ) { 
            imax = i;
            for ( j = i+1 ; j < count ; j++ )
                if ( parts[j].d > parts[imax].d )
                    imax = j;
            if ( imax != i ) {
                temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                }
            } */

        /* loop over the sorted list of particles */
        for ( i = 0 ; i < count ; i++ ) {

            /* is this a particle from the left? */
            if ( parts[i].ind < 0 )
                left[lcount++] = -parts[i].ind - 1;

            /* it's from the right, interact with all left particles */
            else {

                /* get a handle on this particle */
                part_j = &( parts_j[parts[i].ind] );
                pjx[0] = part_j->x[0] + pshift[0];
                pjx[1] = part_j->x[1] + pshift[1];
                pjx[2] = part_j->x[2] + pshift[2];
                pjoff = part_j->type * emt;
                pjq = part_j->q;
                #if defined(VECTORIZE)
                    pjf = part_j->f;
                #endif

                /* loop over the left particles */
                for ( j = 0 ; j < lcount ; j++ ) {

                    /* get a handle on the second particle */
                    part_i = &( parts_i[left[j]] );
                    pijq = pjq * part_i->q;

                    /* get the distance between both particles */
                    for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                        dx[k] = part_i->x[k] - pjx[k];
                        r2 += dx[k] * dx[k];
                        }

                    /* is this within cutoff? */
                    if ( r2 > cutoff2 )
                        continue;

                    /* fetch the potential, if any */
                    pot = eng->p[ pjoff + part_i->type ];
                    if ( pot == NULL && pijq == 0.0f )
                        continue;

                    #if defined(VECTORIZE)

                        /* both potential and charge. */
                        if ( pot != 0 && pijq != 0.0 ) {

                            /* add this interaction to the ee interaction queue. */
                            r2q_2[icount] = r2;
                            dxq_2[icount*3] = dx[0];
                            dxq_2[icount*3+1] = dx[1];
                            dxq_2[icount*3+2] = dx[2];
                            effi_2[icount] = part_i->f;
                            effj_2[icount] = part_j->f;
                            potq_2[icount] = pot;
                            icount_2 += 1;

                            }

                        /* only charge or potential. */
                        else {

                            /* add this interaction to the plain interaction queue. */
                            r2q[icount] = r2;
                            dxq[icount*3] = dx[0];
                            dxq[icount*3+1] = dx[1];
                            dxq[icount*3+2] = dx[2];
                            effi[icount] = part_i->f;
                            effj[icount] = part_j->f;
                            if ( pot == NULL ) {
                                potq[icount] = ep;
                                q[icount] = pijq;
                                }
                            else {
                                potq[icount] = pot;
                                q[icount] = 1.0;
                                }
                            icount += 1;

                            }

                        #if defined(FPTYPE_SINGLE)
                            /* evaluate the interactions if the queues are full. */
                            if ( icount == 4 ) {

                                potential_eval_vec_4single( potq , r2q , e , f );

                                /* update the forces and the energy */
                                for ( l = 0 ; l < 4 ; l++ ) {
                                    epot += e[l] * q[l];
                                    for ( k = 0 ; k < 3 ; k++ ) {
                                        w = f[l] * q[l] * dxq[l*3+k];
                                        effi[l][k] -= w;
                                        effj[l][k] += w;
                                        }
                                    }

                                /* re-set the counter. */
                                icount = 0;

                                }
                            else if ( icount_2 == 4 ) {

                                potential_eval_vec_4single_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

                                /* update the forces and the energy */
                                for ( l = 0 ; l < 4 ; l++ ) {
                                    epot += e_2[l];
                                    for ( k = 0 ; k < 3 ; k++ ) {
                                        w = f_2[l] * dxq_2[l*3+k];
                                        effi_2[l][k] -= w;
                                        effj_2[l][k] += w;
                                        }
                                    }

                                /* re-set the counter. */
                                icount_2 = 0;

                                }
                        #elif defined(FPTYPE_DOUBLE)
                            /* evaluate the interactions if the queues are full. */
                            if ( icount == 4 ) {

                                potential_eval_vec_4double( potq , r2q , e , f );

                                /* update the forces and the energy */
                                for ( l = 0 ; l < 2 ; l++ ) {
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
                            /* evaluate the interactions if the queues are full. */
                            if ( icount_2 == 2 ) {

                                potential_eval_vec_2double_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

                                /* update the forces and the energy */
                                for ( l = 0 ; l < 2 ; l++ ) {
                                    epot += e_2[l];
                                    for ( k = 0 ; k < 3 ; k++ ) {
                                        w = f_2[l] * dxq_2[l*3+k];
                                        effi_2[l][k] -= w;
                                        effj_2[l][k] += w;
                                        }
                                    }

                                /* re-set the counter. */
                                icount_2 = 0;

                                }
                        #endif
                    #else
                        if ( pot != NULL ) {

                            /* evaluate the interaction */
                            if ( pijq != 0.0 )
                                potential_eval_ee( pot , ep , r2 , pijq , &e , &f );
                            else
                                potential_eval( pot , r2 , &e , &f );

                            /* update the forces */
                            for ( k = 0 ; k < 3 ; k++ ) {
                                w = f * dx[k];
                                part_i->f[k] -= w;
                                part_j->f[k] += w;
                                }

                            /* tabulate the energy */
                            epot += e;

                            }

                        else {

                            /* evaluate the interaction */
                            potential_eval( ep , r2 , &e , &f );

                            /* update the forces */
                            for ( k = 0 ; k < 3 ; k++ ) {
                                w = f * pijq * dx[k];
                                part_i->f[k] -= w;
                                part_j->f[k] += w;
                                }

                            /* tabulate the energy */
                            epot += e * pijq;

                            }
                    #endif

                    }

                }

            } /* loop over all particles */
            
        } /* genuine pair? */
        
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
                epot += e[l] * q[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * q[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
        if ( icount_2 > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount_2 ; k < 4 ; k++ ) {
                potq_2[k] = potq_2[0];
                r2q_2[k] = r2q_2[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount_2 ; l++ ) {
                epot += e_2[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * dxq_2[l*3+k];
                    effi_2[l][k] -= w;
                    effj_2[l][k] += w;
                    }
                }

            }
    #elif defined(VEC_DOUBLE)
        /* are there any leftovers? */
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
                epot += e[l] * q[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * q[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
        if ( icount_2 > 0 ) {

            /* copy the first potential to the last entries */
            potq_2[1] = potq_2[0];
            r2q_2[1] = r2q_2[0];

            /* evaluate the potentials */
            potential_eval_vec_2double_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

            /* for each entry, update the forces and energy */
            epot += e_2[0];
            for ( k = 0 ; k < 3 ; k++ ) {
                w = f[0] * dxq_2[k];
                effi_2[0][k] -= w;
                effj_2[0][k] += w;
                }

            }
    #endif
        
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
        
    /* Store the potential energy to cell_i. */
    if ( cell_j->flags & cell_flag_ghost || cell_i->flags & cell_flag_ghost )
        cell_i->epot += 0.5 * epot;
    else
        cell_i->epot += epot;
        
    /* since nothing bad happened to us... */
    return runner_err_ok;

    }


/**
 * @brief Compute the pairwise interactions for the given pair.
 *
 * @param r The #runner computing the pair.
 * @param cell_i The first cell.
 * @param cell_j The second cell.
 * @param shift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err)
 *
 * Computes the interactions between all the particles in @c cell_i and all
 * the paritcles in @c cell_j. @c cell_i and @c cell_j may be the same cell.
 *
 * @sa #runner_sortedpair.
 */

int runner_dopair_unsorted ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift ) {

    int i, j, k, emt, pioff, count_i, count_j;
    FPTYPE cutoff2, r2, dx[3], pix[3], w;
    double epot = 0.0;
    struct engine *eng;
    struct part *part_i, *part_j, *parts_i, *parts_j;
    struct potential *pot;
    struct space *s;
#if defined(VECTORIZE)
    int l, icount = 0;
    FPTYPE *effi[4], *effj[4];
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE e[4] __attribute__ ((aligned (16)));
    FPTYPE f[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12];
    struct potential *potq[4];
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
    cutoff2 = s->cutoff2;
        
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
        
    /* is this a genuine pair or a cell against itself */
    if ( cell_i == cell_j ) {
    
        /* loop over all particles */
        for ( i = 1 ; i < count_i ; i++ ) {
        
            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            pioff = part_i->type * emt;
        
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
            
                /* get the other particle */
                part_j = &(parts_i[j]);
                
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
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL )
                    continue;
                    
                #if defined(VECTORIZE)
                    /* add this interaction to the interaction queue. */
                    r2q[icount] = r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = part_i->f;
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
                        part_i->f[k] -= w;
                        part_j->f[k] += w;
                        }

                    /* tabulate the energy */
                    epot += e;
                #endif
                    
                } /* loop over all other particles */
        
            } /* loop over all particles */
    
        }
        
    /* no, it's a genuine pair */
    else {
    
        /* loop over all particles */
        for ( i = 0 ; i < count_i ; i++ ) {
        
            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0] - shift[0];
            pix[1] = part_i->x[1] - shift[1];
            pix[2] = part_i->x[2] - shift[2];
            pioff = part_i->type * emt;
            
            /* loop over all other particles */
            for ( j = 0 ; j < count_j ; j++ ) {
            
                /* get the other particle */
                part_j = &(parts_j[j]);

                /* fetch the potential, if any */
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */
                    
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL )
                    continue;
                    
                #if defined(VECTORIZE)
                    /* add this interaction to the interaction queue. */
                    r2q[icount] = r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = part_i->f;
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
                        part_i->f[k] -= w;
                        part_j->f[k] += w;
                        }

                    /* tabulate the energy */
                    epot += e;
                #endif
                    
                } /* loop over all other particles */
        
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
        
    /* Store the potential energy to cell_i. */
    if ( cell_j->flags & cell_flag_ghost || cell_i->flags & cell_flag_ghost )
        cell_i->epot += 0.5 * epot;
    else
        cell_i->epot += epot;
        
    /* all is well that ends ok */
    return runner_err_ok;

    }


/**
 * @brief Compute the pairwise interactions for the given pair using
 *      explicit electrostatic interactions.
 *
 * @param r The #runner computing the pair.
 * @param cell_i The first cell.
 * @param cell_j The second cell.
 * @param shift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err)
 *
 * Computes the interactions between all the particles in @c cell_i and all
 * the paritcles in @c cell_j. @c cell_i and @c cell_j may be the same cell.
 *
 * @sa #runner_sortedpair.
 */

int runner_dopair_unsorted_ee ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift ) {

    int i, j, k, emt, pioff;
    FPTYPE cutoff2, r2, dx[3], pix[3], piq, pijq, w;
    double epot = 0.0;
    struct engine *eng;
    struct part *part_i, *part_j, *parts_i, *parts_j = NULL;
    struct potential *pot, *ep;
    struct space *s;
    int count_i, count_j;
#if defined(VECTORIZE)
    int l, icount = 0, icount_2 = 0;
    FPTYPE *effi[4], *effj[4], *effi_2[4], *effj_2[4];
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE e[4] __attribute__ ((aligned (16)));
    FPTYPE f[4] __attribute__ ((aligned (16)));
    FPTYPE q[4] __attribute__ ((aligned (16)));
    FPTYPE r2q_2[4] __attribute__ ((aligned (16)));
    FPTYPE e_2[4] __attribute__ ((aligned (16)));
    FPTYPE f_2[4] __attribute__ ((aligned (16)));
    FPTYPE q_2[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12], dxq_2[12];
    struct potential *potq[4], *potq_2[4];
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
    ep = eng->ep;
    emt = eng->max_type;
    s = &(eng->s);
    cutoff2 = s->cutoff2;
        
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
        
    /* is this a genuine pair or a cell against itself */
    if ( cell_i == cell_j ) {
    
        /* loop over all particles */
        for ( i = 1 ; i < count_i ; i++ ) {
        
            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            pioff = part_i->type * emt;
            piq = part_i->q;
        
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
            
                /* get the other particle */
                part_j = &(parts_i[j]);
                pijq = piq * part_j->q;
                
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                /* fetch the potential, if any */
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL && pijq == 0.0f )
                    continue;
                    
                #if defined(VECTORIZE)
                    
                    /* both potential and charge. */
                    if ( pot != 0 && pijq != 0.0 ) {
                    
                        /* add this interaction to the ee interaction queue. */
                        r2q_2[icount] = r2;
                        dxq_2[icount*3] = dx[0];
                        dxq_2[icount*3+1] = dx[1];
                        dxq_2[icount*3+2] = dx[2];
                        effi_2[icount] = part_i->f;
                        effj_2[icount] = part_j->f;
                        potq_2[icount] = pot;
                        icount_2 += 1;
                        
                        }
                        
                    /* only charge or potential. */
                    else {
                    
                        /* add this interaction to the plain interaction queue. */
                        r2q[icount] = r2;
                        dxq[icount*3] = dx[0];
                        dxq[icount*3+1] = dx[1];
                        dxq[icount*3+2] = dx[2];
                        effi[icount] = part_i->f;
                        effj[icount] = part_j->f;
                        if ( pot == NULL ) {
                            potq[icount] = ep;
                            q[icount] = pijq;
                            }
                        else {
                            potq[icount] = pot;
                            q[icount] = 1.0;
                            }
                        icount += 1;
                        
                        }

                    #if defined(FPTYPE_SINGLE)
                        /* evaluate the interactions if the queues are full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4single( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e[l] * q[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f[l] * q[l] * dxq[l*3+k];
                                    effi[l][k] -= w;
                                    effj[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount = 0;

                            }
                        else if ( icount_2 == 4 ) {

                            potential_eval_vec_4single_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e_2[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f_2[l] * dxq_2[l*3+k];
                                    effi_2[l][k] -= w;
                                    effj_2[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount_2 = 0;

                            }
                    #elif defined(FPTYPE_DOUBLE)
                        /* evaluate the interactions if the queues are full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4double( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 2 ; l++ ) {
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
                        /* evaluate the interactions if the queues are full. */
                        if ( icount_2 == 2 ) {

                            potential_eval_vec_2double_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 2 ; l++ ) {
                                epot += e_2[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f_2[l] * dxq_2[l*3+k];
                                    effi_2[l][k] -= w;
                                    effj_2[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount_2 = 0;

                            }
                    #endif
                #else
                    if ( pot != NULL ) {
                    
                        /* evaluate the interaction */
                        if ( pijq != 0.0 )
                            potential_eval_ee( pot , ep , r2 , pijq , &e , &f );
                        else
                            potential_eval( pot , r2 , &e , &f );

                        /* update the forces */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f * dx[k];
                            part_i->f[k] -= w;
                            part_j->f[k] += w;
                            }

                        /* tabulate the energy */
                        epot += e;
                    
                        }
                        
                    else {
                    
                        /* evaluate the interaction */
                        potential_eval( ep , r2 , &e , &f );
                    
                        /* update the forces */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f * pijq * dx[k];
                            part_i->f[k] -= w;
                            part_j->f[k] += w;
                            }

                        /* tabulate the energy */
                        epot += e * pijq;
                    
                        }
                #endif
                    
                } /* loop over all other particles */
        
            } /* loop over all particles */
    
        }
        
    /* no, it's a genuine pair */
    else {
    
        /* loop over all particles */
        for ( i = 0 ; i < count_i ; i++ ) {
        
            /* get the particle */
            part_i = &(parts_i[i]);
            pix[0] = part_i->x[0] - shift[0];
            pix[1] = part_i->x[1] - shift[1];
            pix[2] = part_i->x[2] - shift[2];
            pioff = part_i->type * emt;
            piq = part_i->q;
            
            /* loop over all other particles */
            for ( j = 0 ; j < count_j ; j++ ) {
            
                /* get the other particle */
                part_j = &(parts_j[j]);
                pijq = piq * part_j->q;
                
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                    
                /* fetch the potential, if any */
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL && pijq == 0.0f )
                    continue;
                    
                #if defined(VECTORIZE)
                    
                    /* both potential and charge. */
                    else if ( pot != 0 && pijq != 0.0 ) {
                    
                        /* add this interaction to the ee interaction queue. */
                        r2q_2[icount] = r2;
                        dxq_2[icount*3] = dx[0];
                        dxq_2[icount*3+1] = dx[1];
                        dxq_2[icount*3+2] = dx[2];
                        effi_2[icount] = part_i->f;
                        effj_2[icount] = part_j->f;
                        potq_2[icount] = pot;
                        icount_2 += 1;
                        
                        }
                        
                    /* only charge or potential. */
                    else {
                    
                        /* add this interaction to the plain interaction queue. */
                        r2q[icount] = r2;
                        dxq[icount*3] = dx[0];
                        dxq[icount*3+1] = dx[1];
                        dxq[icount*3+2] = dx[2];
                        effi[icount] = part_i->f;
                        effj[icount] = part_j->f;
                        if ( pot == NULL ) {
                            potq[icount] = ep;
                            q[icount] = pijq;
                            }
                        else {
                            potq[icount] = pot;
                            q[icount] = 1.0;
                            }
                        icount += 1;
                        
                        }

                    #if defined(FPTYPE_SINGLE)
                        /* evaluate the interactions if the queues are full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4single( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e[l] * q[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f[l] * q[l] * dxq[l*3+k];
                                    effi[l][k] -= w;
                                    effj[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount = 0;

                            }
                        else if ( icount_2 == 4 ) {

                            potential_eval_vec_4single_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 4 ; l++ ) {
                                epot += e_2[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f_2[l] * dxq_2[l*3+k];
                                    effi_2[l][k] -= w;
                                    effj_2[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount_2 = 0;

                            }
                    #elif defined(FPTYPE_DOUBLE)
                        /* evaluate the interactions if the queues are full. */
                        if ( icount == 4 ) {

                            potential_eval_vec_4double( potq , r2q , e , f );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 2 ; l++ ) {
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
                        /* evaluate the interactions if the queues are full. */
                        if ( icount_2 == 2 ) {

                            potential_eval_vec_2double_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

                            /* update the forces and the energy */
                            for ( l = 0 ; l < 2 ; l++ ) {
                                epot += e_2[l];
                                for ( k = 0 ; k < 3 ; k++ ) {
                                    w = f_2[l] * dxq_2[l*3+k];
                                    effi_2[l][k] -= w;
                                    effj_2[l][k] += w;
                                    }
                                }

                            /* re-set the counter. */
                            icount_2 = 0;

                            }
                    #endif
                #else
                    if ( pot != NULL ) {
                    
                        /* evaluate the interaction */
                        if ( pijq != 0.0 )
                            potential_eval_ee( pot , ep , r2 , pijq , &e , &f );
                        else
                            potential_eval( pot , r2 , &e , &f );

                        /* update the forces */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f * dx[k];
                            part_i->f[k] -= w;
                            part_j->f[k] += w;
                            }

                        /* tabulate the energy */
                        epot += e;
                    
                        }
                        
                    else {
                    
                        /* evaluate the interaction */
                        potential_eval( ep , r2 , &e , &f );
                    
                        /* update the forces */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f * pijq * dx[k];
                            part_i->f[k] -= w;
                            part_j->f[k] += w;
                            }

                        /* tabulate the energy */
                        epot += e * pijq;
                    
                        }
                #endif
                    
                } /* loop over all other particles */
        
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
                epot += e[l] * q[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * q[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
        if ( icount_2 > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount_2 ; k < 4 ; k++ ) {
                potq_2[k] = potq_2[0];
                r2q_2[k] = r2q_2[0];
                }

            /* evaluate the potentials */
            potential_eval_vec_4single_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

            /* for each entry, update the forces and energy */
            for ( l = 0 ; l < icount_2 ; l++ ) {
                epot += e_2[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * dxq_2[l*3+k];
                    effi_2[l][k] -= w;
                    effj_2[l][k] += w;
                    }
                }

            }
    #elif defined(VEC_DOUBLE)
        /* are there any leftovers? */
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
                epot += e[l] * q[l];
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = f[l] * q[l] * dxq[l*3+k];
                    effi[l][k] -= w;
                    effj[l][k] += w;
                    }
                }

            }
        if ( icount_2 > 0 ) {

            /* copy the first potential to the last entries */
            potq_2[1] = potq_2[0];
            r2q_2[1] = r2q_2[0];

            /* evaluate the potentials */
            potential_eval_vec_2double_ee( potq_2 , ep , r2q_2 , q_2 , e_2 , f_2 );

            /* for each entry, update the forces and energy */
            epot += e_2[0];
            for ( k = 0 ; k < 3 ; k++ ) {
                w = f[0] * dxq_2[k];
                effi_2[0][k] -= w;
                effj_2[0][k] += w;
                }

            }
    #endif
        
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
        
    /* Store the potential energy to cell_i. */
    if ( cell_j->flags & cell_flag_ghost || cell_i->flags & cell_flag_ghost )
        cell_i->epot += 0.5 * epot;
    else
        cell_i->epot += epot;
        
    /* all is well that ends ok */
    return runner_err_ok;

    }


