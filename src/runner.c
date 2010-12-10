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


/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>
#ifdef CELL
    #include <libspe2.h>
    #include <libmisc.h>
    #define ceil128(v) (((v) + 127) & ~127)
#endif
#ifdef __SSE__
    #include <xmmintrin.h>
#endif

/* Include local headers */
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


/* Global variables. */
/** The ID of the last error. */
int runner_err = runner_err_ok;
unsigned int runner_rcount = 0;

/* the error macro. */
#define error(id)				( runner_err = errs_register( id , runner_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *runner_err_msg[9] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "An error occured when calling a space function.",
    "A call to a pthread routine failed.",
    "An error occured when calling an engine function.",
    "An error occured when calling an SPE function.",
    "An error occured with the memory flow controler.",
    "The requested functionality is not available."
	};
    
    
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
 
#if defined(__SSE__) && defined(FPTYPE_SINGLE)
int runner_sortedpair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift ) {

    struct part *part_i, *part_j;
    struct space *s;
    struct potential *potq[4];
    struct engine *eng;
    int icount = 0;
    FPTYPE cutoff, cutoff2, r2, *effi[4], *effj[4];
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE e[4] __attribute__ ((aligned (16)));
    FPTYPE f[4] __attribute__ ((aligned (16)));
    FPTYPE dx[3], dxq[12];
    FPTYPE d[runner_maxparts], temp, pivot;
    FPTYPE shift[3];
    int ind[runner_maxparts], left[runner_maxparts], count = 0, lcount = 0;
    int i, j, k, l, imax, qpos, lo, hi;
    struct {
        int lo, hi;
        } qstack[runner_maxqstack];
    struct part *parts_i, *parts_j;
    
    /* get the space and cutoff */
    eng = r->e;
    s = &(eng->s);
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    
    /* break early if one of the cells is empty */
    if ( cell_i->count == 0 || cell_j->count == 0 )
        return runner_err_ok;
    
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * cell_i->count );
        parts_j = (struct part *)alloca( sizeof(struct part) * cell_j->count );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * cell_i->count );
        memcpy( parts_j , cell_j->parts , sizeof(struct part) * cell_j->count );
        }
        
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
    /* start by filling the particle ids of both cells into ind and d */
    temp = 1.0 / sqrt( pshift[0]*pshift[0] + pshift[1]*pshift[1] + pshift[2]*pshift[2] );
    shift[0] = pshift[0]*temp; shift[1] = pshift[1]*temp; shift[2] = pshift[2]*temp;
    for ( i = 0 ; i < cell_i->count ; i++ ) {
        part_i = &( parts_i[i] );
        ind[count] = -i - 1;
        d[count] = part_i->x[0]*shift[0] + part_i->x[1]*shift[1] + part_i->x[2]*shift[2];
        count += 1;
        }
    for ( i = 0 ; i < cell_j->count ; i++ ) {
        part_i = &( parts_j[i] );
        ind[count] = i;
        d[count] = (part_i->x[0]+pshift[0])*shift[0] + (part_i->x[1]+pshift[1])*shift[1] + (part_i->x[2]+pshift[2])*shift[2] - cutoff;
        count += 1;
        }
        
    /* sort with quicksort */
    qstack[0].lo = 0; qstack[0].hi = count - 1; qpos = 0;
    while ( qpos >= 0 ) {
        lo = qstack[qpos].lo; hi = qstack[qpos].hi;
        qpos -= 1;
        if ( hi - lo < 10 ) {
            for ( i = lo ; i < hi ; i++ ) {
                imax = i;
                for ( j = i+1 ; j <= hi ; j++ )
                    if ( d[j] > d[imax] )
                        imax = j;
                if ( imax != i ) {
                    k = ind[imax]; ind[imax] = ind[i]; ind[i] = k;
                    temp = d[imax]; d[imax] = d[i]; d[i] = temp;
                    }
                }
            }
        else {
            pivot = d[ ( lo + hi ) / 2 ];
            i = lo; j = hi;
            while ( i <= j ) {
                while ( d[i] > pivot ) i++;
                while ( d[j] < pivot ) j--;
                if ( i <= j ) {
                    if ( i < j ) {
                        k = ind[i]; ind[i] = ind[j]; ind[j] = k;
                        temp = d[i]; d[i] = d[j]; d[j] = temp;
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
            if ( d[j] > d[imax] )
                imax = j;
        if ( imax != i ) {
            k = ind[imax]; ind[imax] = ind[i]; ind[i] = k;
            temp = d[imax]; d[imax] = d[i]; d[i] = temp;
            }
        } */
    
    /* loop over the sorted list of particles */
    for ( i = 0 ; i < count ; i++ ) {
    
        /* is this a particle from the left? */
        if ( ind[i] < 0 )
            left[lcount++] = -ind[i] - 1;
            
        /* it's from the right, interact with all left particles */
        else {
        
            /* get a handle on this particle */
            part_j = &( parts_j[ind[i]] );
        
            /* loop over the left particles */
            for ( j = lcount-1 ; j >= 0 ; j-- ) {
            
                /* get a handle on the second particle */
                part_i = &( parts_i[left[j]] );
            
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = part_i->x[k] - part_j->x[k] - pshift[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                /* fetch the potential, if any */
                potq[icount] = eng->p[ part_i->type * eng->max_type + part_j->type ];
                if ( potq[icount] == NULL )
                    continue;
                
                /* add this interaction to the interaction queue. */
                r2q[icount] = r2;
                dxq[icount*3] = dx[0];
                dxq[icount*3+1] = dx[1];
                dxq[icount*3+2] = dx[2];
                effi[icount] = part_i->f;
                effj[icount] = part_j->f;
                icount += 1;
                    
                /* evaluate the interactions if the queue is full. */
                if ( icount == 4 ) {
                
                    potential_eval_vec_single( potq , r2q , e , f );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        cell_i->epot += e[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            effi[l][k] += -f[l] * dxq[l*3+k];
                            effj[l][k] += f[l] * dxq[l*3+k];
                            }
                        }
                        
                    /* re-set the counter. */
                    icount = 0;
                    
                    }
                    
                }
        
            }
    
        } /* loop over all particles */
        
    /* are there any leftovers? */
    if ( icount > 0 ) {
    
        /* copy the first potential to the last entries */
        for ( k = icount ; k < 4 ; k++ )
            potq[k] = potq[0];
            
        /* evaluate the potentials */
        potential_eval_vec_single( potq , r2q , e , f );

        /* for each entry, update the forces and energy */
        for ( l = 0 ; l < icount ; l++ ) {
            cell_i->epot += e[l];
            for ( k = 0 ; k < 3 ; k++ ) {
                effi[l][k] += -f[l] * dxq[l*3+k];
                effj[l][k] += f[l] * dxq[l*3+k];
                }
            }
                    
        }
        
    /* Write local data back if needed. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* copy the particle data back */
        for ( i = 0 ; i < cell_i->count ; i++ ) {
            cell_i->parts[i].f[0] = parts_i[i].f[0];
            cell_i->parts[i].f[1] = parts_i[i].f[1];
            cell_i->parts[i].f[2] = parts_i[i].f[2];
            }
        for ( i = 0 ; i < cell_j->count ; i++ ) {
            cell_j->parts[i].f[0] = parts_j[i].f[0];
            cell_j->parts[i].f[1] = parts_j[i].f[1];
            cell_j->parts[i].f[2] = parts_j[i].f[2];
            }
        }
        
    /* since nothing bad happened to us... */
    return runner_err_ok;

    }
#else
int runner_sortedpair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift ) {

    struct part *part_i, *part_j;
    struct space *s;
    struct potential *pot;
    struct engine *eng;
    FPTYPE cutoff, cutoff2, r2, dx[3], e, f;
    FPTYPE d[runner_maxparts], temp, pivot;
    FPTYPE shift[3];
    int ind[runner_maxparts], left[runner_maxparts], count = 0, lcount = 0;
    int i, j, k, imax, qpos, lo, hi;
    struct {
        int lo, hi;
        } qstack[runner_maxqstack];
    struct part *parts_i, *parts_j;
    
    /* get the space and cutoff */
    eng = r->e;
    s = &(eng->s);
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    
    /* break early if one of the cells is empty */
    if ( cell_i->count == 0 || cell_j->count == 0 )
        return runner_err_ok;
    
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * cell_i->count );
        parts_j = (struct part *)alloca( sizeof(struct part) * cell_j->count );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * cell_i->count );
        memcpy( parts_j , cell_j->parts , sizeof(struct part) * cell_j->count );
        }
        
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
    /* start by filling the particle ids of both cells into ind and d */
    temp = 1.0 / sqrt( pshift[0]*pshift[0] + pshift[1]*pshift[1] + pshift[2]*pshift[2] );
    shift[0] = pshift[0]*temp; shift[1] = pshift[1]*temp; shift[2] = pshift[2]*temp;
    for ( i = 0 ; i < cell_i->count ; i++ ) {
        part_i = &( parts_i[i] );
        ind[count] = -i - 1;
        d[count] = part_i->x[0]*shift[0] + part_i->x[1]*shift[1] + part_i->x[2]*shift[2];
        count += 1;
        }
    for ( i = 0 ; i < cell_j->count ; i++ ) {
        part_i = &( parts_j[i] );
        ind[count] = i;
        d[count] = (part_i->x[0]+pshift[0])*shift[0] + (part_i->x[1]+pshift[1])*shift[1] + (part_i->x[2]+pshift[2])*shift[2] - cutoff;
        count += 1;
        }
        
    /* sort with quicksort */
    qstack[0].lo = 0; qstack[0].hi = count - 1; qpos = 0;
    while ( qpos >= 0 ) {
        lo = qstack[qpos].lo; hi = qstack[qpos].hi;
        qpos -= 1;
        if ( hi - lo < 10 ) {
            for ( i = lo ; i < hi ; i++ ) {
                imax = i;
                for ( j = i+1 ; j <= hi ; j++ )
                    if ( d[j] > d[imax] )
                        imax = j;
                if ( imax != i ) {
                    k = ind[imax]; ind[imax] = ind[i]; ind[i] = k;
                    temp = d[imax]; d[imax] = d[i]; d[i] = temp;
                    }
                }
            }
        else {
            pivot = d[ ( lo + hi ) / 2 ];
            i = lo; j = hi;
            while ( i <= j ) {
                while ( d[i] > pivot ) i++;
                while ( d[j] < pivot ) j--;
                if ( i <= j ) {
                    if ( i < j ) {
                        k = ind[i]; ind[i] = ind[j]; ind[j] = k;
                        temp = d[i]; d[i] = d[j]; d[j] = temp;
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
            if ( d[j] > d[imax] )
                imax = j;
        if ( imax != i ) {
            k = ind[imax]; ind[imax] = ind[i]; ind[i] = k;
            temp = d[imax]; d[imax] = d[i]; d[i] = temp;
            }
        } */
    
    /* loop over the sorted list of particles */
    for ( i = 0 ; i < count ; i++ ) {
    
        /* is this a particle from the left? */
        if ( ind[i] < 0 )
            left[lcount++] = -ind[i] - 1;
            
        /* it's from the right, interact with all left particles */
        else {
        
            /* get a handle on this particle */
            part_j = &( parts_j[ind[i]] );
        
            /* loop over the left particles */
            for ( j = lcount-1 ; j >= 0 ; j-- ) {
            
                /* get a handle on the second particle */
                part_i = &( parts_i[left[j]] );
            
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = part_i->x[k] - part_j->x[k] - pshift[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                /* fetch the potential, if any */
                pot = eng->p[ part_i->type * eng->max_type + part_j->type ];
                if ( pot == NULL )
                    continue;
                    
                /* evaluate the interaction */
                /* runner_rcount += 1; */
                #ifdef EXPLICIT_POTENTIALS
                    potential_eval_expl( pot , r2 , &e , &f );
                #else
                    potential_eval( pot , r2 , &e , &f );
                #endif
                
                /* update the forces */
                for ( k = 0 ; k < 3 ; k++ ) {
                    part_i->f[k] += -f * dx[k];
                    part_j->f[k] += f * dx[k];
                    }
                    
                /* tabulate the energy */
                cell_i->epot += e;
                
                }
        
            }
    
        } /* loop over all particles */
        
    /* Write local data back if needed. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* copy the particle data back */
        for ( i = 0 ; i < cell_i->count ; i++ ) {
            cell_i->parts[i].f[0] = parts_i[i].f[0];
            cell_i->parts[i].f[1] = parts_i[i].f[1];
            cell_i->parts[i].f[2] = parts_i[i].f[2];
            }
        for ( i = 0 ; i < cell_j->count ; i++ ) {
            cell_j->parts[i].f[0] = parts_j[i].f[0];
            cell_j->parts[i].f[1] = parts_j[i].f[1];
            cell_j->parts[i].f[2] = parts_j[i].f[2];
            }
        }
        
    /* since nothing bad happened to us... */
    return runner_err_ok;

    }
#endif


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

#if defined(__SSE__) && defined(FPTYPE_SINGLE)
int runner_dopair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift ) {

    int i, j, k, l, icount = 0;
    FPTYPE cutoff2, r2, *effi[4], *effj[4];
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE e[4] __attribute__ ((aligned (16)));
    FPTYPE f[4] __attribute__ ((aligned (16)));
    FPTYPE dx[3], dxq[12];
    struct space *s;
    struct part *part_i, *part_j;
    struct potential *potq[4];
    struct engine *eng;
    struct part *parts_i, *parts_j = NULL;

    /* get the space and cutoff */
    eng = r->e;
    s = &(eng->s);
    cutoff2 = s->cutoff2;
        
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * cell_i->count );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * cell_i->count );
        if ( cell_i != cell_j ) {
            parts_j = (struct part *)alloca( sizeof(struct part) * cell_j->count );
            memcpy( parts_j , cell_j->parts , sizeof(struct part) * cell_j->count );
            }
        }
        
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
    /* is this a genuine pair or a cell against itself */
    if ( cell_i == cell_j ) {
    
        /* loop over all particles */
        for ( i = 1 ; i < cell_i->count ; i++ ) {
        
            /* get the particle */
            part_i = &(cell_i->parts[i]);
        
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
            
                /* get the other particle */
                part_j = &(cell_i->parts[j]);
                
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = part_i->x[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                /* fetch the potential, if any */
                potq[icount] = eng->p[ part_i->type * eng->max_type + part_j->type ];
                if ( potq[icount] == NULL )
                    continue;
                    
                /* add this interaction to the interaction queue. */
                r2q[icount] = r2;
                dxq[icount*3] = dx[0];
                dxq[icount*3+1] = dx[1];
                dxq[icount*3+2] = dx[2];
                effi[icount] = part_i->f;
                effj[icount] = part_j->f;
                icount += 1;
                    
                /* evaluate the interactions if the queue is full. */
                if ( icount == 4 ) {
                
                    potential_eval_vec_single( potq , r2q , e , f );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        cell_i->epot += e[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            effi[l][k] += -f[l] * dxq[l*3+k];
                            effj[l][k] += f[l] * dxq[l*3+k];
                            }
                        }
                        
                    /* re-set the counter. */
                    icount = 0;
                    
                    }
                    
                } /* loop over all other particles */
        
            } /* loop over all particles */
    
        }
        
    /* no, it's a genuine pair */
    else {
    
        /* loop over all particles */
        for ( i = 0 ; i < cell_i->count ; i++ ) {
        
            /* get the particle */
            part_i = &(cell_i->parts[i]);
            
            /* loop over all other particles */
            for ( j = 0 ; j < cell_j->count ; j++ ) {
            
                /* get the other particle */
                part_j = &(cell_j->parts[j]);

                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = part_i->x[k] - part_j->x[k] - shift[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                    
                /* fetch the potential, if any */
                potq[icount] = eng->p[ part_i->type * eng->max_type + part_j->type ];
                if ( potq[icount] == NULL )
                    continue;
                    
                /* add this interaction to the interaction queue. */
                r2q[icount] = r2;
                dxq[icount*3] = dx[0];
                dxq[icount*3+1] = dx[1];
                dxq[icount*3+2] = dx[2];
                effi[icount] = part_i->f;
                effj[icount] = part_j->f;
                icount += 1;
                    
                /* evaluate the interactions if the queue is full. */
                if ( icount == 4 ) {
                
                    potential_eval_vec_single( potq , r2q , e , f );

                    /* update the forces and the energy */
                    for ( l = 0 ; l < 4 ; l++ ) {
                        cell_i->epot += e[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            effi[l][k] += -f[l] * dxq[l*3+k];
                            effj[l][k] += f[l] * dxq[l*3+k];
                            }
                        }
                        
                    /* re-set the counter. */
                    icount = 0;
                    
                    }
                    
                } /* loop over all other particles */
        
            } /* loop over all particles */

        }
        
    /* are there any leftovers? */
    if ( icount > 0 ) {
    
        /* copy the first potential to the last entries */
        for ( k = icount ; k < 4 ; k++ )
            potq[k] = potq[0];
            
        /* evaluate the potentials */
        potential_eval_vec_single( potq , r2q , e , f );

        /* for each entry, update the forces and energy */
        for ( l = 0 ; l < icount ; l++ ) {
            cell_i->epot += e[l];
            for ( k = 0 ; k < 3 ; k++ ) {
                effi[l][k] += -f[l] * dxq[l*3+k];
                effj[l][k] += f[l] * dxq[l*3+k];
                }
            }
                    
        }
        
    /* Write local data back if needed. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* copy the particle data back */
        for ( i = 0 ; i < cell_i->count ; i++ ) {
            cell_i->parts[i].f[0] = parts_i[i].f[0];
            cell_i->parts[i].f[1] = parts_i[i].f[1];
            cell_i->parts[i].f[2] = parts_i[i].f[2];
            }
        if ( cell_i != cell_j )
            for ( i = 0 ; i < cell_j->count ; i++ ) {
                cell_j->parts[i].f[0] = parts_j[i].f[0];
                cell_j->parts[i].f[1] = parts_j[i].f[1];
                cell_j->parts[i].f[2] = parts_j[i].f[2];
                }
        }
        
    /* all is well that ends ok */
    return runner_err_ok;

    }
#else
int runner_dopair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift ) {

    int i, j, k;
    FPTYPE cutoff2, dx[3], r2, e, f;
    struct space *s;
    struct part *part_i, *part_j;
    struct potential *pot;
    struct engine *eng;
    struct part *parts_i, *parts_j = NULL;

    /* get the space and cutoff */
    eng = r->e;
    s = &(eng->s);
    cutoff2 = s->cutoff2;
        
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * cell_i->count );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * cell_i->count );
        if ( cell_i != cell_j ) {
            parts_j = (struct part *)alloca( sizeof(struct part) * cell_j->count );
            memcpy( parts_j , cell_j->parts , sizeof(struct part) * cell_j->count );
            }
        }
        
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
    /* is this a genuine pair or a cell against itself */
    if ( cell_i == cell_j ) {
    
        /* loop over all particles */
        for ( i = 1 ; i < cell_i->count ; i++ ) {
        
            /* get the particle */
            part_i = &(cell_i->parts[i]);
        
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
            
                /* get the other particle */
                part_j = &(cell_i->parts[j]);
                
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = part_i->x[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                /* fetch the potential, if any */
                pot = eng->p[ part_i->type * eng->max_type + part_j->type ];
                if ( pot == NULL )
                    continue;
                    
                /* evaluate the interaction */
                /* runner_rcount += 1; */
                #ifdef EXPLICIT_POTENTIALS
                    potential_eval_expl( pot , r2 , &e , &f );
                #else
                    potential_eval( pot , r2 , &e , &f );
                #endif
                
                /* update the forces */
                for ( k = 0 ; k < 3 ; k++ ) {
                    part_i->f[k] += -f * dx[k];
                    part_j->f[k] += f * dx[k];
                    }
                    
                /* tabulate the energy */
                cell_i->epot += e;
                
                } /* loop over all other particles */
        
            } /* loop over all particles */
    
        }
        
    /* no, it's a genuine pair */
    else {
    
        /* loop over all particles */
        for ( i = 0 ; i < cell_i->count ; i++ ) {
        
            /* get the particle */
            part_i = &(cell_i->parts[i]);
            
            /* loop over all other particles */
            for ( j = 0 ; j < cell_j->count ; j++ ) {
            
                /* get the other particle */
                part_j = &(cell_j->parts[j]);

                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = part_i->x[k] - part_j->x[k] - shift[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                    
                /* fetch the potential, if any */
                pot = eng->p[ part_i->type * eng->max_type + part_j->type ];
                if ( pot == NULL )
                    continue;
                    
                /* evaluate the interaction */
                /* runner_rcount += 1; */
                #ifdef EXPLICIT_POTENTIALS
                    potential_eval_expl( pot , r2 , &e , &f );
                #else
                    potential_eval( pot , r2 , &e , &f );
                #endif
                
                /* update the forces */
                for ( k = 0 ; k < 3 ; k++ ) {
                    part_i->f[k] += -f * dx[k];
                    part_j->f[k] += f * dx[k];
                    }
                
                /* tabulate the energy */
                cell_i->epot += e;
                
                } /* loop over all other particles */
        
            } /* loop over all particles */

        }
        
    /* Write local data back if needed. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* copy the particle data back */
        for ( i = 0 ; i < cell_i->count ; i++ ) {
            cell_i->parts[i].f[0] = parts_i[i].f[0];
            cell_i->parts[i].f[1] = parts_i[i].f[1];
            cell_i->parts[i].f[2] = parts_i[i].f[2];
            }
        if ( cell_i != cell_j )
            for ( i = 0 ; i < cell_j->count ; i++ ) {
                cell_j->parts[i].f[0] = parts_j[i].f[0];
                cell_j->parts[i].f[1] = parts_j[i].f[1];
                cell_j->parts[i].f[2] = parts_j[i].f[2];
                }
        }
        
    /* all is well that ends ok */
    return runner_err_ok;

    }
#endif


/**
 * @brief The #runner's main routine (for the Cell/BE SPU).
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having passed, calls #space_getpair until there are no pairs
 * available.
 *
 * Note that this routine is only compiled if @c CELL has been defined.
 *
 * @sa #runner_run_cell.
 */

int runner_run_cell ( struct runner *r ) {

#ifdef CELL
    int err = 0;
    struct cellpair *p[runner_qlen];
    unsigned int buff[2];
    int i, k, count = 0;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running...\n",r->id); fflush(stdout);
    
    /* init the cellpair pointers */
    for ( k = 0 ; k < runner_qlen ; k++ )
        p[k] = NULL;
        
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(r->e) < 0)
            return error(runner_err_engine);
            
        /* write the current cell data */
        for ( i = 0 ; i < r->e->s.nr_cells ; i++ ) {
            r->celldata[i].ni = r->e->s.cells[i].count;
            r->celldata[i].ai = (unsigned long long)r->e->s.cells[i].parts;
            }

        /* emit a reload message */
        buff[0] = 0xFFFFFFFF;
        /* printf("runner_run: runner %i sending reload message...\n",r->id); */
        if ( spe_in_mbox_write( r->spe , buff , 2 , SPE_MBOX_ALL_BLOCKING ) != 2 )
            return runner_err_spe;


        /* while there are pairs... */
        while ( r->e->s.next_pair < r->e->s.nr_pairs || count > 0 ) {

            /* if we have no p[0], try to get some... */
            if ( p[0] == NULL && r->e->s.next_pair < r->e->s.nr_pairs ) {
                p[0] = space_getpair( &(r->e->s) , r->id , runner_bitesize , NULL , &err , count == 0 );
                if ( err < 0 )
                    return runner_err_space;
                }

            /* if we got a pair, send it to the SPU... */
            if ( p[0] != NULL ) {

                /* we've got an active slot! */
                count += 1;

                /* pack this pair's data */
                buff[0] = ( p[0]->i << 20 ) + ( p[0]->j << 8 ) + 1;
                if ( p[0]->shift[0] == r->e->s.cutoff )
                    buff[0] += 1 << 6;
                else if ( p[0]->shift[0] == -r->e->s.cutoff )
                    buff[0] += 2 << 6;
                if ( p[0]->shift[1] == r->e->s.cutoff )
                    buff[0] += 1 << 4;
                else if ( p[0]->shift[1] == -r->e->s.cutoff )
                    buff[0] += 2 << 4;
                if ( p[0]->shift[2] == r->e->s.cutoff )
                    buff[0] += 1 << 2;
                else if ( p[0]->shift[2] == -r->e->s.cutoff )
                    buff[0] += 2 << 2;

                /* wait for the buffer to be free... */
                /* while ( !spe_in_mbox_status( r->spe ) ) */
                /*     sched_yield(); */

                /* write the data to the mailbox */
                /* printf("runner_run: sending pair 0x%llx (n=%i), 0x%llx (n=%i) with shift=[%e,%e,%e].\n", */
                /*     (unsigned long long)ci->parts,ci->count,(unsigned long long)cj->parts,cj->count, */
                /*     p->shift[0], p->shift[1], p->shift[2]); fflush(stdout); */
                /* printf("runner_run: runner %i sending pair to SPU...\n",r->id); fflush(stdout); */
                if ( spe_in_mbox_write( r->spe , buff , 2 , SPE_MBOX_ALL_BLOCKING ) != 2 )
                    return runner_err_spe;
                /* printf("runner_run: runner %i sent pair to SPU.\n",r->id); fflush(stdout); */


                /* wait for the last pair to have been processed */
                if ( p[runner_qlen-1] != NULL ) {

                    /* read a word from the spe */
                    /* printf("runner_run: runner %i waiting for SPU response...\n",r->id); fflush(stdout); */
                    /* if ( spe_out_intr_mbox_read( r->spe , &buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 ) */
                    /*     return runner_err_spe; */
                    /* printf("runner_run: runner %i got SPU response.\n",r->id); fflush(stdout); */

                    /* release the last pair */
                    if ( space_releasepair( &(r->e->s) , p[runner_qlen-1] ) < 0 )
                        return runner_err_space;

                    /* we've got one less... */
                    count -= 1;

                    }

                /* move on in the chain */
                for ( k = runner_qlen-1 ; k > 0 ; k-- )
                    p[k] = p[k-1];
                if ( p[0] != NULL )
                    p[0] = p[0]->next;

                /* take a breather... */
                /* sched_yield(); */

                }

            /* is there a non-empy slot, send a flush */
            else if ( count > 0 ) {

                /* send a flush message... */
                buff[0] = 0;
                if ( spe_in_mbox_write( r->spe , buff , 2 , SPE_MBOX_ALL_BLOCKING ) != 2 )
                    return runner_err_spe;

                /* wait for the reply... */
                if ( spe_out_intr_mbox_read( r->spe , buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 )
                    return runner_err_spe;
                /* printf("runner_run: got rcount=%u.\n",buff[0]); */

                /* release the pairs still in the queue */
                for ( k = 1 ; k < runner_qlen ; k++ )
                    if ( p[k] != NULL ) {
                        if ( space_releasepair( &(r->e->s) , p[k] ) < 0 )
                            return runner_err_space;
                        p[k] = NULL;
                        count -= 1;
                        }

                }

            }
                
        /* did things go wrong? */
        /* printf("runner_run: runner %i done pairs.\n",r->id); fflush(stdout); */
        if ( err < 0 )
            return error(runner_err_space);
    
        }

    /* end well... */
    return runner_err_ok;

#else

    /* This functionality is not available */
    return runner_err_unavail;
    
#endif

    }


/**
 * @brief The #runner's main routine.
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having passed, calls #space_getpair until there are no pairs
 * available.
 */

int runner_run ( struct runner *r ) {

    int err = 0;
    struct cellpair *p = NULL;
    struct cellpair *finger;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running...\n",r->id); fflush(stdout);
    
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(r->e) < 0)
            return error(runner_err_engine);
                        
        /* while i can still get a pair... */
        /* printf("runner_run: runner %i passed barrier, getting pairs...\n",r->id); */
        while ( ( p = space_getpair( &r->e->s , r->id , runner_bitesize , NULL , &err , 1 ) ) != NULL ) {

            /* work this list of pair... */
            for ( finger = p ; finger != NULL ; finger = finger->next ) {

                /* is this cellpair playing with itself? */
                if ( finger->i == finger->j ) {
                    if ( runner_dopair( r , &(r->e->s.cells[finger->i]) , &(r->e->s.cells[finger->j]) , finger->shift ) < 0 )
                        return runner_err;
                    }

                /* nope, good cell-on-cell action. */
                else
                    if ( runner_sortedpair( r , &(r->e->s.cells[finger->i]) , &(r->e->s.cells[finger->j]) , finger->shift ) < 0 )
                        return runner_err;

                /* release this pair */
                if ( space_releasepair( &(r->e->s) , finger->i , finger->j ) < 0 )
                    return runner_err_space;

                }

            }

        /* give the reaction count */
        /* printf("runner_run: last count was %u.\n",runner_rcount); */
            
        /* did things go wrong? */
        /* printf("runner_run: runner %i done pairs.\n",r->id); fflush(stdout); */
        if ( err < 0 )
            return error(runner_err_space);
    
        }

    /* end well... */
    return runner_err_ok;

    }

    
/**
 * @brief The #runner's main routine (#celltuple model).
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having passed, calls #space_gettuple until there are no
 * tuples available.
 */

int runner_run_tuples ( struct runner *r ) {

    int res, i, j, k, ci, cj;
    struct celltuple t;
    FPTYPE shift[3];
    struct space *s;
    struct engine *e;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* Remember who the engine and the space are. */
    e = r->e;
    s = &(r->e->s);
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running (tuples)...\n",r->id); fflush(stdout);
    
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(e) < 0 )
            return r->err = runner_err_engine;
                        
        /* Loop over tuples. */
        while ( 1 ) {
        
            /* Get a tuple. */
            if ( ( res = space_gettuple( s , r->id , &t ) ) < 0 )
                return r->err = runner_err_space;
                
            /* If there were no tuples left, bail. */
            if ( res < 1 )
                break;
                
            /* Loop over all pairs in this tuple. */
            for ( i = 0 ; i < t.n ; i++ ) { 
                        
                /* Get the cell ID. */
                ci = t.cellid[i];
                    
                for ( j = i ; j < t.n ; j++ ) {
                
                    /* Is this pair active? */
                    if ( !( t.pairs & ( 1 << ( i * space_maxtuples + j ) ) ) )
                        continue;
                        
                    /* Get the cell ID. */
                    cj = t.cellid[j];

                    /* Compute the shift between ci and cj. */
                    for ( k = 0 ; k < 3 ; k++ ) {
                        shift[k] = s->cells[cj].origin[k] - s->cells[ci].origin[k];
                        if ( shift[k] * 2 > s->dim[k] )
                            shift[k] -= s->dim[k];
                        else if ( shift[k] * 2 < -s->dim[k] )
                            shift[k] += s->dim[k];
                        }
                    
                    /* is this cellpair playing with itself? */
                    if ( ci == cj ) {
                        if ( runner_dopair( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                            return runner_err;
                        }

                    /* nope, good cell-on-cell action. */
                    else
                        if ( runner_sortedpair( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                            return runner_err;

                    /* release this pair */
                    if ( space_releasepair( s , ci , cj ) < 0 )
                        return runner_err_space;
                        
                    }
                    
                }

            }

        /* give the reaction count */
        /* printf("runner_run: last count was %u.\n",runner_rcount); */
            
        }

    /* end well... */
    return runner_err_ok;

    }

    
/**
 * @brief Initialize the runner associated to the given engine.
 * 
 * @param r The #runner to be initialized.
 * @param e The #engine with which it is associated.
 * @param id The ID of this #runner.
 * 
 * @return #runner_err_ok or < 0 on error (see #runner_err).
 */

int runner_init ( struct runner *r , struct engine *e , int id ) {

    #ifdef CELL
        static void *data = NULL;
        static int size_data = 0;
        void *finger;
        int nr_pots = 0, size_pots = 0, *pots, i, j, k;
        struct potential *p;
        unsigned int buff;
    #endif

    /* make sure the inputs are ok */
    if ( r == NULL || e == NULL )
        return error(runner_err_null);
        
    /* remember who i'm working for */
    r->e = e;
    r->id = id;
    
    #ifdef CELL
    
        /* if this has not been done before, init the runner data */
        if ( data == NULL ) {
    
            /* run through the potentials and count them and their size */
            for ( i = 0 ; i < e->max_type ; i++ )
                for ( j = i ; j < e->max_type ; j++ )
                    if ( e->p[ i * e->max_type + j ] != NULL ) {
                        nr_pots += 1;
                        size_pots += e->p[ i * e->max_type + j ]->n;
                        }

            /* the main data consists of a pointer to the cell data (64 bit), */
            /* the nr of cells (int), the cutoff (double), the width of */
            /* each cell, the max nr of types (int) */
            /* and an array of size max_type*max_type of offsets (int) */
            size_data = sizeof(void *) + sizeof(int) + 4 * sizeof(float) + sizeof(int) * ( 1 + e->max_type*e->max_type );

            /* stretch this data until we are aligned to 8 bytes */
            while ( size_data % 8 ) size_data++;
            
            /* we then append nr_pots potentials consisting of three floats (alphas) */
            /* and two ints with other data */
            size_data += nr_pots * ( 3 * sizeof(float) + 2 * sizeof(int) );

            /* finally, we append the data of each interval of each potential */
            /* which consists of eight floats */
            size_data += size_pots * sizeof(float) * 8;
            
            /* raise to multiple of 128 */
            if ( ( size_data & 127 ) > 0 )
                size_data = ( ( size_data >> 7 ) + 1 ) << 7;
            
            /* allocate memory for the SPU data */
            if ( ( data = malloc_align( size_data , 7 ) ) == NULL )
                return error(runner_err_malloc);

            /* fill-in the engine data (without the pots) */
            finger = data;
            *((unsigned long long *)finger) = 0; finger += sizeof(unsigned long long);
            *((int *)finger) = e->s.nr_cells; finger += sizeof(int);
            *((float *)finger) = e->s.cutoff; finger += sizeof(float);
            *((float *)finger) = e->s.h[0]; finger += sizeof(float);
            *((float *)finger) = e->s.h[1]; finger += sizeof(float);
            *((float *)finger) = e->s.h[2]; finger += sizeof(float);
            *((int *)finger) = e->max_type; finger += sizeof(int);
            pots = (int *)finger; finger += e->max_type * e->max_type * sizeof(int);
            for ( i = 0 ; i < e->max_type*e->max_type ; i++ )
                pots[i] = 0;
                
            /* move the finger until we are at an 8-byte boundary */
            while ( (unsigned long long)finger % 8 ) finger++;

            /* loop over the potentials */
            for ( i = 0 ; i < e->max_type ; i++ )
                for ( j = i ; j < e->max_type ; j++ )
                    if ( pots[ i * e->max_type + j ] == 0 && e->p[ i * e->max_type + j ] != NULL ) {
                        p = e->p[ i * e->max_type + j ];
                        for ( k = 0 ; k < e->max_type*e->max_type ; k++ )
                            if ( e->p[k] == p )
                                pots[k] = finger - data;
                        *((int *)finger) = p->n; finger += sizeof(int);
                        *((int *)finger) = p->flags; finger += sizeof(int);
                        *((float *)finger) = p->alpha[0]; finger += sizeof(float);
                        *((float *)finger) = p->alpha[1]; finger += sizeof(float);
                        *((float *)finger) = p->alpha[2]; finger += sizeof(float);
                        for ( k = 0 ; k < p->n ; k++ ) {
                            *((float *)finger) = p->mi[k]; finger += sizeof(float);
                            *((float *)finger) = p->hi[k]; finger += sizeof(float);
                            *((float *)finger) = p->c[k*6+0]; finger += sizeof(float);
                            *((float *)finger) = p->c[k*6+1]; finger += sizeof(float);
                            *((float *)finger) = p->c[k*6+2]; finger += sizeof(float);
                            *((float *)finger) = p->c[k*6+3]; finger += sizeof(float);
                            *((float *)finger) = p->c[k*6+4]; finger += sizeof(float);
                            *((float *)finger) = p->c[k*6+5]; finger += sizeof(float);
                            }
                        }

            /* raise to multiple of 128 */
            if ( ( (unsigned long long)finger & 127 ) > 0 )
                finger = (void *)( ( ( (unsigned long long)finger >> 7 ) + 1 ) << 7 );
            
            /* if the effective size is smaller than the allocated size */
            /* (e.g. duplicate potentials), be clean and re-allocate the data */
            if ( finger - data < size_data ) {
                size_data = finger - data;
                if ( ( data = realloc_align( data , size_data , 7 ) ) == NULL )
                    return error(runner_err_malloc);
                }
            
            /* say something about it all */
            /* printf("runner_init: initialized data with %i bytes.\n",size_data); */
                
            } /* init runner data */
            
        /* remember where the data is */
        r->data = data;
        
        /* allocate and set the cell data */
        if ( ( r->celldata = (struct celldata *)malloc_align( ceil128( sizeof(struct celldata) * r->e->s.nr_cells ) , 7 ) ) == NULL )
            return error(runner_err_malloc);
        *((unsigned long long *)data) = (unsigned long long)r->celldata;
            
        /* get a handle on an SPU */
        if ( ( r->spe = spe_context_create(0, NULL) ) == NULL )
            return error(runner_err_spe);
        
        /* load the image onto the SPU */
        if ( spe_program_load( r->spe , &runner_spu ) != 0 )
            return error(runner_err_spe);
            
        /* dummy function that just starts the SPU... */
        int dummy ( struct runner *r ) {
            return spe_context_run( r->spe , &(r->entry) , 0 , r->data , (void *)(unsigned long long)size_data , NULL );
            }
        
        /* start the runner with a pointer to the data */
        r->entry = SPE_DEFAULT_ENTRY;
	    if (pthread_create(&r->spe_thread,NULL,(void *(*)(void *))dummy,r) != 0)
		    return error(runner_err_pthread);
            
        /* wait until the SPU is ready... */
        if ( spe_out_intr_mbox_read( r->spe , &buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 )
            return runner_err_spe;
    
    #endif
    
    /* init the thread */
    if ( e->flags & engine_flag_cell ) {
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run_cell,r) != 0)
		    return error(runner_err_pthread);
        }
    else if ( e->flags & engine_flag_tuples ) {
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run_tuples,r) != 0)
		    return error(runner_err_pthread);
        }
    else {
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run,r) != 0)
		    return error(runner_err_pthread);
        }
    
    /* all is well... */
    return runner_err_ok;
    
    }
