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
 * @brief Compute the interactions between the particles in the given
 *        cell using the verlet list.
 *
 * @param r The #runner.
 * @param c The #cell containing the particles to traverse.
 * @param f A pointer to an array of #FPTYPE in which to aggregate the
 *        interaction forces.
 * 
 * @return #runner_err_ok or <0 on error (see #runner_err)
 */
 
int runner_verlet_eval ( struct runner *r , struct cell *c , FPTYPE *f_out ) {

    struct space *s;
    struct part *part_i, *part_j, **partlist;
    struct verlet_entry *verlet_list;
    struct potential *pot;
    int pid, i, j, k, nrpairs;
    FPTYPE pix[3];
    FPTYPE cutoff, cutoff2, r2, dx[3], w, h[3];
    double epot = 0.0;
#if defined(VECTORIZE)
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE *effi[4], *effj[4], *pif;
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE ee[4] __attribute__ ((aligned (16)));
    FPTYPE eff[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12];
#else
    FPTYPE ee, eff;
#endif

    /* Get a direct pointer on the space and some other useful things. */
    s = &(r->e->s);
    partlist = s->partlist;
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    h[0] = s->h[0]; h[1] = s->h[1]; h[2] = s->h[2];
    
    /* Loop over all entries. */
    for ( i = 0 ; i < c->count ; i++ ) {
    
        /* Get a hold of the ith particle. */
        part_i = &( c->parts[i] );
        pid = part_i->id;
        verlet_list = &( s->verlet_list[ pid * space_verlet_maxpairs ] );
        pix[0] = part_i->x[0];
        pix[1] = part_i->x[1];
        pix[2] = part_i->x[2];
        nrpairs = s->verlet_nrpairs[ pid ];
        #if defined(VECTORIZE)
            pif = &( f_out[ pid*4 ] );
        #endif
        
        /* loop over all other particles */
        for ( j = 0 ; j < nrpairs ; j++ ) {

            /* get the other particle */
            part_j = verlet_list[j].p;

            /* get the distance between both particles */
            for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                dx[k] = -part_j->x[k] + pix[k] + verlet_list[j].shift[k]*h[k];
                r2 += dx[k] * dx[k];
                }

            /* is this within cutoff? */
            if ( r2 > cutoff2 )
                continue;
            // runner_rcount += 1;
                
            /* fetch the potential, should be non-NULL by design! */
            pot = verlet_list[j].pot;

            #ifdef VECTORIZE
                /* add this interaction to the interaction queue. */
                r2q[icount] = r2;
                dxq[icount*3] = dx[0];
                dxq[icount*3+1] = dx[1];
                dxq[icount*3+2] = dx[2];
                effi[icount] = pif;
                effj[icount] = &( f_out[part_j->id*4] );
                potq[icount] = pot;
                icount += 1;

                #if defined(FPTYPE_SINGLE)
                    /* evaluate the interactions if the queue is full. */
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
                    /* evaluate the interactions if the queue is full. */
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
                /* evaluate the interaction */
                #ifdef EXPLICIT_POTENTIALS
                    potential_eval_expl( pot , r2 , &ee , &eff );
                #else
                    potential_eval( pot , r2 , &ee , &eff );
                #endif

                /* update the forces */
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff * dx[k];
                    f_out[i*4+k] -= w;
                    f_out[part_j->id*4+k] += w;
                    }

                /* tabulate the energy */
                epot += ee;
            #endif

            } /* loop over all other particles */
            
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
        
    /* Store the accumulated potential energy. */
    r->epot += epot;

    /* All has gone well. */
    return runner_err_ok;

    }
    
    
/**
 * @brief Fill in the Verlet list entries for the given cell pair.
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
 
int runner_verlet_fill ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift ) {

    struct part *part_i, *part_j;
    struct space *s;
    int *left, count = 0, lcount = 0;
    int i, j, k, imax, qpos, lo, hi;
    struct {
        short int lo, hi;
        } *qstack;
    struct part *parts_i, *parts_j;
    struct potential *pot, **pots;
    struct engine *eng;
    int emt, pjoff, pioff, count_i, count_j;
    FPTYPE cutoff, cutoff2, skin, skin2, r2, dx[3], w;
    struct {
        short int d, ind;
        } *parts, temp;
    short int pivot;
    FPTYPE dscale;
    FPTYPE shift[3], inshift;
    FPTYPE pjx[3], pix[3], *pif, *pjf;
    int pid, pind, ishift[3];
    struct verlet_entry *vbuff;
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
    skin = fmin( s->h[0] , fmin( s->h[1] , s->h[2] ) );
    skin2 = skin * skin;
    cutoff = s->cutoff;
    cutoff2 = cutoff*cutoff;
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
            pid = part_i->id;
            pind = s->verlet_nrpairs[ pid ];
            vbuff = &(s->verlet_list[ pid * space_verlet_maxpairs ]);
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
                if ( r2 > skin2 )
                    continue;
                /* runner_rcount += 1; */
                    
                /* fetch the potential, if any */
                pot = pots[ pioff + part_j->type ];
                if ( pot == NULL )
                    continue;
                    
                /* Add this pair to the verlet list. */
                vbuff[pind].shift[0] = 0;
                vbuff[pind].shift[1] = 0;
                vbuff[pind].shift[2] = 0;
                vbuff[pind].pot = pot;
                vbuff[pind].p = &(cell_j->parts[j]);
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

                } /* loop over all other particles */
                
            /* Adjust verlet_nrpairs. */
            s->verlet_nrpairs[pid] = pind;
        
            } /* loop over all particles */
    
        }
        
    /* No, genuine pair. */
    else {
    
        /* Get the integer shift. */
        ishift[0] = round( pshift[0] * s->ih[0] );
        ishift[1] = round( pshift[1] * s->ih[1] );
        ishift[2] = round( pshift[2] * s->ih[2] );
        
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
            parts[count].d = 1 + dscale * ( (part_i->x[0]+pshift[0])*shift[0] + (part_i->x[1]+pshift[1])*shift[1] + (part_i->x[2]+pshift[2])*shift[2] - skin );
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
                pid = part_j->id;
                pind = s->verlet_nrpairs[ pid ];
                vbuff = &(s->verlet_list[ pid * space_verlet_maxpairs ]);
                pjf = &( part_j->f[0] );

                /* loop over the left particles */
                for ( j = 0 ; j < lcount ; j++ ) {

                    /* get a handle on the second particle */
                    part_i = &( parts_i[left[j]] );

                    /* get the distance between both particles */
                    r2 = 0.0;
                    for ( k = 0 ; k < 3 ; k++ ) {
                        dx[k] = part_i->x[k] - pjx[k];
                        r2 += dx[k] * dx[k];
                        }

                    /* is this within cutoff? */
                    if ( r2 > skin2 )
                        continue;
                    /* runner_rcount += 1; */
                        
                    /* fetch the potential, if any */
                    pot = pots[ pjoff + part_i->type ];
                    if ( pot == NULL )
                        continue;

                    /* Add this pair to the verlet list. */
                    vbuff[pind].shift[0] = ishift[0];
                    vbuff[pind].shift[1] = ishift[1];
                    vbuff[pind].shift[2] = ishift[2];
                    vbuff[pind].pot = pot;
                    vbuff[pind].p = &( cell_i->parts[left[j]] );
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
                        effi[icount] = part_i->f;
                        effj[icount] = pjf;
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
                            pjf[k] += w;
                            }

                        /* tabulate the energy */
                        epot += e;
                    #endif

                    }

                /* Adjust verlet_nrpairs. */
                s->verlet_nrpairs[pid] = pind;
        
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
    
    
