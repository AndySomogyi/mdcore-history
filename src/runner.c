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
 * @brief Compute the interactions between the particles in the given
 *        segment of the verlet list.
 *
 * @param r The #runner.
 * @param ind The index of the first entry in the Verlet list to compute.
 * @param count The number of entries to compute
 * @param f A pointer to an array of #FPTYPE in which to aggregate the
 *        interaction forces.
 * 
 * @return #runner_err_ok or <0 on error (see #runner_err)
 */
 
int runner_verlet_eval ( struct runner *r , int ind , int count , FPTYPE *f_out ) {

    struct space *s;
    struct part *part_i, *part_j, **partlist;
    struct verlet_entry *verlet_list;
    struct potential *pot;
    int i, j, k, nrpairs;
    FPTYPE pix[3];
    FPTYPE cutoff, cutoff2, r2, dx[3], w, h[3];
    double epot = 0.0;
#if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE *effi[4], *effj[4], *pif;
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE e[4] __attribute__ ((aligned (16)));
    FPTYPE f[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12];
#else
    FPTYPE e, f;
#endif

    /* Get a direct pointer on the space and some other useful things. */
    s = &(r->e->s);
    partlist = s->partlist;
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    h[0] = s->h[0]; h[1] = s->h[1]; h[2] = s->h[2];
    
    /* Loop over all entries. */
    for ( i = ind ; i < ind+count ; i++ ) {
    
        /* Get a hold of the ith particle. */
        part_i = partlist[i];
        verlet_list = &( s->verlet_list[ i * space_verlet_maxpairs ] );
        pix[0] = part_i->x[0];
        pix[1] = part_i->x[1];
        pix[2] = part_i->x[2];
        nrpairs = s->verlet_nrpairs[i];
        #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
            pif = &( f_out[i*4] );
        #endif
        
        /* loop over all other particles */
        for ( j = 0 ; j < nrpairs ; j++ ) {

            /* get the other particle */
            part_j = verlet_list[j].p;

            /* fetch the potential, should be non-NULL by design! */
            pot = verlet_list[j].pot;

            /* get the distance between both particles */
            for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                dx[k] = -part_j->x[k] + pix[k] + verlet_list[j].shift[k]*h[k];
                r2 += dx[k] * dx[k];
                }

            /* is this within cutoff? */
            if ( r2 > cutoff2 )
                continue;
            /* runner_rcount += 1; */
                
            #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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
                    f_out[i*4+k] -= w;
                    f_out[j*4+k] += w;
                    }

                /* tabulate the energy */
                epot += e;
            #endif

            } /* loop over all other particles */
            
        }
        
    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
    int left[runner_maxparts], count = 0, lcount = 0;
    int i, j, k, imax, qpos, lo, hi;
    struct {
        int lo, hi;
        } qstack[runner_maxqstack];
    struct part *parts_i, *parts_j;
    struct potential *pot, **pots;
    struct engine *eng;
    int emt, pjoff, pioff, count_i, count_j;
    FPTYPE cutoff, cutoff2, skin, skin2, r2, dx[3], w;
    struct {
        short int d, ind;
        } parts[2*runner_maxparts], temp;
    short int pivot;
    FPTYPE dscale;
    FPTYPE shift[3], inshift;
    FPTYPE pjx[3], pix[3], *pif, *pjf;
    int pid, pind, ishift[3];
    struct verlet_entry *vbuff;
    double epot = 0.0;
#if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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
    count_i = cell_i->count;
    count_j = cell_j->count;
    
    /* break early if one of the cells is empty */
    if ( count_i == 0 || count_j == 0 )
        return runner_err_ok;
    
    /* Get pointers to the particle arrays. */
    parts_i = cell_i->parts;
    parts_j = cell_j->parts;
    
    /* Is this a self interaction? */
    if ( cell_i == cell_j ) {
    
        /* loop over all particles */
        for ( i = 1 ; i < count_i ; i++ ) {
        
            /* get the particle */
            part_i = &(cell_i->parts[i]);
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
                
                /* fetch the potential, if any */
                pot = pots[ pioff + part_j->type ];
                if ( pot == NULL )
                    continue;
                    
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > skin2 )
                    continue;
                /* runner_rcount += 1; */
                    
                /* Add this pair to the verlet list. */
                vbuff[pind].shift[0] = 0;
                vbuff[pind].shift[1] = 0;
                vbuff[pind].shift[2] = 0;
                vbuff[pind].pot = pot;
                vbuff[pind].p = part_j;
                pind += 1;
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */

                #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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

                    /* fetch the potential, if any */
                    pot = pots[ pjoff + part_i->type ];
                    if ( pot == NULL )
                        continue;

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
                        
                    /* Add this pair to the verlet list. */
                    vbuff[pind].shift[0] = ishift[0];
                    vbuff[pind].shift[1] = ishift[1];
                    vbuff[pind].shift[2] = ishift[2];
                    vbuff[pind].pot = pot;
                    vbuff[pind].p = part_i;
                    pind += 1;

                    /* is this within cutoff? */
                    if ( r2 > cutoff2 )
                        continue;
                    /* runner_rcount += 1; */

                    #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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
        
    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
    cell_i->epot += epot;
        
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
 */
 
int runner_dopair_verlet ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift , struct verlet_pairwise_list *list ) {

    struct part *part_i, *part_j;
    struct space *s;
    int left[runner_maxparts], count = 0, lcount = 0;
    int i, j, k, imax, qpos, lo, hi;
    struct {
        int lo, hi;
        } qstack[runner_maxqstack];
    struct part *parts_i, *parts_j;
    struct potential *pot, **pots;
    struct engine *eng;
    int emt, pioff, pjoff;
    FPTYPE cutoff, cutoff2, skin, skin2, r2, dx[3], w;
    struct {
        short int d, ind;
        } parts[2*runner_maxparts], temp;
    short int pivot;
    FPTYPE dscale;
    FPTYPE shift[3], inshift;
    FPTYPE pix[3], pjx[3], *pif, *pjf;
    int pind, pid, nr_pairs, count_i, count_j;
    double epot = 0.0;
    struct part **pairs;
#if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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
    count_i = cell_i->count;
    count_j = cell_j->count;
    
    /* break early if one of the cells is empty */
    if ( count_i == 0 || count_j == 0 )
        return runner_err_ok;
    
    /* Get pointers to the particle arrays. */
    parts_i = cell_i->parts;
    parts_j = cell_j->parts;
    
    /* Do we need to re-compute the pairwise Verlet list? */
    if ( s->verlet_rebuild ) {
    
        /* Has the memory for the Verlet list been allocated? */
        if ( list->pairs == NULL ) {
            if ( ( list->pairs = (struct part **)malloc( sizeof(void *) * runner_maxparts * runner_maxparts ) ) == NULL )
                return error(runner_err_malloc);
            if ( ( list->nr_pairs = (unsigned char *)malloc( sizeof(char) * runner_maxparts ) ) == NULL )
                return error(runner_err_malloc);
            }
    
        /* Is this a self interaction? */
        if ( cell_i == cell_j ) {
        
            /* The first particle has, by definition, no interactions. */
            list->nr_pairs[0] = 0;

            /* loop over all particles */
            for ( i = 1 ; i < count_i ; i++ ) {

                /* get the particle */
                part_i = &(cell_i->parts[i]);
                pix[0] = part_i->x[0];
                pix[1] = part_i->x[1];
                pix[2] = part_i->x[2];
                pioff = part_i->type * emt;
                pind = 0;
                pairs = &( list->pairs[ i * runner_maxparts ] );
                pif = &( part_i->f[0] );

                /* loop over all other particles */
                for ( j = 0 ; j < i ; j++ ) {

                    /* get the other particle */
                    part_j = &(parts_j[j]);

                    /* fetch the potential, if any */
                    pot = pots[ pioff + part_j->type ];
                    if ( pot == NULL )
                        continue;

                    /* get the distance between both particles */
                    for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                        dx[k] = pix[k] - part_j->x[k];
                        r2 += dx[k] * dx[k];
                        }

                    /* is this within cutoff? */
                    if ( r2 > skin2 )
                        continue;
                    /* runner_rcount += 1; */

                    /* Add this pair to the (pairwise) verlet list. */
                    pairs[pind] = part_j; // j;
                    pind += 1;

                    /* is this within cutoff? */
                    if ( r2 > cutoff2 )
                        continue;
                    /* runner_rcount += 1; */

                    #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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

                /* Store the number of pairs for this particle. */
                list->nr_pairs[i] = pind;

                } /* loop over all particles */

            }

        /* No, genuine pair. */
        else {

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
                    pid = parts[i].ind;
                    part_j = &( parts_j[pid] );
                    pjx[0] = part_j->x[0] + pshift[0];
                    pjx[1] = part_j->x[1] + pshift[1];
                    pjx[2] = part_j->x[2] + pshift[2];
                    pjoff = part_j->type * emt;
                    pind = 0;
                    pairs = &( list->pairs[ pid * runner_maxparts ] );
                    pjf = &( part_j->f[0] );

                    /* loop over the left particles */
                    for ( j = 0 ; j < lcount ; j++ ) {

                        /* get a handle on the second particle */
                        part_i = &( parts_i[left[j]] );

                        /* fetch the potential, if any */
                        pot = pots[ pjoff + part_i->type ];
                        if ( pot == NULL )
                            continue;

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

                        /* Add this pair to the verlet list. */
                        pairs[pind] = part_i; // left[j];
                        pind += 1;

                        /* is this within cutoff? */
                        if ( r2 > cutoff2 )
                            continue;
                        /* runner_rcount += 1; */

                        #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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
                        
                    /* Store the number of pairs for pid. */
                    list->nr_pairs[pid] = pind;
                        
                    }

                } /* loop over all particles */
            
            }

        } /* do we need to re-build the pairwise verlet list? */
        
    /* Otherwise, just evaluate using the list. */
    else {
    
        /* Loop over the particles in cell_j. */
        for ( j = 0 ; j < count_j ; j++ ) {
        
            /* Skip this particle before we get too involved. */
            if ( ( nr_pairs = list->nr_pairs[j] ) == 0 )
                continue;
                
            /* Get the particle data. */
            part_j = &(cell_j->parts[j]);
            pjx[0] = part_j->x[0] + pshift[0];
            pjx[1] = part_j->x[1] + pshift[1];
            pjx[2] = part_j->x[2] + pshift[2];
            pjf = &( part_j->f[0] );
            pairs = &( list->pairs[ j * runner_maxparts ] );
            pjoff = part_j->type * emt;

            /* Loop over the entries in the Verlet list. */
            for ( i = 0 ; i < nr_pairs ; i++ ) {
            
                /* Get the other particle */
                /* part_i = &(parts_i[ pairs[i] ]); */
                part_i = pairs[i];

                /* fetch the potential (non-NULL by design). */
                pot = pots[ pjoff + part_i->type ];

                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = part_i->x[k] - pjx[k];
                    r2 += dx[k] * dx[k];
                    }

                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */

                #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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

                } /* loop over pairs. */
        
            } /* loop over particles in cell_i. */
    
        }
        
    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
    cell_i->epot += epot;
        
    /* since nothing bad happened to us... */
    return runner_err_ok;

    }
    
    
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
 
int runner_sortedpair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift ) {

    struct part *part_i, *part_j;
    struct space *s;
    int left[runner_maxparts], count = 0, lcount = 0;
    int i, j, k, imax, qpos, lo, hi;
    struct {
        int lo, hi;
        } qstack[runner_maxqstack];
    struct part *parts_i, *parts_j;
    double epot = 0.0;
    struct potential *pot, **pots;
    struct engine *eng;
    int emt, pjoff, count_i, count_j;
    FPTYPE cutoff, cutoff2, r2, dx[3], w;
    struct {
        short int d, ind;
        } parts[2*runner_maxparts], temp;
    short int pivot;
    FPTYPE dscale;
    FPTYPE shift[3], inshift;
    FPTYPE pjx[3];
#if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
    struct potential *potq[4];
    int icount = 0, l;
    FPTYPE *effi[4], *effj[4], *pjf;
    FPTYPE r2q[4] __attribute__ ((aligned (16)));
    FPTYPE e[4] __attribute__ ((aligned (16)));
    FPTYPE f[4] __attribute__ ((aligned (16)));
    FPTYPE dxq[12];
#else
    FPTYPE e, f;
#endif
    
    /* get some useful data */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    pots = eng->p;
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    dscale = (FPTYPE)SHRT_MAX / sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] );
    count_i = cell_i->count;
    count_j = cell_j->count;
    
    /* break early if one of the cells is empty */
    if ( count_i == 0 || count_j == 0 )
        return runner_err_ok;
    
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * count_i );
        parts_j = (struct part *)alloca( sizeof(struct part) * count_j );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * count_i );
        memcpy( parts_j , cell_j->parts , sizeof(struct part) * count_j );
        }
        
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
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
        parts[count].d = 1 + dscale * ( (part_i->x[0]+pshift[0])*shift[0] + (part_i->x[1]+pshift[1])*shift[1] + (part_i->x[2]+pshift[2])*shift[2] - cutoff );
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
            if ( d[j] > d[imax] )
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
            #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
                pjf = part_j->f;
            #endif
        
            /* loop over the left particles */
            for ( j = 0 ; j < lcount ; j++ ) {
            
                /* get a handle on the second particle */
                part_i = &( parts_i[left[j]] );
                
                /* fetch the potential, if any */
                pot = pots[ pjoff + part_i->type ];
                if ( pot == NULL )
                    continue;
                
                /* get the distance between both particles */
                r2 = 0.0;
                for ( k = 0 ; k < 3 ; k++ ) {
                    dx[k] = part_i->x[k] - pjx[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */
                
                #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
                    /* add this interaction to the interaction queue. */
                    r2q[icount] = r2;
                    dxq[icount*3] = dx[0];
                    dxq[icount*3+1] = dx[1];
                    dxq[icount*3+2] = dx[2];
                    effi[icount] = part_i->f;
                    effj[icount] = pjf;
                    potq[icount] = pot;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
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
                    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
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
                    
                }
        
            }
    
        } /* loop over all particles */
        
    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
        for ( i = 0 ; i < count_j ; i++ ) {
            cell_j->parts[i].f[0] = parts_j[i].f[0];
            cell_j->parts[i].f[1] = parts_j[i].f[1];
            cell_j->parts[i].f[2] = parts_j[i].f[2];
            }
        }
        
    /* store the amaSSEd potential energy. */
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
 
int runner_sortedpair_ee ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift ) {

    struct part *part_i, *part_j;
    struct space *s;
    int left[runner_maxparts], count = 0, lcount = 0;
    int i, j, k, imax, qpos, lo, hi;
    struct {
        int lo, hi;
        } qstack[runner_maxqstack];
    struct part *parts_i, *parts_j;
    double epot = 0.0;
    struct potential *pot, *ep;
    struct engine *eng;
    int emt, pjoff, count_i, count_j;
    FPTYPE cutoff, cutoff2, r2, dx[3], dscale, w;
    struct {
        short int d, ind;
        } parts[2*runner_maxparts], temp;
    short int pivot;
    FPTYPE shift[3], inshift;
    FPTYPE pjx[3], pjq, pijq;
#if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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
    
    /* get the space and cutoff */
    eng = r->e;
    emt = eng->max_type;
    ep = eng->ep;
    s = &(eng->s);
    cutoff = s->cutoff;
    cutoff2 = s->cutoff2;
    dscale = (FPTYPE)SHRT_MAX / sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] );
    count_i = cell_i->count;
    count_j = cell_j->count;
    
    /* break early if one of the cells is empty */
    if ( count_i == 0 || count_j == 0 )
        return runner_err_ok;
    
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * count_i );
        parts_j = (struct part *)alloca( sizeof(struct part) * count_j );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * count_i );
        memcpy( parts_j , cell_j->parts , sizeof(struct part) * count_j );
        }
        
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
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
            #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
                pjf = part_j->f;
            #endif
        
            /* loop over the left particles */
            for ( j = 0 ; j < lcount ; j++ ) {
            
                /* get a handle on the second particle */
                part_i = &( parts_i[left[j]] );
                pijq = pjq * part_i->q;
                
                /* fetch the potential, if any */
                pot = eng->p[ pjoff + part_i->type ];
                if ( pot == NULL && pijq == 0.0f )
                    continue;
                    
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = part_i->x[k] - pjx[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
                    
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
        
    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
            for ( k = icount_2 ; k < 4 ; k++ )
                potq_2[k] = potq_2[0];

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
    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
        for ( i = 0 ; i < count_j ; i++ ) {
            cell_j->parts[i].f[0] = parts_j[i].f[0];
            cell_j->parts[i].f[1] = parts_j[i].f[1];
            cell_j->parts[i].f[2] = parts_j[i].f[2];
            }
        }
        
    /* store the amaSSEd potential energy. */
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

int runner_dopair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift ) {

    int i, j, k, emt, pioff, count_i, count_j;
    FPTYPE cutoff2, r2, dx[3], pix[3], w;
    double epot = 0.0;
    struct engine *eng;
    struct part *part_i, *part_j, *parts_i, *parts_j = NULL;
    struct potential *pot;
    struct space *s;
#if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || defined(__SSE2__) && defined(FPTYPE_DOUBLE)
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
    
    /* get the space and cutoff */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    cutoff2 = s->cutoff2;
    count_i = cell_i->count;
    count_j = cell_j->count;
        
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * count_i );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * count_i );
        if ( cell_i != cell_j ) {
            parts_j = (struct part *)alloca( sizeof(struct part) * count_j );
            memcpy( parts_j , cell_j->parts , sizeof(struct part) * count_j );
            }
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
            part_i = &(cell_i->parts[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            pioff = part_i->type * emt;
        
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
            
                /* get the other particle */
                part_j = &(cell_i->parts[j]);
                
                /* fetch the potential, if any */
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL )
                    continue;
                    
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */
                
                #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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
            part_i = &(cell_i->parts[i]);
            pix[0] = part_i->x[0] - shift[0];
            pix[1] = part_i->x[1] - shift[1];
            pix[2] = part_i->x[2] - shift[2];
            pioff = part_i->type * emt;
            
            /* loop over all other particles */
            for ( j = 0 ; j < count_j ; j++ ) {
            
                /* get the other particle */
                part_j = &(cell_j->parts[j]);

                /* fetch the potential, if any */
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL )
                    continue;
                    
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                /* runner_rcount += 1; */
                    
                #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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
        
    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        /* are there any leftovers (single entry)? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
        
    /* store the amaSSEd potential energy. */
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

int runner_dopair_ee ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift ) {

    int i, j, k, emt, pioff;
    FPTYPE cutoff2, r2, dx[3], pix[3], piq, pijq, w;
    double epot = 0.0;
    struct engine *eng;
    struct part *part_i, *part_j, *parts_i, *parts_j = NULL;
    struct potential *pot, *ep;
    struct space *s;
    int count_i, count_j;
#if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
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
    
    /* get the space and cutoff */
    eng = r->e;
    ep = eng->ep;
    emt = eng->max_type;
    s = &(eng->s);
    cutoff2 = s->cutoff2;
    count_i = cell_i->count;
    count_j = cell_j->count;
        
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* set pointers to the particle lists */
        parts_i = (struct part *)alloca( sizeof(struct part) * count_i );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * count_i );
        if ( cell_i != cell_j ) {
            parts_j = (struct part *)alloca( sizeof(struct part) * count_j );
            memcpy( parts_j , cell_j->parts , sizeof(struct part) * count_j );
            }
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
            part_i = &(cell_i->parts[i]);
            pix[0] = part_i->x[0];
            pix[1] = part_i->x[1];
            pix[2] = part_i->x[2];
            pioff = part_i->type * emt;
            piq = part_i->q;
        
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
            
                /* get the other particle */
                part_j = &(cell_i->parts[j]);
                pijq = piq * part_j->q;
                
                /* fetch the potential, if any */
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL && pijq == 0.0f )
                    continue;
                    
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                
                #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
                    
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
            part_i = &(cell_i->parts[i]);
            pix[0] = part_i->x[0] - shift[0];
            pix[1] = part_i->x[1] - shift[1];
            pix[2] = part_i->x[2] - shift[2];
            pioff = part_i->type * emt;
            piq = part_i->q;
            
            /* loop over all other particles */
            for ( j = 0 ; j < count_j ; j++ ) {
            
                /* get the other particle */
                part_j = &(cell_j->parts[j]);
                pijq = piq * part_j->q;

                /* fetch the potential, if any */
                pot = eng->p[ pioff + part_j->type ];
                if ( pot == NULL && pijq == 0.0f )
                    continue;
                    
                /* get the distance between both particles */
                for ( r2 = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - part_j->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* is this within cutoff? */
                if ( r2 > cutoff2 )
                    continue;
                    
                #if (defined(__SSE__) && defined(FPTYPE_SINGLE)) || (defined(__SSE2__) && defined(FPTYPE_DOUBLE))
                    
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
        
    #if defined(__SSE__) && defined(FPTYPE_SINGLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
            for ( k = icount_2 ; k < 4 ; k++ )
                potq_2[k] = potq_2[0];

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
    #elif defined(__SSE2__) && defined(FPTYPE_DOUBLE)
        /* are there any leftovers? */
        if ( icount > 0 ) {

            /* copy the first potential to the last entries */
            for ( k = icount ; k < 4 ; k++ )
                potq[k] = potq[0];

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
        
    /* store the amaSSEd potential energy. */
    cell_i->epot += epot;
        
    /* all is well that ends ok */
    return runner_err_ok;

    }


/**
 * @brief The #runner's main routine (for the Cell/BE SPU).
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having paSSEd, calls #space_getpair until there are no pairs
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
    printf("runner_run: runner %i is up and running (SPU)...\n",r->id); fflush(stdout);
    
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


                /* wait for the last pair to have been proceSSEd */
                if ( p[runner_qlen-1] != NULL ) {

                    /* read a word from the spe */
                    /* printf("runner_run: runner %i waiting for SPU response...\n",r->id); fflush(stdout); */
                    /* if ( spe_out_intr_mbox_read( r->spe , &buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 ) */
                    /*     return runner_err_spe; */
                    /* printf("runner_run: runner %i got SPU response.\n",r->id); fflush(stdout); */

                    /* release the last pair */
                    if ( space_releasepair( &(r->e->s) , p[runner_qlen-1]->i , p[runner_qlen-1]->j ) < 0 )
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
                        if ( space_releasepair( &(r->e->s) , p[k]->i , p[k]->j ) < 0 )
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
 * @brief The #runner's main routine (for Verlet lists).
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having paSSEd, checks first if the Verlet list should
 * be re-built and then proceeds to acquire chunks of the Verlet
 * list and computes its interactions.
 */

int runner_run_verlet ( struct runner *r ) {

    int res, i, ci, j, cj, k, eff_size = 0;
    struct engine *e;
    struct space *s;
    struct celltuple *t;
    FPTYPE shift[3], *eff = NULL;
    int count, from;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* get a pointer on the engine. */
    e = r->e;
    s = &(e->s);
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running (Verlet)...\n",r->id); fflush(stdout);
    
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(e) < 0)
            return error(runner_err_engine);
            
        /* runner_rcount = 0; */
            
        /* Does the Verlet list need to be reconstructed? */
        if ( s->verlet_rebuild ) {
        
            /* Loop over tuples. */
            while ( 1 ) {

                /* Get a tuple. */
                if ( ( res = space_gettuple( s , &t ) ) < 0 )
                    return r->err = runner_err_space;

                /* If there were no tuples left, bail. */
                if ( res < 1 )
                    break;

                /* Loop over all pairs in this tuple. */
                for ( i = 0 ; i < t->n ; i++ ) { 

                    /* Get the cell ID. */
                    ci = t->cellid[i];

                    for ( j = i ; j < t->n ; j++ ) {

                        /* Is this pair active? */
                        if ( !( t->pairs & ( 1 << ( i * space_maxtuples + j ) ) ) )
                            continue;

                        /* Get the cell ID. */
                        cj = t->cellid[j];

                        /* Compute the shift between ci and cj. */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            shift[k] = s->cells[cj].origin[k] - s->cells[ci].origin[k];
                            if ( shift[k] * 2 > s->dim[k] )
                                shift[k] -= s->dim[k];
                            else if ( shift[k] * 2 < -s->dim[k] )
                                shift[k] += s->dim[k];
                            }

                        /* Rebuild the Verlet entries for this cell pair. */
                        if ( runner_verlet_fill( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                            return error(runner_err);
                            
                        /* release this pair */
                        if ( space_releasepair( s , ci , cj ) < 0 )
                            return error(runner_err_space);

                        }

                    }
                    
                } /* loop over tuples. */

            /* did anything go wrong? */
            if ( res < 0 )
                return error(runner_err_space);
                
            /* printf("runner_run_verlet: runner_rcount=%i.\n", runner_rcount); */
            
            } /* reconstruct the Verlet list. */
            
        /* Otherwise, just run through the Verlet list. */
        else {
            
            /* Check if eff is large enough and re-allocate if needed. */
            if ( eff_size < s->nr_parts ) {

                /* Free old eff? */
                if ( eff != NULL )
                    free( eff );

                /* Allocate new eff. */
                eff_size = s->nr_parts;
                if ( ( eff = (FPTYPE *)malloc( sizeof(FPTYPE) * eff_size * 4 ) ) == NULL )
                    return error(runner_err_malloc);

                }

            /* Reset the force vector. */
            bzero( eff , sizeof(FPTYPE) * s->nr_parts * 4 );

            /* Re-set the potential energy. */
            r->epot = 0.0;

            /* While there are still chunks of the Verlet list out there... */
            while ( ( count = space_verlet_get( s , runner_verlet_bitesize , &from ) ) > 0 ) {

                /* Did anything go wrong? */
                if ( count < 0 )
                    return error(runner_err_space);

                /* Dispatch the interactions to runner_verlet_eval. */
                runner_verlet_eval ( r , from , count , eff );

                }

            /* did things go wrong? */
            if ( count < 0 )
                return error(runner_err_space);

            /* Send the forces and energy back to the space. */
            if ( space_verlet_force( s , eff , r->epot ) < 0 )
                return error(runner_err_space);
            
            }

        }

    /* end well... */
    return runner_err_ok;

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
 * and, once having paSSEd, calls #space_getpair until there are no pairs
 * available.
 */

int runner_run_pairs ( struct runner *r ) {

    int err = 0;
    struct cellpair *p = NULL;
    struct cellpair *finger;
    struct engine *e;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* get a pointer on the engine. */
    e = r->e;
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running...\n",r->id); fflush(stdout);
    
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(e) < 0)
            return error(runner_err_engine);
                        
        /* while i can still get a pair... */
        /* printf("runner_run: runner %i paSSEd barrier, getting pairs...\n",r->id); */
        while ( ( p = space_getpair( &e->s , r->id , runner_bitesize , NULL , &err , 1 ) ) != NULL ) {

            /* work this list of pair... */
            for ( finger = p ; finger != NULL ; finger = finger->next ) {

                /* is this cellpair playing with itself? */
                if ( finger->i == finger->j ) {
                    if ( e->flags & engine_flag_explepot ) {
                        if ( runner_dopair_ee( r , &(e->s.cells[finger->i]) , &(e->s.cells[finger->j]) , finger->shift ) < 0 )
                            return error(runner_err);
                        }
                    else {
                        if ( runner_dopair( r , &(e->s.cells[finger->i]) , &(e->s.cells[finger->j]) , finger->shift ) < 0 )
                            return error(runner_err);
                        }
                    }

                /* nope, good cell-on-cell action. */
                else {
                    if ( e->flags & engine_flag_explepot ) {
                        if ( runner_sortedpair_ee( r , &(e->s.cells[finger->i]) , &(e->s.cells[finger->j]) , finger->shift ) < 0 )
                            return error(runner_err);
                        }
                    else {
                        if ( runner_sortedpair( r , &(e->s.cells[finger->i]) , &(e->s.cells[finger->j]) , finger->shift ) < 0 )
                            return error(runner_err);
                        }
                    }

                /* release this pair */
                if ( space_releasepair( &(e->s) , finger->i , finger->j ) < 0 )
                    return error(runner_err_space);

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
    struct celltuple *t;
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
            
        /* runner_rcount = 0; */
                        
        /* Loop over tuples. */
        while ( 1 ) {
        
            /* Get a tuple. */
            if ( ( res = space_gettuple( s , &t ) ) < 0 )
                return r->err = runner_err_space;
                
            /* If there were no tuples left, bail. */
            if ( res < 1 )
                break;
                
            /* Loop over all pairs in this tuple. */
            for ( i = 0 ; i < t->n ; i++ ) { 
                        
                /* Get the cell ID. */
                ci = t->cellid[i];
                    
                for ( j = i ; j < t->n ; j++ ) {
                
                    /* Is this pair active? */
                    if ( !( t->pairs & ( 1 << ( i * space_maxtuples + j ) ) ) )
                        continue;
                        
                    /* Get the cell ID. */
                    cj = t->cellid[j];

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
                        if ( e->flags & engine_flag_explepot ) {
                            if ( runner_dopair_ee( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                                return error(runner_err);
                            }
                        else {
                            if ( runner_dopair( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                                return error(runner_err);
                            }
                        }

                    /* nope, good cell-on-cell action. */
                    else {
                        if ( e->flags & engine_flag_explepot ) {
                            if ( runner_sortedpair_ee( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                                return error(runner_err);
                            }
                        else {
                            if ( runner_sortedpair( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                                return error(runner_err);
                            }
                        }

                    /* release this pair */
                    if ( space_releasepair( s , ci , cj ) < 0 )
                        return error(runner_err_space);
                        
                    }
                    
                }
                
            } /* loop over the tuples. */

        /* give the reaction count */
        /* printf("runner_run_tuples: runner_rcount=%u.\n",runner_rcount); */
            
        }

    /* end well... */
    return runner_err_ok;

    }

    
/**
 * @brief The #runner's main routine (#celltuple model with pairwise Verlet lists).
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

int runner_run_verlet_pairwise ( struct runner *r ) {

    int res, i, j, k, ci, cj;
    struct celltuple *t;
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
    printf("runner_run: runner %i is up and running (pairwise Verlet)...\n",r->id); fflush(stdout);
    
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(e) < 0 )
            return r->err = runner_err_engine;
            
        /* runner_rcount = 0; */
                        
        /* Loop over tuples. */
        while ( 1 ) {
        
            /* Get a tuple. */
            if ( ( res = space_gettuple( s , &t ) ) < 0 )
                return r->err = runner_err_space;
                
            /* If there were no tuples left, bail. */
            if ( res < 1 )
                break;
                
            /* Loop over all pairs in this tuple. */
            for ( i = 0 ; i < t->n ; i++ ) { 
                        
                /* Get the cell ID. */
                ci = t->cellid[i];
                    
                for ( j = i ; j < t->n ; j++ ) {
                
                    /* Is this pair active? */
                    if ( !( t->pairs & ( 1 << ( i * space_maxtuples + j ) ) ) )
                        continue;
                        
                    /* Get the cell ID. */
                    cj = t->cellid[j];

                    /* Compute the shift between ci and cj. */
                    if ( i == j )
                        for ( k = 0 ; k < 3 ; k++ )
                            shift[k] = 0.0;
                    else
                        for ( k = 0 ; k < 3 ; k++ ) {
                            shift[k] = s->cells[cj].origin[k] - s->cells[ci].origin[k];
                            if ( shift[k] * 2 > s->dim[k] )
                                shift[k] -= s->dim[k];
                            else if ( shift[k] * 2 < -s->dim[k] )
                                shift[k] += s->dim[k];
                            }
                    
                    /* Compute the interactions of this pair. */
                    if ( runner_dopair_verlet( r , &(s->cells[ci]) , &(s->cells[cj]) , shift , &(t->verlet_lists[ i * space_maxtuples + j ]) ) < 0 )
                        return error(runner_err);

                    /* release this pair */
                    if ( space_releasepair( s , ci , cj ) < 0 )
                        return error(runner_err_space);
                        
                    }
                    
                }
                
            } /* loop over the tuples. */

        /* give the reaction count */
        /* printf("runner_run_verlet_pairwise: runner_rcount=%u.\n",runner_rcount); */
            
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
        int nr_pots = 0, size_pots = 0, *pots, i, j, k, l;
        struct potential *p;
        unsigned int buff;
    #endif
    #if defined(HAVE_SETAFFINITY) && !defined(CELL)
        cpu_set_t cpuset;
    #endif

    /* make sure the inputs are ok */
    if ( r == NULL || e == NULL )
        return error(runner_err_null);
        
    /* remember who i'm working for */
    r->e = e;
    r->id = id;
    
    /* If this runner will run on an SPU, it needs to init some data. */
    if ( e->flags & engine_flag_useSPU ) {
    
        #ifdef CELL
        /* if this has not been done before, init the runner data */
        if ( data == NULL ) {
    
            /* run through the potentials and count them and their size */
            for ( i = 0 ; i < e->max_type ; i++ )
                for ( j = i ; j < e->max_type ; j++ )
                    if ( e->p[ i * e->max_type + j ] != NULL ) {
                        nr_pots += 1;
                        size_pots += e->p[ i * e->max_type + j ]->n + 1;
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
            size_data += size_pots * sizeof(float) * (potential_degree+3);
            
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
                        /* loop explicitly in case FPTYPE is not float. */
                        for ( k = 0 ; k <= p->n ; k++ ) {
                            for ( l = 0 ; l < potential_degree + 3 ; l++ ) {
                                *((float *)finger) = p->c[k*(potential_degree+3)+l];
                                finger += sizeof(float);
                                }
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
        
        /* start the SPU with a pointer to the data */
        r->entry = SPE_DEFAULT_ENTRY;
	    if (pthread_create(&r->spe_thread,NULL,(void *(*)(void *))dummy,r) != 0)
		    return error(runner_err_pthread);
            
        /* wait until the SPU is ready... */
        if ( spe_out_intr_mbox_read( r->spe , &buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 )
            return runner_err_spe;
            
        /* start the runner. */
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run_cell,r) != 0)
		    return error(runner_err_pthread);
            
        #else
        
            /* if not compiled for cell, then this option is not available. */
            return error(runner_err_unavail);
        
        #endif
        
        }
    
    /* init the thread using a pairwise Verlet list. */
    else if ( e->flags & engine_flag_verlet_pairwise ) {
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run_verlet_pairwise,r) != 0)
		    return error(runner_err_pthread);
        }
        
    /* init the thread using a global Verlet list. */
    else if ( e->flags & engine_flag_verlet ) {
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run_verlet,r) != 0)
		    return error(runner_err_pthread);
        }
        
    /* init the thread using tuples. */
    else if ( e->flags & engine_flag_tuples ) {
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run_tuples,r) != 0)
		    return error(runner_err_pthread);
        }
        
    /* default: use the normal pair-list instead. */
    else {
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run_pairs,r) != 0)
		    return error(runner_err_pthread);
        }
    
    /* If we can, try to restrict this runner to a single CPU. */
    #if defined(HAVE_SETAFFINITY) && !defined(CELL)
        if ( e->flags & engine_flag_affinity ) {
        
            /* Set the cpu mask to zero | r->id. */
            CPU_ZERO( &cpuset );
            CPU_SET( r->id , &cpuset );

            /* Apply this mask to the runner's pthread. */
            if ( pthread_setaffinity_np( r->thread , sizeof(cpu_set_t) , &cpuset ) != 0 )
                return error(runner_err_pthread);

            }
    #endif
    
    /* all is well... */
    return runner_err_ok;
    
    }
