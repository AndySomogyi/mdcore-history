/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
#ifdef HAVE_SETAFFINITY
    #include <sched.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include "part.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "potential_eval.h"
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
 * @brief Rational approximation for inverf.
 */
 
inline float invndist ( float x ) {
    float xp = 2 * fabsf( x - 0.5f );
    float s = copysignf( M_SQRT2 , x - 0.5f );
    return s * 
        ((((((-4.305322784612908e-02*xp + 1.437964951099291e-02)*xp + 1.347179662670975e-01)*xp - 6.453457250071062e-01)*xp + 1.131146515368456e+00)*xp - 5.921099753875319e-01)*xp - 6.245992421211825e-06) /
        ((-6.186894481555618e-01*xp + 1.287412264927493e+00)*xp - 6.688177594499652e-01);
    }


/**
 * @brief Compute the self-interactions for the given cell.
 *
 * @param r The #runner computing the pair.
 * @param cell_i The first cell.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err)
 */

__attribute__ ((flatten)) int runner_doself_dpd ( struct runner *ru , struct cell *c ) {

    struct part *part_i, *part_j;
    struct space *s;
    int count = 0;
    int i, j, k;
    struct part *parts;
    struct engine *eng;
    int emt, pioff;
    unsigned int *seed;
    float *dpd_a, *dpd_g, alpha, gamma;
    FPTYPE cutoff2, icutoff, r2, r, ir, dprod, w;
    FPTYPE *pif;
    FPTYPE f, dx[4], vx[4], pix[4], piv[4];
    FPTYPE irmax, isqrtdt, Z;
    FPTYPE *vproj = c->vproj;
    
    /* break early if one of the cells is empty */
    count = c->count;
    if ( count == 0 )
        return runner_err_ok;
    
    /* get some useful data */
    eng = ru->e;
    emt = eng->max_type;
    dpd_a = eng->dpd_a;
    dpd_g = eng->dpd_g;
    s = &(eng->s);
    icutoff = FPTYPE_ONE / s->cutoff;
    cutoff2 = s->cutoff2;
    pix[3] = FPTYPE_ZERO;
    vx[3] = FPTYPE_ZERO;
    seed = &c->seed;
    irmax = FPTYPE_ONE / RAND_MAX;
    isqrtdt = M_SQRT2 / FPTYPE_SQRT( eng->dt );
    
    /* Make local copies of the parts if requested. */
    if ( ru->e->flags & engine_flag_localparts ) {
        parts = (struct part *)alloca( sizeof(struct part) * count );
        memcpy( parts , c->parts , sizeof(struct part) * count );
        }
    else
        parts = c->parts;
        
    /* loop over all particles */
    for ( i = 1 ; i < count ; i++ ) {

        /* get the particle */
        part_i = &(parts[i]);
        pix[0] = part_i->x[0];
        pix[1] = part_i->x[1];
        pix[2] = part_i->x[2];
        piv[0] = vproj[ 4*i + 0 ];
        piv[1] = vproj[ 4*i + 1 ];
        piv[2] = vproj[ 4*i + 2 ];
        pioff = part_i->type * emt;
        pif = &( part_i->f[0] );

        /* loop over all other particles */
        for ( j = 0 ; j < i ; j++ ) {

            /* get the other particle */
            part_j = &(parts[j]);

            /* get the distance between both particles */
            r2 = fptype_r2( pix , part_j->x , dx );

            /* is this within cutoff? */
            if ( r2 > cutoff2 )
                continue;

            /* evaluate the interaction */
            ir = FPTYPE_ONE / sqrt( r2 );
            r = r2 * ir;
            w = FPTYPE_ONE - r * icutoff;
            alpha = dpd_a[ pioff + part_j->type ];
            gamma = dpd_g[ pioff + part_j->type ];
            vx[0] = piv[0] - vproj[ 4*j + 0 ];
            vx[1] = piv[1] - vproj[ 4*j + 1 ];
            vx[2] = piv[2] - vproj[ 4*j + 2 ];
            dprod = fptype_dprod( dx , vx );
            Z = invndist( rand_r( seed ) * irmax );
            f = ( (-gamma * w * dprod * ir + isqrtdt * Z ) * gamma + alpha ) * w * ir;

            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = f * dx[k];
                part_j->f[k] -= w;
                pif[k] += w;
                }

            } /* loop over all other particles */

        } /* loop over all particles */
        
        
    /* Write local data back if needed. */
    if ( ru->e->flags & engine_flag_localparts ) {
    
        /* copy the particle data back */
        for ( i = 0 ; i < count ; i++ ) {
            c->parts[i].f[0] = parts[i].f[0];
            c->parts[i].f[1] = parts[i].f[1];
            c->parts[i].f[2] = parts[i].f[2];
            }
            
        }
        
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
 * Computes the DPD interactions between all the particles in @c cell_i and all
 * the paritcles in @c cell_j. @c cell_i and @c cell_j may be the same cell.
 *
 * @sa #runner_sortedpair.
 */

__attribute__ ((flatten)) int runner_dopair_dpd ( struct runner *ru , struct cell *cell_i , struct cell *cell_j , int sid ) {

    struct part *part_i, *part_j;
    struct space *s;
    int i, j, k;
    struct part *parts_i, *parts_j;
    float *dpd_a, *dpd_g, alpha, gamma;
    unsigned int *seed;
    struct engine *eng;
    int emt, pioff, dmaxdist, dnshift;
    FPTYPE cutoff, cutoff2, icutoff, r2, r, ir, dprod, w;
    FPTYPE irmax, Z, isqrtdt;
    unsigned int *iparts, *jparts;
    FPTYPE dscale;
    FPTYPE shift[3], nshift, bias;
    FPTYPE *pif;
    int pid, pjd, count_i, count_j;
    FPTYPE f, dx[4], vx[4], pix[4], piv[4];
    FPTYPE *vproj_i, *vproj_j;
    
    /* break early if one of the cells is empty */
    if ( cell_i->count == 0 || cell_j->count == 0 )
        return runner_err_ok;
    
    /* get the space and cutoff */
    eng = ru->e;
    emt = eng->max_type;
    s = &(eng->s);
    dpd_a = eng->dpd_a;
    dpd_g = eng->dpd_g;
    cutoff = s->cutoff;
    icutoff = FPTYPE_ONE / s->cutoff;
    cutoff2 = cutoff*cutoff;
    bias = sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] );
    dscale = (FPTYPE)SHRT_MAX / (2 * bias );
    dmaxdist = 2 + dscale * (cutoff + cell_i->maxdx + cell_j->maxdx);
    pix[3] = FPTYPE_ZERO;
    vx[3] = FPTYPE_ZERO;
    seed = &cell_i->seed;
    irmax = FPTYPE_ONE / RAND_MAX;
    isqrtdt = M_SQRT2 / FPTYPE_SQRT( eng->dt );
    
    /* Get the sort ID. */
    sid = space_getsid( s , &cell_i , &cell_j , shift );
    
    /* Get the counts. */
    count_i = cell_i->count;
    count_j = cell_j->count;
    vproj_i = cell_i->vproj;
    vproj_j = cell_j->vproj;
    
    /* Make local copies of the parts if requested. */
    if ( ru->e->flags & engine_flag_localparts ) {
        parts_i = (struct part *)alloca( sizeof(struct part) * count_i );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * count_i );
        parts_j = (struct part *)alloca( sizeof(struct part) * count_j );
        memcpy( parts_j , cell_j->parts , sizeof(struct part) * count_j );
        }
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
    /* Get the discretized shift norm. */
    nshift = sqrt( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    dnshift = dscale * nshift;

    /* Get the pointers to the left and right particle data. */
    iparts = &cell_i->sortlist[ count_i * sid ];
    jparts = &cell_j->sortlist[ count_j * sid ];

    /* loop over the sorted list of particles in i */
    for ( i = 0 ; i < count_i ; i++ ) {

        /* Quit early? */
        if ( (jparts[count_j-1] & 0xffff) + dnshift - (iparts[i] & 0xffff) > dmaxdist )
            break;

        /* get a handle on this particle */
        pid = iparts[i] >> 16;
        part_i = &( parts_i[pid] );
        pix[0] = part_i->x[0] - shift[0];
        pix[1] = part_i->x[1] - shift[1];
        pix[2] = part_i->x[2] - shift[2];
        piv[0] = vproj_i[ 4*pid + 0 ];
        piv[1] = vproj_i[ 4*pid + 1 ];
        piv[2] = vproj_i[ 4*pid + 2 ];
        pioff = part_i->type * emt;
        pif = &( part_i->f[0] );

        /* loop over the left particles */
        for ( j = count_j-1 ; j >= 0 && (jparts[j] & 0xffff) + dnshift - (iparts[i] & 0xffff) < dmaxdist ; j-- ) {

            /* get a handle on the second particle */
            pjd = jparts[j] >> 16;
            part_j = &( parts_j[ pjd ] );

            /* get the distance between both particles */
            r2 = fptype_r2( pix , part_j->x , dx );

            /* is this within cutoff? */
            if ( r2 > cutoff2 )
                continue;

            /* evaluate the interaction */
            ir = FPTYPE_ONE / sqrt( r2 );
            r = r2 * ir;
            w = FPTYPE_ONE - r * icutoff;
            alpha = dpd_a[ pioff + part_j->type ];
            gamma = dpd_g[ pioff + part_j->type ];
            vx[0] = piv[0] - vproj_j[ 4*pjd + 0 ];
            vx[1] = piv[1] - vproj_j[ 4*pjd + 1 ];
            vx[2] = piv[2] - vproj_j[ 4*pjd + 2 ];
            dprod = fptype_dprod( dx , vx );
            Z = invndist( rand_r( seed ) * irmax );
            f = ( (-gamma * w * dprod * ir + isqrtdt * Z ) * gamma + alpha ) * w * ir;

            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = f * dx[k];
                part_j->f[k] -= w;
                pif[k] += w;
                }

            }

        } /* loop over all particles */
            
        
    /* Write local data back if needed. */
    if ( ru->e->flags & engine_flag_localparts ) {
    
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

__attribute__ ((flatten)) int runner_dopair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , int sid ) {

    struct part *part_i, *part_j;
    struct space *s;
    int i, j, k;
    struct part *parts_i, *parts_j;
    struct potential *pot, **pots;
    struct engine *eng;
    int emt, pioff, dmaxdist, dnshift;
    FPTYPE cutoff, cutoff2, r2, w;
    unsigned int *iparts, *jparts;
    FPTYPE dscale;
    FPTYPE shift[3], nshift, bias;
    FPTYPE *pif;
    int pid, count_i, count_j;
    double epot = 0.0;
    FPTYPE e, f, dx[4], pix[4];
    
    /* break early if one of the cells is empty */
    if ( cell_i->count == 0 || cell_j->count == 0 )
        return runner_err_ok;
    
    /* get the space and cutoff */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    pots = eng->p;
    cutoff = s->cutoff;
    cutoff2 = cutoff*cutoff;
    bias = sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] );
    dscale = (FPTYPE)SHRT_MAX / (2 * bias );
    dmaxdist = 2 + dscale * (cutoff + cell_i->maxdx + cell_j->maxdx);
    pix[3] = FPTYPE_ZERO;
    
    /* Get the sort ID. */
    sid = space_getsid( s , &cell_i , &cell_j , shift );
    
    /* Get the counts. */
    count_i = cell_i->count;
    count_j = cell_j->count;
    
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
        parts_i = (struct part *)alloca( sizeof(struct part) * count_i );
        memcpy( parts_i , cell_i->parts , sizeof(struct part) * count_i );
        parts_j = (struct part *)alloca( sizeof(struct part) * count_j );
        memcpy( parts_j , cell_j->parts , sizeof(struct part) * count_j );
        }
    else {
        parts_i = cell_i->parts;
        parts_j = cell_j->parts;
        }
        
    /* Get the discretized shift norm. */
    nshift = sqrt( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    dnshift = dscale * nshift;

    /* Get the pointers to the left and right particle data. */
    iparts = &cell_i->sortlist[ count_i * sid ];
    jparts = &cell_j->sortlist[ count_j * sid ];

    /* loop over the sorted list of particles in i */
    for ( i = 0 ; i < count_i ; i++ ) {

        /* Quit early? */
        if ( (jparts[count_j-1] & 0xffff) + dnshift - (iparts[i] & 0xffff) > dmaxdist )
            break;

        /* get a handle on this particle */
        pid = iparts[i] >> 16;
        part_i = &( parts_i[pid] );
        pix[0] = part_i->x[0] - shift[0];
        pix[1] = part_i->x[1] - shift[1];
        pix[2] = part_i->x[2] - shift[2];
        pioff = part_i->type * emt;
        pif = &( part_i->f[0] );

        /* loop over the left particles */
        for ( j = count_j-1 ; j >= 0 && (jparts[j] & 0xffff) + dnshift - (iparts[i] & 0xffff) < dmaxdist ; j-- ) {

            /* get a handle on the second particle */
            part_j = &( parts_j[ jparts[j] >> 16 ] );

            /* fetch the potential, if any */
            pot = pots[ pioff + part_j->type ];
            if ( pot == NULL )
                continue;

            /* get the distance between both particles */
            r2 = fptype_r2( pix , part_j->x , dx );

            /* is this within cutoff? */
            if ( r2 > cutoff2 )
                continue;
            // runner_rcount += 1;

            /* evaluate the interaction */
            potential_eval( pot , r2 , &e , &f );

            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = f * dx[k];
                part_j->f[k] -= w;
                pif[k] += w;
                }

            /* tabulate the energy */
            epot += e;

            }

        } /* loop over all particles */
            
        
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
    
    
__attribute__ ((flatten)) int runner_dopair_vec ( struct runner *r , struct cell *cell_i , struct cell *cell_j , int sid ) {

#if defined(VECTORIZE) && defined(FPTYPE_SINGLE)

    struct part *part_i, *part_j[VEC_SIZE];
    struct space *s;
    int i, j, k, l, kk;
    struct part *parts_i, *parts_j;
    struct potential *pot[VEC_SIZE], **pots;
    struct engine *eng;
    int emt, pioff, dmaxdist, dnshift;
    FPTYPE cutoff, cutoff2, w;
    unsigned int *iparts, *jparts;
    FPTYPE dscale;
    FPTYPE shift[3], nshift, bias;
    FPTYPE pif[3];
    int pid, count_i, count_j;
    double epot = 0.0;
    struct potential *potq[VEC_SIZE];
    int icount = 0, fill;
    FPTYPE *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE e[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE f[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[3*VEC_SIZE];
    union {
        VEC_TYPE v;
        float f[VEC_SIZE];
        } pix[3], pjx[3], dx[3], r2 __attribute__ ((aligned (VEC_ALIGN)));
    
    /* break early if one of the cells is empty */
    if ( cell_i->count == 0 || cell_j->count == 0 )
        return runner_err_ok;
    
    /* get the space and cutoff */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    pots = eng->p;
    cutoff = s->cutoff;
    cutoff2 = cutoff*cutoff;
    bias = sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] );
    dscale = (FPTYPE)SHRT_MAX / (2 * bias );
    dmaxdist = 2 + dscale * (cutoff + cell_i->maxdx + cell_j->maxdx);
    
    /* Get the sort ID. */
    sid = space_getsid( s , &cell_i , &cell_j , shift );
    
    /* Get the parts and counts. */
    count_i = cell_i->count;
    count_j = cell_j->count;
    parts_i = cell_i->parts;
    parts_j = cell_j->parts;
    
    /* Get the discretized shift norm. */
    nshift = sqrt( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    dnshift = dscale * nshift;

    /* Get the pointers to the left and right particle data. */
    iparts = &cell_i->sortlist[ count_i * sid ];
    jparts = &cell_j->sortlist[ count_j * sid ];

    /* loop over the sorted list of particles in i */
    for ( i = 0 ; i < count_i ; i++ ) {

        /* Quit early? */
        if ( (jparts[count_j-1] & 0xffff) + dnshift - (iparts[i] & 0xffff) > dmaxdist )
            break;

        /* get a handle on this particle */
        pid = iparts[i] >> 16;
        part_i = &( parts_i[pid] );
        pix[0].v = VEC_SET1( part_i->x[0] - shift[0] );
        pix[1].v = VEC_SET1( part_i->x[1] - shift[1] );
        pix[2].v = VEC_SET1( part_i->x[2] - shift[2] );
        pioff = part_i->type * emt;
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f;
        icount = 0;

        /* loop over the left particles */
        for ( j = count_j-1 ; j >= 0 && (jparts[j] & 0xffff) + dnshift - (iparts[i] & 0xffff) < dmaxdist ; j -= VEC_SIZE ) {

            /* get a handle on the second particle */
            part_j[0] = &( parts_j[ jparts[j] >> 16 ] );
            for ( fill = 1, k = 1 ; k < VEC_SIZE ; k++ )
                if ( j - k >= 0 ) {
                    part_j[k] = &( parts_j[ jparts[j-k] >> 16 ] );
                    fill += 1;
                    }
                else
                    part_j[k] = part_j[0];

            /* fetch the potentials, if any */
            for ( k = 0 ; k < VEC_SIZE ; k++ )
                pot[k] = pots[ pioff + part_j[k]->type ];
                
            /* Get the pjx coordinates. */
            #if VEC_SIZE==8
                pjx[0].v = VEC_SET( part_j[0]->x[0] , part_j[1]->x[0] , part_j[2]->x[0] , part_j[3]->x[0] , part_j[4]->x[0] , part_j[5]->x[0] , part_j[6]->x[0] , part_j[7]->x[0] );
                pjx[1].v = VEC_SET( part_j[0]->x[1] , part_j[1]->x[1] , part_j[2]->x[1] , part_j[3]->x[1] , part_j[4]->x[1] , part_j[5]->x[1] , part_j[6]->x[1] , part_j[7]->x[1] );
                pjx[2].v = VEC_SET( part_j[0]->x[2] , part_j[1]->x[2] , part_j[2]->x[2] , part_j[3]->x[2] , part_j[4]->x[2] , part_j[5]->x[2] , part_j[6]->x[2] , part_j[7]->x[2] );
            #elif VEC_SIZE==4
                pjx[0].v = VEC_SET( part_j[0]->x[0] , part_j[1]->x[0] , part_j[2]->x[0] , part_j[3]->x[0] );
                pjx[1].v = VEC_SET( part_j[0]->x[1] , part_j[1]->x[1] , part_j[2]->x[1] , part_j[3]->x[1] );
                pjx[2].v = VEC_SET( part_j[0]->x[2] , part_j[1]->x[2] , part_j[2]->x[2] , part_j[3]->x[2] );
            #else
                #error "Unknown vector size."
            #endif

            /* get the distance between both particles */
            dx[0].v = pix[0].v - pjx[0].v;
            dx[1].v = pix[1].v - pjx[1].v;
            dx[2].v = pix[2].v - pjx[2].v;
            r2.v = dx[0].v*dx[0].v + dx[1].v*dx[1].v + dx[2].v*dx[2].v;
            
            /* Loop over the vector entries. */
            for ( kk = 0 ; kk < fill ; kk++ ) {

                /* is this within cutoff and has a potential? */
                if ( pot[kk] == NULL || r2.f[kk] > cutoff2 )
                    continue;
                // runner_rcount += 1;

                /* add this interaction to the interaction queue. */
                r2q[icount] = r2.f[kk];
                dxq[icount*3] = dx[0].f[kk];
                dxq[icount*3+1] = dx[1].f[kk];
                dxq[icount*3+2] = dx[2].f[kk];
                effj[icount] = part_j[kk]->f;
                potq[icount] = pot[kk];
                icount += 1;

                /* evaluate the interactions if the queue is full. */
                if ( icount == VEC_SIZE ) {

                    #if defined(FPTYPE_SINGLE)
                        #if VEC_SIZE==8
                        potential_eval_vec_8single( potq , r2q , e , f );
                        #else
                        potential_eval_vec_4single( potq , r2q , e , f );
                        #endif
                    #elif defined(FPTYPE_DOUBLE)
                        #if VEC_SIZE==4
                        potential_eval_vec_4double( potq , r2q , e , f );
                        #else
                        potential_eval_vec_2double( potq , r2q , e , f );
                        #endif
                    #endif

                    /* update the forces and the energy */
                    for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                        epot += e[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f[l] * dxq[l*3+k];
                            pif[k] -= w;
                            effj[l][k] += w;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
                    
                } /* loop over vector entries. */

            }

        /* are there any leftovers? */
        for ( l = 0 ; l < icount ; l++ ) {
            potential_eval( potq[l] , r2q[l] , e , f );
            epot += e[0];
            for ( k = 0 ; k < 3 ; k++ ) {
                w = f[0] * dxq[l*3+k];
                pif[k] -= w;
                effj[l][k] += w;
                }
            }
            
        /* Store the force. */
        for ( k = 0 ; k < 3 ; k++ )
            part_i->f[k] += pif[k];
        
        } /* loop over all particles */
            
        
    /* Store the potential energy to cell_i. */
    if ( cell_j->flags & cell_flag_ghost || cell_i->flags & cell_flag_ghost )
        cell_i->epot += 0.5 * epot;
    else
        cell_i->epot += epot;
        
    /* since nothing bad happened to us... */
    return runner_err_ok;
    
#else

    return runner_err_unavail;
    
#endif

    }
    
    
/**
 * @brief Compute the self-interactions for the given cell.
 *
 * @param r The #runner computing the pair.
 * @param cell_i The first cell.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err)
 */

__attribute__ ((flatten)) int runner_doself ( struct runner *r , struct cell *c ) {

    struct part *part_i, *part_j;
    struct space *s;
    int count = 0;
    int i, j, k;
    struct part *parts;
    double epot = 0.0;
    struct potential *pot, **pots;
    struct engine *eng;
    int emt, pioff;
    FPTYPE cutoff2, r2, w;
    FPTYPE *pif;
    FPTYPE e, f, dx[4], pix[4];
    
    /* break early if one of the cells is empty */
    count = c->count;
    if ( count == 0 )
        return runner_err_ok;
    
    /* get some useful data */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    pots = eng->p;
    cutoff2 = s->cutoff2;
    pix[3] = FPTYPE_ZERO;
    
    /* Make local copies of the parts if requested. */
    if ( r->e->flags & engine_flag_localparts ) {
        parts = (struct part *)alloca( sizeof(struct part) * count );
        memcpy( parts , c->parts , sizeof(struct part) * count );
        }
    else
        parts = c->parts;
        
    /* loop over all particles */
    for ( i = 1 ; i < count ; i++ ) {

        /* get the particle */
        part_i = &(parts[i]);
        pix[0] = part_i->x[0];
        pix[1] = part_i->x[1];
        pix[2] = part_i->x[2];
        pioff = part_i->type * emt;
        pif = &( part_i->f[0] );

        /* loop over all other particles */
        for ( j = 0 ; j < i ; j++ ) {

            /* get the other particle */
            part_j = &(parts[j]);

            /* get the distance between both particles */
            r2 = fptype_r2( pix , part_j->x , dx );

            /* is this within cutoff? */
            if ( r2 > cutoff2 )
                continue;

            /* fetch the potential, if any */
            pot = pots[ pioff + part_j->type ];
            if ( pot == NULL )
                continue;
            // runner_rcount += 1;

            /* evaluate the interaction */
            potential_eval( pot , r2 , &e , &f );

            /* update the forces */
            for ( k = 0 ; k < 3 ; k++ ) {
                w = f * dx[k];
                pif[k] -= w;
                part_j->f[k] += w;
                }

            /* tabulate the energy */
            epot += e;

            } /* loop over all other particles */

        } /* loop over all particles */
        
        
    /* Write local data back if needed. */
    if ( r->e->flags & engine_flag_localparts ) {
    
        /* copy the particle data back */
        for ( i = 0 ; i < count ; i++ ) {
            c->parts[i].f[0] = parts[i].f[0];
            c->parts[i].f[1] = parts[i].f[1];
            c->parts[i].f[2] = parts[i].f[2];
            }
            
        }
        
    /* Store the potential energy to c. */
    c->epot += epot;
        
    /* since nothing bad happened to us... */
    return runner_err_ok;

    }


__attribute__ ((flatten)) int runner_doself_vec ( struct runner *r , struct cell *c ) {

#if defined(VECTORIZE) && defined(FPTYPE_SINGLE)

    struct part *part_i, *part_j[VEC_SIZE];
    struct space *s;
    int count = 0;
    int i, j, k, kk, fill;
    struct part *parts;
    double epot = 0.0;
    struct potential *pot[VEC_SIZE], **pots;
    struct engine *eng;
    int emt, pioff;
    FPTYPE cutoff2, w;
    FPTYPE pif[3];
    struct potential *potq[VEC_SIZE];
    int icount = 0, l;
    FPTYPE *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE e[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE f[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
    union {
        VEC_TYPE v;
        float f[VEC_SIZE];
        } pix[3], pjx[3], dx[3], r2 __attribute__ ((aligned (VEC_ALIGN)));
    
    /* break early if one of the cells is empty */
    count = c->count;
    if ( count == 0 )
        return runner_err_ok;
    
    /* get some useful data */
    eng = r->e;
    emt = eng->max_type;
    s = &(eng->s);
    pots = eng->p;
    cutoff2 = s->cutoff2;
    parts = c->parts;
        
    /* loop over all particles */
    for ( i = 1 ; i < count ; i++ ) {

        /* get the particle */
        part_i = &(parts[i]);
        pix[0].v = VEC_SET1( part_i->x[0] );
        pix[1].v = VEC_SET1( part_i->x[1] );
        pix[2].v = VEC_SET1( part_i->x[2] );
        pioff = part_i->type * emt;
        pif[0] = FPTYPE_ZERO; pif[1] = FPTYPE_ZERO; pif[2] = FPTYPE_ZERO;
        icount = 0;

        /* loop over all other particles */
        for ( j = 0 ; j < i ; j += VEC_SIZE ) {

            /* get the other particle */
            part_j[0] = &(parts[j]);
            for ( fill = 1 , k = 1 ; k < VEC_SIZE ; k++ )
                if ( j + k < i ) {
                    part_j[k] = &( parts[ j + k ] );
                    fill += 1;
                    }
                else
                    part_j[k] = part_j[0];

            /* fetch the potentials, if any */
            for ( k = 0 ; k < VEC_SIZE ; k++ )
                pot[k] = pots[ pioff + part_j[k]->type ];
                
            /* Get the pjx coordinates. */
            #if VEC_SIZE==8
                pjx[0].v = VEC_SET( part_j[0]->x[0] , part_j[1]->x[0] , part_j[2]->x[0] , part_j[3]->x[0] , part_j[4]->x[0] , part_j[5]->x[0] , part_j[6]->x[0] , part_j[7]->x[0] );
                pjx[1].v = VEC_SET( part_j[0]->x[1] , part_j[1]->x[1] , part_j[2]->x[1] , part_j[3]->x[1] , part_j[4]->x[1] , part_j[5]->x[1] , part_j[6]->x[1] , part_j[7]->x[1] );
                pjx[2].v = VEC_SET( part_j[0]->x[2] , part_j[1]->x[2] , part_j[2]->x[2] , part_j[3]->x[2] , part_j[4]->x[2] , part_j[5]->x[2] , part_j[6]->x[2] , part_j[7]->x[2] );
            #elif VEC_SIZE==4
                pjx[0].v = VEC_SET( part_j[0]->x[0] , part_j[1]->x[0] , part_j[2]->x[0] , part_j[3]->x[0] );
                pjx[1].v = VEC_SET( part_j[0]->x[1] , part_j[1]->x[1] , part_j[2]->x[1] , part_j[3]->x[1] );
                pjx[2].v = VEC_SET( part_j[0]->x[2] , part_j[1]->x[2] , part_j[2]->x[2] , part_j[3]->x[2] );
            #else
                #error "Unknown vector size."
            #endif

            /* get the distance between both particles */
            dx[0].v = pix[0].v - pjx[0].v;
            dx[1].v = pix[1].v - pjx[1].v;
            dx[2].v = pix[2].v - pjx[2].v;
            r2.v = dx[0].v*dx[0].v + dx[1].v*dx[1].v + dx[2].v*dx[2].v;
            
            /* Loop over the vector entries. */
            for ( kk = 0 ; kk < fill ; kk++ ) {

                /* is this within cutoff? */
                if ( pot[kk] == NULL || r2.f[kk] > cutoff2 )
                    continue;

                /* add this interaction to the interaction queue. */
                r2q[icount] = r2.f[kk];
                dxq[icount*3] = dx[0].f[kk];
                dxq[icount*3+1] = dx[1].f[kk];
                dxq[icount*3+2] = dx[2].f[kk];
                effj[icount] = part_j[kk]->f;
                potq[icount] = pot[kk];
                icount += 1;

                /* evaluate the interactions if the queue is full. */
                if ( icount == VEC_SIZE ) {

                    /* evaluate the potentials */
                    #if defined(FPTYPE_SINGLE)
                        #if VEC_SIZE==8
                        potential_eval_vec_8single( potq , r2q , e , f );
                        #else
                        potential_eval_vec_4single( potq , r2q , e , f );
                        #endif
                    #elif defined(FPTYPE_DOUBLE)
                        #if VEC_SIZE==4
                        potential_eval_vec_4double( potq , r2q , e , f );
                        #else
                        potential_eval_vec_2double( potq , r2q , e , f );
                        #endif
                    #endif

                    /* update the forces and the energy */
                    for ( l = 0 ; l < VEC_SIZE ; l++ ) {
                        epot += e[l];
                        for ( k = 0 ; k < 3 ; k++ ) {
                            w = f[l] * dxq[l*3+k];
                            pif[k] -= w;
                            effj[l][k] += w;
                            }
                        }

                    /* re-set the counter. */
                    icount = 0;

                    }
                    
                } /* loop over vector entries. */

            } /* loop over all other particles */

        /* are there any leftovers? */
        for ( l = 0 ; l < icount ; l++ ) {
            potential_eval( potq[l] , r2q[l] , e , f );
            epot += e[0];
            for ( k = 0 ; k < 3 ; k++ ) {
                w = f[0] * dxq[l*3+k];
                pif[k] -= w;
                effj[l][k] += w;
                }
            }
            
        /* Store the force. */
        for ( k = 0 ; k < 3 ; k++ )
            part_i->f[k] += pif[k];
        
        } /* loop over all particles */
        

    /* Store the potential energy to c. */
    c->epot += epot;
        
    /* since nothing bad happened to us... */
    return runner_err_ok;

#else

    return runner_err_unavail;
    
#endif

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

__attribute__ ((flatten)) int runner_dopair_unsorted ( struct runner *r , struct cell *cell_i , struct cell *cell_j ) {

    int i, j, k, emt, pioff, count_i, count_j;
    FPTYPE cutoff2, r2, w, shift[3];
    FPTYPE *pif;
    double epot = 0.0;
    struct engine *eng;
    struct part *part_i, *part_j, *parts_i, *parts_j;
    struct potential *pot;
    struct space *s;
#if defined(VECTORIZE)
    int l, icount = 0;
    FPTYPE dx[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE pix[4] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE *effi[VEC_SIZE], *effj[VEC_SIZE];
    FPTYPE r2q[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE e[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE f[VEC_SIZE] __attribute__ ((aligned (VEC_ALIGN)));
    FPTYPE dxq[VEC_SIZE*3];
    struct potential *potq[VEC_SIZE];
#else
    FPTYPE e, f, dx[4], pix[4];
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
    pix[3] = FPTYPE_ZERO;
        
    /* Get the sort ID. */
    space_getsid( s , &cell_i , &cell_j , shift );
    
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
            pif = part_i->f;
            pioff = part_i->type * emt;
        
            /* loop over all other particles */
            for ( j = 0 ; j < i ; j++ ) {
            
                /* get the other particle */
                part_j = &(parts_i[j]);
                
                /* get the distance between both particles */
                r2 = fptype_r2( pix , part_j->x , dx );
                    
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
                    effi[icount] = pif;
                    effj[icount] = part_j->f;
                    potq[icount] = pot;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if ( icount == VEC_SIZE ) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single( potq , r2q , e , f );
                            #else
                            potential_eval_vec_4single( potq , r2q , e , f );
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double( potq , r2q , e , f );
                            #else
                            potential_eval_vec_2double( potq , r2q , e , f );
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for ( l = 0 ; l < VEC_SIZE ; l++ ) {
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
            pif = part_i->f;
            pioff = part_i->type * emt;
            
            /* loop over all other particles */
            for ( j = 0 ; j < count_j ; j++ ) {
            
                /* get the other particle */
                part_j = &(parts_j[j]);

                /* fetch the potential, if any */
                /* get the distance between both particles */
                r2 = fptype_r2( pix , part_j->x , dx );
                    
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
                    effi[icount] = pif;
                    effj[icount] = part_j->f;
                    potq[icount] = pot;
                    icount += 1;

                    /* evaluate the interactions if the queue is full. */
                    if ( icount == VEC_SIZE ) {

                        #if defined(FPTYPE_SINGLE)
                            #if VEC_SIZE==8
                            potential_eval_vec_8single( potq , r2q , e , f );
                            #else
                            potential_eval_vec_4single( potq , r2q , e , f );
                            #endif
                        #elif defined(FPTYPE_DOUBLE)
                            #if VEC_SIZE==4
                            potential_eval_vec_4double( potq , r2q , e , f );
                            #else
                            potential_eval_vec_2double( potq , r2q , e , f );
                            #endif
                        #endif

                        /* update the forces and the energy */
                        for ( l = 0 ; l < VEC_SIZE ; l++ ) {
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
                potential_eval_vec_8single( potq , r2q , e , f );
                #else
                potential_eval_vec_4single( potq , r2q , e , f );
                #endif
            #elif defined(VEC_DOUBLE)
                #if VEC_SIZE==4
                potential_eval_vec_4double( potq , r2q , e , f );
                #else
                potential_eval_vec_2double( potq , r2q , e , f );
                #endif
            #endif

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



