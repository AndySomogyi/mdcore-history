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
#ifdef HAVE_MPI
    #include <mpi.h>
#endif

/* Force single precision. */
#define FPTYPE_SINGLE 1

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


/* Set the max number of parts for shared buffers. */
#define cuda_maxparts 128
#define cuda_frame 32


/* The constant null potential. */
__constant__ struct potential *potential_null_cuda = NULL;

/* The number of cells and pairs. */
__constant__ int cuda_nr_pairs = 0;
__constant__ int cuda_nr_cells = 0;

/* The mutex for accessing the cell pair list. */
__device__ int cuda_cell_mutex = 0;

/* The list of cell pairs. */
__device__ struct cellpair_cuda *cuda_pairs;
__device__ int *cuda_taboo;

/* The index of the next free cell pair. */
__device__ int cuda_pair_next = 0;

/* Some constants. */
__constant__ float cuda_cutoff2 = 0.0f;
__constant__ struct potential **cuda_p;
__constant__ int cuda_maxtype = 0;

/* The part and count vectors. */
__device__


/**
 * @brief Lock a device mutex.
 *
 * @param m The mutex.
 *
 * Loops until the mutex can be set. Note that only one thread
 * can do this at a time, so to synchronize blocks, only a single thread of
 * each block should call it.
 */

__device__ inline void cuda_mutex_lock ( int *m ) {
    while ( atomicCAS( m , 0 , 1 ) != 0 );
    }


/**
 * @brief Unlock a device mutex.
 *
 * @param m The mutex.
 *
 * Does not check if the mutex had been locked.
 */

__device__ inline void cuda_mutex_unlock ( int *m ) {
    atomicExch( m , 0 );
    }


/** 
 * @brief Evaluates the given potential at the given point (interpolated).
 *
 * @param p The #potential to be evaluated.
 * @param r2 The radius at which it is to be evaluated, squared.
 * @param e Pointer to a floating-point value in which to store the
 *      interaction energy.
 * @param f Pointer to a floating-point value in which to store the
 *      magnitude of the interaction force divided by r.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p.
 */

__device__ inline void potential_eval_cuda ( struct potential *p , float r2 , float *e , float *f ) {

    int ind, k;
    float x, ee, eff, *c, r;
    
    /* Get r for the right type. */
    r = sqrtf(r2);
    
    /* compute the interval index */
    ind = fmaxf( 0.0f , p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]) );
    
    /* get the table offset */
    c = &(p->c[ind * potential_chunk]);
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    ee = c[2] * x + c[3];
    eff = c[2];
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        }

    /* store the result */
    *e = ee; *f = eff * c[1] / r;
        
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param cell_i The first cell.
 * @param cell_j The second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 *
 * @sa #runner_dopair.
 */
 
__device__ void runner_dopair_cuda ( struct part *iparts_i , int count_i , struct part *iparts_j , int count_j, float *pshift ) {

    int k, pid, pjd, threadID;
    int pjoff;
    struct part_cuda *pi, *pj;
    struct part *temp;
    struct potential *pot;
    float epot = 0.0f, dx[3], pjx[3], pjf[3], r2, w, ee, eff;
    __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i < ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        temp = iparts_i; iparts_i = iparts_j; iparts_j = temp;
        k = count_i; count_i = count_j; count_j = k;
        }
    
    /* Copy the particle data to the local buffers */
    for ( k = threadID ; k < count_i ; k += cuda_frame ) {
        parts_i[k].x[0] = iparts_i[k].x[0];
        parts_i[k].x[1] = iparts_i[k].x[1];
        parts_i[k].x[2] = iparts_i[k].x[2];
        parts_i[k].f[0] = iparts_i[k].f[0];
        parts_i[k].f[1] = iparts_i[k].f[1];
        parts_i[k].f[2] = iparts_i[k].f[2];
        parts_i[k].type = iparts_i[k].type;
        }
    for ( k = threadID ; k < count_j ; k += cuda_frame ) {
        parts_j[k].x[0] = iparts_j[k].x[0];
        parts_j[k].x[1] = iparts_j[k].x[1];
        parts_j[k].x[2] = iparts_j[k].x[2];
        parts_j[k].f[0] = iparts_j[k].f[0];
        parts_j[k].f[1] = iparts_j[k].f[1];
        parts_j[k].f[2] = iparts_j[k].f[2];
        parts_j[k].type = iparts_j[k].type;
        }
    
    /* Loop over the particles in cell_j, frame-wise. */
    for ( pjd = threadID ; pjd < count_j ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        pj = &parts_j[pjd];
        pjoff = pj->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pjx[k] = pj->x[k] + pshift[k];
            pjf[k] = 0.0f;
            }
        
        /* Loop over the particles in cell_i. */
        for ( pid = 0 ; pid < count_i ; pid++ ) {
        
            /* Get a handle on the wrapped particle pid in cell_i. */
            pi = &parts_i[ (pid + threadID) % count_i ];
            
            /* Compute the radius between pi and pj. */
            for ( r2 = 0.0f , k = 0 ; k < 3 ; k++ ) {
                dx[k] = pi->x[k] - pjx[k];
                r2 += dx[k] * dx[k];
                }
                
            /* Set the null potential if anything is bad. */
            if ( r2 > cuda_cutoff2 || ( pot = cuda_p[ pjoff + pi->type ] ) == potential_null_cuda )
                continue;
            
            /* Interact particles pi and pj. */
            potential_eval_cuda( pot , r2 , &ee , &eff );
            
            /* Store the interaction force and energy. */
            epot += ee;
            for ( k = 0 ; k < 3 ; k++ ) {
                w = eff * dx[k];
                pi->f[k] -= w;
                pjf[k] += w;
                }
        
            } /* loop over parts in cell_i. */
            
            /* Update the force on pj. */
            for ( k = 0 ; k < 3 ; k++ )
                pj->f[k] += pjf[k];
    
        } /* loop over the particles in cell_j. */
    
    /* Copy the particle data back from the local buffers */
    for ( k = threadID ; k < count_i ; k += cuda_frame ) {
        iparts_i[k].x[0] = parts_i[k].x[0];
        iparts_i[k].x[1] = parts_i[k].x[1];
        iparts_i[k].x[2] = parts_i[k].x[2];
        iparts_i[k].f[0] = parts_i[k].f[0];
        iparts_i[k].f[1] = parts_i[k].f[1];
        iparts_i[k].f[2] = parts_i[k].f[2];
        }
    for ( k = threadID ; k < count_j ; k += cuda_frame ) {
        iparts_j[k].x[0] = parts_j[k].x[0];
        iparts_j[k].x[1] = parts_j[k].x[1];
        iparts_j[k].x[2] = parts_j[k].x[2];
        iparts_j[k].f[0] = parts_j[k].f[0];
        iparts_j[k].f[1] = parts_j[k].f[1];
        iparts_j[k].f[2] = parts_j[k].f[2];
        }
        
    }


/**
 * @brief Loop over the cell pairs and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_cuda ( struct part *parts[] , int *counts ) {

    int blockID, threadID;
    int i;
    struct cellpair_cuda temp;
    __shared__ int finger, cid, cjd;
    
    /* Get the block and thread ids. */
    blockID = threadIdx.x / 32;
    threadID = threadIdx.x % 32;
    
    /* If I'm the first thread in the first block, re-set the next pair. */
    if ( blockID == 0 && threadID == 0 )
        atomicExch( &cuda_pair_next , 0 );
        
    /* Main loop... */
    while ( cuda_pair_next < cuda_nr_pairs ) {
    
        /* Try to catch a pair. */
        if ( threadID == 0 ) {
        
            /* Lock the mutex. */
            cuda_mutex_lock( &cuda_cell_mutex );
            
            /* Loop over the remaining pairs... */
            for ( i = cuda_pair_next ; i < cuda_nr_pairs ; i++ )
                if ( cuda_taboo[ cuda_pairs[i].i ] == 0 &&
                     cuda_taboo[ cuda_pairs[i].j ] == 0 )
                    break;
                    
            /* If we actually got a pair, flip it to the top and decrease
               cuda_pair_next. */
            if ( i < cuda_nr_pairs ) {
                temp = cuda_pairs[i];
                cuda_pairs[i] = cuda_pairs[ cuda_pair_next ];
                cuda_pairs[ cuda_pair_next ] = temp;
                finger = cuda_pair_next;
                cid = cuda_pairs[finger].i; cjd = cuda_pairs[finger].j;
                atomicExch( &cuda_pair_next , cuda_pair_next + 1 );
                atomicExch( &cuda_taboo[ cid ] , 1 );
                atomicExch( &cuda_taboo[ cjd ] , 1 );
                }
            else
                finger = -1;
            
            /* Un-lock the mutex. */
            cuda_mutex_unlock( &cuda_cell_mutex );
            
            /* Make sure everybody is on the same page. */
            __threadfence();
        
            }
            
        /* If we actually got a pair, do it! */
        if ( finger >= 0 ) {
        
            /* Do the pair. */
            runner_dopair_cuda( parts[cid] , counts[cid] , parts[cjd] , counts[cjd] , cuda_pairs[finger].shift );
        
            /* Release the cells in the taboo list. */
            atomicExch( &cuda_taboo[ cid ] , 0 );
            atomicExch( &cuda_taboo[ cjd ] , 0 );
            
            /* Make sure everybody is on the same page. */
            __threadfence();
        
            }
    
        } /* main loop. */

    }
