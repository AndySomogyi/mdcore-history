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

/* Include headers for overloaded vector functions. */
#include "cutil_math.h"

/* Include some conditional headers. */
#ifdef HAVE_MPI
    #include <mpi.h>
#endif

/* Force single precision. */
#define FPTYPE_SINGLE 1

/* Disable vectorization for the nvcc compiler's sake. */
#undef __SSE__
#undef __SSE2__
#undef __ALTIVEC__
#undef __AVX__

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include "part.h"
#include "cell.h"
#include "fifo.h"
#include "space.h"
#include "potential.h"
#include "engine.h"
#include "runner.h"
#include "runner_cuda.h"


/* the error macro. */
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )


/* The constant null potential. */
__constant__ struct potential *potential_null_cuda = NULL;

/* The number of cells and pairs. */
__constant__ int cuda_nr_pairs = 0;
__device__ int cuda_pairs_done = 0;
__constant__ int cuda_nr_tuples = 0;
__constant__ int cuda_nr_cells = 0;

/* The parts (non-texture access). */
__constant__ float4 *cuda_parts;

/* The mutex for accessing the cell pair list. */
__device__ int cuda_cell_mutex = 0;
__device__ int cuda_barrier = 0;

/* The list of cell pairs. */
__constant__ struct cellpair_cuda *cuda_pairs;
__device__ int *cuda_taboo;

/* The index of the next free cell pair. */
__device__ int cuda_pair_next = 0;

/* Indices for the "new" queue. */
__device__ int cuda_pair_count = 0;
__device__ int cuda_pair_curr = 0;
__device__ int *cuda_pairIDs;

/* Some constants. */
__constant__ float cuda_cutoff2 = 0.0f;
__constant__ float cuda_cutoff = 0.0f;
__constant__ float cuda_dscale = 0.0f;
__constant__ float cuda_maxdist = 0.0f;
__constant__ struct potential **cuda_p;
__constant__ int cuda_maxtype = 0;
__constant__ struct potential *cuda_pots;

/* Sortlists for the Verlet algorithm. */
__device__ unsigned int *cuda_sortlists = NULL;
__device__ int *cuda_sortlists_ind;

/* The potential coefficients, as a texture. */
texture< float4 , cudaTextureType2D > tex_coeffs;
texture< float4 , cudaTextureType2D > tex_parts;

/* Other textures. */
texture< int , cudaTextureType1D > tex_pind;
texture< unsigned int , cudaTextureType1D > tex_diags;

/* Arrays to hold the textures. */
cudaArray *cuda_coeffs, *cuda_pind, *cuda_diags;

/* The potential parameters (hard-wired size for now). */
__constant__ float cuda_eps[ 100 ];
__constant__ float cuda_rmin[ 100 ];

/* The list of fifos to work with. */
__device__ struct fifo_cuda cuda_fifos_in[ cuda_maxblocks ];
__device__ struct fifo_cuda cuda_fifos_out[ cuda_maxblocks ];

/* Use a set of variables to communicate with the outside world. */
__device__ float cuda_fio[32];
__device__ int cuda_io[32];
__device__ int cuda_rcount = 0;

/* Timers. */
__device__ float cuda_timers[ tid_count ];


/**
 * @brief Lock a device mutex.
 *
 * @param m The mutex.
 *
 * Loops until the mutex can be set. Note that only one thread
 * can do this at a time, so to synchronize blocks, only a single thread of
 * each block should call it.
 */

__device__ void cuda_mutex_lock ( int *m ) {
    TIMER_TIC
    while ( atomicCAS( m , 0 , 1 ) != 0 );
    TIMER_TOC( tid_mutex )
    }


/**
 * @brief Lock a device mutex with an additional condition.
 *
 * @param m The mutex.
 * @param c the condition
 *
 * @return @c 1 if the mutex could be locked or zero if the condition @c c
 * was reached first.
 *
 * Loops until the mutex can be set or until @c *c is non-zero.
 * Note that only one thread
 * can do this at a time, so to synchronize blocks, only a single thread of
 * each block should call it.
 */

__device__ int cuda_mutex_lock_cond ( int *m , int *c ) {
    TIMER_TIC
    while ( atomicCAS( c , 0 , 0 ) == 0 )
        if ( atomicCAS( m , 0 , 1 ) == 0 ) {
            TIMER_TOC( tid_mutex )
            return 1;
            }
    TIMER_TOC( tid_mutex )
    return 0;
    }


/**
 * @brief Unlock a device mutex.
 *
 * @param m The mutex.
 *
 * Does not check if the mutex had been locked.
 */

__device__ void cuda_mutex_unlock ( int *m ) {
    atomicExch( m , 0 );
    }
    
    
/**
 * @brief Push an element onto a #fifo_cuda, blocking.
 *
 * @return The number of elements in the #fifo_cuda.
 */
 
__device__ inline int cuda_fifo_push ( struct fifo_cuda *f , unsigned int e ) {

    /* Wait for there to be space in the list. */
    while ( f->count == cuda_fifo_size );

    /* Put the element in the list. */
    atomicExch( &(f->data[ f->last ]) , e );
    
    /* Increase the "last" counter. */
    atomicExch( &f->last , (f->last + 1) % cuda_fifo_size );
    
    /* Increase the count. */
    atomicAdd( &f->count , 1 );
    
    /* Return the fifo size. */
    return f->count;

    }
    
    
/**
 * @brief Pop an element from a #fifo_cuda, blocking.
 *
 * @return The popped element.
 */
 
__device__ inline unsigned int cuda_fifo_pop ( struct fifo_cuda *f ) {

    /* Wait for there to be something in the fifo. */
    while ( f->count == 0 );

    unsigned int res = f->data[ f->first ];

    /* Increase the "fist" counter. */
    atomicExch( &f->first , (f->first + 1) % cuda_fifo_size );
    
    /* Decrease the count. */
    atomicSub( &f->count , 1 );
    
    /* Return the first element. */
    return res;

    }
    
    
/**
 * @brief Copy bulk memory in a strided way.
 *
 * @param dest Pointer to destination memory.
 * @param source Pointer to source memory.
 * @param count Number of bytes to copy, must be a multiple of sizeof(int).
 */
 
__device__ inline void cuda_memcpy ( void *dest , void *source , int count ) {

    int j, k, icount = count / sizeof(int) / cuda_frame / cuda_memcpy_chunk;
    int *idest = (int *)dest, *isource = (int *)source;
    int chunk[cuda_memcpy_chunk];
    int threadID = threadIdx.x;
    
    TIMER_TIC
    
    /* Copy the data in chunks of sizeof(int). */
    for ( k = 0 ; k < icount ; k += 1 ) {
        #pragma unroll
        for ( j = 0 ; j < cuda_memcpy_chunk ; j++ )
            chunk[j] = isource[ (cuda_memcpy_chunk*k+j)*cuda_frame + threadID ];
        #pragma unroll
        for ( j = 0 ; j < cuda_memcpy_chunk ; j++ )
            idest[ (cuda_memcpy_chunk*k+j)*cuda_frame + threadID ] = chunk[j];
        }
    for ( k = cuda_memcpy_chunk*cuda_frame*icount + threadID ; k < count/sizeof(int) ; k += cuda_frame )
        idest[k] = isource[k];
        
    TIMER_TOC(tid_memcpy)
        
    }
    
    
__device__ inline void cuda_memcpy_old ( void *dest , void *source , int count ) {

    int k;
    volatile int *idest = (int *)dest, *isource = (int *)source;
    
    TIMER_TIC
    
    /* Copy the data in chunks of sizeof(int). */
    for ( k = 0 + threadIdx.x ; k < count/sizeof(int) ; k += cuda_frame )
        idest[k] = isource[k];
        
    TIMER_TOC(tid_memcpy)
        
    }


/**
 * @brief Sum two vectors in a strided way.
 *
 * @param a Pointer to destination memory.
 * @param b Pointer to source memory.
 * @param count Number of floats to sum.
 *
 * Computes @c a[k] += b[k] for k=1..count.
 */
 
__device__ inline void cuda_sum ( float *a , float *b , int count ) {

    int i, j, k, icount = count / cuda_frame / cuda_sum_chunk;
    float chunk[cuda_memcpy_chunk];
    int threadID = threadIdx.x;
    
    TIMER_TIC
    
    /* Copy the data in chunks of sizeof(int). */
    for ( k = 0 ; k < icount ; k += 1 ) {
        #pragma unroll
        for ( j = 0 ; j < cuda_sum_chunk ; j++ ) {
            i = (cuda_sum_chunk*k+j)*cuda_frame + threadID;
            chunk[j] = a[i] + b[i];
            }
        #pragma unroll
        for ( j = 0 ; j < cuda_sum_chunk ; j++ )
            a[ (cuda_sum_chunk*k+j)*cuda_frame + threadID ] = chunk[j];
        }
    for ( k = cuda_sum_chunk*cuda_frame*icount + threadID ; k < count ; k += cuda_frame )
        a[k] += b[k];
        
    TIMER_TOC(tid_update)
        
    }
    
    
/**
 * @brief Sort the given data w.r.t. the lowest 16 bits in decending order.
 *
 * @param a The array to sort.
 * @param count The number of elements.
 */
 
__device__ void cuda_sort_descending ( unsigned int *a , int count ) {

    int i, j, k, threadID = threadIdx.x;
    int hi[2], lo[2], ind[2], jnd[2];
    unsigned int swap_i[2], swap_j[2];

    TIMER_TIC

    /* Sort using normalized bitonic sort. */
    for ( k = 1 ; k < count ; k *= 2 ) {
    
        /* First step. */
        for ( i = threadID ;  i < count ; i += 2*cuda_frame ) {
            hi[0] = i & ~(k-1); lo[0] = i & (k-1);
            hi[1] = (i + cuda_frame) & ~(k-1); lo[1] = (i + cuda_frame) & (k-1);
            ind[0] = i + hi[0]; jnd[0] = 2*(hi[0]+k) - lo[0] - 1;
            ind[1] = i + cuda_frame + hi[1]; jnd[1] = 2*(hi[1]+k) - lo[1] - 1;
            swap_i[0] = ( jnd[0] < count ) ? a[ind[0]] : 0;
            swap_i[1] = ( jnd[1] < count ) ? a[ind[1]] : 0;
            swap_j[0] = ( jnd[0] < count ) ? a[jnd[0]] : 0;
            swap_j[1] = ( jnd[1] < count ) ? a[jnd[1]] : 0;
            if  ( ( swap_i[0] & 0xffff ) < ( swap_j[0] & 0xffff ) ) {
                a[ind[0]] = swap_j[0];
                a[jnd[0]] = swap_i[0];
                }
            if  ( ( swap_i[1] & 0xffff ) < ( swap_j[1] & 0xffff ) ) {
                a[ind[1]] = swap_j[1];
                a[jnd[1]] = swap_i[1];
                }
            }
            
        /* Let that last step sink in. */
        __threadfence_block();
    
        /* Second step(s). */
        for ( j = k/2 ; j > 0 ; j /= 2 ) {
            for ( i = threadID ;  i < count ; i += 2*cuda_frame ) {
                hi[0] = i & ~(j-1);
                hi[1] = (i + cuda_frame) & ~(j-1);
                ind[0] = i + hi[0]; jnd[0] = ind[0] + j;
                ind[1] = i + cuda_frame + hi[1]; jnd[1] = ind[1] + j;
                swap_i[0] = ( jnd[0] < count ) ? a[ind[0]] : 0;
                swap_i[1] = ( jnd[1] < count ) ? a[ind[1]] : 0;
                swap_j[0] = ( jnd[0] < count ) ? a[jnd[0]] : 0;
                swap_j[1] = ( jnd[1] < count ) ? a[jnd[1]] : 0;
                if  ( ( swap_i[0] & 0xffff ) < ( swap_j[0] & 0xffff ) ) {
                    a[ind[0]] = swap_j[0];
                    a[jnd[0]] = swap_i[0];
                    }
                if  ( ( swap_i[1] & 0xffff ) < ( swap_j[1] & 0xffff ) ) {
                    a[ind[1]] = swap_j[1];
                    a[jnd[1]] = swap_i[1];
                    }
                }
            __threadfence_block();
            }
            
        }
        
    TIMER_TOC(tid_sort)
        
    }

    
    
/**
 * @brief Sort the given data w.r.t. the lowest 16 bits in ascending order.
 *
 * @param a The array to sort.
 * @param count The number of elements.
 */
 
__device__ void cuda_sort_ascending ( unsigned int *a , int count ) {

    int i, j, k, threadID = threadIdx.x;
    int hi[2], lo[2], ind[2], jnd[2];
    unsigned int swap_i[2], swap_j[2];

    TIMER_TIC

    /* Sort using normalized bitonic sort. */
    for ( k = 1 ; k < count ; k *= 2 ) {
    
        /* First step. */
        for ( i = threadID ;  i < count ; i += 2*cuda_frame ) {
            hi[0] = i & ~(k-1); lo[0] = i & (k-1);
            hi[1] = (i + cuda_frame) & ~(k-1); lo[1] = (i + cuda_frame) & (k-1);
            ind[0] = i + hi[0]; jnd[0] = 2*(hi[0]+k) - lo[0] - 1;
            ind[1] = i + cuda_frame + hi[1]; jnd[1] = 2*(hi[1]+k) - lo[1] - 1;
            swap_i[0] = ( jnd[0] < count ) ? a[ind[0]] : 0;
            swap_i[1] = ( jnd[1] < count ) ? a[ind[1]] : 0;
            swap_j[0] = ( jnd[0] < count ) ? a[jnd[0]] : 0;
            swap_j[1] = ( jnd[1] < count ) ? a[jnd[1]] : 0;
            if  ( ( swap_i[0] & 0xffff ) > ( swap_j[0] & 0xffff ) ) {
                a[ind[0]] = swap_j[0];
                a[jnd[0]] = swap_i[0];
                }
            if  ( ( swap_i[1] & 0xffff ) > ( swap_j[1] & 0xffff ) ) {
                a[ind[1]] = swap_j[1];
                a[jnd[1]] = swap_i[1];
                }
            }
            
        /* Let that last step sink in. */
        __threadfence_block();
    
        /* Second step(s). */
        for ( j = k/2 ; j > 0 ; j /= 2 ) {
            for ( i = threadID ;  i < count ; i += 2*cuda_frame ) {
                hi[0] = i & ~(j-1);
                hi[1] = (i + cuda_frame) & ~(j-1);
                ind[0] = i + hi[0]; jnd[0] = ind[0] + j;
                ind[1] = i + cuda_frame + hi[1]; jnd[1] = ind[1] + j;
                swap_i[0] = ( jnd[0] < count ) ? a[ind[0]] : 0;
                swap_i[1] = ( jnd[1] < count ) ? a[ind[1]] : 0;
                swap_j[0] = ( jnd[0] < count ) ? a[jnd[0]] : 0;
                swap_j[1] = ( jnd[1] < count ) ? a[jnd[1]] : 0;
                if  ( ( swap_i[0] & 0xffff ) > ( swap_j[0] & 0xffff ) ) {
                    a[ind[0]] = swap_j[0];
                    a[jnd[0]] = swap_i[0];
                    }
                if  ( ( swap_i[1] & 0xffff ) > ( swap_j[1] & 0xffff ) ) {
                    a[ind[1]] = swap_j[1];
                    a[jnd[1]] = swap_i[1];
                    }
                }
            __threadfence_block();
            }
            
        }
        
    TIMER_TOC(tid_sort)
        
    }

    
    
/** 
 * @brief Evaluates the given potential at the given point (interpolated) using
 *      texture memory on the device.
 *
 * @param pid The index of the #potential to be evaluated.
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

__device__ inline void potential_eval_cuda_tex ( int pid , float r2 , float *e , float *f ) {

    int ind;
    float x, ee, eff, r, ir;
    float4 alpha, c1, c2;
    
    TIMER_TIC
    
    /* Get r for the right type. */
    ir = rsqrtf(r2);
    r = r2*ir;
    
    /* compute the interval index */
    alpha = tex2D( tex_coeffs , 0 , pid );
    // alpha = tex1D( tex_alphas , pid );
    if ( ( ind = alpha.x + r * ( alpha.y + r * alpha.z ) ) < 0 )
        ind = 0;
    
    /* pre-load the coefficients. */
    c1 = tex2D( tex_coeffs , 2*ind+2 , pid );
    c2 = tex2D( tex_coeffs , 2*ind+3 , pid );
    
    /* adjust x to the interval */
    x = (r - c1.x) * c1.y;
    
    /* compute the potential and its derivative */
    eff = c1.z;
    ee = c1.z * x + c1.w;
    eff = eff * x + ee;
    ee = ee * x + c2.x;
    eff = eff * x + ee;
    ee = ee * x + c2.y;
    eff = eff * x + ee;
    ee = ee * x + c2.z;
    eff = eff * x + ee;
    ee = ee * x + c2.w;

    /* store the result */
    *e = ee; *f = eff * c1.y * ir;
        
    TIMER_TOC(tid_potential)
        
    }


/** 
 * @brief Evaluates the given potential at the given point (interpolated) using
 *      texture memory on the device.
 *
 * @param pid The index of the #potential to be evaluated.
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

__device__ inline void potential_eval4_cuda_tex ( int4 pid , float4 r2 , float4 *e , float4 *f ) {

    int4 ind;
    float4 x, ee, eff, r, ir, c1[4], c2[4], a[4];
    
    TIMER_TIC
    
    /* Get r for the right type. */
    ir.x = rsqrtf(r2.x);
    ir.y = rsqrtf(r2.y);
    ir.z = rsqrtf(r2.z);
    ir.w = rsqrtf(r2.w);
    r = r2*ir;
    
    /* compute the interval index */
    a[0] = tex2D( tex_coeffs , 0 , pid.x );
    a[1] = tex2D( tex_coeffs , 0 , pid.y );
    a[2] = tex2D( tex_coeffs , 0 , pid.z );
    a[3] = tex2D( tex_coeffs , 0 , pid.w );
    /* a[0] = tex1D( tex_alphas , pid.x );
    a[1] = tex1D( tex_alphas , pid.y );
    a[2] = tex1D( tex_alphas , pid.z );
    a[3] = tex1D( tex_alphas , pid.w ); */
    ind.x = max( 0 , (int)( a[0].x + r.x * ( a[0].y + r.x * a[0].z ) ) );
    ind.y = max( 0 , (int)( a[1].x + r.y * ( a[1].y + r.y * a[1].z ) ) );
    ind.z = max( 0 , (int)( a[2].x + r.z * ( a[2].y + r.z * a[2].z ) ) );
    ind.w = max( 0 , (int)( a[3].x + r.w * ( a[3].y + r.w * a[3].z ) ) );
    
    /* pre-load the coefficients. */
    c1[0] = tex2D( tex_coeffs , 2*ind.x+2 , pid.x );
    c2[0] = tex2D( tex_coeffs , 2*ind.x+3 , pid.x );
    c1[1] = tex2D( tex_coeffs , 2*ind.y+2 , pid.y );
    c2[1] = tex2D( tex_coeffs , 2*ind.y+3 , pid.y );
    c1[2] = tex2D( tex_coeffs , 2*ind.z+2 , pid.z );
    c2[2] = tex2D( tex_coeffs , 2*ind.z+3 , pid.z );
    c1[3] = tex2D( tex_coeffs , 2*ind.w+2 , pid.w );
    c2[3] = tex2D( tex_coeffs , 2*ind.w+3 , pid.w );
    
    /* adjust x to the interval */
    x.x = (r.x - c1[0].x) * c1[0].y;
    x.y = (r.y - c1[1].x) * c1[1].y;
    x.z = (r.z - c1[2].x) * c1[2].y;
    x.w = (r.w - c1[3].x) * c1[3].y;
    
    /* compute the potential and its derivative */
    eff.x = c1[0].z;
    eff.y = c1[1].z;
    eff.z = c1[2].z;
    eff.w = c1[3].z;
    ee.x = c1[0].z * x.x + c1[0].w;
    ee.y = c1[1].z * x.y + c1[1].w;
    ee.z = c1[2].z * x.z + c1[2].w;
    ee.w = c1[3].z * x.w + c1[3].w;
    eff.x = eff.x * x.x + ee.x;
    eff.y = eff.y * x.y + ee.y;
    eff.z = eff.z * x.z + ee.z;
    eff.w = eff.w * x.w + ee.w;
    ee.x = ee.x * x.x + c2[0].x;
    ee.y = ee.y * x.y + c2[1].x;
    ee.z = ee.z * x.z + c2[2].x;
    ee.w = ee.w * x.w + c2[3].x;
    eff.x = eff.x * x.x + ee.x;
    eff.y = eff.y * x.y + ee.y;
    eff.z = eff.z * x.z + ee.z;
    eff.w = eff.w * x.w + ee.w;
    ee.x = ee.x * x.x + c2[0].y;
    ee.y = ee.y * x.y + c2[1].y;
    ee.z = ee.z * x.z + c2[2].y;
    ee.w = ee.w * x.w + c2[3].y;
    eff.x = eff.x * x.x + ee.x;
    eff.y = eff.y * x.y + ee.y;
    eff.z = eff.z * x.z + ee.z;
    eff.w = eff.w * x.w + ee.w;
    ee.x = ee.x * x.x + c2[0].z;
    ee.y = ee.y * x.y + c2[1].z;
    ee.z = ee.z * x.z + c2[2].z;
    ee.w = ee.w * x.w + c2[3].z;
    eff.x = eff.x * x.x + ee.x;
    eff.y = eff.y * x.y + ee.y;
    eff.z = eff.z * x.z + ee.z;
    eff.w = eff.w * x.w + ee.w;
    ee.x = ee.x * x.x + c2[0].w;
    ee.y = ee.y * x.y + c2[1].w;
    ee.z = ee.z * x.z + c2[2].w;
    ee.w = ee.w * x.w + c2[3].w;

    /* Scale the derivative accordingly. */
    eff.x *= c1[0].y * ir.x;
    eff.y *= c1[1].y * ir.y;
    eff.z *= c1[2].y * ir.z;
    eff.w *= c1[3].y * ir.w;
    
    /* store the result */
    *e = ee; *f = eff;
        
    TIMER_TOC(tid_potential4)
        
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
    float x, ee, eff, *c, ir, r;
    
    TIMER_TIC
    
    /* Get r for the right type. */
    ir = rsqrtf(r2);
    r = r2*ir;
    
    /* compute the interval index */
    ind = fmaxf( 0.0f , p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]) );
    
    /* get the table offset */
    c = &(p->c[ind * potential_chunk]);
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    ee = c[2] * x + c[3];
    eff = c[2];
    #pragma unroll
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        }

    /* store the result */
    *e = ee; *f = eff * c[1] * ir;
        
    TIMER_TOC(tid_potential)
        
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */

#ifdef PARTS_TEX 
__device__ void runner_dopair_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , float *pshift ) {
#else
__device__ void runner_dopair_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , float *pshift ) {
#endif

    int k, pid, pjd, ind, wrap_i, threadID;
    int pjoff;
    int pot;
    float epot = 0.0f, dx[3], pjf[3], shift[3], r2, w;
    float ee = 0.0f, eff = 0.0f, *temp;
    float4 pi, pj;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i < ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        #ifdef PARTS_TEX
            k = cid; cid = cjd; cjd = k;
        #else
            float4 *temp4 = parts_i; parts_i = parts_j; parts_j = temp4;
        #endif
        k = count_i; count_i = count_j; count_j = k;
        temp = forces_i; forces_i = forces_j; forces_j = temp;
        shift[0] = -pshift[0]; shift[1] = -pshift[1]; shift[2] = -pshift[2];
        }
    else {
        shift[0] = pshift[0]; shift[1] = pshift[1]; shift[2] = pshift[2];
        }

    /* Get the wraps. */
    wrap_i = (count_i < cuda_frame) ? cuda_frame : count_i;
    
    /* Make sure everybody is in the same place. */
    __threadfence_block();

    /* Loop over the particles in cell_j, frame-wise. */
    for ( pjd = threadID ; pjd < count_j ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        #ifdef PARTS_TEX
            pj = tex2D( tex_parts , pjd , cjd );
        #else
            pj = parts_j[ pjd ];
        #endif
        pjoff = pj.w * cuda_maxtype;
        pj.x += shift[0]; pj.y += shift[1]; pj.z += shift[2];
        pjf[0] = 0.0f; pjf[1] = 0.0f; pjf[2] = 0.0f;
        
        /* Loop over the particles in cell_i. */
        for ( ind = 0 ; ind < wrap_i ; ind++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pid = ind + threadID ) >= wrap_i )
                pid -= wrap_i;
            if ( pid < count_i ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                #ifdef PARTS_TEX
                    pi = tex2D( tex_parts , pid , cid );
                #else
                    pi = parts_i[ pid ];
                #endif

                /* Compute the radius between pi and pj. */
                r2 = 0.0f;
                dx[0] = pi.x - pj.x; r2 += dx[0]*dx[0];
                dx[1] = pi.y - pj.y; r2 += dx[1]*dx[1];
                dx[2] = pi.z - pj.z; r2 += dx[2]*dx[2];

                /* Set the null potential if anything is bad. */
                if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pjoff + pi.w ) ) != 0 ) {

                    // atomicAdd( &cuda_rcount , 1 );
                
                    /* Interact particles pi and pj. */
                    potential_eval_cuda_tex( pot , r2 , &ee , &eff );

                    /* Store the interaction force and energy. */
                    epot += ee;
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff * dx[k];
                        forces_i[ 3*pid + k ] -= w;
                        pjf[k] += w;
                        }

                    /* Sync the shared memory values. */
                    __threadfence_block();
                
                    } /* in range and potential. */

                } /* valid pid? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_j[ 3*pjd + k ] += pjf[k];

        /* Sync the shared memory values. */
        __threadfence_block();
            
        } /* loop over the particles in cell_j. */
        
    TIMER_TOC(tid_pair)
        
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */

#ifdef PARTS_TEX 
__device__ void runner_dopair4_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , float *pshift ) {
#else
__device__ void runner_dopair4_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , float *pshift ) {
#endif

    int k, pjd, ind, wrap_i, threadID;
    int pjoff;
    float4 pi[4], pj;
    int4 pot, pid, valid;
    float4 r2, ee, eff;
    float epot = 0.0f, dx[12], pjf[3], shift[3], w, *temp;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i < ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        #ifdef PARTS_TEX
            k = cid; cid = cjd; cjd = k;
        #else
            float4 *temp4 = parts_i; parts_i = parts_j; parts_j = temp4;
        #endif
        k = count_i; count_i = count_j; count_j = k;
        temp = forces_i; forces_i = forces_j; forces_j = temp;
        shift[0] = -pshift[0]; shift[1] = -pshift[1]; shift[2] = -pshift[2];
        }
    else {
        shift[0] = pshift[0]; shift[1] = pshift[1]; shift[2] = pshift[2];
        }

    /* Get the wraps. */
    wrap_i = (count_i < cuda_frame) ? cuda_frame : count_i;
    
    /* Make sure everybody is in the same place. */
    __threadfence_block();

    /* Loop over the particles in cell_j, frame-wise. */
    for ( pjd = threadID ; pjd < count_j ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        #ifdef PARTS_TEX
            pj = tex2D( tex_parts , pjd , cjd );
        #else
            pj = parts_j[ pjd ];
        #endif
        pjoff = pj.w * cuda_maxtype;
        pj.x += shift[0]; pj.y += shift[1]; pj.z += shift[2];
        for ( k = 0 ; k < 3 ; k++ )
            pjf[k] = 0.0f;
        
        /* Loop over the particles in cell_i. */
        for ( ind = 0 ; ind < wrap_i ; ind += 4 ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pid.x = ind + threadID ) >= wrap_i )
                pid.x -= wrap_i;
            if ( ( pid.y = ind + threadID + 1 ) >= wrap_i )
                pid.y -= wrap_i;
            if ( ( pid.z = ind + threadID + 2 ) >= wrap_i )
                pid.z -= wrap_i;
            if ( ( pid.w = ind + threadID + 3 ) >= wrap_i )
                pid.w -= wrap_i;
                
            /* Get the particle pointers. */
            #ifdef PARTS_TEX
                pi[0] = ( valid.x = ( pid.x < count_i ) ) ? tex2D( tex_parts , pid.x , cid ) : pj;
                pi[1] = ( valid.y = ( pid.y < count_i ) && ( ind + 1 < wrap_i ) ) ? tex2D( tex_parts , pid.y , cid ) : pj;
                pi[2] = ( valid.z = ( pid.z < count_i ) && ( ind + 2 < wrap_i ) ) ? tex2D( tex_parts , pid.z , cid ) : pj;
                pi[3] = ( valid.w = ( pid.w < count_i ) && ( ind + 3 < wrap_i ) ) ? tex2D( tex_parts , pid.w , cid ) : pj;
            #else
                pi[0] = ( valid.x = ( pid.x < count_i ) ) ? parts_i[ pid.x] : pj;
                pi[1] = ( valid.y = ( pid.y < count_i ) && ( ind + 1 < wrap_i ) ) ? parts_i[ pid.y ] : pj;
                pi[2] = ( valid.z = ( pid.z < count_i ) && ( ind + 2 < wrap_i ) ) ? parts_i[ pid.z ] : pj;
                pi[3] = ( valid.w = ( pid.w < count_i ) && ( ind + 3 < wrap_i ) ) ? parts_i[ pid.w ] : pj;
            #endif
            
            /* Compute the pairwise distances. */
            r2 = make_float4( 0.0f );
            dx[0] = pj.x - pi[0].x; r2.x += dx[0] * dx[0];
            dx[1] = pj.y - pi[0].y; r2.x += dx[1] * dx[1];
            dx[2] = pj.z - pi[0].z; r2.x += dx[2] * dx[2];
            dx[3] = pj.x - pi[1].x; r2.y += dx[3] * dx[3];
            dx[4] = pj.y - pi[1].y; r2.y += dx[4] * dx[4];
            dx[5] = pj.z - pi[1].z; r2.y += dx[5] * dx[5];
            dx[6] = pj.x - pi[2].x; r2.z += dx[6] * dx[6];
            dx[7] = pj.y - pi[2].y; r2.z += dx[7] * dx[7];
            dx[8] = pj.z - pi[2].z; r2.z += dx[8] * dx[8];
            dx[9] = pj.x - pi[3].x; r2.w += dx[9] * dx[9];
            dx[10] = pj.y - pi[3].y; r2.w += dx[10] * dx[10];
            dx[11] = pj.z - pi[3].z; r2.w += dx[11] * dx[11];

                
            /* Get the potentials. */
            valid.x = ( valid.x && r2.x < cuda_cutoff2 );
            valid.y = ( valid.y && r2.y < cuda_cutoff2 );
            valid.z = ( valid.z && r2.z < cuda_cutoff2 );
            valid.w = ( valid.w && r2.w < cuda_cutoff2 );
            pot.x = valid.x ? tex1D( tex_pind , pjoff + pi[0].w ) : 0;
            pot.y = valid.y ? tex1D( tex_pind , pjoff + pi[1].w ) : 0;
            pot.z = valid.z ? tex1D( tex_pind , pjoff + pi[2].w ) : 0;
            pot.w = valid.w ? tex1D( tex_pind , pjoff + pi[3].w ) : 0;
            
            /* if ( pot.x != 0 )
                atomicAdd( &cuda_rcount , 1 );
            if ( pot.y != 0 )
                atomicAdd( &cuda_rcount , 1 );
            if ( pot.z != 0 )
                atomicAdd( &cuda_rcount , 1 );
            if ( pot.w != 0 )
                atomicAdd( &cuda_rcount , 1 ); */
            
            /* Compute the interaction. */
            potential_eval4_cuda_tex( pot , r2 , &ee , &eff );
            
            /* Store the interaction energy. */
            epot += ee.x + ee.y + ee.z + ee.w;
            
            /* Update the forces. */
            if ( valid.x ) {
                pjf[0] -= ( w = eff.x * dx[0] ); forces_i[ 3*pid.x + 0 ] += w;
                pjf[1] -= ( w = eff.x * dx[1] ); forces_i[ 3*pid.x + 1 ] += w;
                pjf[2] -= ( w = eff.x * dx[2] ); forces_i[ 3*pid.x + 2 ] += w;
                }
            __threadfence_block();
            if ( valid.y ) {
                pjf[0] -= ( w = eff.y * dx[3] ); forces_i[ 3*pid.y + 0 ] += w;
                pjf[1] -= ( w = eff.y * dx[4] ); forces_i[ 3*pid.y + 1 ] += w;
                pjf[2] -= ( w = eff.y * dx[5] ); forces_i[ 3*pid.y + 2 ] += w;
                }
            __threadfence_block();
            if ( valid.z ) {
                pjf[0] -= ( w = eff.z * dx[6] ); forces_i[ 3*pid.z + 0 ] += w;
                pjf[1] -= ( w = eff.z * dx[7] ); forces_i[ 3*pid.z + 1 ] += w;
                pjf[2] -= ( w = eff.z * dx[8] ); forces_i[ 3*pid.z + 2 ] += w;
                }
            __threadfence_block();
            if ( valid.w ) {
                pjf[0] -= ( w = eff.w * dx[9] ); forces_i[ 3*pid.w + 0 ] += w;
                pjf[1] -= ( w = eff.w * dx[10] ); forces_i[ 3*pid.w + 1 ] += w;
                pjf[2] -= ( w = eff.w * dx[11] ); forces_i[ 3*pid.w + 2 ] += w;
                }
            __threadfence_block();
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_j[ 3*pjd + k ] += pjf[k];

        /* Sync the shared memory values. */
        __threadfence_block();
            
        } /* loop over the particles in cell_j. */
        
    TIMER_TOC(tid_pair)
        
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
#ifdef PARTS_TEX
__device__ void runner_dopair_verlet_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *pshift , int verlet_rebuild , unsigned int *sortlist ) {
#else
__device__ void runner_dopair_verlet_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *pshift , int verlet_rebuild , unsigned int *sortlist ) {
#endif

    int k, pid, pjd, spid, spjd, pjdid, threadID, wrap, cj;
    int pioff;
    unsigned int dmaxdist;
    float4 pi, pj;
    int pot;
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f, nshift, inshift;
    float dx[3], pif[3], shift[3], shiftn[3], *temp;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i > ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        #ifdef PARTS_TEX
            k = cid; cid = cjd; cjd = k;
        #else
            float4 *temp4 = parts_i; parts_i = parts_j; parts_j = temp4;
        #endif
        k = count_i; count_i = count_j; count_j = k;
        temp = forces_i; forces_i = forces_j; forces_j = temp;
        shift[0] = -pshift[0]; shift[1] = -pshift[1]; shift[2] = -pshift[2];
        }
    else {
        shift[0] = pshift[0]; shift[1] = pshift[1]; shift[2] = pshift[2];
        }

        
    /* Pre-compute the inverse norm of the shift. */
    nshift = sqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    inshift = 1.0f / nshift;
    shiftn[0] = inshift*shift[0]; shiftn[1] = inshift*shift[1]; shiftn[2] = inshift*shift[2];
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;
       
    TIMER_TIC2
        
    /* Re-build sorted pairs list? */
    if ( verlet_rebuild ) {
    
        /* Pack the parts of i and j into the sort arrays. */
        for ( k = threadID ; k < count_i ; k += cuda_frame ) {
            #ifdef PARTS_TEX
                pi = tex2D( tex_parts , k , cid );
            #else
                pi = parts_i[ k ];
            #endif
            sort_i[k] = ( k << 16 ) |
                (unsigned int)( cuda_dscale * (nshift + pi.x*shiftn[0] + pi.y*shiftn[1] + pi.z*shiftn[2]) );
            }
        for ( k = threadID ; k < count_j ; k += cuda_frame ) {
            #ifdef PARTS_TEX
                pj = tex2D( tex_parts , k , cjd );
            #else
                pj = parts_j[ k ];
            #endif
            sort_j[k] = ( k << 16 ) | 
                (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj.x)*shiftn[0] + (shift[1]+pj.y)*shiftn[1] + (shift[2]+pj.z)*shiftn[2]) );
            }
            
        /* Make sure all the memory is in the right place. */
        __threadfence_block();
        
        /* Sort using normalized bitonic sort. */
        cuda_sort_descending( sort_i , count_i );
        cuda_sort_ascending( sort_j , count_j );

        /* Store the sorted list back to global memory. */
        cuda_memcpy( sortlist , sort_i , sizeof(int) * count_i );
        cuda_memcpy( &sortlist[count_i] , sort_j , sizeof(int) * count_j );
            
        } /* re-build sorted pairs list. */
        
    /* Otherwise, just read it from memory. */
    else {
        cuda_memcpy( sort_i , sortlist , sizeof(int) * count_i );
        cuda_memcpy( sort_j , &sortlist[count_i] , sizeof(int) * count_j );
        __threadfence_block();
        }
        
    TIMER_TOC2(tid_sort)
        
        
    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && ( sort_j[cj-1] & 0xffff ) - ( sort_i[pid & ~(cuda_frame - 1)] & 0xffff ) > dmaxdist )
            cj -= 1;
        if ( cj == 0 )
            break;
        else if ( cj < cuda_frame )
            wrap = max( cj , min( count_i - (pid & ~(cuda_frame - 1)) , cuda_frame ) );
        else
            wrap = cj;
            
        /* Get a direct pointer on the pjdth part in cell_j. */
        spid = sort_i[pid] >> 16;
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , spid , cid );
        #else
            pi = parts_i[ spid ];
        #endif
        pioff = pi.w * cuda_maxtype;
        pi.x -= shift[0]; pi.y -= shift[1]; pi.z -= shift[2];
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f;
        
        /* Loop over the particles in cell_i. */
        for ( pjdid = 0 ; pjdid < wrap ; pjdid++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pjd = pjdid + threadID ) >= wrap )
                pjd -= wrap;
            
            /* Do we have a pair? */
            if ( pjd < cj ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                spjd = sort_j[pjd] >> 16;
                #ifdef PARTS_TEX
                    pj = tex2D( tex_parts , spjd , cjd );
                #else
                    pj = parts_j[ spjd ];
                #endif

                /* Compute the radius between pi and pj. */
                r2 = 0.0f;
                dx[0] = pi.x - pj.x; r2 += dx[0]*dx[0];
                dx[1] = pi.y - pj.y; r2 += dx[1]*dx[1];
                dx[2] = pi.z - pj.z; r2 += dx[2]*dx[2];
                    
                /* Set the null potential if anything is bad. */
                if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pioff + pj.w ) ) != 0 ) {

                    /* printf( "runner_dopair_cuda[%i]: doing pair [%i,%i] with r=%i (d=%i).\n" ,
                        threadID , sort_i[pid].ind , sort_j[pjd].ind , (int)(sqrtf(r2)*1000.0) , (int)((sort_j[pjd].d - sort_i[pid].d)*1000) ); */

                    // atomicAdd( &cuda_pairs_done , 1 );
                    
                    /* Interact particles pi and pj. */
                    potential_eval_cuda_tex( pot , r2 , &ee , &eff );

                    /* Store the interaction force and energy. */
                    epot += ee;
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff * dx[k];
                        pif[k] -= w;
                        forces_j[ 3*spjd + k ] += w;
                        }

                    /* Sync the shared memory values. */
                    __threadfence_block();
                
                    } /* in range and potential. */

                } /* do we have a pair? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_i[ 3*spid + k ] += pif[k];
    
        /* Sync the shared memory values. */
        __threadfence_block();
        
        } /* loop over the particles in cell_j. */
        
    TIMER_TOC(tid_pair)
    
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
#ifdef PARTS_TEX
__device__ void runner_dopair4_verlet_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *pshift , int verlet_rebuild , unsigned int *sortlist ) {
#else
__device__ void runner_dopair4_verlet_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *pshift , int verlet_rebuild , unsigned int *sortlist ) {
#endif

    int k, pid, spid, pjdid, threadID, wrap, cj;
    int pioff;
    unsigned int dmaxdist;
    float4 pi, pj[4];
    int4 pot, pjd, spjd, valid;
    float4 ee, eff, r2;
    float epot = 0.0f, w, nshift, inshift;
    float dx[12], pif[3], shift[3], shiftn[3], *temp;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i > ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        #ifdef PARTS_TEX
            k = cid; cid = cjd; cjd = k;
        #else
            float4 *temp4 = parts_i; parts_i = parts_j; parts_j = temp4;
        #endif
        k = count_i; count_i = count_j; count_j = k;
        temp = forces_i; forces_i = forces_j; forces_j = temp;
        shift[0] = -pshift[0]; shift[1] = -pshift[1]; shift[2] = -pshift[2];
        }
    else {
        shift[0] = pshift[0]; shift[1] = pshift[1]; shift[2] = pshift[2];
        }
        
    /* Pre-compute the inverse norm of the shift. */
    nshift = sqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    inshift = 1.0f / nshift;
    shiftn[0] = inshift*shift[0]; shiftn[1] = inshift*shift[1]; shiftn[2] = inshift*shift[2];
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;
       
    /* Re-build sorted pairs list? */
    if ( verlet_rebuild ) {
    
        TIMER_TIC2
        
        /* Pack the parts of i and j into the sort arrays. */
        for ( k = threadID ; k < count_i ; k += 4*cuda_frame ) {
            #ifdef PARTS_TEX
                pj[0] = tex2D( tex_parts , k + 0*cuda_frame , cid );
                pj[1] = tex2D( tex_parts , k + 1*cuda_frame , cid );
                pj[2] = tex2D( tex_parts , k + 2*cuda_frame , cid );
                pj[3] = tex2D( tex_parts , k + 3*cuda_frame , cid );
            #else
                pj[0] = parts_i[ k + 0*cuda_frame ];
                pj[1] = parts_i[ k + 1*cuda_frame ];
                pj[2] = parts_i[ k + 2*cuda_frame ];
                pj[3] = parts_i[ k + 3*cuda_frame ];
            #endif
            spjd.x = ( k << 16 ) | (unsigned int)( cuda_dscale * (nshift + pj[0].x*shiftn[0] + pj[0].y*shiftn[1] + pj[0].z*shiftn[2]) );
            spjd.y = ( (k + 1*cuda_frame) << 16 ) | (unsigned int)( cuda_dscale * (nshift + pj[1].x*shiftn[0] + pj[1].y*shiftn[1] + pj[1].z*shiftn[2]) );
            spjd.z = ( (k + 2*cuda_frame) << 16 ) | (unsigned int)( cuda_dscale * (nshift + pj[2].x*shiftn[0] + pj[2].y*shiftn[1] + pj[2].z*shiftn[2]) );
            spjd.w = ( (k + 3*cuda_frame) << 16 ) | (unsigned int)( cuda_dscale * (nshift + pj[3].x*shiftn[0] + pj[3].y*shiftn[1] + pj[3].z*shiftn[2]) );
            sort_i[k] = spjd.x;
            if ( k + 1*cuda_frame < count_i ) sort_i[ k + 1*cuda_frame ] = spjd.y;
            if ( k + 2*cuda_frame < count_i ) sort_i[ k + 2*cuda_frame ] = spjd.z;
            if ( k + 3*cuda_frame < count_i ) sort_i[ k + 3*cuda_frame ] = spjd.w;
            }
        for ( k = threadID ; k < count_j ; k += 4*cuda_frame ) {
            #ifdef PARTS_TEX
                pj[0] = tex2D( tex_parts , k + 0*cuda_frame , cjd );
                pj[1] = tex2D( tex_parts , k + 1*cuda_frame , cjd );
                pj[2] = tex2D( tex_parts , k + 2*cuda_frame , cjd );
                pj[3] = tex2D( tex_parts , k + 3*cuda_frame , cjd );
            #else
                pj[0] = parts_j[ k + 0*cuda_frame ];
                pj[1] = parts_j[ k + 1*cuda_frame ];
                pj[2] = parts_j[ k + 2*cuda_frame ];
                pj[3] = parts_j[ k + 3*cuda_frame ];
            #endif
            spjd.x = ( k << 16 ) | (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj[0].x)*shiftn[0] + (shift[1]+pj[0].y)*shiftn[1] + (shift[2]+pj[0].z)*shiftn[2]) );
            spjd.y = ( k + 1*cuda_frame << 16 ) | (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj[1].x)*shiftn[0] + (shift[1]+pj[1].y)*shiftn[1] + (shift[2]+pj[1].z)*shiftn[2]) );
            spjd.z = ( k + 2*cuda_frame << 16 ) | (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj[2].x)*shiftn[0] + (shift[1]+pj[2].y)*shiftn[1] + (shift[2]+pj[2].z)*shiftn[2]) );
            spjd.w = ( k + 3*cuda_frame << 16 ) | (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj[3].x)*shiftn[0] + (shift[1]+pj[3].y)*shiftn[1] + (shift[2]+pj[3].z)*shiftn[2]) );
            sort_j[k] = spjd.x;
            if ( k + 1*cuda_frame < count_j ) sort_j[ k + 1*cuda_frame ] = spjd.y;
            if ( k + 2*cuda_frame < count_j ) sort_j[ k + 2*cuda_frame ] = spjd.z;
            if ( k + 3*cuda_frame < count_j ) sort_j[ k + 3*cuda_frame ] = spjd.w;
            }
        
        TIMER_TOC2(tid_pack)
            
        /* Make sure all the memory is in the right place. */
        __threadfence_block();
        
        /* Sort using normalized bitonic sort. */
        cuda_sort_descending( sort_i , count_i );
        cuda_sort_ascending( sort_j , count_j );

        /* Store the sorted list back to global memory. */
        cuda_memcpy( sortlist , sort_i , sizeof(int) * count_i );
        cuda_memcpy( &sortlist[count_i] , sort_j , sizeof(int) * count_j );
            
        } /* re-build sorted pairs list. */
        
    /* Otherwise, just read it from memory. */
    else {
        cuda_memcpy( sort_i , sortlist , sizeof(int) * count_i );
        cuda_memcpy( sort_j , &sortlist[count_i] , sizeof(int) * count_j );
        __threadfence_block();
        }
        
        
    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && ( sort_j[cj-1] & 0xffff ) - ( sort_i[pid & ~(cuda_frame - 1)] & 0xffff ) > dmaxdist )
            cj -= 1;
        if ( cj == 0 )
            break;
        else if ( cj < cuda_frame )
            wrap = max( cj , min( count_i - (pid & ~(cuda_frame - 1)) , cuda_frame ) );
        else
            wrap = cj;
            
        /* Get a direct pointer on the pjdth part in cell_j. */
        spid = sort_i[pid] >> 16;
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , spid , cid );
        #else
            pi = parts_i[ spid ];
        #endif
        pioff = pi.w * cuda_maxtype;
        pi.x -= shift[0]; pi.y -= shift[1]; pi.z -= shift[2];
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f;
        
        /* Loop over the particles in cell_i. */
        for ( pjdid = 0 ; pjdid < wrap ; pjdid += 4 ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pjd.x = pjdid + threadID ) >= wrap )
                pjd.x -= wrap;
            if ( ( pjd.y = pjdid + threadID + 1 ) >= wrap )
                pjd.y -= wrap;
            if ( ( pjd.z = pjdid + threadID + 2 ) >= wrap )
                pjd.z -= wrap;
            if ( ( pjd.w = pjdid + threadID + 3 ) >= wrap )
                pjd.w -= wrap;
                
            /* Get the particle pointers. */
            spjd.x = sort_j[pjd.x] >> 16; spjd.y = sort_j[pjd.y] >> 16; spjd.z = sort_j[pjd.z] >> 16; spjd.w = sort_j[pjd.w] >> 16; 
            #ifdef PARTS_TEX
                pj[0] = ( valid.x = ( pjd.x < cj ) ) ? tex2D( tex_parts , spjd.x , cjd ) : pi;
                pj[1] = ( valid.y = ( pjd.y < cj ) && ( pjdid + 1 < wrap ) ) ? tex2D( tex_parts , spjd.y , cjd ) : pi;
                pj[2] = ( valid.z = ( pjd.z < cj ) && ( pjdid + 2 < wrap ) ) ? tex2D( tex_parts , spjd.z , cjd ) : pi;
                pj[3] = ( valid.w = ( pjd.w < cj ) && ( pjdid + 3 < wrap ) ) ? tex2D( tex_parts , spjd.w , cjd ) : pi;
            #else
                pj[0] = ( valid.x = ( pjd.x < cj ) ) ? parts_j[ spjd.x ] : pi;
                pj[1] = ( valid.y = ( pjd.y < cj ) && ( pjdid + 1 < wrap ) ) ? parts_j[ spjd.y ] : pi;
                pj[2] = ( valid.z = ( pjd.z < cj ) && ( pjdid + 2 < wrap ) ) ? parts_j[ spjd.z ] : pi;
                pj[3] = ( valid.w = ( pjd.w < cj ) && ( pjdid + 3 < wrap ) ) ? parts_j[ spjd.w ] : pi;
            #endif
            
            /* Compute the pairwise distances. */
            r2 = make_float4( 0.0f );
            dx[0] = pi.x - pj[0].x; r2.x += dx[0] * dx[0];
            dx[1] = pi.y - pj[0].y; r2.x += dx[1] * dx[1];
            dx[2] = pi.z - pj[0].z; r2.x += dx[2] * dx[2];
            dx[3] = pi.x - pj[1].x; r2.y += dx[3] * dx[3];
            dx[4] = pi.y - pj[1].y; r2.y += dx[4] * dx[4];
            dx[5] = pi.z - pj[1].z; r2.y += dx[5] * dx[5];
            dx[6] = pi.x - pj[2].x; r2.z += dx[6] * dx[6];
            dx[7] = pi.y - pj[2].y; r2.z += dx[7] * dx[7];
            dx[8] = pi.z - pj[2].z; r2.z += dx[8] * dx[8];
            dx[9] = pi.x - pj[3].x; r2.w += dx[9] * dx[9];
            dx[10] = pi.y - pj[3].y; r2.w += dx[10] * dx[10];
            dx[11] = pi.z - pj[3].z; r2.w += dx[11] * dx[11];
                
            /* Get the potentials. */
            valid.x = ( valid.x && r2.x < cuda_cutoff2 );
            valid.y = ( valid.y && r2.y < cuda_cutoff2 );
            valid.z = ( valid.z && r2.z < cuda_cutoff2 );
            valid.w = ( valid.w && r2.w < cuda_cutoff2 );
            pot.x = valid.x ? tex1D( tex_pind , pioff + pj[0].w ) : 0;
            pot.y = valid.y ? tex1D( tex_pind , pioff + pj[1].w ) : 0;
            pot.z = valid.z ? tex1D( tex_pind , pioff + pj[2].w ) : 0;
            pot.w = valid.w ? tex1D( tex_pind , pioff + pj[3].w ) : 0;
            
            /* Compute the interaction. */
            potential_eval4_cuda_tex( pot , r2 , &ee , &eff );
            
            /* Store the interaction energy. */
            epot += ee.x + ee.y + ee.z + ee.w;
            
            /* Update the particle forces. */
            if ( valid.x ) {
                pif[0] -= ( w = eff.x * dx[0] ); forces_j[ 3*spjd.x + 0 ] += w;
                pif[1] -= ( w = eff.x * dx[1] ); forces_j[ 3*spjd.x + 1 ] += w;
                pif[2] -= ( w = eff.x * dx[2] ); forces_j[ 3*spjd.x + 2 ] += w;
                }
            __threadfence_block();
            if ( valid.y ) {
                pif[0] -= ( w = eff.y * dx[3] ); forces_j[ 3*spjd.y + 0 ] += w;
                pif[1] -= ( w = eff.y * dx[4] ); forces_j[ 3*spjd.y + 1 ] += w;
                pif[2] -= ( w = eff.y * dx[5] ); forces_j[ 3*spjd.y + 2 ] += w;
                }
            __threadfence_block();
            if ( valid.z ) {
                pif[0] -= ( w = eff.z * dx[6] ); forces_j[ 3*spjd.z + 0 ] += w;
                pif[1] -= ( w = eff.z * dx[7] ); forces_j[ 3*spjd.z + 1 ] += w;
                pif[2] -= ( w = eff.z * dx[8] ); forces_j[ 3*spjd.z + 2 ] += w;
                }
            __threadfence_block();
            if ( valid.w ) {
                pif[0] -= ( w = eff.w * dx[9] ); forces_j[ 3*spjd.w + 0 ] += w;
                pif[1] -= ( w = eff.w * dx[10] ); forces_j[ 3*spjd.w + 1 ] += w;
                pif[2] -= ( w = eff.w * dx[11] ); forces_j[ 3*spjd.w + 2 ] += w;
                }
            __threadfence_block();
            
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_i[ 3*spid + k ] += pif[k];
    
        /* Sync the shared memory values. */
        __threadfence_block();
        
        } /* loop over the particles in cell_j. */
        
    TIMER_TOC(tid_pair)
    
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
#ifdef PARTS_TEX
__device__ void runner_dopair_sorted_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *pshift ) {
#else
__device__ void runner_dopair_sorted_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *pshift ) {
#endif

    int k, pid, pjd, spid, spjd, pjdid, threadID, wrap, cj;
    int pioff, dcutoff;
    float4 pi, pj;
    int pot;
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f, nshift, inshift;
    float dx[3], pif[3], shift[3], shiftn[3], *temp;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i > ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        #ifdef PARTS_TEX
            k = cid; cid = cjd; cjd = k;
        #else
            float4 *temp4 = parts_i; parts_i = parts_j; parts_j = temp4;
        #endif
        k = count_i; count_i = count_j; count_j = k;
        temp = forces_i; forces_i = forces_j; forces_j = temp;
        shift[0] = -pshift[0]; shift[1] = -pshift[1]; shift[2] = -pshift[2];
        }
    else {
        shift[0] = pshift[0]; shift[1] = pshift[1]; shift[2] = pshift[2];
        }
        
    /* Pre-compute the inverse norm of the shift. */
    nshift = sqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    inshift = 1.0f / nshift;
    shiftn[0] = inshift*shift[0]; shiftn[1] = inshift*shift[1]; shiftn[2] = inshift*shift[2];
    dcutoff = 2 + cuda_dscale * cuda_cutoff;
       
    TIMER_TIC2
       
    /* Pack the parts of i and j into the sort arrays. */
    for ( k = threadID ; k < count_i ; k += cuda_frame ) {
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , k , cid );
        #else
            pi = parts_i[ k ];
        #endif
        sort_i[k] = ( k << 16 ) |
            (unsigned int)( cuda_dscale * (nshift + pi.x*shiftn[0] + pi.y*shiftn[1] + pi.z*shiftn[2]) );
        }
    for ( k = threadID ; k < count_j ; k += cuda_frame ) {
        #ifdef PARTS_TEX
            pj = tex2D( tex_parts , k , cjd );
        #else
            pj = parts_j[ k ];
        #endif
        sort_j[k] = ( k << 16 ) | 
            (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj.x)*shiftn[0] + (shift[1]+pj.y)*shiftn[1] + (shift[2]+pj.z)*shiftn[2]) );
        }
        
    /* Make sure all the memory is in the right place. */
    __threadfence_block();
    
    /* Sort using normalized bitonic sort. */
    cuda_sort_descending( sort_i , count_i );
    cuda_sort_ascending( sort_j , count_j );

        
    TIMER_TOC2(tid_sort)
        

    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && ( sort_j[cj-1] & 0xffff ) - ( sort_i[pid & ~(cuda_frame - 1)] & 0xffff ) > dcutoff )
            cj -= 1;
        if ( cj == 0 )
            break;
        else if ( cj < cuda_frame )
            wrap = max( cj , min( count_i - (pid & ~(cuda_frame - 1)) , cuda_frame ) );
        else
            wrap = cj;
            
        /* Get a direct pointer on the pjdth part in cell_j. */
        spid = sort_i[pid] >> 16;
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , spid , cid );
        #else
            pi = parts_i[ spid ];
        #endif
        pioff = pi.w * cuda_maxtype;
        pi.x -= shift[0]; pi.y -= shift[1]; pi.z -= shift[2];
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f;
        
        /* Loop over the particles in cell_i. */
        for ( pjdid = 0 ; pjdid < wrap ; pjdid++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pjd = pjdid + threadID ) >= wrap )
                pjd -= wrap;
            
            /* Do we have a pair? */
            if ( pjd < cj ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                spjd = sort_j[pjd] >> 16;
                #ifdef PARTS_TEX
                    pj = tex2D( tex_parts , spjd , cjd );
                #else
                    pj = parts_j[ spjd ];
                #endif

                /* Compute the radius between pi and pj. */
                r2 = 0.0f;
                dx[0] = pi.x - pj.x; r2 += dx[0]*dx[0];
                dx[1] = pi.y - pj.y; r2 += dx[1]*dx[1];
                dx[2] = pi.z - pj.z; r2 += dx[2]*dx[2];
                    
                /* Set the null potential if anything is bad. */
                if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pioff + pj.w ) ) != 0 ) {

                    /* printf( "runner_dopair_cuda[%i]: doing pair [%i,%i] with r=%i (d=%i).\n" ,
                        threadID , sort_i[pid].ind , sort_j[pjd].ind , (int)(sqrtf(r2)*1000.0) , (int)((sort_j[pjd].d - sort_i[pid].d)*1000) ); */

                    // atomicAdd( &cuda_pairs_done , 1 );
                    
                    /* Interact particles pi and pj. */
                    potential_eval_cuda_tex( pot , r2 , &ee , &eff );

                    /* Store the interaction force and energy. */
                    epot += ee;
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff * dx[k];
                        pif[k] -= w;
                        forces_j[ 3*spjd + k ] += w;
                        }

                    /* Sync the shared memory values. */
                    __threadfence_block();
                
                    } /* in range and potential. */

                } /* do we have a pair? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_i[ 3*spid + k ] += pif[k];
    
        /* Sync the shared memory values. */
        __threadfence_block();
        
        } /* loop over the particles in cell_j. */
    
    TIMER_TOC(tid_pair)
    
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param icid Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param icjd Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param cid Part buffer in local memory.
 * @param cjd Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
#ifdef PARTS_TEX
__device__ void runner_dopair4_sorted_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *pshift ) {
#else
__device__ void runner_dopair4_sorted_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *pshift ) {
#endif

    int k, pid, spid, pjdid, threadID, wrap, cj;
    int pioff, dcutoff;
    float4 pi, pj[4];
    int4 pot, pjd, spjd, valid;
    float4 ee, eff, r2;
    float epot = 0.0f, w, nshift, inshift;
    float dx[12], pif[3], shift[3], shiftn[3], *temp;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i > ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        #ifdef PARTS_TEX
            k = cid; cid = cjd; cjd = k;
        #else
            float4 *temp4 = parts_i; parts_i = parts_j; parts_j = temp4;
        #endif
        k = count_i; count_i = count_j; count_j = k;
        temp = forces_i; forces_i = forces_j; forces_j = temp;
        shift[0] = -pshift[0]; shift[1] = -pshift[1]; shift[2] = -pshift[2];
        }
    else {
        shift[0] = pshift[0]; shift[1] = pshift[1]; shift[2] = pshift[2];
        }
        
    /* Pre-compute the inverse norm of the shift. */
    nshift = sqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    inshift = 1.0f / nshift;
    shiftn[0] = inshift*shift[0]; shiftn[1] = inshift*shift[1]; shiftn[2] = inshift*shift[2];
    dcutoff = 2 + cuda_dscale * cuda_cutoff;
       
    TIMER_TIC2
       
    /* Pack the parts of i and j into the sort arrays. */
    /* for ( k = threadID ; k < count_i ; k += cuda_frame ) {
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , k , cid );
        #else
            pi = parts_i[ k ];
        #endif
        sort_i[k] = ( k << 16 ) |
            (unsigned int)( cuda_dscale * (nshift + pi.x*shiftn[0] + pi.y*shiftn[1] + pi.z*shiftn[2]) );
        }
    for ( k = threadID ; k < count_j ; k += cuda_frame ) {
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , k , cjd );
        #else
            pi = parts_j[ k ];
        #endif
        sort_j[k] = ( k << 16 ) | 
            (unsigned int)( cuda_dscale * (nshift + (shift[0]+pi.x)*shiftn[0] + (shift[1]+pi.y)*shiftn[1] + (shift[2]+pi.z)*shiftn[2]) );
        } */
        
    /* Pack the parts of i and j into the sort arrays. */
    for ( k = threadID ; k < count_i ; k += 4*cuda_frame ) {
        #ifdef PARTS_TEX
            pj[0] = tex2D( tex_parts , k + 0*cuda_frame , cid );
            pj[1] = tex2D( tex_parts , k + 1*cuda_frame , cid );
            pj[2] = tex2D( tex_parts , k + 2*cuda_frame , cid );
            pj[3] = tex2D( tex_parts , k + 3*cuda_frame , cid );
        #else
            pj[0] = parts_i[ k + 0*cuda_frame ];
            pj[1] = parts_i[ k + 1*cuda_frame ];
            pj[2] = parts_i[ k + 2*cuda_frame ];
            pj[3] = parts_i[ k + 3*cuda_frame ];
        #endif
        spjd.x = ( k << 16 ) | (unsigned int)( cuda_dscale * (nshift + pj[0].x*shiftn[0] + pj[0].y*shiftn[1] + pj[0].z*shiftn[2]) );
        spjd.y = ( (k + 1*cuda_frame) << 16 ) | (unsigned int)( cuda_dscale * (nshift + pj[1].x*shiftn[0] + pj[1].y*shiftn[1] + pj[1].z*shiftn[2]) );
        spjd.z = ( (k + 2*cuda_frame) << 16 ) | (unsigned int)( cuda_dscale * (nshift + pj[2].x*shiftn[0] + pj[2].y*shiftn[1] + pj[2].z*shiftn[2]) );
        spjd.w = ( (k + 3*cuda_frame) << 16 ) | (unsigned int)( cuda_dscale * (nshift + pj[3].x*shiftn[0] + pj[3].y*shiftn[1] + pj[3].z*shiftn[2]) );
        sort_i[k] = spjd.x;
        if ( k + 1*cuda_frame < count_i ) sort_i[ k + 1*cuda_frame ] = spjd.y;
        if ( k + 2*cuda_frame < count_i ) sort_i[ k + 2*cuda_frame ] = spjd.z;
        if ( k + 3*cuda_frame < count_i ) sort_i[ k + 3*cuda_frame ] = spjd.w;
        }
    for ( k = threadID ; k < count_j ; k += 4*cuda_frame ) {
        #ifdef PARTS_TEX
            pj[0] = tex2D( tex_parts , k + 0*cuda_frame , cjd );
            pj[1] = tex2D( tex_parts , k + 1*cuda_frame , cjd );
            pj[2] = tex2D( tex_parts , k + 2*cuda_frame , cjd );
            pj[3] = tex2D( tex_parts , k + 3*cuda_frame , cjd );
        #else
            pj[0] = parts_j[ k + 0*cuda_frame ];
            pj[1] = parts_j[ k + 1*cuda_frame ];
            pj[2] = parts_j[ k + 2*cuda_frame ];
            pj[3] = parts_j[ k + 3*cuda_frame ];
        #endif
        spjd.x = ( k << 16 ) | (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj[0].x)*shiftn[0] + (shift[1]+pj[0].y)*shiftn[1] + (shift[2]+pj[0].z)*shiftn[2]) );
        spjd.y = ( k + 1*cuda_frame << 16 ) | (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj[1].x)*shiftn[0] + (shift[1]+pj[1].y)*shiftn[1] + (shift[2]+pj[1].z)*shiftn[2]) );
        spjd.z = ( k + 2*cuda_frame << 16 ) | (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj[2].x)*shiftn[0] + (shift[1]+pj[2].y)*shiftn[1] + (shift[2]+pj[2].z)*shiftn[2]) );
        spjd.w = ( k + 3*cuda_frame << 16 ) | (unsigned int)( cuda_dscale * (nshift + (shift[0]+pj[3].x)*shiftn[0] + (shift[1]+pj[3].y)*shiftn[1] + (shift[2]+pj[3].z)*shiftn[2]) );
        sort_j[k] = spjd.x;
        if ( k + 1*cuda_frame < count_j ) sort_j[ k + 1*cuda_frame ] = spjd.y;
        if ( k + 2*cuda_frame < count_j ) sort_j[ k + 2*cuda_frame ] = spjd.z;
        if ( k + 3*cuda_frame < count_j ) sort_j[ k + 3*cuda_frame ] = spjd.w;
        }
        
    /* Make sure all the memory is in the right place. */
    __threadfence_block();
    
    TIMER_TOC2(tid_pack)
    
    /* Sort using normalized bitonic sort. */
    cuda_sort_descending( sort_i , count_i );
    cuda_sort_ascending( sort_j , count_j );
    

    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && ( sort_j[cj-1] & 0xffff ) - ( sort_i[pid & ~(cuda_frame - 1)] & 0xffff ) > dcutoff )
            cj -= 1;
        if ( cj == 0 )
            break;
        else if ( cj < cuda_frame )
            wrap = max( cj , min( count_i - (pid & ~(cuda_frame - 1)) , cuda_frame ) );
        else
            wrap = cj;
            
        /* Get a direct pointer on the pjdth part in cell_j. */
        spid = sort_i[pid] >> 16;
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , spid , cid );
        #else
            pi = parts_i[ spid ];
        #endif
        pioff = pi.w * cuda_maxtype;
        pi.x -= shift[0]; pi.y -= shift[1]; pi.z -= shift[2];
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f;
        
        /* Loop over the particles in cell_i. */
        for ( pjdid = 0 ; pjdid < wrap ; pjdid += 4 ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pjd.x = pjdid + threadID ) >= wrap )
                pjd.x -= wrap;
            if ( ( pjd.y = pjdid + threadID + 1 ) >= wrap )
                pjd.y -= wrap;
            if ( ( pjd.z = pjdid + threadID + 2 ) >= wrap )
                pjd.z -= wrap;
            if ( ( pjd.w = pjdid + threadID + 3 ) >= wrap )
                pjd.w -= wrap;
                
            /* Get the particle pointers. */
            spjd.x = sort_j[pjd.x] >> 16; spjd.y = sort_j[pjd.y] >> 16; spjd.z = sort_j[pjd.z] >> 16; spjd.w = sort_j[pjd.w] >> 16; 
            #ifdef PARTS_TEX
                pj[0] = ( valid.x = ( pjd.x < cj ) ) ? tex2D( tex_parts , spjd.x , cjd ) : pi;
                pj[1] = ( valid.y = ( pjd.y < cj ) && ( pjdid + 1 < wrap ) ) ? tex2D( tex_parts , spjd.y , cjd ) : pi;
                pj[2] = ( valid.z = ( pjd.z < cj ) && ( pjdid + 2 < wrap ) ) ? tex2D( tex_parts , spjd.z , cjd ) : pi;
                pj[3] = ( valid.w = ( pjd.w < cj ) && ( pjdid + 3 < wrap ) ) ? tex2D( tex_parts , spjd.w , cjd ) : pi;
            #else
                pj[0] = ( valid.x = ( pjd.x < cj ) ) ? parts_j[ spjd.x ] : pi;
                pj[1] = ( valid.y = ( pjd.y < cj ) && ( pjdid + 1 < wrap ) ) ? parts_j[ spjd.y ] : pi;
                pj[2] = ( valid.z = ( pjd.z < cj ) && ( pjdid + 2 < wrap ) ) ? parts_j[ spjd.z ] : pi;
                pj[3] = ( valid.w = ( pjd.w < cj ) && ( pjdid + 3 < wrap ) ) ? parts_j[ spjd.w ] : pi;
            #endif
            
            /* Compute the pairwise distances. */
            r2 = make_float4( 0.0f );
            dx[0] = pi.x - pj[0].x; r2.x += dx[0] * dx[0];
            dx[1] = pi.y - pj[0].y; r2.x += dx[1] * dx[1];
            dx[2] = pi.z - pj[0].z; r2.x += dx[2] * dx[2];
            dx[3] = pi.x - pj[1].x; r2.y += dx[3] * dx[3];
            dx[4] = pi.y - pj[1].y; r2.y += dx[4] * dx[4];
            dx[5] = pi.z - pj[1].z; r2.y += dx[5] * dx[5];
            dx[6] = pi.x - pj[2].x; r2.z += dx[6] * dx[6];
            dx[7] = pi.y - pj[2].y; r2.z += dx[7] * dx[7];
            dx[8] = pi.z - pj[2].z; r2.z += dx[8] * dx[8];
            dx[9] = pi.x - pj[3].x; r2.w += dx[9] * dx[9];
            dx[10] = pi.y - pj[3].y; r2.w += dx[10] * dx[10];
            dx[11] = pi.z - pj[3].z; r2.w += dx[11] * dx[11];
                
            /* Get the potentials. */
            valid.x = ( valid.x && r2.x < cuda_cutoff2 );
            valid.y = ( valid.y && r2.y < cuda_cutoff2 );
            valid.z = ( valid.z && r2.z < cuda_cutoff2 );
            valid.w = ( valid.w && r2.w < cuda_cutoff2 );
            pot.x = valid.x ? tex1D( tex_pind , pioff + pj[0].w ) : 0;
            pot.y = valid.y ? tex1D( tex_pind , pioff + pj[1].w ) : 0;
            pot.z = valid.z ? tex1D( tex_pind , pioff + pj[2].w ) : 0;
            pot.w = valid.w ? tex1D( tex_pind , pioff + pj[3].w ) : 0;
            
            /* if ( pot.x != 0 )
                atomicAdd( &cuda_rcount , 1 );
            if ( pot.y != 0 )
                atomicAdd( &cuda_rcount , 1 );
            if ( pot.z != 0 )
                atomicAdd( &cuda_rcount , 1 );
            if ( pot.w != 0 )
                atomicAdd( &cuda_rcount , 1 ); */
            
            /* Compute the interaction. */
            potential_eval4_cuda_tex( pot , r2 , &ee , &eff );
            
            /* Store the interaction energy. */
            epot += ee.x + ee.y + ee.z + ee.w;
            
            /* Update the particle forces. */
            if ( valid.x ) {
                pif[0] -= ( w = eff.x * dx[0] ); forces_j[ 3*spjd.x + 0 ] += w;
                pif[1] -= ( w = eff.x * dx[1] ); forces_j[ 3*spjd.x + 1 ] += w;
                pif[2] -= ( w = eff.x * dx[2] ); forces_j[ 3*spjd.x + 2 ] += w;
                }
            __threadfence_block();
            if ( valid.y ) {
                pif[0] -= ( w = eff.y * dx[3] ); forces_j[ 3*spjd.y + 0 ] += w;
                pif[1] -= ( w = eff.y * dx[4] ); forces_j[ 3*spjd.y + 1 ] += w;
                pif[2] -= ( w = eff.y * dx[5] ); forces_j[ 3*spjd.y + 2 ] += w;
                }
            __threadfence_block();
            if ( valid.z ) {
                pif[0] -= ( w = eff.z * dx[6] ); forces_j[ 3*spjd.z + 0 ] += w;
                pif[1] -= ( w = eff.z * dx[7] ); forces_j[ 3*spjd.z + 1 ] += w;
                pif[2] -= ( w = eff.z * dx[8] ); forces_j[ 3*spjd.z + 2 ] += w;
                }
            __threadfence_block();
            if ( valid.w ) {
                pif[0] -= ( w = eff.w * dx[9] ); forces_j[ 3*spjd.w + 0 ] += w;
                pif[1] -= ( w = eff.w * dx[10] ); forces_j[ 3*spjd.w + 1 ] += w;
                pif[2] -= ( w = eff.w * dx[11] ); forces_j[ 3*spjd.w + 2 ] += w;
                }
            __threadfence_block();
            
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_i[ 3*spid + k ] += pif[k];
    
        /* Sync the shared memory values. */
        __threadfence_block();
        
        } /* loop over the particles in cell_j. */
    
    TIMER_TOC(tid_pair)
    
    }


/**
 * @brief Compute the self interactions for the given cell on a CUDA device.
 *
 * @param iparts Array of parts in this cell.
 * @param count Number of parts in the cell.
 * @param parts Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
#ifdef PARTS_TEX
__device__ void runner_doself_cuda ( int cid , int count , float *forces ) {
#else
__device__ void runner_doself_cuda ( float4 *parts , int count , float *forces ) {
#endif

    int k, pid, pjd, threadID;
    int pjoff;
    float4 pi, pj;
    int pot;
    float epot = 0.0f, dx[3], pjf[3], r2, w, ee, eff;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Make sure everybody is in the same place. */
    __threadfence_block();

    /* Loop over the particles in the cell, frame-wise. */
    for ( pjd = threadID ; pjd < count-1 ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        #ifdef PARTS_TEX
            pj = tex2D( tex_parts , pjd , cid );
        #else
            pj = parts[ pjd ];
        #endif
        pjoff = pj.w * cuda_maxtype;
        pjf[0] = 0.0f; pjf[1] = 0.0f; pjf[2] = 0.0f;
            
        /* Loop over the particles in cell_i. */
        for ( pid = pjd+1 ; pid < count ; pid++ ) {
        
            /* Get a handle on the wrapped particle pid in cell_i. */
            #ifdef PARTS_TEX
                pi = tex2D( tex_parts , pid , cid );
            #else
                pi = parts[ pid ];
            #endif

            /* Compute the radius between pi and pj. */
            r2 = 0.0f;
            dx[0] = pi.x - pj.x; r2 += dx[0]*dx[0];
            dx[1] = pi.y - pj.y; r2 += dx[1]*dx[1];
            dx[2] = pi.z - pj.z; r2 += dx[2]*dx[2];

            /* Set the null potential if anything is bad. */
            if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pjoff + pi.w ) ) != 0 ) {

                // atomicAdd( &cuda_pairs_done , 1 );
            
                /* Interact particles pi and pj. */
                potential_eval_cuda_tex( pot , r2 , &ee , &eff );

                /* Store the interaction force and energy. */
                epot += ee;
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff * dx[k];
                    forces[ 3*pid + k ] -= w;
                    pjf[k] += w;
                    }

                /* Sync the shared memory values. */
                __threadfence_block();
            
                } /* in range and potential. */

            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces[ 3*pjd + k ] += pjf[k];

        /* Sync the shared memory values. */
        __threadfence_block();

        } /* loop over the particles in cell_j. */
        
    TIMER_TOC(tid_self)
    
    }
    
    
/**
 * @brief Compute the self interactions for the given cell on a CUDA device.
 *
 * @param iparts Array of parts in this cell.
 * @param count Number of parts in the cell.
 * @param parts Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
#ifdef PARTS_TEX
__device__ void runner_doself_diag_cuda ( int cid , int count , float *forces ) {
#else
__device__ void runner_doself_diag_cuda ( float4 *parts , int count , float *forces ) {
#endif

    int diag, k, diag_max, step, pid, pjd, threadID;
    unsigned int packed;
    float4 pi, pj;
    int pot;
    float epot = 0.0f, dx[3], r2, w[3], ee, eff;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Step along the number of diagonal entries. */
    diag_max = count * (count - 1) / 2; step = 1;
    for ( diag = 0 ; diag < diag_max ; diag += step ) {
    
        /* is it time for this thread to step in? */
        if ( diag == threadID ) {
            step = diag;
            diag = (diag + 2) * (diag + 1) / 2 - 1;
            }
            
        /* If running, continue with the interactions. */
        if ( diag >= threadID && diag < diag_max ) {
        
            /* Increase the step if necessary. */
            if ( step < cuda_frame )
                step += 1;
    
            /* Get the location of the kth entry on the diagonal. */
            packed = tex1D( tex_diags , diag );
            pid = packed >> 16; pjd = count - (packed & 0xffff);
            
            /* Get a handle on the particles. */
            #ifdef PARTS_TEX
                pi = tex2D( tex_parts , pid , cid );
                pj = tex2D( tex_parts , pjd , cid );
            #else
                pi = parts[ pid ];
                pj = parts[ pjd ];
            #endif

            /* Compute the radius between pi and pj. */
            r2 = 0.0f;
            dx[0] = pi.x - pj.x; r2 += dx[0]*dx[0];
            dx[1] = pi.y - pj.y; r2 += dx[1]*dx[1];
            dx[2] = pi.z - pj.z; r2 += dx[2]*dx[2];

            /* Set the null potential if anything is bad. */
            if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pj.w*cuda_maxtype + pi.w ) ) != 0 ) {

                // atomicAdd( &cuda_rcount , 1 );
                    
                /* Interact particles pi and pj. */
                potential_eval_cuda_tex( pot , r2 , &ee , &eff );

                /* Store the interaction force on pi and energy. */
                for ( k = 0 ; k < 3 ; k++ ) {
                    w[k] = eff * dx[k];
                    forces[ 3*pid + k ] -= w[k];
                    }

                /* Sync the shared memory values. */
                __threadfence_block();

                /* Store the interaction force on pj. */
                epot += ee;
                for ( k = 0 ; k < 3 ; k++ )
                    forces[ 3*pjd + k ] += w[k];

                /* Sync the shared memory values. */
                __threadfence_block();

                } /* range and potential? */

            /* printf( "runner_doself_diag_cuda[%i]: diag=%i, step=%i, i=%i, j=%i.\n" ,
                threadID , diag , step , pid , pjd ); */

            } /* is it this thread's turn? */
    
        } /* Loop over diagonal indices. */
        
    TIMER_TOC(tid_self)
    
    }
    
    
/**
 * @brief Compute the self interactions for the given cell on a CUDA device.
 *
 * @param iparts Array of parts in this cell.
 * @param count Number of parts in the cell.
 * @param parts Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
#ifdef PARTS_TEX
__device__ void runner_doself4_diag_cuda ( int cid , int count , float *forces ) {
#else
__device__ void runner_doself4_diag_cuda ( float4 *parts , int count , float *forces ) {
#endif

    int diag, k, diag_max, step, threadID;
    uint4 packed;
    float4 pi[4], pj[4], r2, ee, eff, w[3];
    int4 pot, ldiag, valid, pid, pjd;
    float epot = 0.0f, dx[12];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Step along the number of diagonal entries. */
    valid = make_int4(0);
    diag_max = count * (count - 1) / 2; step = 1; diag = 0;
    while ( diag < diag_max ) {
    
        /* Get the diagonals and validity for each 4-step. */
        ldiag.x = diag;
        if ( ldiag.x == threadID ) {
            step = ldiag.x;
            ldiag.x = (diag + 2) * (diag + 1) / 2 - 1;
            }
        if ( ( valid.x = ( ldiag.x >= threadID && ldiag.x < diag_max ) ) )
            if ( step < cuda_frame )
                step += 1;
        ldiag.y = ldiag.x + step;
        if ( ldiag.y == threadID ) {
            step = ldiag.y;
            ldiag.y = (ldiag.y + 2) * (ldiag.y + 1) / 2 - 1;
            }
        if ( ( valid.y = ( ldiag.y >= threadID && ldiag.y < diag_max ) ) )
            if ( step < cuda_frame )
                step += 1;               
        ldiag.z = ldiag.y + step;
        if ( ldiag.z == threadID ) {
            step = ldiag.z;
            ldiag.z = (ldiag.z + 2) * (ldiag.z + 1) / 2 - 1;
            }
        if ( ( valid.z = ( ldiag.z >= threadID && ldiag.z < diag_max ) ) )
            if ( step < cuda_frame )
                step += 1;
        ldiag.w = ldiag.z + step;
        if ( ldiag.w == threadID ) {
            step = ldiag.w;
            ldiag.w = (ldiag.w + 2) * (ldiag.w + 1) / 2 - 1;
            }
        if ( ( valid.w = ( ldiag.w >= threadID && ldiag.w < diag_max ) ) )
            if ( step < cuda_frame )
                step += 1;
                
        /* Update diag for next iteration. */
        diag = ldiag.w + step;
    
        /* Get the location of the kth entry on the diagonal. */
        packed.x = ( valid.x ) ? tex1D( tex_diags , ldiag.x ) : 0;
        packed.y = ( valid.y ) ? tex1D( tex_diags , ldiag.y ) : 0;
        packed.z = ( valid.z ) ? tex1D( tex_diags , ldiag.z ) : 0;
        packed.w = ( valid.w ) ? tex1D( tex_diags , ldiag.w ) : 0;
        pid.x = packed.x >> 16; pjd.x = count - (packed.x & 0xffff);
        pid.y = packed.y >> 16; pjd.y = count - (packed.y & 0xffff);
        pid.z = packed.z >> 16; pjd.z = count - (packed.z & 0xffff);
        pid.w = packed.w >> 16; pjd.w = count - (packed.w & 0xffff);
        
        /* Get a handle on the particles. */
        #ifdef PARTS_TEX
            pi[0] = tex2D( tex_parts , pid.x , cid ); pj[0] = tex2D( tex_parts , pjd.x , cid );
            pi[1] = tex2D( tex_parts , pid.y , cid ); pj[1] = tex2D( tex_parts , pjd.y , cid );
            pi[2] = tex2D( tex_parts , pid.z , cid ); pj[2] = tex2D( tex_parts , pjd.z , cid );
            pi[3] = tex2D( tex_parts , pid.w , cid ); pj[3] = tex2D( tex_parts , pjd.w , cid );
        #else
            pi[0] = parts[ pid.x ]; pj[0] = parts[ pjd.x ];
            pi[1] = parts[ pid.y ]; pj[1] = parts[ pjd.y ];
            pi[2] = parts[ pid.z ]; pj[2] = parts[ pjd.z ];
            pi[3] = parts[ pid.w ]; pj[3] = parts[ pjd.w ];
        #endif

        /* Compute the radius between pi and pj. */
        r2 = make_float4( 0.0f );
        dx[0] = pi[0].x - pj[0].x; r2.x += dx[0]*dx[0];
        dx[1] = pi[0].y - pj[0].y; r2.x += dx[1]*dx[1];
        dx[2] = pi[0].z - pj[0].z; r2.x += dx[2]*dx[2];
        dx[3] = pi[1].x - pj[1].x; r2.y += dx[3]*dx[3];
        dx[4] = pi[1].y - pj[1].y; r2.y += dx[4]*dx[4];
        dx[5] = pi[1].z - pj[1].z; r2.y += dx[5]*dx[5];
        dx[6] = pi[2].x - pj[2].x; r2.z += dx[6]*dx[6];
        dx[7] = pi[2].y - pj[2].y; r2.z += dx[7]*dx[7];
        dx[8] = pi[2].z - pj[2].z; r2.z += dx[8]*dx[8];
        dx[9] = pi[3].x - pj[3].x; r2.w += dx[9]*dx[9];
        dx[10] = pi[3].y - pj[3].y; r2.w += dx[10]*dx[10];
        dx[11] = pi[3].z - pj[3].z; r2.w += dx[11]*dx[11];
        
        /* Get the potential for each pair. */
        pot.x = ( valid.x = ( valid.x && r2.x < cuda_cutoff2) ) ? tex1D( tex_pind , pj[0].w*cuda_maxtype + pi[0].w ) : 0;
        pot.y = ( valid.y = ( valid.y && r2.y < cuda_cutoff2) ) ? tex1D( tex_pind , pj[1].w*cuda_maxtype + pi[1].w ) : 0;
        pot.z = ( valid.z = ( valid.z && r2.z < cuda_cutoff2) ) ? tex1D( tex_pind , pj[2].w*cuda_maxtype + pi[2].w ) : 0;
        pot.w = ( valid.w = ( valid.w && r2.w < cuda_cutoff2) ) ? tex1D( tex_pind , pj[3].w*cuda_maxtype + pi[3].w ) : 0;

        /* Interact particles pi and pj. */
        potential_eval4_cuda_tex( pot , r2 , &ee , &eff );
        
        /* Store the interaction energy. */
        epot += ee.x + ee.y + ee.z + ee.w;

        /* Store the interaction force on pi and energy. */
        if ( valid.x )
            for ( k = 0 ; k < 3 ; k++ )
                forces[ 3*pid.x + k ] -= ( w[k].x = eff.x * dx[k] );
        __threadfence_block();
        if ( valid.y )
            for ( k = 0 ; k < 3 ; k++ )
                forces[ 3*pid.y + k ] -= ( w[k].y = eff.y * dx[3+k] );
        __threadfence_block();
        if ( valid.z )
            for ( k = 0 ; k < 3 ; k++ )
                forces[ 3*pid.z + k ] -= ( w[k].z = eff.z * dx[6+k] );
        __threadfence_block();
        if ( valid.w )
            for ( k = 0 ; k < 3 ; k++ )
                forces[ 3*pid.w + k ] -= ( w[k].w = eff.w * dx[9+k] );
        __threadfence_block();

        /* Store the interaction force on pj. */
        if ( valid.x )
            for ( k = 0 ; k < 3 ; k++ )
                forces[ 3*pjd.x + k ] += w[k].x;
        __threadfence_block();
        if ( valid.y )
            for ( k = 0 ; k < 3 ; k++ )
                forces[ 3*pjd.y + k ] += w[k].y;
        __threadfence_block();
        if ( valid.z )
            for ( k = 0 ; k < 3 ; k++ )
                forces[ 3*pjd.z + k ] += w[k].z;
        __threadfence_block();
        if ( valid.w )
            for ( k = 0 ; k < 3 ; k++ )
                forces[ 3*pjd.w + k ] += w[k].w;
        __threadfence_block();

        } /* Loop over diagonal indices. */
        
    TIMER_TOC(tid_self)
    
    }
    
    
/**
 * @brief Bind textures to the given cuda Arrays.
 *
 *
 * Hack to get around the fact that textures are static and can thus not
 * be externalized.
 */
 
int runner_bind ( cudaArray *cuArray_coeffs , cudaArray *cuArray_pind , cudaArray *cuArray_diags ) {

    /* Set the coeff properties. */
    tex_coeffs.addressMode[0] = cudaAddressModeClamp;
    tex_coeffs.addressMode[1] = cudaAddressModeClamp;
    tex_coeffs.filterMode = cudaFilterModePoint;
    tex_coeffs.normalized = false;

    /* Bind the coeffs. */
    cuda_coeffs = cuArray_coeffs;
    if ( cudaBindTextureToArray( tex_coeffs , cuArray_coeffs ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Set the coeff properties. */
    tex_pind.addressMode[0] = cudaAddressModeClamp;
    tex_pind.filterMode = cudaFilterModePoint;
    tex_pind.normalized = false;

    /* Bind the pinds. */
    cuda_pind = cuArray_pind;
    if ( cudaBindTextureToArray( tex_pind , cuArray_pind ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Set the coeff properties. */
    tex_diags.addressMode[0] = cudaAddressModeClamp;
    tex_diags.filterMode = cudaFilterModePoint;
    tex_diags.normalized = false;

    /* Bind the diags. */
    cuda_diags = cuArray_diags;
    if ( cudaBindTextureToArray( tex_diags , cuArray_diags ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Rock and roll. */
    return runner_err_ok;

    }


/**
 * @brief Bind textures to the given cuda Arrays.
 *
 *
 * Hack to get around the fact that textures are static and can thus not
 * be externalized.
 */
 
int runner_parts_bind ( cudaArray *cuArray_parts ) {

    /* Set the texture properties. */
    tex_parts.addressMode[0] = cudaAddressModeClamp;
    tex_parts.addressMode[1] = cudaAddressModeClamp;
    tex_parts.filterMode = cudaFilterModePoint;
    tex_parts.normalized = false;

    /* Bind the parts. */
    if ( cudaBindTextureToArray( tex_parts , cuArray_parts ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Rock and roll. */
    return runner_err_ok;

    }


/**
 * @brief Bind textures to the given cuda Arrays.
 *
 *
 * Hack to get around the fact that textures are static and can thus not
 * be externalized.
 */
 
int runner_parts_unbind ( ) {

    /* Bind the coeffs. */
    if ( cudaUnbindTexture( tex_parts ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Rock and roll. */
    return runner_err_ok;

    }


/** This set of defines and includes produces kernels with buffers for multiples
 *  of 32 particles up to 512 cuda_maxparts.
 */
 
#define cuda_nrparts 32
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 64
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 96
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 128
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 160
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 192
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 224
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 256
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 288
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 320
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 352
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 384
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 416
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 448
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 480
    #include "runner_cuda_main.h"
#undef cuda_nrparts

#define cuda_nrparts 512
    #include "runner_cuda_main.h"





