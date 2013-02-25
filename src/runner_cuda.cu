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
#include "space.h"
#include "task.h"
#include "potential.h"
#include "engine.h"
#include "runner.h"
#include "runner_cuda.h"


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )


/* The constant null potential. */
__constant__ struct potential *potential_null_cuda = NULL;

/* The number of cells and pairs. */
__constant__ int cuda_nr_cells = 0;

/* The parts (non-texture access). */
__constant__ float4 *cuda_parts;

/* Diagonal entries and potential index lookup table. */
__constant__ unsigned int *cuda_pind;

/* The mutex for accessing the cell pair list. */
__device__ int cuda_cell_mutex = 0;
__device__ int cuda_barrier = 0;

/* The index of the next free cell pair. */
__device__ int cuda_pair_next = 0;

/* The list of cell pairs. */
__constant__ struct cellpair_cuda *cuda_pairs;
__device__ int *cuda_taboo;

/* The list of tasks. */
__constant__ struct task_cuda *cuda_tasks;
__constant__ int cuda_nr_tasks = 0;

/* The per-SM task queues. */
__device__ struct queue_cuda cuda_queues[ cuda_maxqueues ];
__constant__ int cuda_nrqueues;
__constant__ int cuda_queue_size;

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

/* The potential coefficients, as a texture. */
texture< float4 , cudaTextureType2D > tex_coeffs;
texture< float4 , cudaTextureType2D > tex_parts;

/* Other textures. */
texture< int , cudaTextureType1D > tex_pind;

/* Arrays to hold the textures. */
cudaArray *cuda_coeffs;

/* The potential parameters (hard-wired size for now). */
__constant__ float cuda_eps[ 100 ];
__constant__ float cuda_rmin[ 100 ];

/* Use a set of variables to communicate with the outside world. */
__device__ float cuda_fio[32];
__device__ int cuda_io[32];
__device__ int cuda_rcount = 0;

/* Potential energy. */
__device__ float cuda_epot = 0.0f, cuda_epot_out;

/* Timers. */
__device__ float cuda_timers[ tid_count ];

/* Map sid to shift vectors. */
__constant__ float cuda_shift[13*3] = {
     1.0 ,  1.0 ,  1.0 ,
     1.0 ,  1.0 ,  0.0 ,
     1.0 ,  1.0 , -1.0 ,
     1.0 ,  0.0 ,  1.0 ,
     1.0 ,  0.0 ,  0.0 ,
     1.0 ,  0.0 , -1.0 ,
     1.0 , -1.0 ,  1.0 ,
     1.0 , -1.0 ,  0.0 ,
     1.0 , -1.0 , -1.0 ,
     0.0 ,  1.0 ,  1.0 ,
     0.0 ,  1.0 ,  0.0 ,
     0.0 ,  1.0 , -1.0 ,
     0.0 ,  0.0 ,  1.0 ,
    };
    
/* The cell edge lengths. */
__constant__ float cuda_h[3];
    
    
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
 * @brief Attempt to lock a device mutex.
 *
 * @param m The mutex.
 *
 * Try to grab the mutex. Note that only one thread
 * can do this at a time, so to synchronize blocks, only a single thread of
 * each block should call it.
 */

__device__ int cuda_mutex_trylock ( int *m ) {
    TIMER_TIC
    int res = atomicCAS( m , 0 , 1 ) == 0;
    TIMER_TOC( tid_mutex )
    return res;
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
    TIMER_TIC
    atomicExch( m , 0 );
    TIMER_TOC( tid_mutex )
    }
    
    
/**
 * @brief Get a task ID from the given queue.
 *
 */
 
__device__ int cuda_queue_gettask ( struct queue_cuda *q ) {

    int ind, tid = -1;
    
    /* Don't even try... */
    if ( q->rec_count == q->count )
        return -1;

    /* Get the index of the next task. */
    ind = atomicAdd( &q->first , 1 );
        
    /* Wrap the index. */
    ind %= cuda_queue_size; 

    /* Loop until there is a valid task at that index. */
    while ( q->rec_count < q->count && ( tid = q->data[ind] ) < 0 );
    
    /* Scratch the task from the queue */
    if ( tid >= 0 )
        q->data[ind] = -1;

    /* Return the acquired task ID. */
    return tid;
    
    }


/**
 * @brief Put a task onto the given queue.
 *
 * @param tid The task ID to add to the end of the queue.
 */
 
__device__ void cuda_queue_puttask ( struct queue_cuda *q , int tid ) {

    int ind;

    /* Get the index of the next task. */
    ind = atomicAdd( &q->last , 1 ) % cuda_queue_size;
    
    /* Wait for the slot in the queue to be empty. */
    while ( q->data[ind] != -1 );

    /* Write the task back to the queue. */
    q->data[ind] = tid;
    
    }
    
    
/**
 * @brief Get the ID of the block's SM.
 */
 
__noinline__ __device__ uint get_smid ( void ) {
    uint ret;
    asm("mov.u32 %0, %smid;" : "=r"(ret) );
    return ret;
    }


/**
 * @brief Get a task from the given task queue.
 *
 * Picks tasks from the queue sequentially and checks if they
 * can be computed. If not, they are returned to the queue.
 *
 * This routine blocks until a valid task is picked up, or the
 * specified queue is empty.
 */
 
__device__ int runner_cuda_gettask ( struct queue_cuda *q , int steal ) {

    int tid = -1;
    #ifndef FORCES_LOCAL
        int cid, cjd;
    #endif
    
    TIMER_TIC
    
    /* Main loop. */
    while ( ( tid = cuda_queue_gettask( q ) ) >= 0 ) {
    
        /* If this task is not even free, don't even bother. */
        if ( !cuda_tasks[tid].wait ) {
    
            #ifdef FORCES_LOCAL
                break;
            #else
                /* Dfferent options for different tasks. */
                if ( cuda_tasks[tid].type == task_type_sort ) {
                
                    /* No locking needed. */
                    break;
                
                    }
                else if ( cuda_tasks[tid].type == task_type_self ) {
                
                    /* Decode this task. */
                    cid = cuda_tasks[tid].i;

                    /* Lock down this task? */
                    if ( cuda_mutex_trylock( &cuda_taboo[ cid ] ) )
                        break;
                            
                    }
                else if ( cuda_tasks[tid].type == task_type_pair ) {
                
                    /* Decode this task. */
                    cid = cuda_tasks[tid].i;
                    cjd = cuda_tasks[tid].j;

                    /* Lock down this task? */
                    if ( cuda_mutex_trylock( &cuda_taboo[ cid ] ) )
                        if ( cuda_mutex_trylock( &cuda_taboo[ cjd ] ) ) 
                            break;
                        else
                            cuda_mutex_unlock( &cuda_taboo[ cid ] );
                            
                    }
            #endif

            }
                
        /* Put this task back into the queue. */
        cuda_queue_puttask( q , tid );
    
        }
        
    /* Put this task into the recycling queue, if needed. */
    if ( tid >= 0 ) {
        if ( steal )
            atomicSub( (int *)&q->count , 1 );
        else
            q->rec_data[ atomicAdd( (int *)&q->rec_count , 1 ) ] = tid;
        }
        
    TIMER_TOC(tid_queue);
        
    /* Return whatever we got. */
    return tid;

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
        // __threadfence_block();
    
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
            // __threadfence_block();
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
        // __threadfence_block();
    
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
            // __threadfence_block();
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
__device__ void runner_dopair_unsorted_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , int sid , float *epot_global ) {
#else
__device__ void runner_dopair_unsorted_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , int sid , float *epot_global ) {
#endif

    int k, pid, pjd, ind, wrap_i, threadID;
    int pjoff;
    int pot;
    float epot = 0.0f, dx[3], pjf[3], shift[3], r2, w;
    float ee = 0.0f, eff = 0.0f;
    float4 pi, pj;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Get the shift vector from the sid. */
    shift[0] = cuda_shift[ 3*sid + 0 ] * cuda_h[0];
    shift[1] = cuda_shift[ 3*sid + 1 ] * cuda_h[1];
    shift[2] = cuda_shift[ 3*sid + 2 ] * cuda_h[2];

    /* Get the wraps. */
    wrap_i = (count_i < cuda_frame) ? cuda_frame : count_i;
    
    /* Make sure everybody is in the same place. */
    // __threadfence_block();

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
                if ( r2 < cuda_cutoff2 && ( pot = cuda_pind[ pjoff + (int)pi.w ] ) != 0 ) {

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
                    // __threadfence_block();
                
                    } /* in range and potential. */

                } /* valid pid? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_j[ 3*pjd + k ] += pjf[k];

        /* Sync the shared memory values. */
        // __threadfence_block();
            
        } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
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
__device__ void runner_dopair4_unsorted_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , int sid , float *epot_global ) {
#else
__device__ void runner_dopair4_unsorted_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , int sid , float *epot_global ) {
#endif

    int k, pjd, ind, wrap_i, threadID;
    int pjoff;
    float4 pi[4], pj;
    int4 pot, pid;
    char4 valid;
    float4 r2, ee, eff;
    float epot = 0.0f, dx[12], pjf[3], shift[3], w;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Get the shift vector from the sid. */
    shift[0] = cuda_shift[ 3*sid + 0 ] * cuda_h[0];
    shift[1] = cuda_shift[ 3*sid + 1 ] * cuda_h[1];
    shift[2] = cuda_shift[ 3*sid + 2 ] * cuda_h[2];

    /* Get the wraps. */
    wrap_i = (count_i < cuda_frame) ? cuda_frame : count_i;
    
    /* Make sure everybody is in the same place. */
    // __threadfence_block();

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
            pot.x = valid.x ? cuda_pind[ pjoff + (int)pi[0].w ] : 0;
            pot.y = valid.y ? cuda_pind[ pjoff + (int)pi[1].w ] : 0;
            pot.z = valid.z ? cuda_pind[ pjoff + (int)pi[2].w ] : 0;
            pot.w = valid.w ? cuda_pind[ pjoff + (int)pi[3].w ] : 0;
            
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
            
            /* Update the forces. */
            if ( valid.x ) {
                pjf[0] -= ( w = eff.x * dx[0] ); forces_i[ 3*pid.x + 0 ] += w;
                pjf[1] -= ( w = eff.x * dx[1] ); forces_i[ 3*pid.x + 1 ] += w;
                pjf[2] -= ( w = eff.x * dx[2] ); forces_i[ 3*pid.x + 2 ] += w;
                epot += ee.x;
                }
            // __threadfence_block();
            if ( valid.y ) {
                pjf[0] -= ( w = eff.y * dx[3] ); forces_i[ 3*pid.y + 0 ] += w;
                pjf[1] -= ( w = eff.y * dx[4] ); forces_i[ 3*pid.y + 1 ] += w;
                pjf[2] -= ( w = eff.y * dx[5] ); forces_i[ 3*pid.y + 2 ] += w;
                epot += ee.y;
                }
            // __threadfence_block();
            if ( valid.z ) {
                pjf[0] -= ( w = eff.z * dx[6] ); forces_i[ 3*pid.z + 0 ] += w;
                pjf[1] -= ( w = eff.z * dx[7] ); forces_i[ 3*pid.z + 1 ] += w;
                pjf[2] -= ( w = eff.z * dx[8] ); forces_i[ 3*pid.z + 2 ] += w;
                epot += ee.z;
                }
            // __threadfence_block();
            if ( valid.w ) {
                pjf[0] -= ( w = eff.w * dx[9] ); forces_i[ 3*pid.w + 0 ] += w;
                pjf[1] -= ( w = eff.w * dx[10] ); forces_i[ 3*pid.w + 1 ] += w;
                pjf[2] -= ( w = eff.w * dx[11] ); forces_i[ 3*pid.w + 2 ] += w;
                epot += ee.w;
                }
            // __threadfence_block();
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_j[ 3*pjd + k ] += pjf[k];

        /* Sync the shared memory values. */
        // __threadfence_block();
            
        } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
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
__device__ void runner_dosort_cuda ( int cid , int count_i , unsigned int *sort_i , int sid ) {
#else
__device__ void runner_dosort_cuda ( float4 *parts_i , int count_i , unsigned int *sort_i , int sid ) {
#endif

    int k, threadID = threadIdx.x;
    float4 pi;
    // int4 spid;
    float nshift, shift[3], shiftn[3];
    
    TIMER_TIC
    
    /* Get the shift vector from the sid. */
    shift[0] = cuda_shift[ 3*sid + 0 ];
    shift[1] = cuda_shift[ 3*sid + 1 ];
    shift[2] = cuda_shift[ 3*sid + 2 ];

    /* Pre-compute the inverse norm of the shift. */
    nshift = sqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    shiftn[0] = shift[0] / nshift;
    shiftn[1] = shift[1] / nshift;
    shiftn[2] = shift[2] / nshift;

    /* Pack the parts of i into the sort arrays. */
    /* for ( k = threadID ; k < count_i ; k += 4*cuda_frame ) {
        #ifdef PARTS_TEX
            pi[0] = tex2D( tex_parts , k + 0*cuda_frame , cid );
            pi[1] = tex2D( tex_parts , k + 1*cuda_frame , cid );
            pi[2] = tex2D( tex_parts , k + 2*cuda_frame , cid );
            pi[3] = tex2D( tex_parts , k + 3*cuda_frame , cid );
        #else
            pi[0] = parts_i[ k + 0*cuda_frame ];
            if ( k + 1*cuda_frame < count_i ) pi[1] = parts_i[ k + 1*cuda_frame ];
            if ( k + 2*cuda_frame < count_i ) pi[2] = parts_i[ k + 2*cuda_frame ];
            if ( k + 3*cuda_frame < count_i ) pi[3] = parts_i[ k + 3*cuda_frame ];
        #endif
        spid.x = ( k << 16 ) | (unsigned int)( cuda_dscale * (nshift + pi[0].x*shiftn[0] + pi[0].y*shiftn[1] + pi[0].z*shiftn[2]) );
        spid.y = ( (k + 1*cuda_frame) << 16 ) | (unsigned int)( cuda_dscale * (nshift + pi[1].x*shiftn[0] + pi[1].y*shiftn[1] + pi[1].z*shiftn[2]) );
        spid.z = ( (k + 2*cuda_frame) << 16 ) | (unsigned int)( cuda_dscale * (nshift + pi[2].x*shiftn[0] + pi[2].y*shiftn[1] + pi[2].z*shiftn[2]) );
        spid.w = ( (k + 3*cuda_frame) << 16 ) | (unsigned int)( cuda_dscale * (nshift + pi[3].x*shiftn[0] + pi[3].y*shiftn[1] + pi[3].z*shiftn[2]) );
        sort_i[k] = spid.x;
        if ( k + 1*cuda_frame < count_i ) sort_i[ k + 1*cuda_frame ] = spid.y;
        if ( k + 2*cuda_frame < count_i ) sort_i[ k + 2*cuda_frame ] = spid.z;
        if ( k + 3*cuda_frame < count_i ) sort_i[ k + 3*cuda_frame ] = spid.w;
        } */

    /* Pack the parts into the sort arrays. */
    for ( k = threadID ; k < count_i ; k += cuda_frame ) {
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , k , cid );
        #else
            pi = parts_i[ k ];
        #endif
        sort_i[k] = ( k << 16 ) |
            (unsigned int)( cuda_dscale * (nshift + pi.x*shiftn[0] + pi.y*shiftn[1] + pi.z*shiftn[2]) );
        }

    TIMER_TOC(tid_pack)
        
    /* Sort using normalized bitonic sort. */
    cuda_sort_descending( sort_i , count_i );

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
__device__ void runner_dopair_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , int sid , float *epot_global ) {
#else
__device__ void runner_dopair_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , int sid , float *epot_global ) {
#endif

    int k, pid, pjd, spid, spjd, pjdid, threadID, wrap, cj;
    int pioff;
    unsigned int dmaxdist, dshift;
    float4 pi, pj;
    int pot;
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f;
    float dx[3], pif[3], shift[3];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Get the shift vector from the sid. */
    shift[0] = cuda_shift[ 3*sid + 0 ] * cuda_h[0];
    shift[1] = cuda_shift[ 3*sid + 1 ] * cuda_h[1];
    shift[2] = cuda_shift[ 3*sid + 2 ] * cuda_h[2];
    dshift = cuda_dscale * sqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
        
    /* Pre-compute the inverse norm of the shift. */
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;
       

    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && ( sort_j[count_j-cj] & 0xffff ) - ( sort_i[pid & ~(cuda_frame - 1)] & 0xffff ) > dmaxdist - dshift )
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
                spjd = sort_j[count_j-1-pjd] >> 16;
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
                if ( r2 < cuda_cutoff2 && ( pot = cuda_pind[ pioff + (int)pj.w ] ) != 0 ) {

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
                    // __threadfence_block();
                
                    } /* in range and potential. */

                } /* do we have a pair? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_i[ 3*spid + k ] += pif[k];
    
        /* Sync the shared memory values. */
        // __threadfence_block();
        
        } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
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
__device__ void runner_dopair4_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , int sid , float *epot_global ) {
#else
__device__ void runner_dopair4_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , int sid , float *epot_global ) {
#endif

    int k, pid, spid, pjdid, threadID, wrap, cj;
    int pioff;
    unsigned int dmaxdist, dshift;
    float4 pi, pj[4];
    int4 pot, pjd, spjd;
    char4 valid;
    float4 ee, eff, r2;
    float epot = 0.0f, w;
    float dx[12], pif[3], shift[3];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Get the shift vector from the sid. */
    shift[0] = cuda_shift[ 3*sid + 0 ] * cuda_h[0];
    shift[1] = cuda_shift[ 3*sid + 1 ] * cuda_h[1];
    shift[2] = cuda_shift[ 3*sid + 2 ] * cuda_h[2];
    dshift = cuda_dscale * sqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
        
    /* Pre-compute the inverse norm of the shift. */
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;
               
        
    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && ( sort_j[count_j-cj] & 0xffff ) - ( sort_i[pid & ~(cuda_frame - 1)] & 0xffff ) > dmaxdist - dshift )
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
            spjd.x = sort_j[count_j-1-pjd.x] >> 16; spjd.y = sort_j[count_j-1-pjd.y] >> 16; spjd.z = sort_j[count_j-1-pjd.z] >> 16; spjd.w = sort_j[count_j-1-pjd.w] >> 16; 
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
            pot.x = valid.x ? cuda_pind[ pioff + (int)pj[0].w ] : 0;
            pot.y = valid.y ? cuda_pind[ pioff + (int)pj[1].w ] : 0;
            pot.z = valid.z ? cuda_pind[ pioff + (int)pj[2].w ] : 0;
            pot.w = valid.w ? cuda_pind[ pioff + (int)pj[3].w ] : 0;
            
            /* Compute the interaction. */
            potential_eval4_cuda_tex( pot , r2 , &ee , &eff );
            
            /* Update the particle forces. */
            if ( valid.x ) {
                pif[0] -= ( w = eff.x * dx[0] ); forces_j[ 3*spjd.x + 0 ] += w;
                pif[1] -= ( w = eff.x * dx[1] ); forces_j[ 3*spjd.x + 1 ] += w;
                pif[2] -= ( w = eff.x * dx[2] ); forces_j[ 3*spjd.x + 2 ] += w;
                epot += ee.x;
                }
            // __threadfence_block();
            if ( valid.y ) {
                pif[0] -= ( w = eff.y * dx[3] ); forces_j[ 3*spjd.y + 0 ] += w;
                pif[1] -= ( w = eff.y * dx[4] ); forces_j[ 3*spjd.y + 1 ] += w;
                pif[2] -= ( w = eff.y * dx[5] ); forces_j[ 3*spjd.y + 2 ] += w;
                epot += ee.y;
                }
            // __threadfence_block();
            if ( valid.z ) {
                pif[0] -= ( w = eff.z * dx[6] ); forces_j[ 3*spjd.z + 0 ] += w;
                pif[1] -= ( w = eff.z * dx[7] ); forces_j[ 3*spjd.z + 1 ] += w;
                pif[2] -= ( w = eff.z * dx[8] ); forces_j[ 3*spjd.z + 2 ] += w;
                epot += ee.z;
                }
            // __threadfence_block();
            if ( valid.w ) {
                pif[0] -= ( w = eff.w * dx[9] ); forces_j[ 3*spjd.w + 0 ] += w;
                pif[1] -= ( w = eff.w * dx[10] ); forces_j[ 3*spjd.w + 1 ] += w;
                pif[2] -= ( w = eff.w * dx[11] ); forces_j[ 3*spjd.w + 2 ] += w;
                epot += ee.w;
                }
            // __threadfence_block();
            
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces_i[ 3*spid + k ] += pif[k];
    
        /* Sync the shared memory values. */
        // __threadfence_block();
        
        } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
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
__device__ void runner_doself_cuda ( int cid , int count , float *forces , float *epot_global ) {
#else
__device__ void runner_doself_cuda ( float4 *parts , int count , float *forces , float *epot_global ) {
#endif

    int k, pid, pjd, threadID;
    int pjoff;
    float4 pi, pj;
    int pot;
    float epot = 0.0f, dx[3], pjf[3], r2, w, ee, eff;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Make sure everybody is in the same place. */
    // __threadfence_block();

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
            if ( r2 < cuda_cutoff2 && ( pot = cuda_pind[ pjoff + (int)pi.w ] ) != 0 ) {

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
                // __threadfence_block();
            
                } /* in range and potential. */

            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces[ 3*pjd + k ] += pjf[k];

        /* Sync the shared memory values. */
        // __threadfence_block();

        } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
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
__device__ void runner_doself4_cuda ( int cid , int count , float *forces , float *epot_global ) {
#else
__device__ void runner_doself4_cuda ( float4 *parts , int count , float *forces , float *epot_global ) {
#endif

    int k, pjd, ind, threadID;
    int pjoff;
    float4 pi[4], pj;
    int4 pot, pid;
    char4 valid;
    float4 r2, ee, eff;
    float epot = 0.0f, dx[12], pjf[3], w;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Make sure everybody is in the same place. */
    // __threadfence_block();

    /* Loop over the particles in cell_j, frame-wise. */
    for ( pjd = threadID ; pjd < count-1 ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        #ifdef PARTS_TEX
            pj = tex2D( tex_parts , pjd , cid );
        #else
            pj = parts[ pjd ];
        #endif
        pjoff = pj.w * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ )
            pjf[k] = 0.0f;
        
        /* Loop over the particles in cell_i. */
        for ( ind = pjd+1 ; ind < count ; ind += 4 ) {
        
            /* Get the particle indices. */
            pid.x = ind + 0;
            pid.y = ind + 1;
            pid.z = ind + 2;
            pid.w = ind + 3;
                
            /* Get the particle pointers. */
            #ifdef PARTS_TEX
                pi[0] = ( valid.x = ( pid.x < count ) ) ? tex2D( tex_parts , pid.x , cid ) : pj;
                pi[1] = ( valid.y = ( pid.y < count ) ) ? tex2D( tex_parts , pid.y , cid ) : pj;
                pi[2] = ( valid.z = ( pid.z < count ) ) ? tex2D( tex_parts , pid.z , cid ) : pj;
                pi[3] = ( valid.w = ( pid.w < count ) ) ? tex2D( tex_parts , pid.w , cid ) : pj;
            #else
                pi[0] = ( valid.x = ( pid.x < count ) ) ? parts[ pid.x ] : pj;
                pi[1] = ( valid.y = ( pid.y < count ) ) ? parts[ pid.y ] : pj;
                pi[2] = ( valid.z = ( pid.z < count ) ) ? parts[ pid.z ] : pj;
                pi[3] = ( valid.w = ( pid.w < count ) ) ? parts[ pid.w ] : pj;
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
            pot.x = valid.x ? cuda_pind[ pjoff + (int)pi[0].w ] : 0;
            pot.y = valid.y ? cuda_pind[ pjoff + (int)pi[1].w ] : 0;
            pot.z = valid.z ? cuda_pind[ pjoff + (int)pi[2].w ] : 0;
            pot.w = valid.w ? cuda_pind[ pjoff + (int)pi[3].w ] : 0;
            
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
            
            /* Update the forces and energy. */
            if ( valid.x ) {
                pjf[0] -= ( w = eff.x * dx[0] ); forces[ 3*pid.x + 0 ] += w;
                pjf[1] -= ( w = eff.x * dx[1] ); forces[ 3*pid.x + 1 ] += w;
                pjf[2] -= ( w = eff.x * dx[2] ); forces[ 3*pid.x + 2 ] += w;
                epot += ee.x;
                }
            // __threadfence_block();
            if ( valid.y ) {
                pjf[0] -= ( w = eff.y * dx[3] ); forces[ 3*pid.y + 0 ] += w;
                pjf[1] -= ( w = eff.y * dx[4] ); forces[ 3*pid.y + 1 ] += w;
                pjf[2] -= ( w = eff.y * dx[5] ); forces[ 3*pid.y + 2 ] += w;
                epot += ee.y;
                }
            // __threadfence_block();
            if ( valid.z ) {
                pjf[0] -= ( w = eff.z * dx[6] ); forces[ 3*pid.z + 0 ] += w;
                pjf[1] -= ( w = eff.z * dx[7] ); forces[ 3*pid.z + 1 ] += w;
                pjf[2] -= ( w = eff.z * dx[8] ); forces[ 3*pid.z + 2 ] += w;
                epot += ee.z;
                }
            // __threadfence_block();
            if ( valid.w ) {
                pjf[0] -= ( w = eff.w * dx[9] ); forces[ 3*pid.w + 0 ] += w;
                pjf[1] -= ( w = eff.w * dx[10] ); forces[ 3*pid.w + 1 ] += w;
                pjf[2] -= ( w = eff.w * dx[11] ); forces[ 3*pid.w + 2 ] += w;
                epot += ee.w;
                }
            // __threadfence_block();
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            forces[ 3*pjd + k ] += pjf[k];

        /* Sync the shared memory values. */
        // __threadfence_block();
            
        } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
    TIMER_TOC(tid_pair)
        
    }


/**
 * @brief Bind textures to the given cuda Arrays.
 *
 *
 * Hack to get around the fact that textures are static and can thus not
 * be externalized.
 */
 
int runner_bind ( cudaArray *cuArray_coeffs , cudaArray *cuArray_pind ) {

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
    if ( cudaBindTextureToArray( tex_pind , cuArray_pind ) != cudaSuccess )
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

// #define cuda_nrparts 512
//     #include "runner_cuda_main.h"



/**
 * @brief Offload and compute the nonbonded interactions on a CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_nonbond_cuda ( struct engine *e ) {

    dim3 nr_threads( cuda_frame , 1 );
    dim3 nr_blocks( e->nr_runners , 1 );
    int maxcount;
    ticks tic;
    // int zero = 0;
    // int cuda_io[32];
    // float cuda_fio[32];
    #ifdef TIMERS
        float timers[ tid_count ];
        double icpms = 1000.0 / 1.4e9; 
    #endif
    
    /* Load the particle data onto the device. */
    tic = getticks();
    if ( ( maxcount = engine_cuda_load_parts( e ) ) < 0 )
        return error(engine_err);
    e->timers[ engine_timer_cuda_load ] += getticks() - tic;

    /* Init the pointer to the next entry. */    
    /* if ( cudaMemcpyToSymbol( "cuda_pair_next" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_tuple_next" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */
    /* if ( cudaMemcpyToSymbol( cuda_rcount , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */
    /* if ( cudaMemcpyToSymbol( "cuda_pair_curr" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */
    /* if ( cudaMemcpyToSymbol( "cuda_pairs_done" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */
    /* if ( cudaMemcpyToSymbol( "cuda_cell_mutex" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */
        
    /* Re-set timers */
    #ifdef TIMERS
        for ( int k = 0 ; k < tid_count ; k++ )
            timers[k] = 0.0f;
        if ( cudaMemcpyToSymbol( cuda_timers , timers , sizeof(float) * tid_count , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    #endif
    
    /* Start the appropriate kernel. */
    tic = getticks();
    switch ( (maxcount + 31) / 32 ) {
        case 1:
            runner_run_cuda_32 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 2:
            runner_run_cuda_64 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 3:
            runner_run_cuda_96 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 4:
            runner_run_cuda_128 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 5:
            runner_run_cuda_160 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 6:
            runner_run_cuda_192 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 7:
            runner_run_cuda_224 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 8:
            runner_run_cuda_256 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 9:
            runner_run_cuda_288 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 10:
            runner_run_cuda_320 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 11:
            runner_run_cuda_352 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 12:
            runner_run_cuda_384 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 13:
            runner_run_cuda_416 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 14:
            runner_run_cuda_448 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        case 15:
            runner_run_cuda_480 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
            break;
        // case 16:
        //     runner_run_verlet_cuda_512 <<<nr_blocks,nr_threads>>> ( e->s.forces_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
        //     break;
        default:
            return error(engine_err_maxparts);
        }
    if ( cudaDeviceSynchronize() != cudaSuccess )
        return cuda_error(engine_err_cuda);
    e->timers[ engine_timer_cuda_dopairs ] += getticks() - tic;
    
    /* Check for CUDA errors. */
    if ( cudaPeekAtLastError() != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Get and dump timers. */
    #ifdef TIMERS
        if ( cudaMemcpyFromSymbol( timers , cuda_timers , sizeof(float) * tid_count , 0 , cudaMemcpyDeviceToHost ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        printf( "engine_nonbond_cuda: timers = [ %.2f " , icpms * timers[0] );
        for ( int k = 1 ; k < tid_count ; k++ )
            printf( "%.2f " , icpms * timers[k] );
        printf( "] ms\n" );
    #endif
    
    /* int queue[ e->s.nr_pairs ], queue2[ e->s.nr_pairs ];
    void *dptr;
    if ( cudaMemcpyFromSymbol( &dptr , cuda_queue_data , sizeof(int *) , 0 , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);   
    if ( cudaMemcpy( queue , dptr , sizeof(int) * e->s.nr_pairs , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);   
    if ( cudaMemcpyFromSymbol( &dptr , cuda_queue2_data , sizeof(int *) , 0 , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);   
    if ( cudaMemcpy( queue2 , dptr , sizeof(int) * e->s.nr_pairs , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    for ( int k = 0 ; k < e->s.nr_pairs ; k++ )
        printf( "\t%5i\t%5i\t%5i\n" , k , queue[k] , queue2[k] ); */
        
    /* Get the IO data. */
    /*if ( cudaMemcpyFromSymbol( cuda_io , "cuda_io" , sizeof(int) * 32 , 0 , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyFromSymbol( cuda_fio , "cuda_fio" , sizeof(float) * 32 , 0 , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    printf( "engine_nonbond_cuda: cuda_io is [ %i , %i , %i , %i , %i , %i , %i , %i , %i , %i ].\n" , 
        cuda_io[0] , cuda_io[1] , cuda_io[2] , cuda_io[3] , cuda_io[4] , cuda_io[5] , cuda_io[6] , cuda_io[7] , cuda_io[8] , cuda_io[9] );
    printf( "engine_nonbond_cuda: cuda_fio is [ %f " , cuda_fio[0] );
    for ( int k = 1 ; k < 32 ; k++ )
        printf( ", %f " , cuda_fio[k] );
    printf( "]\n" ); */
    
    /* if ( cudaMemcpyFromSymbol( &zero , cuda_rcount , sizeof(int) , 0 , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    printf( "engine_nonbond_cuda: computed %i pairs.\n" , zero ); */

    /* Unload the particle data from the device. */
    tic = getticks();
    if ( engine_cuda_unload_parts( e ) < 0 )
        return error(engine_err);
    e->timers[ engine_timer_cuda_unload ] += getticks() - tic;

    /* Go away. */
    return engine_err_ok;
    
    }



/**
 * @brief Load the cell data onto the CUDA device.
 *
 * @param e The #engine.
 *
 * @return The maximum number of parts per cell or < 0
 *      on error (see #engine_err).
 */
 
extern "C" int engine_cuda_load_parts ( struct engine *e ) {
    
    int k, cid, pid, maxcount = 0;
    struct part *p;
    float4 *parts_cuda, *buff;
    struct space *s = &e->s;
    FPTYPE maxdist = s->cutoff + 2*s->maxdx;
    cudaChannelFormatDesc channelDesc_float4 = cudaCreateChannelDesc<float4>();
    
    /* Clear the counts array. */
    bzero( s->counts_cuda_local , sizeof(int) * s->nr_cells );
    
    /* Load the counts. */
    for ( k = 0 ; k < s->nr_marked ; k++ )
        if ( ( s->counts_cuda_local[ s->cid_marked[k] ] = s->cells[ s->cid_marked[k] ].count ) > maxcount )
            maxcount = s->counts_cuda_local[ s->cid_marked[k] ];
            
    /* Raise maxcount to the next multiple of 32. */
    maxcount = ( maxcount + (cuda_frame - 1) ) & ~(cuda_frame - 1);
    // printf( "engine_cuda_load_parts: maxcount=%i.\n" , maxcount );

    /* Compute the indices. */
    s->ind_cuda_local[0] = 0;
    for ( k = 1 ; k < s->nr_cells ; k++ )
        s->ind_cuda_local[k] = s->ind_cuda_local[k-1] + s->counts_cuda_local[k-1];
        
    /* Start by setting the maxdist on the device. */
    if ( cudaMemcpyToSymbol( cuda_maxdist , &maxdist , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Allocate the particle buffer. */
    #ifdef PARTS_TEX
        if ( ( parts_cuda = (float4 *)malloc( sizeof( float4 ) * s->nr_cells * maxcount ) ) == NULL )
            return error(engine_err_malloc);
    #else
        if ( ( parts_cuda = (float4 *)malloc( sizeof( float4 ) * s->nr_parts ) ) == NULL )
            return error(engine_err_malloc);
    #endif
    
    /* Loop over the marked cells. */
    for ( k = 0 ; k < s->nr_marked ; k++ ) {
    
        /* Get the cell id. */
        cid = s->cid_marked[k];
        
        /* Copy the particle data to the device. */
        #ifdef PARTS_TEX
            buff = (float4 *)&parts_cuda[ maxcount * cid ];
        #else
            buff = (float4 *)&parts_cuda[ s->ind_cuda_local[cid] ];
        #endif
        for ( pid = 0 ; pid < s->counts_cuda_local[cid] ; pid++ ) {
            p = &s->cells[cid].parts[pid];
            buff[ pid ].x = p->x[0];
            buff[ pid ].y = p->x[1];
            buff[ pid ].z = p->x[2];
            buff[ pid ].w = p->type;
            }
    
        }
        
    // printf( "engine_cuda_load_parts: packed %i cells with %i parts each (%i kB).\n" , s->nr_cells , maxcount , (sizeof(float4)*maxcount*s->nr_cells)/1024 );
        
    /* Copy the counts onto the device. */
    if ( cudaMemcpy( s->counts_cuda , s->counts_cuda_local , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Copy the inds onto the device. */
    if ( cudaMemcpy( s->ind_cuda , s->ind_cuda_local , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Bind the particle positions to a texture. */
    #ifdef PARTS_TEX
        if ( cudaMallocArray( (cudaArray **)&s->cuArray_parts , &channelDesc_float4 , maxcount , s->nr_cells ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToArray( (cudaArray *)s->cuArray_parts , 0 , 0 , parts_cuda , sizeof(float4) * s->nr_cells * maxcount , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( runner_parts_bind( (cudaArray *)s->cuArray_parts ) < 0 )
            return error(engine_err_runner);
    #else
        if ( cudaMalloc( &s->parts_cuda , sizeof( float4 ) * s->nr_parts ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( s->parts_cuda , parts_cuda , sizeof(float4) * s->nr_parts , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_parts , &s->parts_cuda , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    #endif
    free( parts_cuda );
        
    /* Finally, init the forces on the device. */
    if ( cudaMalloc( &s->forces_cuda , sizeof( float ) * 3 * s->nr_parts ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemset( s->forces_cuda , 0 , sizeof( float ) * 3 * s->nr_parts ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Our work is done here. */
    return maxcount;

    }
    
    

/**
 * @brief Load the cell data from the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_cuda_unload_parts ( struct engine *e ) {
    
    int k, cid, pid;
    struct part *p;
    float *forces_cuda, *buff, epot;
    struct space *s = &e->s;
    
    /* Get the forces from the device. */
    if ( ( forces_cuda = (float *)malloc( sizeof(float) * 3 * s->nr_parts ) ) == NULL )
        return error(engine_err_malloc);
    if ( cudaMemcpy( forces_cuda , s->forces_cuda , sizeof(float) * 3 * s->nr_parts , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Get the potential energy. */
    if ( cudaMemcpyFromSymbol( &epot , cuda_epot_out , sizeof(float) , 0 , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    e->s.epot += epot;
                
    /* Loop over the marked cells. */
    for ( k = 0 ; k < s->nr_marked ; k++ ) {
    
        /* Get the cell id. */
        cid = s->cid_marked[k];
        
        /* Copy the particle data from the device. */
        buff = &forces_cuda[ 3*s->ind_cuda_local[cid] ];
        for ( pid = 0 ; pid < e->s.cells[cid].count ; pid++ ) {
            p = &e->s.cells[cid].parts[pid];
            p->f[0] += buff[ 3*pid ];
            p->f[1] += buff[ 3*pid + 1 ];
            p->f[2] += buff[ 3*pid + 2 ];
            }
            
        }

    /* Deallocate the parts array and counts array. */
    free( forces_cuda );
    if ( cudaFree( e->s.forces_cuda ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Unbind and free the parts data. */
    #ifdef PARTS_TEX
        if ( runner_parts_unbind( ) < 0 )
            return error(engine_err_runner);
        if ( cudaFreeArray( (cudaArray *)e->s.cuArray_parts ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    #else
        if ( cudaFree( e->s.parts_cuda ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    #endif
        
    /* Our work is done here. */
    return engine_err_ok;

    }


/**
 * @brief Load the queues onto the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_cuda_queues_load ( struct engine *e ) {
    
    int did, nr_queues, qid, k, qsize, nr_tasks = e->s.nr_tasks;
    struct cudaDeviceProp prop;
    int *data;
    struct queue_cuda queues[ cuda_maxqueues ];
    
    /* Get the device properties. */
    if ( cudaGetDevice( &did ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaGetDeviceProperties( &prop , did ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Get the number of SMs on the current device. */
    nr_queues = 1; // prop.multiProcessorCount;
    
    /* Set the size of each queue. */
    qsize = 3 * nr_tasks / min( nr_queues , e->nr_runners );
    if ( cudaMemcpyToSymbol( cuda_queue_size , &qsize , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Allocate a temporary buffer for the queue data. */
    data = (int *)alloca( sizeof(int) * qsize );
    
    /* Set the number of queues. */
    if ( cudaMemcpyToSymbol( cuda_nrqueues , &nr_queues , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Init each queue separately. */
    for ( qid = 0 ; qid < nr_queues ; qid++ ) {
    
        /* Fill the data for this queue. */
        queues[qid].count = 0;
        for ( k = qid ; k < nr_tasks ; k += nr_queues )
            data[ queues[qid].count++ ] = k;
        for ( k = queues[qid].count ; k < qsize ; k++ )
            data[k] = -1;
            
        /* Allocate and copy the data. */
        if ( cudaMalloc( &queues[qid].data , sizeof(int) * qsize ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( (void *)queues[qid].data , data , sizeof(int) * qsize , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        
        /* Allocate and copy the recycling data. */
        for ( k = 0 ; k < queues[qid].count ; k++ )
            data[k] = -1;
        if ( cudaMalloc( &queues[qid].rec_data , sizeof(int) * qsize ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( (void *)queues[qid].rec_data , data , sizeof(int) * qsize , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        
        /* Set some other values. */
        queues[qid].first = 0;
        queues[qid].last = queues[qid].count;
        queues[qid].rec_count = 0;
            
        }
        
    /* Copy the queue structures to the device. */
    if ( cudaMemcpyToSymbol( cuda_queues , queues , sizeof(struct queue_cuda) * nr_queues , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Fade to grey. */
    return engine_err_ok;

    }
    

/**
 * @brief Load the potentials and cell pairs onto the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_cuda_load ( struct engine *e ) {

    int i, j, k, nr_pots, nr_coeffs, max_coeffs = 0;
    int pind[ e->max_type * e->max_type ], *pind_cuda;
    struct space *s = &e->s;
    struct potential *pots[ e->nr_types * (e->nr_types + 1) / 2 + 1 ];
    struct task_cuda *tasks_cuda;
    float *finger, *coeffs_cuda;
    float cutoff = e->s.cutoff, cutoff2 = e->s.cutoff2, dscale; //, buff[ e->nr_types ];
    cudaArray *cuArray_coeffs, *cuArray_pind;
    cudaChannelFormatDesc channelDesc_int = cudaCreateChannelDesc<int>();
    cudaChannelFormatDesc channelDesc_float = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc channelDesc_float4 = cudaCreateChannelDesc<float4>();
    unsigned int *taboo_cuda;
    float h[3];
    
    /* Init the null potential. */
    if ( ( pots[0] = (struct potential *)alloca( sizeof(struct potential) ) ) == NULL )
        return error(engine_err_malloc);
    pots[0]->alpha[0] = pots[0]->alpha[1] = pots[0]->alpha[2] = pots[0]->alpha[3] = 0.0f;
    pots[0]->a = 0.0; pots[0]->b = DBL_MAX;
    pots[0]->flags = potential_flag_none;
    pots[0]->n = 0;
    if ( ( pots[0]->c = (FPTYPE *)alloca( sizeof(float) * potential_chunk ) ) == NULL )
        return error(engine_err_malloc);
    bzero( pots[0]->c , sizeof(float) * potential_chunk );
    nr_pots = 1; nr_coeffs = 1;
    
    /* Start by identifying the unique potentials in the engine. */
    for ( i = 0 ; i < e->max_type * e->max_type ; i++ ) {
    
        /* Skip if there is no potential or no parts of this type. */
        if ( e->p[i] == NULL )
            continue;
            
        /* Check this potential against previous potentials. */
        for ( j = 0 ; j < nr_pots && e->p[i] != pots[j] ; j++ );
        if ( j < nr_pots )
            continue;
            
        /* Store this potential and the number of coefficient entries it has. */
        pots[nr_pots] = e->p[i];
        nr_pots += 1;
        nr_coeffs += e->p[i]->n + 1;
        if ( e->p[i]->n + 1 > max_coeffs )
            max_coeffs = e->p[i]->n + 1;
    
        }
       
    /* Copy eps and rmin to the device. */
    /* for ( i = 0 ; i < e->nr_types ; i++ )
        buff[i] = sqrt( fabs( e->types[i].eps ) );
    if ( cudaMemcpyToSymbol( "cuda_eps" , buff , sizeof(float) * e->nr_types , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    for ( i = 0 ; i < e->nr_types ; i++ )
        buff[i] = e->types[i].rmin;
    if ( cudaMemcpyToSymbol( "cuda_rmin" , buff , sizeof(float) * e->nr_types , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */

    /* Pack the potential matrix. */
    for ( i = 0 ; i < e->max_type * e->max_type ; i++ ) {
        if ( e->p[i] == NULL ) {
            pind[i] = 0;
            }
        else {
            for ( j = 0 ; j < nr_pots && pots[j] != e->p[i] ; j++ );
            pind[i] = j;
            }
        }
        
    /* Pack the coefficients before shipping them off to the device. */
    if ( ( coeffs_cuda = (float *)malloc( sizeof(float4) * (2*max_coeffs + 2) * nr_pots ) ) == NULL )
        return error(engine_err_malloc);
    for ( i = 0 ; i < nr_pots ; i++ ) {
        finger = &coeffs_cuda[ i*4*(2*max_coeffs + 2) ];
        finger[0] = pots[i]->alpha[0];
        finger[1] = pots[i]->alpha[1];
        finger[2] = pots[i]->alpha[2];
        memcpy( &finger[8] , pots[i]->c , sizeof(float) * potential_chunk * (pots[i]->n + 1) );
        }
    /* for ( finger = coeffs_cuda , i = 0 ; i < nr_pots ; i++ ) {
        memcpy( finger , pots[i]->c , sizeof(float) * potential_chunk * (pots[i]->n + 1) );
        finger = &finger[ (pots[i]->n + 1) * potential_chunk ];
        } */
    printf( "engine_cuda_load: packed %i potentials with %i coefficient chunks (%i kB).\n" , nr_pots , max_coeffs , (sizeof(float4)*(2*max_coeffs+2)*nr_pots)/1024 ); fflush(stdout);
        
    /* Bind the potential coefficients to a texture. */
    if ( cudaMallocArray( &cuArray_coeffs , &channelDesc_float4 , 2*max_coeffs + 2 , nr_pots ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToArray( cuArray_coeffs , 0 , 0 , coeffs_cuda , sizeof(float4) * (2*max_coeffs + 2) * nr_pots , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    free( coeffs_cuda );
    
    /* Copy the cell edge lengths to the device. */
    h[0] = s->h[0]; h[1] = s->h[1]; h[2] = s->h[2];
    if ( cudaMemcpyToSymbol( cuda_h , h , sizeof(float) * 3 , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Copy the potential indices to the device. */
    if ( cudaMallocArray( &cuArray_pind , &channelDesc_int , e->max_type * e->max_type , 1 ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToArray( cuArray_pind , 0 , 0 , pind , sizeof(int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Store pind as a constant too. */
    if ( cudaMalloc( &pind_cuda , sizeof(unsigned int) * e->max_type * e->max_type ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( pind_cuda , pind , sizeof(unsigned int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( cuda_pind , &pind_cuda , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
            
    /* Bind the textures on the device. */
    if ( runner_bind( cuArray_coeffs , cuArray_pind ) < 0 )
        return error(engine_err_runner);
        
        
    /* Set the constant pointer to the null potential and other useful values. */
    if ( cudaMemcpyToSymbol( cuda_cutoff2 , &cutoff2 , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( cuda_cutoff , &cutoff , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( cuda_maxdist , &cutoff , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( cuda_maxtype , &(e->max_type) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    dscale = ((float)SHRT_MAX) / ( 3.0 * sqrt( s->h[0]*s->h[0] + s->h[1]*s->h[1] + s->h[2]*s->h[2] ) );
    if ( cudaMemcpyToSymbol( cuda_dscale , &dscale , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Allocate and fill the task list. */
    if ( ( tasks_cuda = (struct task_cuda *)alloca( sizeof(struct task_cuda) * s->nr_tasks ) ) == NULL )
        return error(engine_err_malloc);
    for ( i = 0 ; i < s->nr_tasks ; i++ ) {
        tasks_cuda[i].type = s->tasks[i].type;
        tasks_cuda[i].subtype = s->tasks[i].subtype;
        tasks_cuda[i].wait = 0;
        tasks_cuda[i].flags = s->tasks[i].flags;
        tasks_cuda[i].i = s->tasks[i].i;
        tasks_cuda[i].j = s->tasks[i].j;
        tasks_cuda[i].nr_unlock = s->tasks[i].nr_unlock;
        for ( k = 0 ; k < tasks_cuda[i].nr_unlock ; k++ )
            tasks_cuda[i].unlock[k] = s->tasks[i].unlock[k] - s->tasks;
        }
    for ( i = 0 ; i < s->nr_tasks ; i++ )
        for ( k = 0 ; k < tasks_cuda[i].nr_unlock ; k++ )
            tasks_cuda[ tasks_cuda[i].unlock[k] ].wait += 1;
        
    /* Allocate and fill the pairs list on the device. */
    if ( cudaMalloc( &s->tasks_cuda , sizeof(struct task_cuda) * s->nr_tasks ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( s->tasks_cuda , tasks_cuda , sizeof(struct task_cuda) * s->nr_tasks , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( cuda_tasks , &(s->tasks_cuda) , sizeof(struct task_cuda *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);

        
    /* Allocate the sortlists locally and on the device if needed. */
    if ( cudaMalloc( &e->sortlists_cuda , sizeof(unsigned int) * s->nr_parts * 13 ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( cuda_sortlists , &e->sortlists_cuda , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);

    /* Set the number of cells and tasks. */
    if ( cudaMemcpyToSymbol( cuda_nr_tasks , &(s->nr_tasks) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( cuda_nr_cells , &(s->nr_cells) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Allocate the cell counts and offsets. */
    if ( ( s->counts_cuda_local = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL ||
         ( s->ind_cuda_local = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL )
        return error(engine_err_malloc);
    if ( cudaMalloc( &s->counts_cuda , sizeof(int) * s->nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMalloc( &s->ind_cuda , sizeof(int) * s->nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Allocate and init the taboo list on the device. */
    if ( cudaMalloc( &taboo_cuda , sizeof(int) * s->nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemset( taboo_cuda , 0 , sizeof(int) * s->nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( cuda_taboo , &taboo_cuda , sizeof(int *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Init the pair queue on the device. */
    if ( engine_cuda_queues_load( e ) < 0 )
        return error(engine_err);
        
    /* He's done it! */
    return engine_err_ok;
    
    }
    
    




