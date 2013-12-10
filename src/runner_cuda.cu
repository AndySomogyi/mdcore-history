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

/* Include configuratin header */
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
#include "bond.h"
#include "exclusion.h"
#include "angle.h"
#include "dihedral.h"
#include "engine.h"
#include "runner_cuda.h"


/* the error macro. */
#define error(id)                ( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_error(id)            ( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )


/* The constant null potential. */
__constant__ struct potential *potential_null_cuda = NULL;

/* The number of cells and pairs. */
__constant__ int cuda_nr_cells = 0;

/* The parts (non-texture access). */
__constant__ float4 *cuda_parts;
__constant__ int cuda_nr_parts;

/* Potential index lookup tables. */
__constant__ unsigned int *cuda_pind;
__constant__ unsigned int *cuda_pind_bond;

/* The mutex for accessing the cell pair list. */
__device__ int cuda_barrier = 0;

/* The index of the next free cell pair. */
__device__ int cuda_pair_next = 0;

/* The list of cell pairs. */
__constant__ struct cellpair_cuda *cuda_pairs;
__device__ int *cuda_taboo;
#ifdef TASK_TIMERS
/*x = block y = type z = start w = end*/
__device__ int4 cuda_task_timers[26*10000];
#endif

/* The list of tasks. */
__constant__ struct task_cuda *cuda_tasks;
__constant__ int cuda_nr_tasks = 0;

/* The per-SM task queues. */
__device__ struct queue_cuda cuda_queues[ cuda_maxqueues ];
__device__ struct queue_cuda cuda_sorts[ cuda_maxqueues ];
__constant__ int cuda_nrqueues;
__constant__ int cuda_queue_size;

/* Some constants. */
__constant__ float cuda_cutoff2 = 0.0f;
__constant__ float cuda_cutoff = 0.0f;
__constant__ float cuda_dscale = 0.0f;
__constant__ float cuda_maxdist = 0.0f;
__constant__ int cuda_maxtype = 0;
__constant__ struct potential *cuda_pots;

/* Sortlists for the Verlet algorithm. */
__device__ unsigned int *cuda_sortlists = NULL;

/* The potential coefficients, as a texture. */
texture< float4 , cudaTextureType2D > tex_coeffs;
texture< float4 , cudaTextureType2D > tex_parts;

/* Other textures. */
texture< int , cudaTextureType1D > tex_pind;
texture< int , cudaTextureType1D > tex_pind_bond;

/* Bond, angle, dihedral, and exclusion lists. */
__constant__ int *cuda_bonds;
__constant__ int *cuda_exclusions;
__constant__ int *cuda_angles;
__constant__ int *cuda_dihedrals;
__constant__ int cuda_nrbonds, cuda_nrangles, cuda_nrdihedrals, cuda_nrexclusions;
__constant__ int *cuda_partlist;
__constant__ int *cuda_celllist;

/* Arrays to hold the textures. */
cudaArray *cuda_coeffs;

/* Cell origins. */
__constant__ float *cuda_corig;

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
__constant__ float cuda_shiftn[13*3] = {
     5.773502691896258e-01 ,  5.773502691896258e-01 ,  5.773502691896258e-01 ,
     7.071067811865475e-01 ,  7.071067811865475e-01 ,  0.0                   ,
     5.773502691896258e-01 ,  5.773502691896258e-01 , -5.773502691896258e-01 ,
     7.071067811865475e-01 ,  0.0                   ,  7.071067811865475e-01 ,
     1.0                   ,  0.0                   ,  0.0                   ,
     7.071067811865475e-01 ,  0.0                   , -7.071067811865475e-01 ,
     5.773502691896258e-01 , -5.773502691896258e-01 ,  5.773502691896258e-01 ,
     7.071067811865475e-01 , -7.071067811865475e-01 ,  0.0                   ,
     5.773502691896258e-01 , -5.773502691896258e-01 , -5.773502691896258e-01 ,
     0.0                   ,  7.071067811865475e-01 ,  7.071067811865475e-01 ,
     0.0                   ,  1.0                   ,  0.0                   ,
     0.0                   ,  7.071067811865475e-01 , -7.071067811865475e-01 ,
     0.0                   ,  0.0                   ,  1.0                   ,
     };
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
    
/* The cell edge lengths and space dimensions. */
__constant__ float cuda_h[3];
__constant__ float cuda_dim[3];
    
    
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

__device__ int runner_cuda_gettask_nolock ( struct queue_cuda *q , int steal ) {

    int tid = -1/*,cid,cjd*/;
    
    TIMER_TIC
    
    /* Main loop. */
    while ( ( tid = cuda_queue_gettask( q ) ) >= 0 ) {
    
        /* If this task is not even free, don't even bother. */
        if ( !cuda_tasks[tid].wait ) {
    
            #ifdef FORCES_LOCAL
                break;
            #else
                break;
                    
            #endif

            }
        /*if( cuda_tasks[tid].type == task_type_pair )
        {
               cid = cuda_tasks[tid].i;
               cjd = cuda_tasks[tid].j;
               if(!( cuda_taboo[cid] || cuda_taboo[cjd] ))
                   break;
           }else{
               break;
           }*/
           
                
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

    int k;
    int *idest = (int *)dest, *isource = (int *)source;

    int threadID = threadIdx.x;
    
    TIMER_TIC
    
    /* Copy the data in chunks of sizeof(int). */
    for ( k = threadID ; k < count/sizeof(int) ; k += blockDim.x )
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
 
__device__ inline void cuda_sort_descending ( unsigned int *a , int count ) {

    
    int i, j, k, threadID = threadIdx.x;
    int hi, lo, ind, jnd;
    unsigned int swap_i, swap_j;

    TIMER_TIC

    /* Sort using normalized bitonic sort. */
    for ( k = 1 ; k < count ; k *= 2 ) {
    
        /* First step. */
        for ( i = threadID ;  i < count ; i += blockDim.x ) {
            hi = i & ~(k-1); lo = i & (k-1);
            ind = i + hi; jnd = 2*(hi+k) - lo - 1;
            swap_i = ( jnd < count ) ? a[ind] : 0;
            swap_j = ( jnd < count ) ? a[jnd] : 0;
            if  ( ( swap_i & 0xffff ) < ( swap_j & 0xffff ) ) {
                a[ind] = swap_j;
                a[jnd] = swap_i;
                }
            }
            
        /* Let that last step sink in. */
            __syncthreads();
    
        /* Second step(s). */
        for ( j = k/2 ; j > 0 ; j /= 2 ) {
            for ( i = threadID ;  i < count ; i += blockDim.x ) {
                hi = i & ~(j-1);
                ind = i + hi; jnd = ind + j;
                swap_i = ( jnd < count ) ? a[ind] : 0;
                swap_j = ( jnd < count ) ? a[jnd] : 0;
                if  ( ( swap_i & 0xffff ) < ( swap_j & 0xffff ) ) {
                    a[ind] = swap_j;
                    a[jnd] = swap_i;
                    }
                }
                __syncthreads();
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


__device__ inline void potential_eval_r_cuda_tex ( int pid , float r , float *e , float *f ) {

    int ind;
    float x, ee, eff;
    float4 alpha, c1, c2;
    
    TIMER_TIC
    
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
    *e = ee; *f = eff * c1.y;
        
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
 * @brief Evaluate a list of bonded interactoins
 *
 * @param b Pointer to an array of bond indices.
 * @param count Nr of bonds in @c b.
 * @param forces Pointer to an array with the global particle forces.
 * @param epot_out Potential energy aggregator.
 */
 
__device__ void runner_dobond_cuda ( int *b , int count , float *forces , float *epot_out ) {

    int bid, pid, pjd, cid, cjd, k;
    float epot = 0.0f;
    float4 pi, pj;
    float r2, w;
    float ee, eff, dx[3];
    int pot;
    TIMER_TIC
    /* Loop over the bonds. */
    for ( bid = threadIdx.x ; bid < count ; bid += blockDim.x ) {
    
        /* Get the particles involved. */
        if ( ( pid = cuda_partlist[ b[ 2*bid + 0 ] ] ) < 0 ||
             ( pjd = cuda_partlist[ b[ 2*bid + 1 ] ] ) < 0 )
            continue;
            
        /* Get the cell IDs. */
        cid = cuda_celllist[ b[ 2*bid + 0 ] ];
        cjd = cuda_celllist[ b[ 2*bid + 1 ] ];
            
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , pid % cuda_maxparts , cid );
            pj = tex2D( tex_parts , pjd % cuda_maxparts , cjd );
        #else
            pi = cuda_parts[ pid ];
            pj = cuda_parts[ pjd ];
        #endif
        
        /* Try to get the potential. */
        if ( ( pot = cuda_pind_bond[ ((int)pi.w)*cuda_maxtype + (int)pj.w ] ) <= 0 )
            continue;
            
        /* Compute the shift for the particle cells. */
        for ( k = 0 ; k < 3 ; k++ ) {
            dx[k] = cuda_corig[ 3*cid + k ] - cuda_corig[ 3*cjd + k ];
            if ( 2*dx[k] > cuda_dim[k] )
                dx[k] -= cuda_dim[k];
            else if ( 2*dx[k] < -cuda_dim[k] )
                dx[k] += cuda_dim[k];
            }
            
        /* Get the interparticle distance. */
        dx[0] += pi.x - pj.x;
        dx[1] += pi.y - pj.y;
        dx[2] += pi.z - pj.z;
        r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
                    
        /* evaluate the bonded potential. */
        potential_eval_cuda_tex( pot , r2 , &ee , &eff );

        /* update the forces */
        for ( k = 0 ; k < 3 ; k++ ) {
            w = eff * dx[k];
            atomicAdd( &forces[ 3*pid + k ] , -w );
            atomicAdd( &forces[ 3*pjd + k ] , w );
            }

        /* tabulate the energy */
        epot += ee;

        } /* loop over bonds. */
        
    /* Store the potential energy. */
    *epot_out += 2*epot;
    TIMER_TOC(tid_bonded)
    }



/**
 * @brief Evaluate a list of angle interactoins
 *
 * @param b Pointer to an array of angle indices.
 * @param count Nr of angles in @c b.
 * @param forces Pointer to an array with the global particle forces.
 * @param epot_out Potential energy aggregator.
 */
 
__device__ void runner_doangle_cuda ( int *b , int count , float *forces , float *epot_out ) {

    int bid, pid, pjd, pkd, cid, cjd, ckd, k;
    float epot = 0.0f;
    float4 pi, pj, pk;
    float ee, eff;
    float rji[3], rjk[3];
    float dprod, inji, injk, ctheta;
    float dxi[3], dxk[3], wi, wk;
    
    TIMER_TIC
    /* Loop over the angles. */
    for ( bid = threadIdx.x ; bid < count ; bid += blockDim.x ) {
    
        /* Get the particles involved. */
        if ( ( pid = cuda_partlist[ b[ 4*bid + 0 ] ] ) < 0 ||
             ( pjd = cuda_partlist[ b[ 4*bid + 1 ] ] ) < 0 ||
             ( pkd = cuda_partlist[ b[ 4*bid + 2 ] ] ) < 0 )
            continue;
            
        // if ( threadIdx.x == 0 )
        //     printf( "runner_doangle_cuda: working on angle with pid=%i, pjd=%i, pkd=%i.\n" , pid , pjd , pkd );
            
        /* Get the cell IDs. */
        cid = cuda_celllist[ b[ 4*bid + 0 ] ];
        cjd = cuda_celllist[ b[ 4*bid + 1 ] ];
        ckd = cuda_celllist[ b[ 4*bid + 2 ] ];
            
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , pid % cuda_maxparts , cid );
            pj = tex2D( tex_parts , pjd % cuda_maxparts , cjd );
            pk = tex2D( tex_parts , pkd % cuda_maxparts , ckd );
        #else
            pi = cuda_parts[ pid ];
            pj = cuda_parts[ pjd ];
            pk = cuda_parts[ pkd ];
        #endif
        
        /* Compute the shift for the particle cells. */
        for ( k = 0 ; k < 3 ; k++ ) {
            rji[k] = cuda_corig[ 3*cid + k ] - cuda_corig[ 3*cjd + k ];
            if ( 2*rji[k] > cuda_dim[k] )
                rji[k] -= cuda_dim[k];
            else if ( 2*rji[k] < -cuda_dim[k] )
                rji[k] += cuda_dim[k];
            rjk[k] = cuda_corig[ 3*ckd + k ] - cuda_corig[ 3*cjd + k ];
            if ( 2*rjk[k] > cuda_dim[k] )
                rjk[k] -= cuda_dim[k];
            else if ( 2*rjk[k] < -cuda_dim[k] )
                rjk[k] += cuda_dim[k];
            }
            
        /* Get the interparticle distances. */
        rji[0] += pi.x - pj.x;
        rji[1] += pi.y - pj.y;
        rji[2] += pi.z - pj.z;
        rjk[0] += pk.x - pj.x;
        rjk[1] += pk.y - pj.y;
        rjk[2] += pk.z - pj.z;
                    
        /* Compute some quantities we will re-use. */
        dprod = rji[0]*rjk[0] + rji[1]*rjk[1] + rji[2]*rjk[2];
        inji = rsqrt( rji[0]*rji[0] + rji[1]*rji[1] + rji[2]*rji[2] );
        injk = rsqrt( rjk[0]*rjk[0] + rjk[1]*rjk[1] + rjk[2]*rjk[2] );
        
        /* Compute the cosine. */
        ctheta = fmaxf( -1.0f , fminf( 1.0f , dprod * inji * injk ) );
        
        /* Set the derivatives. */
        for ( k = 0 ; k < 3 ; k++ ) {
            dxi[k] = ( rjk[k]*injk - ctheta * rji[k]*inji ) * inji;
            dxk[k] = ( rji[k]*inji - ctheta * rjk[k]*injk ) * injk;
            }
        
        /* evaluate the angleed potential. */
        potential_eval_r_cuda_tex( b[ 4*bid + 3 ] , ctheta , &ee , &eff );

        /* update the forces */
        for ( k = 0 ; k < 3 ; k++ ) {
            wi = eff * dxi[k];
            wk = eff * dxk[k];
            atomicAdd( &forces[ 3*pid + k ] , -wi );
            atomicAdd( &forces[ 3*pjd + k ] , wi + wk );
            atomicAdd( &forces[ 3*pkd + k ] , -wk );
            }

        /* tabulate the energy */
        epot += ee;

        } /* loop over angles. */
        
    /* Store the potential energy. */
    *epot_out += 2*epot;
    TIMER_TOC(tid_bonded)
    }


/**
 * @brief Evaluate a list of dihedral interactoins
 *
 * @param b Pointer to an array of dihedral indices.
 * @param count Nr of dihedrals in @c b.
 * @param forces Pointer to an array with the global particle forces.
 * @param epot_out Potential energy aggregator.
 */
 
__device__ void runner_dodihedral_cuda ( int *b , int count , float *forces , float *epot_out ) {

    int bid, pid, pjd, pkd, pld, cid, cjd, ckd, cld, k;
    float epot = 0.0f;
    float4 pi, pj, pk, pl;
    float cphi, ee, eff;
    float rji[3], rjk[3], rjl[3];
    float dxi[3], dxj[3], dxl[3], wi, wj, wl;
    float t1, t10, t11, t12, t13, t14, t15, t16, t17, t18, t19, t20, t21,
        t22, t24, t26, t3, t30, t31, t32, t33, t34, t35, t36, t37, t38, t39, t40,
        t41, t42, t43, t44, t45, t46, t47, t5, t6, t7, t8, t9,
        t2, t4, t23, t25, t27, t28, t51, t52, t53, t54, t59;

    TIMER_TIC
    /* Loop over the dihedrals. */
    for ( bid = threadIdx.x ; bid < count ; bid += blockDim.x ) {
    
        /* Get the particles involved. */
        if ( ( pid = cuda_partlist[ b[ 5*bid + 0 ] ] ) < 0 ||
             ( pjd = cuda_partlist[ b[ 5*bid + 1 ] ] ) < 0 ||
             ( pkd = cuda_partlist[ b[ 5*bid + 2 ] ] ) < 0 ||
             ( pld = cuda_partlist[ b[ 5*bid + 3 ] ] ) < 0 )
            continue;
            
        /* Get the cell IDs. */
        cid = cuda_celllist[ b[ 5*bid + 0 ] ];
        cjd = cuda_celllist[ b[ 5*bid + 1 ] ];
        ckd = cuda_celllist[ b[ 5*bid + 2 ] ];
        cld = cuda_celllist[ b[ 5*bid + 3 ] ];
            
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , pid % cuda_maxparts , cid );
            pj = tex2D( tex_parts , pjd % cuda_maxparts , cjd );
            pk = tex2D( tex_parts , pkd % cuda_maxparts , ckd );
            pl = tex2D( tex_parts , pld % cuda_maxparts , cld );
        #else
            pi = cuda_parts[ pid ];
            pj = cuda_parts[ pjd ];
            pk = cuda_parts[ pkd ];
            pl = cuda_parts[ pld ];
        #endif
        
        /* Compute the shift for the particle cells. */
        for ( k = 0 ; k < 3 ; k++ ) {
            rji[k] = cuda_corig[ 3*cid + k ] - cuda_corig[ 3*cjd + k ];
            if ( 2*rji[k] > cuda_dim[k] )
                rji[k] -= cuda_dim[k];
            else if ( 2*rji[k] < -cuda_dim[k] )
                rji[k] += cuda_dim[k];
            rjk[k] = cuda_corig[ 3*ckd + k ] - cuda_corig[ 3*cjd + k ];
            if ( 2*rjk[k] > cuda_dim[k] )
                rjk[k] -= cuda_dim[k];
            else if ( 2*rjk[k] < -cuda_dim[k] )
                rjk[k] += cuda_dim[k];
            rjl[k] = cuda_corig[ 3*cld + k ] - cuda_corig[ 3*cjd + k ];
            if ( 2*rjl[k] > cuda_dim[k] )
                rjl[k] -= cuda_dim[k];
            else if ( 2*rjl[k] < -cuda_dim[k] )
                rjl[k] += cuda_dim[k];
            }
            
        /* Shift the particles accordingly. */
        pi.x += rji[0]; pi.y += rji[1]; pi.z += rji[2];
        pk.x += rjk[0]; pk.y += rjk[1]; pk.z += rjk[2];
        pl.x += rjl[0]; pl.y += rjl[1]; pl.z += rjl[2];
                    
        /* This is Maple-generated code, see "dihedral.maple" for details. */
        t16 = pl.z-pk.z;
        t17 = pl.y-pk.y;
        t18 = pl.x-pk.x;
        t2 = t18*t18;
        t4 = t17*t17;
        t23 = t16*t16;
        t10 = t2+t4+t23;
        t19 = pk.z-pj.z;
        t20 = pk.y-pj.y;
        t21 = pk.x-pj.x;
        t25 = t21*t21;
        t27 = t20*t20;
        t28 = t19*t19;
        t11 = t25+t27+t28;
        t7 = t18*t21+t17*t20+t16*t19;
        t51 = t7*t7;
        t5 = t11*t10-t51;
        t22 = pi.z-pj.z;
        t24 = pi.y-pj.y;
        t26 = pi.x-pj.x;
        t52 = t26*t26;
        t53 = t24*t24;
        t54 = t22*t22;
        t12 = t52+t53+t54;
        t9 = -t26*t21-t24*t20-t22*t19;
        t59 = t9*t9;
        t6 = t12*t11-t59;
        t3 = t6*t5;
        t1 = rsqrt(t3);
        t8 = -t26*t18-t24*t17-t22*t16;
        t47 = (t9*t7-t8*t11)*t1;
        t46 = 2.0f*t8;
        t45 = t6*t7;
        t44 = t9*t5;
        t43 = t6*t10;
        t42 = -t9-t11;
        t41 = t22*t11;
        t40 = t24*t11;
        t39 = t26*t11;
        t38 = t47/t3;
        t37 = -t7*t19+t16*t11;
        t36 = -t7*t20+t17*t11;
        t35 = -t7*t21+t18*t11;
        t34 = t9*t19+t41;
        t33 = t9*t20+t40;
        t32 = t9*t21+t39;
        t31 = t5*t38;
        t30 = t6*t38;
        t15 = pk.x-2.0f*pj.x+pi.x;
        t14 = pk.y-2.0f*pj.y+pi.y;
        t13 = pk.z-2.0f*pj.z+pi.z;
        dxi[0] = t35*t1-t32*t31;
        dxi[1] = t36*t1-t33*t31;
        dxi[2]= t37*t1-t34*t31;
        dxj[0] = (t15*t7+t21*t46+t42*t18)*t1-(-t15*t44+t18*t45+(-t39-t12*t21)*t5-t21*t43)*t38;
        dxj[1] = (t14*t7+t20*t46+t42*t17)*t1-(-t14*t44+t17*t45+(-t40-t12*t20)*t5-t20*t43)*t38;
        dxj[2] = (t13*t7+t19*t46+t42*t16)*t1-(-t13*t44+t16*t45+(-t41-t12*t19)*t5-t19*t43)*t38;
        dxl[0] = t32*t1-t35*t30;
        dxl[1] = t33*t1-t36*t30;
        dxl[2] = t34*t1-t37*t30;
        cphi = fmaxf( -1.0f , fminf( 1.0f , t47 ) );
        
        /* evaluate the dihedraled potential. */
        potential_eval_r_cuda_tex( b[ 5*bid + 4 ] , cphi , &ee , &eff );

        /* update the forces */
        for ( k = 0 ; k < 3 ; k++ ) {
            wi = eff * dxi[k];
            wj = eff * dxj[k];
            wl = eff * dxl[k];
            atomicAdd( &forces[ 3*pid + k ] , -wi );
            atomicAdd( &forces[ 3*pjd + k ] , -wj );
            atomicAdd( &forces[ 3*pkd + k ] , wi+wj+wl );
            atomicAdd( &forces[ 3*pld + k ] , -wl );
            }

        /* tabulate the energy */
        epot += ee;

        } /* loop over dihedrals. */
        
    /* Store the potential energy. */
    *epot_out += 2*epot;
    TIMER_TOC(tid_bonded)
    }



/**
 * @brief Evaluate a list of exclusions
 *
 * @param b Pointer to an array of bond indices.
 * @param count Nr of bonds in @c b.
 * @param forces Pointer to an array with the global particle forces.
 * @param epot_out Potential energy aggregator.
 */
 
__device__ void runner_doexclusion_cuda ( int *b , int count , float *forces , float *epot_out ) {

    int bid, pid, pjd, cid, cjd, k;
    float epot = 0.0f;
    float4 pi, pj;
    float r2, w;
    float ee, eff, dx[3];
    int pot;

    TIMER_TIC
    /* Loop over the bonds. */
    for ( bid = threadIdx.x ; bid < count ; bid += blockDim.x ) {
    
        /* Get the particles involved. */
        if ( ( pid = cuda_partlist[ b[ 2*bid + 0 ] ] ) < 0 ||
             ( pjd = cuda_partlist[ b[ 2*bid + 1 ] ] ) < 0 )
            continue;
            
        /* Get the cell IDs. */
        cid = cuda_celllist[ b[ 2*bid + 0 ] ];
        cjd = cuda_celllist[ b[ 2*bid + 1 ] ];
            
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , pid % cuda_maxparts , cid );
            pj = tex2D( tex_parts , pjd % cuda_maxparts , cjd );
        #else
            pi = cuda_parts[ pid ];
            pj = cuda_parts[ pjd ];
        #endif
        
        /* Try to get the potential. */
        if ( ( pot = cuda_pind[ ((int)pi.w)*cuda_maxtype + (int)pj.w ] ) <= 0 )
            continue;
            
        /* Compute the shift for the particle cells. */
        for ( k = 0 ; k < 3 ; k++ ) {
            dx[k] = cuda_corig[ 3*cid + k ] - cuda_corig[ 3*cjd + k ];
            if ( 2*dx[k] > cuda_dim[k] )
                dx[k] -= cuda_dim[k];
            else if ( 2*dx[k] < -cuda_dim[k] )
                dx[k] += cuda_dim[k];
            }
            
        /* Get the interparticle distance. */
        dx[0] += pi.x - pj.x;
        dx[1] += pi.y - pj.y;
        dx[2] += pi.z - pj.z;
        r2 = dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
                    
        /* evaluate the bonded potential. */
        potential_eval_cuda_tex( pot , r2 , &ee , &eff );

        /* update the forces */
        for ( k = 0 ; k < 3 ; k++ ) {
            w = eff * dx[k];
            atomicAdd( &forces[ 3*pid + k ] , w );
            atomicAdd( &forces[ 3*pjd + k ] , -w );
            }

        /* tabulate the energy */
        epot += ee;

        } /* loop over bonds. */
        
    /* Store the potential energy. */
    *epot_out -= 2*epot;
    TIMER_TOC(tid_bonded)
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
__device__ void runner_dopair_unsorted_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , float *shift , float *epot_global ) {
#else
__device__ void runner_dopair_unsorted_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , float *shift , float *epot_global ) {
#endif

    int k, pid, pjd, ind, wrap_i, threadID;
    int pjoff;
    int pot;
    float epot = 0.0f, dx[3], pjf[3], r2, w;
    float ee = 0.0f, eff = 0.0f;
    float4 pi, pj;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
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
__device__ void runner_dopair4_unsorted_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , float *shift , float *epot_global ) {
#else
__device__ void runner_dopair4_unsorted_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , float *shift , float *epot_global ) {
#endif

    int k, pjd, ind, wrap_i, threadID;
    int pjoff;
    float4 pi[4], pj;
    int4 pot, pid;
    char4 valid;
    float4 r2, ee, eff;
    float epot = 0.0f, dx[12], pjf[3], w;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
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
__device__ inline void runner_dosort_cuda ( float4 *parts_i , int count_i , unsigned int *sort_i , int sid ) {
#endif

    int k, threadID = threadIdx.x;
    float4 pi;
    // int4 spid;
    float nshift, shift[3], shiftn[3];
    
    TIMER_TIC
    
    /* Get the shift vector from the sid. */
    shift[0] = cuda_shift[ 3*sid + 0 ] * cuda_h[0];
    shift[1] = cuda_shift[ 3*sid + 1 ] * cuda_h[1];
    shift[2] = cuda_shift[ 3*sid + 2 ] * cuda_h[2];

    /* Pre-compute the inverse norm of the shift. */
    nshift = sqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
    shiftn[0] = cuda_shiftn[ 3*sid + 0 ];
    shiftn[1] = cuda_shiftn[ 3*sid + 1 ];
    shiftn[2] = cuda_shiftn[ 3*sid + 2 ];

    /* Pack the parts into the sort arrays. */
    for ( k = threadID ; k < count_i ; k += blockDim.x ) {
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , k , cid );
        #else
            pi = parts_i[ k ];
        #endif
        sort_i[k] = ( k << 16 ) | (unsigned int)( cuda_dscale * (nshift + pi.x*shiftn[0] + pi.y*shiftn[1] + pi.z*shiftn[2]) );
        }

    TIMER_TOC(tid_pack)
    __syncthreads();
    
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
__device__ void runner_dopair_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *shift , unsigned int dshift , float *epot_global ) {
#else
__device__ void runner_dopair_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *shift , unsigned int dshift , float *epot_global ) {
#endif

    int k, pid, pjd, spid, spjd, pjdid, threadID, wrap, cj;
    int pioff;
    unsigned int dmaxdist;
    float4 pi, pj;
    int pot;
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f;
    float dx[3], pif[3];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Pre-compute the inverse norm of the shift. */
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;
       

    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && ( sort_j[count_j-cj] & 0xffff ) + dshift - ( sort_i[pid & ~(cuda_frame - 1)] & 0xffff ) > dmaxdist )
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
__device__ void runner_dopair_left_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *shift , unsigned int dshift , float *epot_global ) {
#else
__device__  void runner_dopair_left_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *shift , unsigned int dshift , float *epot_global ) {
#endif

    int k, pjd, spid, spjd, threadID;
    int pioff;
    unsigned int dmaxdist, di;
    float4 pi, pj;
    int pot;
    int i;
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f;
    float dx[3], pif[3];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Pre-compute the inverse norm of the shift. */
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;

    /* nr_threads >= count_i */
    
    for ( i = threadID ; i < count_i ;  i += blockDim.x  ) {
    /*cj = count_j-1;
    while ( cj > 0 && ( sort_j[cj] & 0xffff ) + dshift  <= dmaxdist +( sort_i[threadID] & 0xffff ) )
            cj -= 1;*/
    di= sort_i[i]&0xffff;      
        /* Get a direct pointer on the pjdth part in cell_j. */
        spid = sort_i[i] >> 16;
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , spid , cid );
        #else
            pi = parts_i[ spid ];
        #endif
        pioff = pi.w * cuda_maxtype;
        pi.x -= shift[0]; pi.y -= shift[1]; pi.z -= shift[2];
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f;
        /* Loop over the particles in cell_j. */
        for ( pjd = count_j-1 ; pjd >=0 && (sort_j[pjd]&0xffff)+dshift<=dmaxdist+di ; pjd-- ) {
                 
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
                        }

                    /* Sync the shared memory values. */
                    // __threadfence_block();
                
                    } /* in range and potential. */

            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            atomicAdd( &forces_i[ 3*spid + k], pif[k] );
            //forces_i[ 3*spid + k ] += pif[k];
        
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
__device__ void runner_dopair_right_cuda ( int cid , int count_i , int cjd , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *shift , unsigned int dshift , float *epot_global ) {
#else
__device__ void runner_dopair_right_cuda ( float4 *parts_i , int count_i , float4 *parts_j , int count_j , float *forces_i , float *forces_j , unsigned int *sort_i , unsigned int *sort_j , float *shift , unsigned int dshift , float *epot_global ) {
#endif

    int k, pjd, spid, spjd, threadID;
    int pioff;
    unsigned int dmaxdist, dj;
    float4 pi, pj;
    int pot, i;
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f;
    float dx[3], pif[3];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Pre-compute the inverse norm of the shift. */
    dmaxdist = 2 + cuda_dscale * cuda_maxdist;
       

    /* nr_threads >= count_i */

    for ( i = threadID ; i < count_i ;  i += blockDim.x  ) {
    /*cj = 0;
    while ( cj < count_j && ( sort_i[threadID] & 0xffff ) + dshift  <= dmaxdist +( sort_j[cj] & 0xffff ) )
            cj += 1;*/
    dj = sort_i[i]&0xffff;      
        /* Get a direct pointer on the pjdth part in cell_j. */
        spid = sort_i[i] >> 16;
        #ifdef PARTS_TEX
            pi = tex2D( tex_parts , spid , cid );
        #else
            pi = parts_i[ spid ];
        #endif
        pioff = pi.w * cuda_maxtype;
        pi.x += shift[0]; pi.y += shift[1]; pi.z += shift[2];
        pif[0] = 0.0f; pif[1] = 0.0f; pif[2] = 0.0f;
        
        /* Loop over the particles in cell_j. */
        for ( pjd = 0 ; pjd < count_j && dj+ dshift <= dmaxdist+(sort_j[pjd]&0xffff) ; pjd++ ) {
                  /*if((sort_i[threadID]&0xffff + dshift < dmaxdist))
                     break;                       */
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
                        }

                    /* Sync the shared memory values. */
                    // __threadfence_block();
                
                    } /* in range and potential. */
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            atomicAdd( &forces_i[ 3*spid + k] , pif[k]);
            //forces_i[ 3*spid + k] += pif[k];
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

    int k, pid, threadID;
    int pjoff;
    float4 pi, pj;
    int pot, i;
    float epot = 0.0f, dx[3], pjf[3], r2, w, ee, eff;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x;
    
    /* Make sure everybody is in the same place. */
    // __threadfence_block();

    /* Loop over the particles in the cell, frame-wise. */
    for ( i = threadID ; i < count ;  i += blockDim.x  ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        #ifdef PARTS_TEX
            pj = tex2D( tex_parts , threadID , cid );
        #else
            pj = parts[ i ];
        #endif
        pjoff = pj.w * cuda_maxtype;
        pjf[0] = 0.0f; pjf[1] = 0.0f; pjf[2] = 0.0f;
            
        /* Loop over the particles in cell_i. */
        for ( pid = 0 ; pid < count ; pid++ ) {
            if(i != pid ) {
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
//                    forces[ 3*pid + k ] -= w;
                    //atomicAdd( &forces[ 3*pid + k] , -w );
                    pjf[k] += w;
                    }

                /* Sync the shared memory values. */
                // __threadfence_block();
            
                } /* in range and potential. */
            }
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            atomicAdd( &forces[ 3*i + k], pjf[k] );
            //forces[ 3*threadID + k] += pjf[k];
        /* Sync the shared memory values. */
        // __threadfence_block();

        } /* loop over the particles in cell_j. */
        
    /* Store the potential energy. */
    *epot_global += epot;
        
    TIMER_TOC(tid_self)
    
    }



/**
 * @brief Our very own memset for the particle forces as cudaMemsetAsync requires
 *        a device switch when using streams on different devices.
 *
 */
 
__global__ void cuda_memset_float ( float *data , float val , int N ) {

    int k, tid = threadIdx.x + blockIdx.x * blockDim.x;
    int stride = gridDim.x * blockDim.x;
    
    for ( k = tid ; k < N ; k += stride )
        data[k] = val;

    }


/** This set of defines and includes produces kernels with buffers for multiples
 *  of 32 particles up to 512 cuda_maxparts.
 */
 
#define cuda_nparts 32
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 64
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 96
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 128
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 160
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 192
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 224
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 256
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 288
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 320
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 352
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 384
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 416
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 448
    #include "runner_cuda_main.h"
#undef cuda_nparts

#define cuda_nparts 480
    #include "runner_cuda_main.h"
#undef cuda_nparts

// #define cuda_nparts 512
//     #include "runner_cuda_main.h"



/**
 * @brief Offload and compute the nonbonded interactions on a CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_nonbond_cuda ( struct engine *e ) {

    dim3 nr_threads( cuda_frames_per_block*cuda_frame , 1 );
    dim3 nr_blocks( e->nr_runners , 1 );
    int k, cid, did, pid, maxcount = 0;
    cudaStream_t stream;
    cudaEvent_t tic[engine_maxgpu], toc_load[engine_maxgpu], toc_run[engine_maxgpu], toc_unload[engine_maxgpu];
    float ms_load, ms_run, ms_unload;
    struct part *p;
    float4 *parts_cuda = (float4 *)e->parts_cuda_local, *buff4;
    int *partlist, *celllist;
    struct space *s = &e->s;
    FPTYPE maxdist = s->cutoff + 2*s->maxdx;
    int *counts = e->counts_cuda_local[ 0 ], *inds = e->ind_cuda_local[ 0 ];
    float *forces_cuda[ engine_maxgpu ], epot[ engine_maxgpu ], *buff;
    #ifdef TIMERS
        float timers[ tid_count ];
        double icpms = 1000.0 / 1.4e9; 
    #endif
    
    /* Create the events. */
    for(did = 0; did < e->nr_devices; did++ )
    {
    if ( cudaSetDevice( e->devices[did] ) != cudaSuccess ||
         cudaEventCreate( &tic[did] ) != cudaSuccess ||
         cudaEventCreate( &toc_load[did] ) != cudaSuccess ||
         cudaEventCreate( &toc_run[did] ) != cudaSuccess ||
         cudaEventCreate( &toc_unload[did] ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    }
    
    /* Re-set timers */
    #ifdef TIMERS
        for ( int k = 0 ; k < tid_count ; k++ )
            timers[k] = 0.0f;
        for ( did = 0 ; did < e->nr_devices ; did++ )
            if ( cudaMemcpyToSymbolAsync( cuda_timers , timers , sizeof(float) * tid_count , 0 , cudaMemcpyHostToDevice , (cudaStream_t)e->streams[did] ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
    #endif
    
    /* Loop over the devices and call the different kernels on each stream. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
        if(cudaEventRecord( tic[did], (cudaStream_t)e->streams[did] ) != cudaSuccess )
            cuda_error(engine_err_cuda);
        /* Load the particle data onto the device. */
        // tic = getticks();
        // if ( ( maxcount = engine_cuda_load_parts( e ) ) < 0 )
        //     return error(engine_err);
        // e->timers[ engine_timer_cuda_load ] += getticks() - tic;
        counts = e->counts_cuda_local[ did ];
        inds = e->ind_cuda_local[ did ];
        /* Clear the counts array. */
        bzero( counts , sizeof(int) * s->nr_cells );

        /* Load the counts. */
        for( maxcount = 0, k = 0; k < e->cells_cuda_nr[did] ; k++ )
        if( ( counts[e->cells_cuda_local[did][k]] = s->cells[e->cells_cuda_local[did][k]].count ) > maxcount )
            maxcount = counts[ e->cells_cuda_local[did][k]];
    /*    for ( maxcount = 0 , k = 0 ; k < s->nr_marked ; k++ )
            if ( ( counts[ s->cid_marked[k] ] = s->cells[ s->cid_marked[k] ].count ) > maxcount )
                maxcount = counts[ s->cid_marked[k] ];*/

        /* Raise maxcount to the next multiple of 32. */
        maxcount = ( maxcount + (cuda_frame - 1) ) & ~(cuda_frame - 1);
        // printf( "engine_cuda_load_parts: maxcount=%i.\n" , maxcount );

        /* Compute the indices. */
        inds[0] = 0;
        for ( k = 1 ; k < e->cells_cuda_nr[did] ; k++ )
            inds[k] = inds[k-1] + counts[k-1];
            
        /* Init the partlist. */
        partlist = e->partlist_cuda_local[did];
        celllist = e->celllist_cuda_local[did];
        for ( k = 0 ; k < s->nr_parts ; k++ ) {
            partlist[k] = -1;
            celllist[k] = -1;
            }

        /* Loop over the marked cells. */
        for ( k = 0 ; k < e->cells_cuda_nr[did] ; k++ ) {

            /* Get the cell id. */
            cid = e->cells_cuda_local[did][k];

            /* Copy the particle data to the device. */
            #ifdef PARTS_TEX
                buff4 = (float4 *)&parts_cuda[ maxcount * cid ];
            #else
                buff4 = (float4 *)&parts_cuda[ inds[cid] ];
            #endif
            for ( pid = 0 ; pid < counts[cid] ; pid++ ) {
                p = &s->cells[cid].parts[pid];
                buff4[ pid ].x = p->x[0];
                buff4[ pid ].y = p->x[1];
                buff4[ pid ].z = p->x[2];
                buff4[ pid ].w = p->type;
                partlist[ p->id ] = &buff4[ pid ] - parts_cuda;
                celllist[ p->id ] = cid;
                }

            }

    #ifdef PARTS_TEX
        /* Set the texture properties. */
        tex_parts.addressMode[0] = cudaAddressModeClamp;
        tex_parts.addressMode[1] = cudaAddressModeClamp;
        tex_parts.filterMode = cudaFilterModePoint;
        tex_parts.normalized = false;
    #endif

	    /* Start by setting the maxdist on the device. */
        if ( cudaMemcpyToSymbolAsync( cuda_maxdist , &maxdist , sizeof(float) , 0 , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Copy the counts onto the device. */
        if ( cudaMemcpyAsync( e->counts_cuda[did] , counts , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Copy the inds onto the device. */
        if ( cudaMemcpyAsync( e->ind_cuda[did] , inds , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Bind the particle positions to a texture. */
        #ifdef PARTS_TEX
            if ( cudaMemcpyToArrayAsync( (cudaArray *)s->cuArray_parts , 0 , 0 , parts_cuda , sizeof(float4) * s->nr_cells * maxcount , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
        #else
            if ( cudaMemcpyAsync( e->parts_cuda[did] , parts_cuda , sizeof(float4) * s->nr_parts , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
        #endif
        
        /* Finally, copy the partlist. */
        if ( cudaMemcpyAsync( e->partlist_cuda[did] , partlist , sizeof(int) * s->nr_parts , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyAsync( e->celllist_cuda[did] , celllist , sizeof(int) * s->nr_parts , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Make sure maxtype has been reset to avoid stupid compiler bug. */
        if ( cudaMemcpyToSymbol( cuda_maxtype , &(e->max_type) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        
    
    /* Lap the clock on the last stream. */
    if ( cudaEventRecord( toc_load[did] , (cudaStream_t)e->streams[did] ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
        /* Clear the force array. */
        // if ( cudaMemsetAsync( e->forces_cuda[did] , 0 , sizeof( float ) * 3 * s->nr_parts , stream ) != cudaSuccess )
        //     return cuda_error(engine_err_cuda);
        cuda_memset_float <<<8,512,0,stream>>> ( e->forces_cuda[did] , 0.0f , 3 * s->nr_parts );
            
        /* Start the appropriate kernel. */
        switch ( (maxcount + 31) / 32 ) {
            case 1:
                runner_run_cuda_32 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 2:
                runner_run_cuda_64 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 3:
                runner_run_cuda_96 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 4:
                runner_run_cuda_128 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 5:
                runner_run_cuda_160 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 6:
                runner_run_cuda_192 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 7:
                runner_run_cuda_224 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 8:
                runner_run_cuda_256 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 9:
                runner_run_cuda_288 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 10:
                runner_run_cuda_320 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 11:
                runner_run_cuda_352 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 12:
                runner_run_cuda_384 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 13:
                runner_run_cuda_416 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 14:
                runner_run_cuda_448 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            case 15:
                runner_run_cuda_480 <<<nr_blocks,nr_threads,0,stream>>> ( e->forces_cuda[did] , e->counts_cuda[did] , e->ind_cuda[did] , e->s.verlet_rebuild );
                break;
            // case 16:
            //     runner_run_verlet_cuda_512 <<<nr_blocks,nr_threads>>> ( e->forces_cuda , e->counts_cuda , e->ind_cuda , e->s.verlet_rebuild );
            //     break;
            default:
                return error(engine_err_maxparts);
            }
        if ( cudaEventRecord( toc_run[did] , (cudaStream_t)e->streams[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }

    for( did = 0; did < e->nr_devices ; did ++ ) {
    
        /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
        
        /* Get the forces from the device. */
        if ( ( forces_cuda[did] = (float *)malloc( sizeof(float) * 3 * s->nr_parts ) ) == NULL )
            return error(engine_err_malloc);
        if ( cudaMemcpyAsync( forces_cuda[did] , e->forces_cuda[did] , sizeof(float) * 3 * s->nr_parts , cudaMemcpyDeviceToHost , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the potential energy. */
        if ( cudaMemcpyFromSymbolAsync( &epot[did] , cuda_epot_out , sizeof(float) , 0 , cudaMemcpyDeviceToHost , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        
        }
        
    // e->timers[ engine_timer_cuda_dopairs ] += getticks() - tic;
    
    /* Lap the clock on the last stream. */

    
    /* Get and dump timers. */
    #ifdef TIMERS
        if ( cudaMemcpyFromSymbolAsync( timers , cuda_timers , sizeof(float) * tid_count , 0 , cudaMemcpyDeviceToHost , (cudaStream_t)e->streams[0] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        printf( "engine_nonbond_cuda: timers = [ %.2f " , icpms * timers[0] );
        for ( int k = 1 ; k < tid_count ; k++ )
            printf( "%.2f " , icpms * timers[k] );
        printf( "] ms\n" );
    #endif

    #ifdef TASK_TIMERS
		int4 cuda_task_timers_local[26*cuda_maxcells*3];
		if(cudaMemcpyFromSymbol( cuda_task_timers_local, cuda_task_timers, sizeof(int4)*26*cuda_maxcells*3 , 0 , cudaMemcpyDeviceToHost) != cudaSuccess )
			return cuda_error(engine_err_cuda);	
		for(int i = 0; i < e->s.nr_tasks ; i++)
		printf("Task: %i %i %i %i\n", cuda_task_timers_local[i].x, cuda_task_timers_local[i].y, cuda_task_timers_local[i].z, cuda_task_timers_local[i].w);

    #endif
    
    /* Check for any missed CUDA errors. */
    if ( cudaPeekAtLastError() != cudaSuccess )
        return cuda_error(engine_err_cuda);
        

    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
    
        /* Set the device ID. */
        // if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
        //     return cuda_error(engine_err_cuda);

        /* Wait for the chickens to come home to roost. */
        if ( cudaStreamSynchronize( stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    
        /* Get the potential energy. */
        e->s.epot += epot[did];
        
        /* Loop over the marked cells. */
        for ( k = 0 ; k < e->cells_cuda_nr[did] ; k++ ) {

            /* Get the cell id. */
            cid = e->cells_cuda_local[did][k];

            /* Copy the particle data from the device. */
            buff = &forces_cuda[did][ 3*e->ind_cuda_local[did][cid] ];
            for ( pid = 0 ; pid < s->cells[cid].count ; pid++ ) {
                p = &s->cells[cid].parts[pid];
                p->f[0] += buff[ 3*pid ];
                p->f[1] += buff[ 3*pid + 1 ];
                p->f[2] += buff[ 3*pid + 2 ];
                }

            }

        /* Deallocate the parts array and counts array. */
        free( forces_cuda[did] );
                /* Stop the clock on the last stream. */
    if ( cudaEventRecord( toc_unload[did] , (cudaStream_t)e->streams[did] ) != cudaSuccess ||
         cudaStreamSynchronize( (cudaStream_t)e->streams[did] ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        }
        
    /* Check for any missed CUDA errors. */
    if ( cudaPeekAtLastError() != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    
    /* Check for any missed CUDA errors. */
    if ( cudaPeekAtLastError() != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Store the timers. */
    e->timers_cuda[e->nr_devices*3 + 1] = 0.0f;
    e->timers_cuda[e->nr_devices*3 + 0] = 0.0f;
    e->timers_cuda[e->nr_devices*3 + 2] = 0.0f;
    for(did = 0; did < e->nr_devices; did++)
    {
    if ( cudaSetDevice( e->devices[did] ) != cudaSuccess ||
         cudaEventElapsedTime( &ms_load , tic[did] , toc_load[did] ) != cudaSuccess ||
         cudaEventElapsedTime( &ms_run , toc_load[did] , toc_run[did] ) != cudaSuccess ||
         cudaEventElapsedTime( &ms_unload , toc_run[did] , toc_unload[did] ) != cudaSuccess )
         return cuda_error(engine_err_cuda);
    e->timers_cuda[did*3+1]= ms_run*1.0f;
    e->timers_cuda[e->nr_devices*3 + 1] += ms_run; 
    e->timers_cuda[did*3+0]= ms_load*1.0f;
    e->timers_cuda[e->nr_devices*3 + 0] += ms_load; 
    e->timers_cuda[did*3+2] = ms_unload*1.0f;
    e->timers_cuda[e->nr_devices*3 + 2] += ms_unload; 
    }

    
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
    
    int k, did, cid, pid, maxcount = 0;
    struct part *p;
    float4 *parts_cuda = (float4 *)e->parts_cuda_local, *buff;
    struct space *s = &e->s;
    FPTYPE maxdist = s->cutoff + 2*s->maxdx;
    int *counts = e->counts_cuda_local[0], *inds = e->ind_cuda_local[0];
    cudaStream_t stream;
    
    /* Clear the counts array. */
    bzero( counts , sizeof(int) * s->nr_cells );

    /* Load the counts. */
    for ( maxcount = 0 , k = 0 ; k < s->nr_marked ; k++ )
        if ( ( counts[ s->cid_marked[k] ] = s->cells[ s->cid_marked[k] ].count ) > maxcount )
            maxcount = counts[ s->cid_marked[k] ];

    /* Raise maxcount to the next multiple of 32. */
    maxcount = ( maxcount + (cuda_frame - 1) ) & ~(cuda_frame - 1);
    // printf( "engine_cuda_load_parts: maxcount=%i.\n" , maxcount );

    /* Compute the indices. */
    inds[0] = 0;
    for ( k = 1 ; k < s->nr_cells ; k++ )
        inds[k] = inds[k-1] + counts[k-1];

    /* Loop over the marked cells. */
    for ( k = 0 ; k < s->nr_marked ; k++ ) {

        /* Get the cell id. */
        cid = s->cid_marked[k];

        /* Copy the particle data to the device. */
        #ifdef PARTS_TEX
            buff = (float4 *)&parts_cuda[ maxcount * cid ];
        #else
            buff = (float4 *)&parts_cuda[ inds[cid] ];
        #endif
        for ( pid = 0 ; pid < counts[cid] ; pid++ ) {
            p = &s->cells[cid].parts[pid];
            buff[ pid ].x = p->x[0];
            buff[ pid ].y = p->x[1];
            buff[ pid ].z = p->x[2];
            buff[ pid ].w = p->type;
            }

        }

    #ifdef PARTS_TEX
        /* Set the texture properties. */
        tex_parts.addressMode[0] = cudaAddressModeClamp;
        tex_parts.addressMode[1] = cudaAddressModeClamp;
        tex_parts.filterMode = cudaFilterModePoint;
        tex_parts.normalized = false;
    #endif

    // printf( "engine_cuda_load_parts: packed %i cells with %i parts each (%i kB).\n" , s->nr_cells , maxcount , (sizeof(float4)*maxcount*s->nr_cells)/1024 );

    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
        
        /* Start by setting the maxdist on the device. */
        if ( cudaMemcpyToSymbolAsync( cuda_maxdist , &maxdist , sizeof(float) , 0 , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Copy the counts onto the device. */
        if ( cudaMemcpyAsync( e->counts_cuda[did] , counts , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Copy the inds onto the device. */
        if ( cudaMemcpyAsync( e->ind_cuda[did] , inds , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Bind the particle positions to a texture. */
        #ifdef PARTS_TEX
            if ( cudaMemcpyToArrayAsync( (cudaArray *)s->cuArray_parts , 0 , 0 , parts_cuda , sizeof(float4) * s->nr_cells * maxcount , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
        #else
            if ( cudaMemcpyAsync( e->parts_cuda[did] , parts_cuda , sizeof(float4) * s->nr_parts , cudaMemcpyHostToDevice , stream ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
        #endif

        /* Clear the force array. */
        if ( cudaMemsetAsync( e->forces_cuda[did] , 0 , sizeof( float ) * 3 * s->nr_parts , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        // cuda_memset_float <<<8,512,0,stream>>> ( e->forces_cuda[did] , 0.0f , 3 * s->nr_parts );
            
        }
    
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
    
    int k, did, cid, pid;
    struct part *p;
    float *forces_cuda[ engine_maxgpu ], *buff, epot[ engine_maxgpu ];
    struct space *s = &e->s;
    cudaStream_t stream;
    
    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
    
        /* Get the forces from the device. */
        if ( ( forces_cuda[did] = (float *)malloc( sizeof(float) * 3 * s->nr_parts ) ) == NULL )
            return error(engine_err_malloc);
        if ( cudaMemcpyAsync( forces_cuda[did] , e->forces_cuda[did] , sizeof(float) * 3 * s->nr_parts , cudaMemcpyDeviceToHost , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Get the potential energy. */
        if ( cudaMemcpyFromSymbolAsync( &epot[did] , cuda_epot_out , sizeof(float) , 0 , cudaMemcpyDeviceToHost , stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        
        }

    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Get the stream. */
        stream = (cudaStream_t)e->streams[did];
    
        /* Set the device ID. */
        // if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
        //     return cuda_error(engine_err_cuda);

        /* Wait for the chickens to come home to roost. */
        if ( cudaStreamSynchronize( stream ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    
        /* Get the potential energy. */
        e->s.epot += epot[did];
        
        /* Loop over the marked cells. */
        for ( k = 0 ; k < s->nr_marked ; k++ ) {

            /* Get the cell id. */
            cid = s->cid_marked[k];

            /* Copy the particle data from the device. */
            buff = &forces_cuda[did][ 3*e->ind_cuda_local[did][cid] ];
            for ( pid = 0 ; pid < s->cells[cid].count ; pid++ ) {
                p = &s->cells[cid].parts[pid];
                p->f[0] += buff[ 3*pid ];
                p->f[1] += buff[ 3*pid + 1 ];
                p->f[2] += buff[ 3*pid + 2 ];
                }

            }

        /* Deallocate the parts array and counts array. */
        free( forces_cuda[did] );
        
        }
        
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
    
    /* Loop over the devices. */
    for ( did = 0 ; did < e->nr_devices ; did++ ) {
    
        /* Set the device ID. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Get the device properties. */
        if ( cudaGetDeviceProperties( &prop , e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Get the number of SMs on the current device. */
        nr_queues = 1; // prop.multiProcessorCount;

        /* Get the local number of tasks. */
        nr_tasks = e->nrtasks_cuda[did];

        /* Set the size of each queue. */
        qsize = 3 * nr_tasks / min( nr_queues , e->nr_runners );
        if ( cudaMemcpyToSymbol( cuda_queue_size , &qsize , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);

        /* Allocate a temporary buffer for the queue data. */
        if ( ( data = (int *)malloc( sizeof(int) * qsize ) ) == NULL )
            return error(engine_err_malloc);

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

        /* Wait so that we can re-use the local memory. */            
        if ( cudaDeviceSynchronize() != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Clean up. */
        free( data );
        
        }
        
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

    
    int i, j, k, nr_pots, nr_coeffs, nr_tasks, max_coeffs = 0, c1 ,c2;
    int did, *cellsorts;
    int pind[ e->max_type * e->max_type ], pind_bond[ e->max_type * e->max_type ];
    int pind_angle[ e->nr_anglepots ], pind_dihedral[ e->nr_dihedralpots ];
    int *pind_cuda[ engine_maxgpu ], *pind_bond_cuda[ engine_maxgpu ];
    struct space *s = &e->s;
    int nr_devices = e->nr_devices;
    struct potential **pots;
    struct task_cuda *tasks_cuda, *tc, *ts;
    struct task *t;
    float *finger, *coeffs_cuda;
    float cutoff = e->s.cutoff, cutoff2 = e->s.cutoff2, dscale; //, buff[ e->nr_types ];
    cudaArray *cuArray_coeffs[ engine_maxgpu ], *cuArray_pind[ engine_maxgpu ], *cuArray_pind_bond[ engine_maxgpu ];
    cudaChannelFormatDesc channelDesc_int = cudaCreateChannelDesc<int>();
    cudaChannelFormatDesc channelDesc_float = cudaCreateChannelDesc<float>();
    cudaChannelFormatDesc channelDesc_float4 = cudaCreateChannelDesc<float4>();
    float h[3], dim[3], *corig;
    void *dummy[ engine_maxgpu ];
    int *bonds_cuda, *exclusions_cuda;
    int *angles_cuda;
    int *dihedrals_cuda;

    /*Split the space over the available GPUs*/
    engine_split_METIS( e , nr_devices , engine_split_GPU  );
    
    /* Set the coeff properties. */
    tex_coeffs.addressMode[0] = cudaAddressModeClamp;
    tex_coeffs.addressMode[1] = cudaAddressModeClamp;
    tex_coeffs.filterMode = cudaFilterModePoint;
    tex_coeffs.normalized = false;

    /* Set the pind properties. */
    tex_pind.addressMode[0] = cudaAddressModeClamp;
    tex_pind.filterMode = cudaFilterModePoint;
    tex_pind.normalized = false;
    
    /* Allocate the pots array. */
    if ( ( pots = (struct potential **)malloc( sizeof(void *) * ( e->nr_types * (e->nr_types + 1 ) + e->nr_anglepots + e->nr_dihedralpots ) ) ) == NULL )
        return error(engine_err_malloc);

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
    
    /* Start by identifying the unique bonded/non-bonded potentials in the engine. */
    for ( i = 0 ; i < e->max_type * e->max_type ; i++ ) {
    
        /* Skip if there is no non-bonded potential or no parts of this type. */
        if ( e->p[i] != NULL ) {
            
            /* Check this potential against previous potentials. */
            for ( j = 0 ; j < nr_pots && e->p[i] != pots[j] ; j++ );
            if ( j == nr_pots ) {

                /* Store this potential and the number of coefficient entries it has. */
                pots[nr_pots] = e->p[i];
                nr_pots += 1;
                nr_coeffs += e->p[i]->n + 1;
                if ( e->p[i]->n + 1 > max_coeffs )
                    max_coeffs = e->p[i]->n + 1;

                }
            pind[i] = j;
            
            }
        else
            pind[i] = 0;
    
        /* Skip if there is no bonded potential or no parts of this type. */
        if ( e->p_bond[i] != NULL ) {
            
            /* Check this potential against previous potentials. */
            for ( j = 0 ; j < nr_pots && e->p_bond[i] != pots[j] ; j++ );
            if ( j == nr_pots ) {

                /* Store this potential and the number of coefficient entries it has. */
                pots[nr_pots] = e->p_bond[i];
                nr_pots += 1;
                nr_coeffs += e->p_bond[i]->n + 1;
                if ( e->p_bond[i]->n + 1 > max_coeffs )
                    max_coeffs = e->p_bond[i]->n + 1;

                }
            pind_bond[i] = j;
            
            }
        else
            pind_bond[i] = 0;
    
        }
        
    /* Collect the angular potentials. */
    for ( i = 0 ; i < e->nr_anglepots ; i++ ) {
        pots[nr_pots] = e->p_angle[i];
        pind_angle[i] = nr_pots;
        nr_pots += 1;
        nr_coeffs += e->p_angle[i]->n + 1;
        if ( e->p_angle[i]->n + 1 > max_coeffs )
            max_coeffs = e->p_angle[i]->n + 1;
        }
        
    /* Collect the angular potentials. */
    for ( i = 0 ; i < e->nr_dihedralpots ; i++ ) {
        pots[nr_pots] = e->p_dihedral[i];
        pind_dihedral[i] = nr_pots;
        nr_pots += 1;
        nr_coeffs += e->p_dihedral[i]->n + 1;
        if ( e->p_dihedral[i]->n + 1 > max_coeffs )
            max_coeffs = e->p_dihedral[i]->n + 1;
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
    printf( "engine_cuda_load: packed %i potentials with %i coefficient chunks (%i kB).\n" , nr_pots , max_coeffs , (int)(sizeof(float4)*(2*max_coeffs+2)*nr_pots)/1024 ); fflush(stdout);
    free( pots );
        
    /* Bind the potential coefficients to a texture. */
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMallocArray( &cuArray_coeffs[did] , &channelDesc_float4 , 2*max_coeffs + 2 , nr_pots ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToArray( cuArray_coeffs[did] , 0 , 0 , coeffs_cuda , sizeof(float4) * (2*max_coeffs + 2) * nr_pots , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
    free( coeffs_cuda );
    
    /* Copy the bond lists to the device. */
    if ( ( bonds_cuda = (int *)malloc( sizeof(int) * e->nr_bonds * 2 ) ) == NULL )
        return error(engine_err_malloc);
    for ( k = 0 ; k < e->nr_bonds ; k++ ) {
        bonds_cuda[ k*2 + 0 ] = e->bonds[k].i;
        bonds_cuda[ k*2 + 1 ] = e->bonds[k].j;
        }
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(int) * e->nr_bonds * 2 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( dummy[did] , bonds_cuda , sizeof(int) * e->nr_bonds * 2 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_bonds , &dummy[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_nrbonds , &e->nr_bonds , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
    free( bonds_cuda );
    
    /* Copy the exclusion lists to the device. */
    if ( ( exclusions_cuda = (int *)malloc( sizeof(int) * e->nr_exclusions * 2 ) ) == NULL )
        return error(engine_err_malloc);
    for ( k = 0 ; k < e->nr_exclusions ; k++ ) {
        exclusions_cuda[ k*2 + 0 ] = e->exclusions[k].i;
        exclusions_cuda[ k*2 + 1 ] = e->exclusions[k].j;
        }
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(int) * e->nr_exclusions * 2 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( dummy[did] , exclusions_cuda , sizeof(int) * e->nr_exclusions * 2 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_exclusions , &dummy[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_nrexclusions , &e->nr_exclusions , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
    free( exclusions_cuda );
    
    /* Copy the angle lists to the device. */
    if ( ( angles_cuda = (int *)malloc( sizeof(int) * e->nr_angles * 4 ) ) == NULL )
        return error(engine_err_malloc);
    for ( k = 0 ; k < e->nr_angles ; k++ ) {
        angles_cuda[ k*4 + 0 ] = e->angles[k].i;
        angles_cuda[ k*4 + 1 ] = e->angles[k].j;
        angles_cuda[ k*4 + 2 ] = e->angles[k].k;
        angles_cuda[ k*4 + 3 ] = pind_angle[ e->angles[k].pid ];
        }
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(int) * e->nr_angles * 4 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( dummy[did] , angles_cuda , sizeof(int) * e->nr_angles * 4 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_angles , &dummy[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_nrangles , &e->nr_angles , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
    free( angles_cuda );
    
    /* Copy the dihedral lists to the device. */
    if ( ( dihedrals_cuda = (int *)malloc( sizeof(int) * e->nr_dihedrals * 5 ) ) == NULL )
        return error(engine_err_malloc);
    for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {
        dihedrals_cuda[ k*5 + 0 ] = e->dihedrals[k].i;
        dihedrals_cuda[ k*5 + 1 ] = e->dihedrals[k].j;
        dihedrals_cuda[ k*5 + 2 ] = e->dihedrals[k].k;
        dihedrals_cuda[ k*5 + 3 ] = e->dihedrals[k].l;
        dihedrals_cuda[ k*5 + 4 ] = pind_dihedral[ e->dihedrals[k].pid ];
        }
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(int) * e->nr_dihedrals * 5 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( dummy[did] , dihedrals_cuda , sizeof(int) * e->nr_dihedrals * 5 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_dihedrals , &dummy[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_nrdihedrals , &e->nr_dihedrals , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
    free( dihedrals_cuda );
    
    /* Allocate the partlist on the device. */
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->partlist_cuda[did] , sizeof(int) * s->nr_parts ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_partlist , &e->partlist_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->celllist_cuda[did] , sizeof(int) * s->nr_parts ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_celllist , &e->celllist_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
    
    /* Copy the cell edge lengths to the device. */
    h[0] = s->h[0]*s->span[0];
    h[1] = s->h[1]*s->span[1];
    h[2] = s->h[2]*s->span[2];
    dim[0] = s->dim[0]; dim[1] = s->dim[1]; dim[2] = s->dim[2];
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_h , h , sizeof(float) * 3 , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_dim , dim , sizeof(float) * 3 , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
        
    /* Copy the cell origins to the device. */
    if ( ( corig = (float *)malloc( sizeof(float) * s->nr_cells * 3 ) ) == NULL )
        return error(engine_err_malloc);
    for ( i = 0 ; i < s->nr_cells ; i++ ) {
        corig[ 3*i + 0 ] = s->cells[i].origin[0];
        corig[ 3*i + 1 ] = s->cells[i].origin[1];
        corig[ 3*i + 2 ] = s->cells[i].origin[2];
        }
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(float) * s->nr_cells * 3 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( dummy[did] , corig , sizeof(float) * s->nr_cells * 3 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_corig , &dummy[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
    free( corig );
    
    /* Copy the bonded/non-bonded potential indices to the device. */
    for ( did = 0 ;did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMallocArray( &cuArray_pind[did] , &channelDesc_int , e->max_type * e->max_type , 1 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToArray( cuArray_pind[did] , 0 , 0 , pind , sizeof(int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMallocArray( &cuArray_pind_bond[did] , &channelDesc_int , e->max_type * e->max_type , 1 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToArray( cuArray_pind_bond[did] , 0 , 0 , pind_bond , sizeof(int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
    
    /* Store pind and pind_bond as a constants too. */
    for ( did = 0 ;did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &pind_cuda[did] , sizeof(unsigned int) * e->max_type * e->max_type ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( pind_cuda[did] , pind , sizeof(unsigned int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_pind , &pind_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &pind_bond_cuda[did] , sizeof(unsigned int) * e->max_type * e->max_type ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( pind_bond_cuda[did] , pind_bond , sizeof(unsigned int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_pind_bond , &pind_bond_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
            
    /* Bind the textures on the device. */
    for ( did = 0 ;did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaBindTextureToArray( tex_coeffs , cuArray_coeffs[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaBindTextureToArray( tex_pind , cuArray_pind[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaBindTextureToArray( tex_pind_bond , cuArray_pind_bond[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
        
        
    /* Set the constant pointer to the null potential and other useful values. */
    dscale = ((float)SHRT_MAX) / ( 3.0 * sqrt( s->h[0]*s->h[0]*s->span[0]*s->span[0] + s->h[1]*s->h[1]*s->span[1]*s->span[1] + s->h[2]*s->h[2]*s->span[2]*s->span[2] ) );
    for ( did = 0 ;did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_cutoff2 , &cutoff2 , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_cutoff , &cutoff , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_maxdist , &cutoff , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_maxtype , &(e->max_type) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_dscale , &dscale , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_nr_cells , &(s->nr_cells) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
        
    /* Allocate and fill the task list. */
    int tot_tasks = s->nr_tasks;
    if ( e->flags & engine_flag_parbonded ) {
        tot_tasks += ( e->nr_bonds + ( cuda_frame * cuda_frames_per_block - 1 ) ) / ( cuda_frame * cuda_frames_per_block );
        tot_tasks += ( e->nr_angles + ( cuda_frame * cuda_frames_per_block - 1 ) ) / ( cuda_frame * cuda_frames_per_block );
        tot_tasks += ( e->nr_dihedrals + ( cuda_frame * cuda_frames_per_block - 1 ) ) / ( cuda_frame * cuda_frames_per_block );
        tot_tasks += ( e->nr_exclusions + ( cuda_frame * cuda_frames_per_block - 1 ) ) / ( cuda_frame * cuda_frames_per_block );
        }
    if ( ( tasks_cuda = (struct task_cuda *)malloc( sizeof(struct task_cuda) * tot_tasks ) ) == NULL )
        return error(engine_err_malloc);
    if ( ( cellsorts = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL )
        return error(engine_err_malloc);
    for ( did = 0 ; did < nr_devices ; did++ ) {
    
        /* Allocate the list of cells local to the device. */
	    if( (e->cells_cuda_local[did] = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL)
	        return error(engine_err_malloc);
        e->cells_cuda_nr[did]=0;
        
        /* Set the device. */
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Select the tasks for each device ID. */  
        for ( nr_tasks = 0 , i = 0 ; i < s->nr_tasks ; i++ ) {
            
            /* Get local pointers. */
            t = &s->tasks[i];
            tc = &tasks_cuda[nr_tasks];
            
            /* Skip bonded tasks. */
            if ( t->type == task_type_bonded )
                continue;
        
            /* Skip pairs and self with wrong cid, keep all sorts. */
            if ( ( t->type == task_type_pair && e->s.cells[t->i].GPUID != did  /*t->i % nr_devices != did */) ||
                 ( t->type == task_type_self && e->s.cells[t->i].GPUID != did /*e->s.cells[t->i].loc[1] < e->s.cdim[1] / e->nr_devices * (did + 1) && e->s.cells[t->i].loc[1] >= e->s.cdim[1] / e->nr_devices * did t->i % e->nr_devices != did*/ ) )
                continue;
            
            /* Copy the data. */
            tc->type = t->type;
            tc->subtype = t->subtype;
            tc->wait = 0;
            tc->flags = t->flags;
            tc->i = t->i;
            tc->j = t->j;
            tc->nr_unlock = 0;
            
            /* Remember which task sorts which cell. */
            if ( t->type == task_type_sort ) {
                tc->flags = 0;
                cellsorts[ t->i ] = nr_tasks;
                }

            /*Add the cell to list of cells for this GPU if needed*/
            c1=1; c2=1;
            for(int i = 0; i < e->cells_cuda_nr[did] ; i++ ) {
            
                /* Check cell is valid */
                if(t->i < 0 || t->i == e->cells_cuda_local[did][i])
                    c1 = 0;
                if(t->j < 0 || t->j == e->cells_cuda_local[did][i])
                    c2 = 0;
                   }
                   
            if( c1 )
                e->cells_cuda_local[did][e->cells_cuda_nr[did]++] = t->i;
            if( c2 )
                e->cells_cuda_local[did][e->cells_cuda_nr[did]++] = t->j;                    
                
            /* Add one task. */
            nr_tasks += 1;
        
            }

        /* Link each pair task to its sorts. */
        for ( i = 0 ; i < nr_tasks ; i++ ) {
            tc = &tasks_cuda[i];
            if ( tc->type == task_type_pair ) {
                ts = &tasks_cuda[ cellsorts[ tc->i ] ];
                ts->flags |= (1 << tc->flags);
                ts->unlock[ ts->nr_unlock ] = i;
                ts->nr_unlock += 1;
                ts = &tasks_cuda[ cellsorts[ tc->j ] ];
                ts->flags |= (1 << tc->flags);
                ts->unlock[ ts->nr_unlock ] = i;
                ts->nr_unlock += 1;
                }
            }
        
        /* Set the waits. */
        for ( i = 0 ; i < nr_tasks ; i++ )
            for ( k = 0 ; k < tasks_cuda[i].nr_unlock ; k++ )
                tasks_cuda[ tasks_cuda[i].unlock[k] ].wait += 1;
                
        if ( e->flags & engine_flag_parbonded ) {
                
            /* Generate the bond tasks. */
            int count = ( e->nr_bonds + ( cuda_bondspertask - 1 ) ) / cuda_bondspertask;
            for ( k = did ; k < count ; k+= nr_devices) {
                tc = &tasks_cuda[ nr_tasks ];
                nr_tasks += 1;
                tc->type = task_type_bonded;
                tc->subtype = task_subtype_bond;
                tc->i = k*cuda_bondspertask;
                tc->j = ( k < count-1 ) ? cuda_bondspertask : e->nr_bonds - k*cuda_bondspertask ;
                tc->wait = 0;
                tc->nr_unlock = 0;
                }

            /* Generate the angle tasks. */
            count = ( e->nr_angles + ( cuda_bondspertask - 1 ) ) / cuda_bondspertask;
            for ( k = did ; k < count ; k+= nr_devices) {
                tc = &tasks_cuda[ nr_tasks ];
                nr_tasks += 1;
                tc->type = task_type_bonded;
                tc->subtype = task_subtype_angle;
                tc->i = k*cuda_bondspertask;
                tc->j = ( k < count-1 ) ? cuda_bondspertask : e->nr_angles - k*cuda_bondspertask ;
                tc->wait = 0;
                tc->nr_unlock = 0;
                }

            /* Generate the dihedral tasks. */
            count = ( e->nr_dihedrals + ( cuda_bondspertask - 1 ) ) / cuda_bondspertask;
            for ( k = did ; k < count ; k+= nr_devices) {
                tc = &tasks_cuda[ nr_tasks ];
                nr_tasks += 1;
                tc->type = task_type_bonded;
                tc->subtype = task_subtype_dihedral;
                tc->i = k*cuda_bondspertask;
                tc->j = ( k < count-1 ) ? cuda_bondspertask : e->nr_dihedrals - k*cuda_bondspertask ;
                tc->wait = 0;
                tc->nr_unlock = 0;
                }

            /* Generate the exclusion tasks. */
            count = ( e->nr_exclusions + ( cuda_bondspertask - 1 ) ) / cuda_bondspertask;
            for ( k = did ; k < count ; k+= nr_devices) {
                tc = &tasks_cuda[ nr_tasks ];
                nr_tasks += 1;
                tc->type = task_type_bonded;
                tc->subtype = task_subtype_exclusion;
                tc->i = k*cuda_bondspertask;
                tc->j = ( k < count-1 ) ? cuda_bondspertask : e->nr_exclusions - k*cuda_bondspertask ;
                tc->wait = 0;
                tc->nr_unlock = 0;
                }
               
            }

        /* Allocate and fill the tasks list on the device. */
        if ( cudaMemcpyToSymbol( cuda_nr_tasks , &nr_tasks , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(struct task_cuda) * nr_tasks ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( dummy[did] , tasks_cuda , sizeof(struct task_cuda) * nr_tasks , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_tasks , &dummy[did] , sizeof(struct task_cuda *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Remember the number of tasks. */
        e->nrtasks_cuda[did] = nr_tasks;
            
        }
    
    /* Clean up */
    free( tasks_cuda );
    free( cellsorts );
        
    /* Allocate the sortlists locally and on the device if needed. */
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->sortlists_cuda[did] , sizeof(unsigned int) * s->nr_parts * 13 ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_sortlists , &e->sortlists_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }


    for ( did = 0 ; did < nr_devices ; did++ ) {
        /* Allocate the cell counts and offsets. */
        if ( ( e->counts_cuda_local[did] = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL ||
             ( e->ind_cuda_local[did] = (int *)malloc( sizeof(int) * s->nr_cells ) ) == NULL )
            return error(engine_err_malloc);
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->counts_cuda[did] , sizeof(int) * s->nr_cells ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &e->ind_cuda[did] , sizeof(int) * s->nr_cells ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
        
    /* Allocate and init the taboo list on the device. */
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMalloc( &dummy[did] , sizeof(int) * s->nr_cells ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemset( dummy[did] , 0 , sizeof(int) * s->nr_cells ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_taboo , &dummy[did] , sizeof(int *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }
        
    /* Allocate the particle buffer. */
    #ifdef PARTS_TEX
        if ( ( e->parts_cuda_local = (float4 *)malloc( sizeof( float4 ) * s->nr_cells * 512 ) ) == NULL )
            return error(engine_err_malloc);
    #else
        if ( ( e->parts_cuda_local = (float4 *)malloc( sizeof( float4 ) * s->nr_parts ) ) == NULL )
            return error(engine_err_malloc);
    #endif
    for ( did = 0 ; did < nr_devices ; did++ ) {
        if ( ( e->partlist_cuda_local[did] = (int *)malloc( sizeof(int) * s->nr_parts ) ) == NULL ||
             ( e->celllist_cuda_local[did] = (int *)malloc( sizeof(int) * s->nr_parts ) ) == NULL )
            return error(engine_err_malloc);
        }

    /* Allocate the particle and force data. */
    for ( did = 0 ;did < nr_devices ; did++ ) {
        if ( cudaSetDevice( e->devices[did] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( cuda_nr_parts , &s->nr_parts , sizeof(int *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        #ifdef PARTS_TEX
            tex_parts.addressMode[0] = cudaAddressModeClamp;
            tex_parts.addressMode[1] = cudaAddressModeClamp;
            tex_parts.filterMode = cudaFilterModePoint;
            tex_parts.normalized = false;
            if ( cudaMallocArray( (cudaArray **)&e->cuArray_parts[did] , &channelDesc_float4 , 512 , s->nr_cells ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
            if ( cudaBindTextureToArray( tex_parts , (cudaArray *)e->cuArray_parts[did] ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
        #else
            if ( cudaMalloc( &e->parts_cuda[did] , sizeof( float4 ) * s->nr_parts ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
            if ( cudaMemcpyToSymbol( cuda_parts , &e->parts_cuda[did] , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
        #endif
        if ( cudaMalloc( &e->forces_cuda[did] , sizeof( float ) * 3 * s->nr_parts ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }

    /* Init the pair queue on the device. */
    if ( engine_cuda_queues_load( e ) < 0 )
        return error(engine_err);
        
    /* He's done it! */
    return engine_err_ok;
    
    }
    
    




