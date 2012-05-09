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

/* The mutex for accessing the cell pair list. */
__device__ int cuda_cell_mutex = 0;
__device__ int cuda_barrier = 0;

/* The list of cell pairs. */
__device__ struct cellpair_cuda *cuda_pairs;
__device__ struct celltuple_cuda *cuda_tuples;
__device__ int *cuda_taboo, *cuda_owner;

/* The index of the next free cell pair. */
volatile __device__ int cuda_pair_next = 0;
volatile __device__ int cuda_tuple_next = 0;

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
texture< float , cudaTextureType2D > tex_coeffs;
texture< float4 , cudaTextureType1D > tex_alphas;
texture< int , cudaTextureType1D > tex_offsets;

/* Other textures. */
texture< int , cudaTextureType1D > tex_pind;
texture< int , cudaTextureType2D > tex_diags;

/* Arrays to hold the textures. */
cudaArray *cuda_coeffs, *cuda_alphas, *cuda_offsets, *cuda_pind, *cuda_diags;

/* The potential parameters (hard-wired size for now). */
__constant__ float cuda_eps[ 100 ];
__constant__ float cuda_rmin[ 100 ];

/* The list of fifos to work with. */
__device__ struct fifo_cuda cuda_fifos_in[ cuda_maxblocks ];
__device__ struct fifo_cuda cuda_fifos_out[ cuda_maxblocks ];

/* Use a set of variables to communicate with the outside world. */
__device__ float cuda_fio[32];
__device__ int cuda_io[32];

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
        for ( j = 0 ; j < cuda_memcpy_chunk ; j++ )
            chunk[j] = isource[ (cuda_memcpy_chunk*k+j)*cuda_frame + threadID ];
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
 * @brief Evaluates the given potential at the given point (interpolated) using
 *      texture memory on the device and explicit electrostatics.
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

__device__ inline void potential_eval_cuda_tex_e ( int pid , float q , float r2 , float *e , float *f ) {

    int ind, k;
    float x, ee, eff, r, ir, qir, c[potential_chunk];
    float4 alpha;
    
    TIMER_TIC
    
    /* Get r for the right type. */
    ir = rsqrtf(r2);
    r = r2*ir;
    qir = q*ir;
    
    /* compute the interval index */
    alpha = tex1D( tex_alphas , pid );
    if ( ( ind = alpha.x + r * ( alpha.y + r * alpha.z ) ) < 0 )
        ind = 0;
    ind += alpha.w;
    
    /* pre-load the coefficients. */
    #pragma unroll
    for ( k = 0 ; k < potential_chunk ; k++ )
        c[k] = tex2D( tex_coeffs , k , ind );
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    eff = c[2];
    ee = c[2] * x + c[3];
    #pragma unroll
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        }

    /* store the result */
    *e = ee + qir;
    *f = ( eff * c[1] + qir ) * ir;
        
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

__device__ inline void potential_eval_cuda_tex ( int pid , float r2 , float *e , float *f ) {

    int ind, k;
    float x, ee, eff, r, ir, c[potential_chunk];
    float4 alpha;
    
    TIMER_TIC
    
    /* Get r for the right type. */
    ir = rsqrtf(r2);
    r = r2*ir;
    
    /* compute the interval index */
    alpha = tex1D( tex_alphas , pid );
    if ( ( ind = alpha.x + r * ( alpha.y + r * alpha.z ) ) < 0 )
        ind = 0;
    ind += alpha.w;
    
    /* pre-load the coefficients. */
    #pragma unroll
    for ( k = 0 ; k < potential_chunk ; k++ )
        c[k] = tex2D( tex_coeffs , k , ind );
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    eff = c[2];
    ee = c[2] * x + c[3];
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

    int k;
    int4 ind;
    float4 x, ee, eff, r, ir, c[potential_chunk], a[4];
    
    TIMER_TIC
    
    /* Get r for the right type. */
    ir.x = rsqrtf(r2.x);
    ir.y = rsqrtf(r2.y);
    ir.z = rsqrtf(r2.z);
    ir.w = rsqrtf(r2.w);
    r = r2*ir;
    
    /* compute the interval index */
    a[0] = tex1D( tex_alphas , pid.x );
    a[1] = tex1D( tex_alphas , pid.y );
    a[2] = tex1D( tex_alphas , pid.z );
    a[3] = tex1D( tex_alphas , pid.w );
    ind.x = a[0].w + max( 0 , (int)( a[0].x + r.x * ( a[0].y + r.x * a[0].z ) ) );
    ind.y = a[1].w + max( 0 , (int)( a[1].x + r.y * ( a[1].y + r.y * a[1].z ) ) );
    ind.z = a[2].w + max( 0 , (int)( a[2].x + r.z * ( a[2].y + r.z * a[2].z ) ) );
    ind.w = a[3].w + max( 0 , (int)( a[3].x + r.w * ( a[3].y + r.w * a[3].z ) ) );
    
    /* pre-load the coefficients. */
    #pragma unroll
    for ( k = 0 ; k < potential_chunk ; k++ )
        c[k] = make_float4( tex2D( tex_coeffs , k , ind.x ) , tex2D( tex_coeffs , k , ind.y ) , tex2D( tex_coeffs , k , ind.z ) , tex2D( tex_coeffs , k , ind.w ) );
        
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    eff = c[2];
    ee = c[2] * x + c[3];
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
 * @brief Evaluates the given potential at the given point (explicit).
 *
 * @param type_i The type ID of the first particle.
 * @param type_j The type ID of the second particle
 * @param r2 The radius at which it is to be evaluated, squared.
 * @param e Pointer to a floating-point value in which to store the
 *      interaction energy.
 * @param f Pointer to a floating-point value in which to store the
 *      magnitude of the interaction force divided by r.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL.
 */

__device__ inline void potential_eval_cuda_expl ( int type_i , int type_j , float q , float r2 , float *e , float *f ) {

    float eps, ir, rminr, rminr2, rminr6;
    
    TIMER_TIC
    
    /* Get the inverse of r. */
    ir = rsqrtf(r2);
    
    /* Get eps. */
    eps = cuda_eps[ type_i ] * cuda_eps[ type_j ];
    
    /* Get the powers of rminr. */
    rminr = ( cuda_rmin[ type_i ] + cuda_rmin[ type_j ] ) * ir;
    rminr2 = rminr * rminr;
    rminr6 = rminr2 * rminr2 * rminr2;
    
    /* Compute the energy. */
    *e = eps * rminr6 * ( rminr6 - 2 ) + potential_escale * ir;
    
    /* Compute the force. */
    *f = 12.0f * eps * rminr6 * ir * ( 1 - rminr6 ) - potential_escale / r2;
        
    TIMER_TOC(tid_potential)
        
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param iparts_i Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param iparts_j Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param parts_i Part buffer in local memory.
 * @param parts_j Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
__device__ void runner_dopair_cuda ( struct part_cuda *parts_i , int count_i , struct part_cuda *parts_j , int count_j, float *pshift ) {

    int k, pid, pjd, ind, wrap_i, threadID;
    int pjoff;
    struct part_cuda *pi, *pj;
    struct part_cuda *temp;
    #if defined(USETEX_E) || defined(EXPLPOT)
        float qj, q;
    #endif
    #if defined(USETEX) || defined(USETEX_E)
        int pot;
    #elif !defined(EXPLPOT)
        struct potential *pot;
    #endif
    float epot = 0.0f, dx[3], pjx[3], pjf[3], shift[3], r2, w;
    float ee = 0.0f, eff = 0.0f;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i < ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        temp = parts_i; parts_i = parts_j; parts_j = temp;
        k = count_i; count_i = count_j; count_j = k;
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
        pj = &parts_j[pjd];
        pjoff = pj->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pjx[k] = pj->x[k] + shift[k];
            pjf[k] = 0.0f;
            }
        #if defined(USETEX_E) || defined(EXPLPOT)
        qj = pj->q;
        #endif
        
        /* Loop over the particles in cell_i. */
        for ( ind = 0 ; ind < wrap_i ; ind++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pid = ind + threadID ) >= wrap_i )
                pid -= wrap_i;
            if ( pid < count_i ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                pi = &parts_i[ pid ];
                // printf( "runner_dopair_cuda: doing pair [%i,%i].\n" , pjd , ind );

                /* Compute the radius between pi and pj. */
                for ( r2 = 0.0f , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pi->x[k] - pjx[k];
                    r2 += dx[k] * dx[k];
                    }

                /* Set the null potential if anything is bad. */
                #ifdef USETEX_E
                if ( r2 < cuda_cutoff2 && ( ( pot = tex1D( tex_pind , pjoff + pi->type ) ) != 0 || ( q = qj*pi->q ) != 0.0f ) ) {
                #elif defined(USETEX)
                if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pjoff + pi->type ) ) != 0 ) {
                #elif defined(EXPLPOT)
                if ( r2 < cuda_cutoff2 ) {
                #else
                if ( r2 < cuda_cutoff2 && ( pot = cuda_p[ pjoff + pi->type ] ) != NULL ) {
                #endif

                    // atomicAdd( &cuda_pairs_done , 1 );
                
                    /* Interact particles pi and pj. */
                    #ifdef USETEX_E
                    potential_eval_cuda_tex_e( pot , q , r2 , &ee , &eff );
                    #elif defined(USETEX)
                    potential_eval_cuda_tex( pot , r2 , &ee , &eff );
                    #elif defined(EXPLPOT)
                    potential_eval_cuda_expl( pi->type , pj->type , qj*pi->q , r2 , &ee , &eff );
                    #else
                    potential_eval_cuda( pot , r2 , &ee , &eff );
                    #endif

                    /* Store the interaction force and energy. */
                    epot += ee;
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff * dx[k];
                        pi->f[k] -= w;
                        pjf[k] += w;
                        }

                    /* Sync the shared memory values. */
                    __threadfence_block();
                
                    } /* in range and potential. */

                } /* valid pid? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            pj->f[k] += pjf[k];

        /* Sync the shared memory values. */
        __threadfence_block();
            
        } /* loop over the particles in cell_j. */
        
    TIMER_TOC(tid_pair)
        
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param iparts_i Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param iparts_j Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param parts_i Part buffer in local memory.
 * @param parts_j Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
__device__ void runner_dopair4_cuda ( struct part_cuda *parts_i , int count_i , struct part_cuda *parts_j , int count_j, float *pshift ) {

    int k, pjd, ind, wrap_i, threadID;
    int pjoff;
    struct part_cuda *pi[4], *pj;
    struct part_cuda *temp;
    int4 pot, pid, valid;
    float4 r2, ee, eff;
    float epot = 0.0f, dx[12], pjx[3], pjf[3], shift[3], w;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i < ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        temp = parts_i; parts_i = parts_j; parts_j = temp;
        k = count_i; count_i = count_j; count_j = k;
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
        pj = &parts_j[pjd];
        pjoff = pj->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pjx[k] = pj->x[k] + shift[k];
            pjf[k] = 0.0f;
            }
        #if defined(USETEX_E) || defined(EXPLPOT)
        qj = pj->q;
        #endif
        
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
            pi[0] = ( valid.x = ( pid.x < count_i ) ) ? &parts_i[ pid.x ] : pj;
            pi[1] = ( valid.y = ( pid.y < count_i ) && ( ind + 1 < wrap_i ) ) ? &parts_i[ pid.y ] : pj;
            pi[2] = ( valid.z = ( pid.z < count_i ) && ( ind + 2 < wrap_i ) ) ? &parts_i[ pid.z ] : pj;
            pi[3] = ( valid.w = ( pid.w < count_i ) && ( ind + 3 < wrap_i ) ) ? &parts_i[ pid.w ] : pj;
            
            /* Compute the pairwise distances. */
            r2 = make_float4( 0.0f );
            #pragma unroll
            for ( k = 0 ; k < 3 ; k++ ) {
                dx[k] = pjx[k] - pi[0]->x[k];
                r2.x += dx[k] * dx[k];
                dx[3+k] = pjx[k] - pi[1]->x[k];
                r2.y += dx[3+k] * dx[3+k];
                dx[6+k] = pjx[k] - pi[2]->x[k];
                r2.z += dx[6+k] * dx[6+k];
                dx[9+k] = pjx[k] - pi[3]->x[k];
                r2.w += dx[9+k] * dx[9+k];
                }
                
            /* Get the potentials. */
            valid.x = ( valid.x && r2.x < cuda_cutoff2 );
            valid.y = ( valid.y && r2.y < cuda_cutoff2 );
            valid.z = ( valid.z && r2.z < cuda_cutoff2 );
            valid.w = ( valid.w && r2.w < cuda_cutoff2 );
            pot.x = valid.x ? tex1D( tex_pind , pjoff + pi[0]->type ) : 0;
            pot.y = valid.y ? tex1D( tex_pind , pjoff + pi[1]->type ) : 0;
            pot.z = valid.z ? tex1D( tex_pind , pjoff + pi[2]->type ) : 0;
            pot.w = valid.w ? tex1D( tex_pind , pjoff + pi[3]->type ) : 0;
            
            /* Compute the interaction. */
            potential_eval4_cuda_tex( pot , r2 , &ee , &eff );
            
            /* Store the interaction energy. */
            epot += ee.x + ee.y + ee.z + ee.w;
            
            /* Update the forces. */
            if ( valid.x ) {
                pjf[0] -= ( w = eff.x * dx[0] ); pi[0]->f[0] += w;
                pjf[1] -= ( w = eff.x * dx[1] ); pi[0]->f[1] += w;
                pjf[2] -= ( w = eff.x * dx[2] ); pi[0]->f[2] += w;
                }
            __threadfence_block();
            if ( valid.y ) {
                pjf[0] -= ( w = eff.y * dx[3] ); pi[1]->f[0] += w;
                pjf[1] -= ( w = eff.y * dx[4] ); pi[1]->f[1] += w;
                pjf[2] -= ( w = eff.y * dx[5] ); pi[1]->f[2] += w;
                }
            __threadfence_block();
            if ( valid.z ) {
                pjf[0] -= ( w = eff.z * dx[6] ); pi[2]->f[0] += w;
                pjf[1] -= ( w = eff.z * dx[7] ); pi[2]->f[1] += w;
                pjf[2] -= ( w = eff.z * dx[8] ); pi[2]->f[2] += w;
                }
            __threadfence_block();
            if ( valid.w ) {
                pjf[0] -= ( w = eff.w * dx[9] ); pi[3]->f[0] += w;
                pjf[1] -= ( w = eff.w * dx[10] ); pi[3]->f[1] += w;
                pjf[2] -= ( w = eff.w * dx[11] ); pi[3]->f[2] += w;
                }
            __threadfence_block();
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            pj->f[k] += pjf[k];

        /* Sync the shared memory values. */
        __threadfence_block();
            
        } /* loop over the particles in cell_j. */
        
    TIMER_TOC(tid_pair)
        
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param iparts_i Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param iparts_j Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param parts_i Part buffer in local memory.
 * @param parts_j Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
__device__ void runner_dopair_sorted_verlet_cuda ( struct part_cuda *parts_i , int count_i , struct part_cuda *parts_j , int count_j , float *pshift , int verlet_rebuild , unsigned int *sortlist ) {

    int k, j, i, ind, jnd, pid, pjd, pjdid, threadID, wrap, cj;
    int pioff;
    unsigned int swap_i, dmaxdist;
    struct part_cuda *pi, *pj;
    struct part_cuda *temp;
    #if defined(USETEX_E) || defined(EXPLPOT)
        float qi, q;
    #endif
    #if defined(USETEX) || defined(USETEX_E)
        int pot;
    #elif !defined(EXPLPOT)
        struct potential *pot;
    #endif
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f, nshift, inshift;
    float dx[3], pix[3], pif[3], shift[3], shiftn[3];
    __shared__ unsigned int sort_i[ cuda_maxparts ], sort_j[ cuda_maxparts ];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i > ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        temp = parts_i; parts_i = parts_j; parts_j = temp;
        k = count_i; count_i = count_j; count_j = k;
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
        for ( k = threadID ; k < count_i ; k += cuda_frame )
            sort_i[k] = ( k << 16 ) |
                (unsigned int)( cuda_dscale * (nshift + parts_i[k].x[0]*shiftn[0] + parts_i[k].x[1]*shiftn[1] + parts_i[k].x[2]*shiftn[2]) );
        for ( k = threadID ; k < count_j ; k += cuda_frame )
            sort_j[k] = ( k << 16 ) | 
                (unsigned int)( cuda_dscale * (nshift + (shift[0]+parts_j[k].x[0])*shiftn[0] + (shift[1]+parts_j[k].x[1])*shiftn[1] + (shift[2]+parts_j[k].x[2])*shiftn[2]) );
            
        /* Make sure all the memory is in the right place. */
        __threadfence_block();
        
        /* Sort using normalized bitonic sort. */
        for ( k = 1 ; k < count_i ; k *= 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_i ; i += cuda_frame ) {
                jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
                if ( jnd < count_i && ( sort_i[ind] & 0xffff ) < ( sort_i[jnd] & 0xffff ) ) {
                    swap_i = sort_i[ind]; sort_i[ind] = sort_i[jnd]; sort_i[jnd] = swap_i;
                    }
                }
            __threadfence_block();
            for ( j = k/2 ; j > 0 ; j = j / 2 ) {
                for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_i ; i += cuda_frame ) {
                    jnd = ind + j;
                    if ( ( sort_i[ind] & 0xffff ) < ( sort_i[jnd] & 0xffff ) ) {
                        swap_i = sort_i[ind]; sort_i[ind] = sort_i[jnd]; sort_i[jnd] = swap_i;
                        }
                    }
                __threadfence_block();
                }
            }
        for ( k = 1 ; k < count_j ; k *= 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_j ; i += cuda_frame ) {
                jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
                if ( jnd < count_j && ( sort_j[ind] & 0xffff ) > ( sort_j[jnd] & 0xffff ) ) {
                    swap_i = sort_j[ind]; sort_j[ind] = sort_j[jnd]; sort_j[jnd] = swap_i;
                    }
                }
            __threadfence_block();
            for ( j = k/2 ; j > 0 ; j = j / 2 ) {
                for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_j ; i += cuda_frame ) {
                    jnd = ind + j;
                    if ( ( sort_j[ind] & 0xffff ) > ( sort_j[jnd] & 0xffff ) ) {
                        swap_i = sort_j[ind]; sort_j[ind] = sort_j[jnd]; sort_j[jnd] = swap_i;
                        }
                    }
                __threadfence_block();
                }
            }

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
        pi = &parts_i[ sort_i[pid] >> 16 ];
        pioff = pi->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pix[k] = pi->x[k] - shift[k];
            pif[k] = 0.0f;
            }
        #if defined(USETEX_E) || defined(EXPLPOT)
        qi = pi->q;
        #endif
        
        /* Loop over the particles in cell_i. */
        for ( pjdid = 0 ; pjdid < wrap ; pjdid++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pjd = pjdid + threadID ) >= wrap )
                pjd -= wrap;
            
            /* Do we have a pair? */
            if ( pjd < cj ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                pj = &parts_j[ sort_j[pjd] >> 16 ];

                /* Compute the radius between pi and pj. */
                for ( r2 = 0.0f , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - pj->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* Set the null potential if anything is bad. */
                #ifdef USETEX_E
                if ( r2 < cuda_cutoff2 && ( ( pot = tex1D( tex_pind , pioff + pj->type ) ) != 0 || ( q = qj*pi->q ) != 0.0f ) ) {
                #elif defined(USETEX)
                if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pioff + pj->type ) ) != 0 ) {
                #elif defined(EXPLPOT)
                if ( r2 < cuda_cutoff2 ) {
                #else
                if ( r2 < cuda_cutoff2 && ( pot = cuda_p[ pioff + pj->type ] ) != NULL ) {
                #endif

                    /* printf( "runner_dopair_cuda[%i]: doing pair [%i,%i] with r=%i (d=%i).\n" ,
                        threadID , sort_i[pid].ind , sort_j[pjd].ind , (int)(sqrtf(r2)*1000.0) , (int)((sort_j[pjd].d - sort_i[pid].d)*1000) ); */

                    // atomicAdd( &cuda_pairs_done , 1 );
                    
                    /* Interact particles pi and pj. */
                    #ifdef USETEX_E
                    potential_eval_cuda_tex_e( pot , q , r2 , &ee , &eff );
                    #elif defined(USETEX)
                    potential_eval_cuda_tex( pot , r2 , &ee , &eff );
                    #elif defined(EXPLPOT)
                    potential_eval_cuda_expl( pi->type , pj->type , qi*pj->q , r2 , &ee , &eff );
                    #else
                    potential_eval_cuda( pot , r2 , &ee , &eff );
                    #endif


                    /* Store the interaction force and energy. */
                    epot += ee;
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff * dx[k];
                        pif[k] -= w;
                        pj->f[k] += w;
                        }

                    /* Sync the shared memory values. */
                    __threadfence_block();
                
                    } /* in range and potential. */

                } /* do we have a pair? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            pi->f[k] += pif[k];
    
        /* Sync the shared memory values. */
        __threadfence_block();
        
        } /* loop over the particles in cell_j. */
        
    TIMER_TOC(tid_pair)
    
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param iparts_i Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param iparts_j Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param parts_i Part buffer in local memory.
 * @param parts_j Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
__device__ void runner_dopair_sorted4_verlet_cuda ( struct part_cuda *parts_i , int count_i , struct part_cuda *parts_j , int count_j , float *pshift , int verlet_rebuild , unsigned int *sortlist ) {

    int k, j, i, ind, jnd, pid, pjdid, threadID, wrap, cj;
    int pioff;
    unsigned int swap_i, dmaxdist;
    struct part_cuda *pi, *pj[4];
    struct part_cuda *temp;
    int4 pot, pjd, valid;
    float4 ee, eff, r2;
    float epot = 0.0f, w, nshift, inshift;
    float dx[12], pix[3], pif[3], shift[3], shiftn[3];
    __shared__ unsigned int sort_i[ cuda_maxparts ], sort_j[ cuda_maxparts ];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i > ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        temp = parts_i; parts_i = parts_j; parts_j = temp;
        k = count_i; count_i = count_j; count_j = k;
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
        for ( k = threadID ; k < count_i ; k += cuda_frame )
            sort_i[k] = ( k << 16 ) |
                (unsigned int)( cuda_dscale * (nshift + parts_i[k].x[0]*shiftn[0] + parts_i[k].x[1]*shiftn[1] + parts_i[k].x[2]*shiftn[2]) );
        for ( k = threadID ; k < count_j ; k += cuda_frame )
            sort_j[k] = ( k << 16 ) | 
                (unsigned int)( cuda_dscale * (nshift + (shift[0]+parts_j[k].x[0])*shiftn[0] + (shift[1]+parts_j[k].x[1])*shiftn[1] + (shift[2]+parts_j[k].x[2])*shiftn[2]) );
            
        /* Make sure all the memory is in the right place. */
        __threadfence_block();
        
        /* Sort using normalized bitonic sort. */
        for ( k = 1 ; k < count_i ; k *= 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_i ; i += cuda_frame ) {
                jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
                if ( jnd < count_i && ( sort_i[ind] & 0xffff ) < ( sort_i[jnd] & 0xffff ) ) {
                    swap_i = sort_i[ind]; sort_i[ind] = sort_i[jnd]; sort_i[jnd] = swap_i;
                    }
                }
            __threadfence_block();
            for ( j = k/2 ; j > 0 ; j = j / 2 ) {
                for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_i ; i += cuda_frame ) {
                    jnd = ind + j;
                    if ( ( sort_i[ind] & 0xffff ) < ( sort_i[jnd] & 0xffff ) ) {
                        swap_i = sort_i[ind]; sort_i[ind] = sort_i[jnd]; sort_i[jnd] = swap_i;
                        }
                    }
                __threadfence_block();
                }
            }
        for ( k = 1 ; k < count_j ; k *= 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_j ; i += cuda_frame ) {
                jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
                if ( jnd < count_j && ( sort_j[ind] & 0xffff ) > ( sort_j[jnd] & 0xffff ) ) {
                    swap_i = sort_j[ind]; sort_j[ind] = sort_j[jnd]; sort_j[jnd] = swap_i;
                    }
                }
            __threadfence_block();
            for ( j = k/2 ; j > 0 ; j = j / 2 ) {
                for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_j ; i += cuda_frame ) {
                    jnd = ind + j;
                    if ( ( sort_j[ind] & 0xffff ) > ( sort_j[jnd] & 0xffff ) ) {
                        swap_i = sort_j[ind]; sort_j[ind] = sort_j[jnd]; sort_j[jnd] = swap_i;
                        }
                    }
                __threadfence_block();
                }
            }

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
        pi = &parts_i[ sort_i[pid] >> 16 ];
        pioff = pi->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pix[k] = pi->x[k] - shift[k];
            pif[k] = 0.0f;
            }
        #if defined(USETEX_E) || defined(EXPLPOT)
        qi = pi->q;
        #endif
        
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
            pj[0] = ( valid.x = ( pjd.x < cj ) ) ? &parts_j[ sort_j[pjd.x] >> 16 ] : pi;
            pj[1] = ( valid.y = ( pjd.y < cj ) && ( pjdid + 1 < wrap ) ) ? &parts_j[ sort_j[pjd.y] >> 16 ] : pi;
            pj[2] = ( valid.z = ( pjd.z < cj ) && ( pjdid + 2 < wrap ) ) ? &parts_j[ sort_j[pjd.z] >> 16 ] : pi;
            pj[3] = ( valid.w = ( pjd.w < cj ) && ( pjdid + 3 < wrap ) ) ? &parts_j[ sort_j[pjd.w] >> 16 ] : pi;
            
            /* Compute the pairwise distances. */
            r2 = make_float4( 0.0f );
            #pragma unroll
            for ( k = 0 ; k < 3 ; k++ ) {
                dx[k] = pix[k] - pj[0]->x[k];
                r2.x += dx[k] * dx[k];
                dx[3+k] = pix[k] - pj[1]->x[k];
                r2.y += dx[3+k] * dx[3+k];
                dx[6+k] = pix[k] - pj[2]->x[k];
                r2.z += dx[6+k] * dx[6+k];
                dx[9+k] = pix[k] - pj[3]->x[k];
                r2.w += dx[9+k] * dx[9+k];
                }
                
            /* Get the potentials. */
            valid.x = ( valid.x && r2.x < cuda_cutoff2 );
            valid.y = ( valid.y && r2.y < cuda_cutoff2 );
            valid.z = ( valid.z && r2.z < cuda_cutoff2 );
            valid.w = ( valid.w && r2.w < cuda_cutoff2 );
            pot.x = valid.x ? tex1D( tex_pind , pioff + pj[0]->type ) : 0;
            pot.y = valid.y ? tex1D( tex_pind , pioff + pj[1]->type ) : 0;
            pot.z = valid.z ? tex1D( tex_pind , pioff + pj[2]->type ) : 0;
            pot.w = valid.w ? tex1D( tex_pind , pioff + pj[3]->type ) : 0;
            
            /* Compute the interaction. */
            potential_eval4_cuda_tex( pot , r2 , &ee , &eff );
            
            /* Store the interaction energy. */
            epot += ee.x + ee.y + ee.z + ee.w;
            
            /* Update the particle forces. */
            if ( valid.x ) {
                pif[0] -= ( w = eff.x * dx[0] ); pj[0]->f[0] += w;
                pif[1] -= ( w = eff.x * dx[1] ); pj[0]->f[1] += w;
                pif[2] -= ( w = eff.x * dx[2] ); pj[0]->f[2] += w;
                }
            __threadfence_block();
            if ( valid.y ) {
                pif[0] -= ( w = eff.y * dx[3] ); pj[1]->f[0] += w;
                pif[1] -= ( w = eff.y * dx[4] ); pj[1]->f[1] += w;
                pif[2] -= ( w = eff.y * dx[5] ); pj[1]->f[2] += w;
                }
            __threadfence_block();
            if ( valid.z ) {
                pif[0] -= ( w = eff.z * dx[6] ); pj[2]->f[0] += w;
                pif[1] -= ( w = eff.z * dx[7] ); pj[2]->f[1] += w;
                pif[2] -= ( w = eff.z * dx[8] ); pj[2]->f[2] += w;
                }
            __threadfence_block();
            if ( valid.w ) {
                pif[0] -= ( w = eff.w * dx[9] ); pj[3]->f[0] += w;
                pif[1] -= ( w = eff.w * dx[10] ); pj[3]->f[1] += w;
                pif[2] -= ( w = eff.w * dx[11] ); pj[3]->f[2] += w;
                }
            __threadfence_block();
            
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            pi->f[k] += pif[k];
    
        /* Sync the shared memory values. */
        __threadfence_block();
        
        } /* loop over the particles in cell_j. */
        
    TIMER_TOC(tid_pair)
    
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param iparts_i Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param iparts_j Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param parts_i Part buffer in local memory.
 * @param parts_j Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
__device__ void runner_dopair_sorted_cuda ( struct part_cuda *parts_i , int count_i , struct part_cuda *parts_j , int count_j , float *pshift ) {

    int k, j, i, ind, jnd, pid, pjd, pjdid, threadID, wrap, cj;
    int pioff, dcutoff;
    unsigned int swap_i;
    struct part_cuda *pi, *pj;
    struct part_cuda *temp;
    #if defined(USETEX_E) || defined(EXPLPOT)
        float qi, q;
    #endif
    #if defined(USETEX) || defined(USETEX_E)
        int pot;
    #elif !defined(EXPLPOT)
        struct potential *pot;
    #endif
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f, nshift, inshift;
    float dx[3], pix[3], pif[3], shift[3], shiftn[3];
    __shared__ unsigned int sort_i[ cuda_maxparts ], sort_j[ cuda_maxparts ];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i > ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        temp = parts_i; parts_i = parts_j; parts_j = temp;
        k = count_i; count_i = count_j; count_j = k;
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
    for ( k = threadID ; k < count_i ; k += cuda_frame )
        sort_i[k] = ( k << 16 ) |
            (unsigned int)( cuda_dscale * (nshift + parts_i[k].x[0]*shiftn[0] + parts_i[k].x[1]*shiftn[1] + parts_i[k].x[2]*shiftn[2]) );
    for ( k = threadID ; k < count_j ; k += cuda_frame )
        sort_j[k] = ( k << 16 ) | 
            (unsigned int)( cuda_dscale * (nshift + (shift[0]+parts_j[k].x[0])*shiftn[0] + (shift[1]+parts_j[k].x[1])*shiftn[1] + (shift[2]+parts_j[k].x[2])*shiftn[2]) );
        
    /* Make sure all the memory is in the right place. */
    __threadfence_block();
    
    /* Sort using normalized bitonic sort. */
    for ( k = 1 ; k < count_i ; k *= 2 ) {
        for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_i ; i += cuda_frame ) {
            jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
            if ( jnd < count_i && ( sort_i[ind] & 0xffff ) < ( sort_i[jnd] & 0xffff ) ) {
                swap_i = sort_i[ind]; sort_i[ind] = sort_i[jnd]; sort_i[jnd] = swap_i;
                }
            }
        __threadfence_block();
        for ( j = k/2 ; j > 0 ; j = j / 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_i ; i += cuda_frame ) {
                jnd = ind + j;
                if ( ( sort_i[ind] & 0xffff ) < ( sort_i[jnd] & 0xffff ) ) {
                    swap_i = sort_i[ind]; sort_i[ind] = sort_i[jnd]; sort_i[jnd] = swap_i;
                    }
                }
            __threadfence_block();
            }
        }
    for ( k = 1 ; k < count_j ; k *= 2 ) {
        for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_j ; i += cuda_frame ) {
            jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
            if ( jnd < count_j && ( sort_j[ind] & 0xffff ) > ( sort_j[jnd] & 0xffff ) ) {
                swap_i = sort_j[ind]; sort_j[ind] = sort_j[jnd]; sort_j[jnd] = swap_i;
                }
            }
        __threadfence_block();
        for ( j = k/2 ; j > 0 ; j = j / 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_j ; i += cuda_frame ) {
                jnd = ind + j;
                if ( ( sort_j[ind] & 0xffff ) > ( sort_j[jnd] & 0xffff ) ) {
                    swap_i = sort_j[ind]; sort_j[ind] = sort_j[jnd]; sort_j[jnd] = swap_i;
                    }
                }
            __threadfence_block();
            }
        }
        
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
        pi = &parts_i[ sort_i[pid] >> 16 ];
        pioff = pi->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pix[k] = pi->x[k] - shift[k];
            pif[k] = 0.0f;
            }
        #if defined(USETEX_E) || defined(EXPLPOT)
        qi = pi->q;
        #endif
        
        /* Loop over the particles in cell_i. */
        for ( pjdid = 0 ; pjdid < wrap ; pjdid++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pjd = pjdid + threadID ) >= wrap )
                pjd -= wrap;
            
            /* Do we have a pair? */
            if ( pjd < cj ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                pj = &parts_j[ sort_j[pjd] >> 16 ];

                /* Compute the radius between pi and pj. */
                for ( r2 = 0.0f , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pix[k] - pj->x[k];
                    r2 += dx[k] * dx[k];
                    }
                    
                /* Set the null potential if anything is bad. */
                #ifdef USETEX_E
                if ( r2 < cuda_cutoff2 && ( ( pot = tex1D( tex_pind , pioff + pj->type ) ) != 0 || ( q = qj*pi->q ) != 0.0f ) ) {
                #elif defined(USETEX)
                if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pioff + pj->type ) ) != 0 ) {
                #elif defined(EXPLPOT)
                if ( r2 < cuda_cutoff2 ) {
                #else
                if ( r2 < cuda_cutoff2 && ( pot = cuda_p[ pioff + pj->type ] ) != NULL ) {
                #endif

                    /* printf( "runner_dopair_cuda[%i]: doing pair [%i,%i] with r=%i (d=%i).\n" ,
                        threadID , sort_i[pid].ind , sort_j[pjd].ind , (int)(sqrtf(r2)*1000.0) , (int)((sort_j[pjd].d - sort_i[pid].d)*1000) ); */

                    // atomicAdd( &cuda_pairs_done , 1 );
                    
                    /* Interact particles pi and pj. */
                    #ifdef USETEX_E
                    potential_eval_cuda_tex_e( pot , q , r2 , &ee , &eff );
                    #elif defined(USETEX)
                    potential_eval_cuda_tex( pot , r2 , &ee , &eff );
                    #elif defined(EXPLPOT)
                    potential_eval_cuda_expl( pi->type , pj->type , qi*pj->q , r2 , &ee , &eff );
                    #else
                    potential_eval_cuda( pot , r2 , &ee , &eff );
                    #endif


                    /* Store the interaction force and energy. */
                    epot += ee;
                    for ( k = 0 ; k < 3 ; k++ ) {
                        w = eff * dx[k];
                        pif[k] -= w;
                        pj->f[k] += w;
                        }

                    /* Sync the shared memory values. */
                    __threadfence_block();
                
                    } /* in range and potential. */

                } /* do we have a pair? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            pi->f[k] += pif[k];
    
        /* Sync the shared memory values. */
        __threadfence_block();
        
        } /* loop over the particles in cell_j. */
    
    TIMER_TOC(tid_pair)
    
    }


/**
 * @brief Compute the pairwise interactions for the given pair on a CUDA device.
 *
 * @param iparts_i Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param iparts_j Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 * @param parts_i Part buffer in local memory.
 * @param parts_j Part buffer in local memory.
 *
 * @sa #runner_dopair.
 */
 
__device__ void runner_dopair_sorted4_cuda ( struct part_cuda *parts_i , int count_i , struct part_cuda *parts_j , int count_j , float *pshift ) {

    int k, j, i, ind, jnd, pid, pjdid, threadID, wrap, cj;
    int pioff, dcutoff;
    unsigned int swap_i;
    struct part_cuda *pi, *pj[4];
    struct part_cuda *temp;
    int4 pot, pjd, valid;
    float4 ee, eff, r2;
    float epot = 0.0f, w, nshift, inshift;
    float dx[12], pix[3], pif[3], shift[3], shiftn[3];
    __shared__ unsigned int sort_i[ cuda_maxparts ], sort_j[ cuda_maxparts ];
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i > ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        temp = parts_i; parts_i = parts_j; parts_j = temp;
        k = count_i; count_i = count_j; count_j = k;
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
    for ( k = threadID ; k < count_i ; k += cuda_frame )
        sort_i[k] = ( k << 16 ) |
            (unsigned int)( cuda_dscale * (nshift + parts_i[k].x[0]*shiftn[0] + parts_i[k].x[1]*shiftn[1] + parts_i[k].x[2]*shiftn[2]) );
    for ( k = threadID ; k < count_j ; k += cuda_frame )
        sort_j[k] = ( k << 16 ) | 
            (unsigned int)( cuda_dscale * (nshift + (shift[0]+parts_j[k].x[0])*shiftn[0] + (shift[1]+parts_j[k].x[1])*shiftn[1] + (shift[2]+parts_j[k].x[2])*shiftn[2]) );
        
    /* Make sure all the memory is in the right place. */
    __threadfence_block();
    
    /* Sort using normalized bitonic sort. */
    for ( k = 1 ; k < count_i ; k *= 2 ) {
        for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_i ; i += cuda_frame ) {
            jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
            if ( jnd < count_i && ( sort_i[ind] & 0xffff ) < ( sort_i[jnd] & 0xffff ) ) {
                swap_i = sort_i[ind]; sort_i[ind] = sort_i[jnd]; sort_i[jnd] = swap_i;
                }
            }
        __threadfence_block();
        for ( j = k/2 ; j > 0 ; j = j / 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_i ; i += cuda_frame ) {
                jnd = ind + j;
                if ( ( sort_i[ind] & 0xffff ) < ( sort_i[jnd] & 0xffff ) ) {
                    swap_i = sort_i[ind]; sort_i[ind] = sort_i[jnd]; sort_i[jnd] = swap_i;
                    }
                }
            __threadfence_block();
            }
        }
    for ( k = 1 ; k < count_j ; k *= 2 ) {
        for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_j ; i += cuda_frame ) {
            jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
            if ( jnd < count_j && ( sort_j[ind] & 0xffff ) > ( sort_j[jnd] & 0xffff ) ) {
                swap_i = sort_j[ind]; sort_j[ind] = sort_j[jnd]; sort_j[jnd] = swap_i;
                }
            }
        __threadfence_block();
        for ( j = k/2 ; j > 0 ; j = j / 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_j ; i += cuda_frame ) {
                jnd = ind + j;
                if ( ( sort_j[ind] & 0xffff ) > ( sort_j[jnd] & 0xffff ) ) {
                    swap_i = sort_j[ind]; sort_j[ind] = sort_j[jnd]; sort_j[jnd] = swap_i;
                    }
                }
            __threadfence_block();
            }
        }
        
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
        pi = &parts_i[ sort_i[pid] >> 16 ];
        pioff = pi->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pix[k] = pi->x[k] - shift[k];
            pif[k] = 0.0f;
            }
        #if defined(USETEX_E) || defined(EXPLPOT)
        qi = pi->q;
        #endif
        
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
            pj[0] = ( valid.x = ( pjd.x < cj ) ) ? &parts_j[ sort_j[pjd.x] >> 16 ] : pi;
            pj[1] = ( valid.y = ( pjd.y < cj ) && ( pjdid + 1 < wrap ) ) ? &parts_j[ sort_j[pjd.y] >> 16 ] : pi;
            pj[2] = ( valid.z = ( pjd.z < cj ) && ( pjdid + 2 < wrap ) ) ? &parts_j[ sort_j[pjd.z] >> 16 ] : pi;
            pj[3] = ( valid.w = ( pjd.w < cj ) && ( pjdid + 3 < wrap ) ) ? &parts_j[ sort_j[pjd.w] >> 16 ] : pi;
            
            /* Compute the pairwise distances. */
            r2 = make_float4( 0.0f );
            #pragma unroll
            for ( k = 0 ; k < 3 ; k++ ) {
                dx[k] = pix[k] - pj[0]->x[k]; r2.x += dx[k] * dx[k];
                dx[3+k] = pix[k] - pj[1]->x[k]; r2.y += dx[3+k] * dx[3+k];
                dx[6+k] = pix[k] - pj[2]->x[k]; r2.z += dx[6+k] * dx[6+k];
                dx[9+k] = pix[k] - pj[3]->x[k]; r2.w += dx[9+k] * dx[9+k];
                }
                
            /* Get the potentials. */
            valid.x = ( valid.x && r2.x < cuda_cutoff2 );
            valid.y = ( valid.y && r2.y < cuda_cutoff2 );
            valid.z = ( valid.z && r2.z < cuda_cutoff2 );
            valid.w = ( valid.w && r2.w < cuda_cutoff2 );
            pot.x = valid.x ? tex1D( tex_pind , pioff + pj[0]->type ) : 0;
            pot.y = valid.y ? tex1D( tex_pind , pioff + pj[1]->type ) : 0;
            pot.z = valid.z ? tex1D( tex_pind , pioff + pj[2]->type ) : 0;
            pot.w = valid.w ? tex1D( tex_pind , pioff + pj[3]->type ) : 0;
            
            /* Compute the interaction. */
            potential_eval4_cuda_tex( pot , r2 , &ee , &eff );
            
            /* Store the interaction energy. */
            epot += ee.x + ee.y + ee.z + ee.w;
            
            /* Update the particle forces. */
            if ( valid.x ) {
                pif[0] -= ( w = eff.x * dx[0] ); pj[0]->f[0] += w;
                pif[1] -= ( w = eff.x * dx[1] ); pj[0]->f[1] += w;
                pif[2] -= ( w = eff.x * dx[2] ); pj[0]->f[2] += w;
                }
            __threadfence_block();
            if ( valid.y ) {
                pif[0] -= ( w = eff.y * dx[3] ); pj[1]->f[0] += w;
                pif[1] -= ( w = eff.y * dx[4] ); pj[1]->f[1] += w;
                pif[2] -= ( w = eff.y * dx[5] ); pj[1]->f[2] += w;
                }
            __threadfence_block();
            if ( valid.z ) {
                pif[0] -= ( w = eff.z * dx[6] ); pj[2]->f[0] += w;
                pif[1] -= ( w = eff.z * dx[7] ); pj[2]->f[1] += w;
                pif[2] -= ( w = eff.z * dx[8] ); pj[2]->f[2] += w;
                }
            __threadfence_block();
            if ( valid.w ) {
                pif[0] -= ( w = eff.w * dx[9] ); pj[3]->f[0] += w;
                pif[1] -= ( w = eff.w * dx[10] ); pj[3]->f[1] += w;
                pif[2] -= ( w = eff.w * dx[11] ); pj[3]->f[2] += w;
                }
            __threadfence_block();
            
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            pi->f[k] += pif[k];
    
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
 
__device__ void runner_doself_cuda ( struct part_cuda *parts , int count ) {

    int k, pid, pjd, threadID;
    int pjoff;
    struct part_cuda *pi, *pj;
    #if defined(USETEX_E) || defined(EXPLPOT)
        float qj, q;
    #endif
    #if defined(USETEX) || defined(USETEX_E)
        int pot;
    #elif !defined(EXPLPOT)
        struct potential *pot;
    #endif
    float epot = 0.0f, dx[3], pjx[3], pjf[3], r2, w, ee, eff;
    
    TIMER_TIC
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Make sure everybody is in the same place. */
    __threadfence_block();

    /* Loop over the particles in the cell, frame-wise. */
    for ( pjd = threadID ; pjd < count-1 ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        pj = &parts[pjd];
        pjoff = pj->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pjx[k] = pj->x[k];
            pjf[k] = 0.0f;
            }
        #if defined(USETEX_E) || defined(EXPLPOT)
        qj = pj->q;
        #endif
            
        /* Loop over the particles in cell_i. */
        for ( pid = pjd+1 ; pid < count ; pid++ ) {
        
            /* Get a handle on the wrapped particle pid in cell_i. */
            pi = &parts[ pid ];

            /* Compute the radius between pi and pj. */
            for ( r2 = 0.0f , k = 0 ; k < 3 ; k++ ) {
                dx[k] = pi->x[k] - pjx[k];
                r2 += dx[k] * dx[k];
                }

            /* Set the null potential if anything is bad. */
            #ifdef USETEX_E
            if ( r2 < cuda_cutoff2 && ( ( pot = tex1D( tex_pind , pjoff + pi->type ) ) != 0 || ( q = qj*pi->q ) != 0.0f ) ) {
            #elif defined(USETEX)
            if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pjoff + pi->type ) ) != 0 ) {
            #elif defined(EXPLPOT)
            if ( r2 < cuda_cutoff2 ) {
            #else
            if ( r2 < cuda_cutoff2 && ( pot = cuda_p[ pjoff + pi->type ] ) != NULL ) {
            #endif

                /* Interact particles pi and pj. */
                #ifdef USETEX_E
                potential_eval_cuda_tex_e( pot , q , r2 , &ee , &eff );
                #elif defined(USETEX)
                potential_eval_cuda_tex( pot , r2 , &ee , &eff );
                #elif defined(EXPLPOT)
                potential_eval_cuda_expl( pi->type , pj->type , qj*pi->q , r2 , &ee , &eff );
                #else
                potential_eval_cuda( pot , r2 , &ee , &eff );
                #endif

                /* Store the interaction force and energy. */
                epot += ee;
                for ( k = 0 ; k < 3 ; k++ ) {
                    w = eff * dx[k];
                    pi->f[k] -= w;
                    pjf[k] += w;
                    }

                /* Sync the shared memory values. */
                __threadfence_block();

                } /* range and potential? */

            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            pj->f[k] += pjf[k];
    
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
 
__device__ void runner_doself_diag_cuda ( struct part_cuda *parts , int count ) {

    int diag, k, diag_max, step, pid, pjd, threadID;
    struct part_cuda *pi, *pj;
    #if defined(USETEX_E)
        float q;
    #endif
    #if defined(USETEX) || defined(USETEX_E)
        int pot;
    #elif !defined(EXPLPOT)
        struct potential *pot;
    #endif
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
            // k = tex1D( tex_diags ,  diag ); // ( sqrtf( 8*diag + 1 ) - 1 ) / 2;
            // pid = diag - k*(k+1)/2;
            // pjd = count - 1 - k + pid;
            pid = tex2D( tex_diags , 0 , diag );
            pjd = count - tex2D( tex_diags , 1 , diag );
            
            /* Get a handle on the particles. */
            pi = &parts[ pid ];
            pj = &parts[ pjd ];

            /* Compute the radius between pi and pj. */
            for ( r2 = 0.0f , k = 0 ; k < 3 ; k++ ) {
                dx[k] = pi->x[k] - pj->x[k];
                r2 += dx[k] * dx[k];
                }

            /* Set the null potential if anything is bad. */
            #ifdef USETEX_E
            if ( r2 < cuda_cutoff2 && ( ( pot = tex1D( tex_pind , pj->type*cuda_maxtype + pi->type ) ) != 0 || ( q = pj->q*pi->q ) != 0.0f ) ) {
            #elif defined(USETEX)
            if ( r2 < cuda_cutoff2 && ( pot = tex1D( tex_pind , pj->type*cuda_maxtype + pi->type ) ) != 0 ) {
            #elif defined(EXPLPOT)
            if ( r2 < cuda_cutoff2 ) {
            #else
            if ( r2 < cuda_cutoff2 && ( pot = cuda_p[ pj->type*cuda_maxtype + pi->type ] ) != NULL ) {
            #endif

                // atomicAdd( &cuda_pairs_done , 1 );
                    
                /* Interact particles pi and pj. */
                #ifdef USETEX_E
                potential_eval_cuda_tex_e( pot , q , r2 , &ee , &eff );
                #elif defined(USETEX)
                potential_eval_cuda_tex( pot , r2 , &ee , &eff );
                #elif defined(EXPLPOT)
                potential_eval_cuda_expl( pi->type , pj->type , pi->q*pj->q , r2 , &ee , &eff );
                #else
                potential_eval_cuda( pot , r2 , &ee , &eff );
                #endif

                /* Store the interaction force on pi and energy. */
                for ( k = 0 ; k < 3 ; k++ ) {
                    w[k] = eff * dx[k];
                    pi->f[k] -= w[k];
                    }

                /* Sync the shared memory values. */
                __threadfence_block();

                /* Store the interaction force on pj. */
                epot += ee;
                for ( k = 0 ; k < 3 ; k++ )
                    pj->f[k] += w[k];

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
 * @brief Bind textures to the given cuda Arrays.
 *
 *
 * Hack to get around the fact that textures are static and can thus not
 * be externalized.
 */
 
int runner_bind ( cudaArray *cuArray_coeffs , cudaArray *cuArray_offsets , cudaArray *cuArray_alphas , cudaArray *cuArray_pind , cudaArray *cuArray_diags ) {

    /* Bind the coeffs. */
    cuda_coeffs = cuArray_coeffs;
    if ( cudaBindTextureToArray( tex_coeffs , cuArray_coeffs ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Bind the offsets. */
    cuda_offsets = cuArray_offsets;
    if ( cudaBindTextureToArray( tex_offsets , cuArray_offsets ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Bind the alphas. */
    cuda_alphas = cuArray_alphas;
    if ( cudaBindTextureToArray( tex_alphas , cuArray_alphas ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Bind the pinds. */
    cuda_pind = cuArray_pind;
    if ( cudaBindTextureToArray( tex_pind , cuArray_pind ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Bind the diags. */
    cuda_diags = cuArray_diags;
    if ( cudaBindTextureToArray( tex_diags , cuArray_diags ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Rock and roll. */
    return runner_err_ok;

    }


/**
 * @brief Loop over the cell pairs and process them using Verlet lists.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_verlet_cuda_old ( struct part_cuda *parts , int *counts , int *ind , int verlet_rebuild ) {

    int threadID, blockID;
    int i, nf, cpn, itemp, cid, cjd;
    volatile __shared__ int nr_fingers, finds[max_fingers];
    __shared__ struct cellpair_cuda finger[max_fingers];
    __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    
    /* Get the block and thread ids. */
    blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Check in at the barrier. */
    if ( threadID == 0 )
        atomicAdd( &cuda_barrier , 1 );
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        

    /* Main loop... */
    while ( cuda_pair_next < cuda_nr_pairs ) {
    
        /* Let the first thread get a pair. */
        if ( threadID == 0 ) {
        
            TIMER_TIC
        
            /* Lock the mutex. */
            cuda_mutex_lock( &cuda_cell_mutex );
            
            /* Get as many fingers as possible. */
            cpn = cuda_pair_next;
            for ( i = cpn , nf = 0 ; nf < max_fingers && i < cuda_nr_pairs ; i++ ) {
                    
                /* Pick up this pair? */
                if ( ( atomicCAS( &cuda_taboo[ cuda_pairs[i].i ] , 0 , 0 ) == 0 ) &&
                     ( atomicCAS( &cuda_taboo[ cuda_pairs[i].j ] , 0 , 0 ) == 0 ) ) {
                        
                    /* Swap entries in the pair list. */
                    finds[ nf ] = cpn;
                    finger[ nf ] = cuda_pairs[ i ];
                    if ( cpn != i ) {
                        cuda_pairs[ i ] = cuda_pairs[ cpn ];
                        cuda_pairs[ cpn ] = finger[ nf ];
                        }
                    
                    /* Swap sortlist index. */
                    itemp = cuda_sortlists_ind[ i ];
                    cuda_sortlists_ind[ i ] = cuda_sortlists_ind[ cpn ];
                    cuda_sortlists_ind[ cpn ] = itemp;
                        
                    /* Store this pair to the fingers. */
                    nf += 1;
                    cpn += 1;
                    
                    }
                
                } /* get fingers. */
            
            /* Did we get anything? */
            if ( nf > 0 ) {
            
                /* Mark the cells. */
                for ( i = 0 ; i < nf ; i++ ) {
                    atomicAdd( &cuda_taboo[ finger[i].i ] , 1 );
                    atomicAdd( &cuda_taboo[ finger[i].j ] , 1 );
                    }
                    
                /* Store the modified cell_pair_next. */
                nr_fingers = nf;
                cuda_pair_next = cpn;
                    
                /* Make sure everybody is on the same page. */
                __threadfence();
                
                }
                
            /* No, return empty-handed. */
            else {
                nr_fingers = 0;
                __threadfence_block();
                }
        
            /* Un-lock the mutex. */
            cuda_mutex_unlock( &cuda_cell_mutex );
            
            TIMER_TOC(tid_queue)
            
            } /* threadID=0 doing it's own thing. */
            
            
        /* If we actually got a set of pairs, do them! */
        nf = nr_fingers;
        for ( i = 0 ; i < nf ; i++ ) {
        
            /* Get a hold of the pair. */
            cid = finger[i].i;
            cjd = finger[i].j;
        
            /* Do the pair. */
            if ( cid != cjd ) {
            
                /* Get a local copy of the particle data. */
                cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
                cuda_memcpy( parts_j , &parts[ind[cjd]] , sizeof( struct part_cuda ) * counts[cjd] );
                __threadfence_block();
                
                /* Compute the cell pair interactions. */
                runner_dopair_sorted_verlet_cuda(
                    parts_i , counts[cid] ,
                    parts_j , counts[cjd] ,
                    finger[i].shift ,
                    verlet_rebuild , &cuda_sortlists[ cuda_sortlists_ind[ finds[i] ] ] );
                    
                /* Write the particle data back. */
                cuda_memcpy( &parts[ind[cid]] , parts_i , sizeof( struct part_cuda ) * counts[cid] );
                cuda_memcpy( &parts[ind[cjd]] , parts_j , sizeof( struct part_cuda ) * counts[cjd] ); 
                    
                }
            else {
            
                /* Get a local copy of the particle data. */
                cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
                __threadfence_block();
                
                /* Compute the cell self interactions. */
                if ( counts[cid] <= cuda_frame )
                    runner_doself_cuda( parts_i , counts[cid] );
                else
                    runner_doself_diag_cuda( parts_i , counts[cid] );
                    
                /* Write the particle data back. */
                cuda_memcpy( &parts[ind[cid]] , parts_i , sizeof( struct part_cuda ) * counts[cid] );
                
                }
        
            /* Release the cells in the taboo list. */
            if ( threadID == 0 ) {
                atomicSub( &cuda_taboo[ cid ] , 1 );
                atomicSub( &cuda_taboo[ cjd ] , 1 );
                }
                
            /* Sync the memory before retuning the particle data. */
            __threadfence();

            }
            
        } /* main loop. */

    /* Check out at the barrier. */
    if ( threadID == 0 )
        atomicSub( &cuda_barrier , 1 );
    
    /* The last one out cleans up the mess... */
    if ( threadID == 0 && blockID == 0 ) {
        while ( atomicCAS( &cuda_barrier , 0 , 0 ) != 0 );
        cuda_pair_next = 0;
        __threadfence();
        }

    }
    
    
/**
 * @brief Loop over the cell pairs and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_verlet_cuda ( struct part_cuda *parts , int *counts , int *ind , int verlet_rebuild ) {

    int threadID, blockID;
    int k, cid, cjd;
    volatile __shared__ int pid;
    __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    struct part_cuda *parts_k;
    
    TIMER_TIC2
    
    /* Get the block and thread ids. */
    blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Make a notch on the barrier. */
    if ( threadID == 0 )
        atomicAdd( &cuda_barrier , 1 );
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        

    /* Main loop... */
    while ( 1 ) {
    
        /* Let the first thread grab a pair. */
        if ( threadID == 0 ) {
            pid = atomicAdd( (int *)&cuda_pair_next , 1 );
            __threadfence_block();
            }
            
        /* Are we at the end of the list? */
        if ( pid >= cuda_nr_pairs )
            break;
        
        /* Get a hold of the pair cells. */
        cid = cuda_pairs[pid].i;
        cjd = cuda_pairs[pid].j;
    
        /* Do the pair. */
        if ( cid != cjd ) {
        
            /* Get a local copy of the particle data. */
            cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
            TIMER_TIC
            for ( k = threadID ; k < counts[cid] ; k += cuda_frame ) {
                parts_i[k].f[0] = 0.0f;
                parts_i[k].f[1] = 0.0f;
                parts_i[k].f[2] = 0.0f;
                }
            TIMER_TOC(tid_queue)
            cuda_memcpy( parts_j , &parts[ind[cjd]] , sizeof( struct part_cuda ) * counts[cjd] );
            TIMER_TIC_ND
            for ( k = threadID ; k < counts[cjd] ; k += cuda_frame ) {
                parts_j[k].f[0] = 0.0f;
                parts_j[k].f[1] = 0.0f;
                parts_j[k].f[2] = 0.0f;
                }
            TIMER_TOC_ND(tid_queue)
            __threadfence_block();
            
            /* Compute the cell pair interactions. */
            runner_dopair_sorted4_verlet_cuda(
                parts_i , counts[cid] ,
                parts_j , counts[cjd] ,
                cuda_pairs[pid].shift ,
                verlet_rebuild , &cuda_sortlists[ cuda_sortlists_ind[ pid ] ] );
                
            /* Write the particle data back. */
            parts_k = &parts[ ind[ cid ] ];
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cid] );
            TIMER_TIC_ND
            for ( k = threadID ; k < counts[cid] ; k += cuda_frame ) {
                parts_k[k].f[0] += parts_i[k].f[0];
                parts_k[k].f[1] += parts_i[k].f[1];
                parts_k[k].f[2] += parts_i[k].f[2];
                }
            TIMER_TOC_ND(tid_queue)
            if ( threadID == 0 )
                cuda_mutex_unlock( &cuda_taboo[cid] );
                
            parts_k = &parts[ ind[ cjd ] ];
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cjd] );
            TIMER_TIC_ND
            for ( k = threadID ; k < counts[cjd] ; k += cuda_frame ) {
                parts_k[k].f[0] += parts_j[k].f[0];
                parts_k[k].f[1] += parts_j[k].f[1];
                parts_k[k].f[2] += parts_j[k].f[2];
                }
            TIMER_TOC_ND(tid_queue)
            if ( threadID == 0 )
                cuda_mutex_unlock( &cuda_taboo[cjd] );
            __threadfence();
                
            }
        else {
        
            /* Get a local copy of the particle data. */
            cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
            TIMER_TIC
            for ( k = threadID ; k < counts[cid] ; k += cuda_frame ) {
                parts_i[k].f[0] = 0.0f;
                parts_i[k].f[1] = 0.0f;
                parts_i[k].f[2] = 0.0f;
                }
            TIMER_TOC(tid_queue)
            __threadfence_block();
            
            /* Compute the cell self interactions. */
            if ( counts[cid] <= cuda_frame )
                runner_doself_cuda( parts_i , counts[cid] );
            else
                runner_doself_diag_cuda( parts_i , counts[cid] );
                
            /* Write the particle data back. */
            parts_k = &parts[ ind[ cid ] ];
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cid] );
            TIMER_TIC_ND
            for ( k = threadID ; k < counts[cid] ; k += cuda_frame ) {
                parts_k[k].f[0] += parts_i[k].f[0];
                parts_k[k].f[1] += parts_i[k].f[1];
                parts_k[k].f[2] += parts_i[k].f[2];
                }
            TIMER_TOC_ND(tid_queue)
            if ( threadID == 0 )
                cuda_mutex_unlock( &cuda_taboo[cid] );
            __threadfence();
            
            }
        
        } /* main loop. */

    /* Check out at the barrier. */
    if ( threadID == 0 )
        atomicSub( &cuda_barrier , 1 );
    
    /* The last one out cleans up the mess... */
    if ( threadID == 0 && blockID == 0 ) {
        while ( atomicCAS( &cuda_barrier , 0 , 0 ) != 0 );
        cuda_pair_next = 0;
        __threadfence();
        }
        
    TIMER_TOC2(tid_total)

    }
    
    
/**
 * @brief Loop over the cell pairs and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_cuda_old ( struct part_cuda *parts , int *counts , int *ind ) {

    int threadID, blockID;
    int i, nf, cpn, cid, cjd;
    volatile __shared__ int nr_fingers;
    __shared__ struct cellpair_cuda finger[max_fingers];
    #ifdef SHARED_BUFF
        __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    #else
        struct part_cuda *parts_i, *parts_j;
    #endif
    
    TIMER_TIC2
    
    /* Get the block and thread ids. */
    blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Make a notch on the barrier. */
    if ( threadID == 0 )
        atomicAdd( &cuda_barrier , 1 );
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        

    /* Main loop... */
    while ( cuda_pair_next < cuda_nr_pairs ) {
    
        /* Let the first thread get a pair. */
        if ( threadID == 0 ) {
        
            /* Lock the mutex. */
            cuda_mutex_lock( &cuda_cell_mutex );
            
            TIMER_TIC
        
            /* Get as many fingers as possible. */
            for ( i = cuda_pair_next , nf = 0 ; nf < max_fingers && i < cuda_nr_pairs ; i++ ) {
            
                /* Pick up this pair? */
                if ( ( atomicCAS( &cuda_taboo[ cuda_pairs[i].i ] , 0 , 0 ) == 0 ) &&
                     ( atomicCAS( &cuda_taboo[ cuda_pairs[i].j ] , 0 , 0 ) == 0 ) ) {
                        
                    /* Swap entries in the pair list. */
                    cpn = cuda_pair_next + nf;
                    finger[ nf ] = cuda_pairs[ i ];
                    if ( cpn != i ) {
                        cuda_pairs[ i ] = cuda_pairs[ cpn ];
                        cuda_pairs[ cpn ] = finger[ nf ];
                        }
                    
                    /* Store this pair to the fingers. */
                    nf += 1;
                    
                    }
                    
                } /* get fingers. */
            
            /* Did we get anything? */
            if ( nf > 0 ) {
            
                /* Mark the cells. */
                for ( i = 0 ; i < nf ; i++ ) {
                    atomicAdd( &cuda_taboo[ finger[i].i ] , 1 );
                    atomicAdd( &cuda_taboo[ finger[i].j ] , 1 );
                    }
                    
                /* Store the modified cell_pair_next. */
                nr_fingers = nf;
                cuda_pair_next += nf;
                    
                /* Make sure everybody is on the same page. */
                __threadfence();
                
                }
                
            /* No, return empty-handed. */
            else {
                nr_fingers = 0;
                __threadfence_block();
                }
        
            /* Un-lock the mutex. */
            cuda_mutex_unlock( &cuda_cell_mutex );
            
            TIMER_TOC(tid_queue)
            
            } /* threadID=0 doing it's own thing. */
            

        /* If we actually got a set of pairs, do them! */
        nf = nr_fingers;
        for ( i = 0 ; i < nf ; i++ ) {
        
            /* Get a hold of the pair. */
            cid = finger[i].i;
            cjd = finger[i].j;
        
            /* Do the pair. */
            if ( cid != cjd ) {
            
                /* Get a local copy of the particle data. */
                #ifdef SHARED_BUFF
                    cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
                    cuda_memcpy( parts_j , &parts[ind[cjd]] , sizeof( struct part_cuda ) * counts[cjd] );
                    __threadfence_block();
                #else
                    parts_i = &parts[ ind[cid] ];
                    parts_j = &parts[ ind[cjd] ];
                #endif
                
                /* Compute the cell pair interactions. */
                if ( counts[cid] <= 2*cuda_frame || counts[cjd] <= 2*cuda_frame )
                    runner_dopair_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        finger[i].shift );
                else
                    runner_dopair_sorted_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        finger[i].shift );
                    
                /* Write the particle data back. */
                #ifdef SHARED_BUFF
                    cuda_memcpy( &parts[ind[cid]] , parts_i , sizeof( struct part_cuda ) * counts[cid] );
                    cuda_memcpy( &parts[ind[cjd]] , parts_j , sizeof( struct part_cuda ) * counts[cjd] );
                #endif
                    
                }
            else {
            
                /* Get a local copy of the particle data. */
                #ifdef SHARED_BUFF
                    cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
                    __threadfence_block();
                #else
                    parts_i = &parts[ ind[cid] ];
                #endif
                
                /* Compute the cell self interactions. */
                if ( counts[cid] <= cuda_frame )
                    runner_doself_cuda( parts_i , counts[cid] );
                else
                    runner_doself_diag_cuda( parts_i , counts[cid] );
                    
                /* Write the particle data back. */
                #ifdef SHARED_BUFF
                    cuda_memcpy( &parts[ind[cid]] , parts_i , sizeof( struct part_cuda ) * counts[cid] );
                #endif
                
                }
        
            /* Sync the memory before retuning the particle data. */
            __threadfence();

            /* Release the cells in the taboo list. */
            if ( threadID == 0 ) {
                atomicSub( &cuda_taboo[ cid ] , 1 );
                atomicSub( &cuda_taboo[ cjd ] , 1 );
                } 
                
            }
            
        } /* main loop. */

    /* Check out at the barrier. */
    if ( threadID == 0 )
        atomicSub( &cuda_barrier , 1 );
    
    /* The last one out cleans up the mess... */
    if ( threadID == 0 && blockID == 0 ) {
        while ( atomicCAS( &cuda_barrier , 0 , 0 ) != 0 );
        cuda_pair_next = 0;
        __threadfence();
        }
        
    TIMER_TOC2(tid_total)

    }
    
    
/**
 * @brief Loop over the cell pairs and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_cuda ( struct part_cuda *parts , int *counts , int *ind ) {

    int threadID, blockID;
    int k, cid, cjd;
    volatile __shared__ int pid;
    __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    struct part_cuda *parts_k;
    
    TIMER_TIC2
    
    /* Get the block and thread ids. */
    blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Make a notch on the barrier. */
    if ( threadID == 0 )
        atomicAdd( &cuda_barrier , 1 );
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        

    /* Main loop... */
    while ( 1 ) {
    
        /* Let the first thread grab a pair. */
        if ( threadID == 0 ) {
            pid = atomicAdd( (int *)&cuda_pair_next , 1 );
            __threadfence_block();
            }
            
        /* Are we at the end of the list? */
        if ( pid >= cuda_nr_pairs )
            break;
        
        /* Get a hold of the pair cells. */
        cid = cuda_pairs[pid].i;
        cjd = cuda_pairs[pid].j;
    
        /* Do the pair. */
        if ( cid != cjd ) {
        
            /* Get a local copy of the particle data. */
            cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
            TIMER_TIC
            for ( k = threadID ; k < counts[cid] ; k += cuda_frame ) {
                parts_i[k].f[0] = 0.0f;
                parts_i[k].f[1] = 0.0f;
                parts_i[k].f[2] = 0.0f;
                }
            TIMER_TOC(tid_queue)
            cuda_memcpy( parts_j , &parts[ind[cjd]] , sizeof( struct part_cuda ) * counts[cjd] );
            TIMER_TIC_ND
            for ( k = threadID ; k < counts[cjd] ; k += cuda_frame ) {
                parts_j[k].f[0] = 0.0f;
                parts_j[k].f[1] = 0.0f;
                parts_j[k].f[2] = 0.0f;
                }
            TIMER_TOC_ND(tid_queue)
            __threadfence_block();
            
            /* Compute the cell pair interactions. */
            if ( counts[cid] <= cuda_frame || counts[cjd] <= cuda_frame )
                runner_dopair4_cuda(
                    parts_i , counts[cid] ,
                    parts_j , counts[cjd] ,
                    cuda_pairs[pid].shift );
            else
                runner_dopair_sorted4_cuda(
                    parts_i , counts[cid] ,
                    parts_j , counts[cjd] ,
                    cuda_pairs[pid].shift );
                
            /* Write the particle data back. */
            parts_k = &parts[ ind[ cid ] ];
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cid] );
            TIMER_TIC_ND
            for ( k = threadID ; k < counts[cid] ; k += cuda_frame ) {
                parts_k[k].f[0] += parts_i[k].f[0];
                parts_k[k].f[1] += parts_i[k].f[1];
                parts_k[k].f[2] += parts_i[k].f[2];
                }
            TIMER_TOC_ND(tid_queue)
            if ( threadID == 0 )
                cuda_mutex_unlock( &cuda_taboo[cid] );
                
            parts_k = &parts[ ind[ cjd ] ];
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cjd] );
            TIMER_TIC_ND
            for ( k = threadID ; k < counts[cjd] ; k += cuda_frame ) {
                parts_k[k].f[0] += parts_j[k].f[0];
                parts_k[k].f[1] += parts_j[k].f[1];
                parts_k[k].f[2] += parts_j[k].f[2];
                }
            TIMER_TOC_ND(tid_queue)
            if ( threadID == 0 )
                cuda_mutex_unlock( &cuda_taboo[cjd] );
            __threadfence();
                
            }
        else {
        
            /* Get a local copy of the particle data. */
            cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
            TIMER_TIC
            for ( k = threadID ; k < counts[cid] ; k += cuda_frame ) {
                parts_i[k].f[0] = 0.0f;
                parts_i[k].f[1] = 0.0f;
                parts_i[k].f[2] = 0.0f;
                }
            TIMER_TOC(tid_queue)
            __threadfence_block();
            
            /* Compute the cell self interactions. */
            if ( counts[cid] <= cuda_frame )
                runner_doself_cuda( parts_i , counts[cid] );
            else
                runner_doself_diag_cuda( parts_i , counts[cid] );
                
            /* Write the particle data back. */
            parts_k = &parts[ ind[ cid ] ];
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cid] );
            TIMER_TIC_ND
            for ( k = threadID ; k < counts[cid] ; k += cuda_frame ) {
                parts_k[k].f[0] += parts_i[k].f[0];
                parts_k[k].f[1] += parts_i[k].f[1];
                parts_k[k].f[2] += parts_i[k].f[2];
                }
            TIMER_TOC_ND(tid_queue)
            if ( threadID == 0 )
                cuda_mutex_unlock( &cuda_taboo[cid] );
            __threadfence();
            
            }
        
        } /* main loop. */

    /* Check out at the barrier. */
    if ( threadID == 0 )
        atomicSub( &cuda_barrier , 1 );
    
    /* The last one out cleans up the mess... */
    if ( threadID == 0 && blockID == 0 ) {
        while ( atomicCAS( &cuda_barrier , 0 , 0 ) != 0 );
        cuda_pair_next = 0;
        __threadfence();
        }
        
    TIMER_TOC2(tid_total)

    }
    
    
/**
 * @brief Loop over the cell pairs and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_cuda_new ( struct part_cuda *parts , int *counts , int *ind ) {

    int threadID, blockID;
    int i, pid, cid, cjd, oid, ojd;
    __shared__ int spid;
    #ifdef SHARED_BUFF
        __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    #else
        struct part_cuda *parts_i, *parts_j;
    #endif
    
    TIMER_TIC2
    
    /* Get the block and thread ids. */
    blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Make a notch on the barrier. */
    if ( threadID == 0 )
        atomicAdd( &cuda_barrier , 1 );
    
    /* Main loop... */
    while ( cuda_pair_count < cuda_nr_pairs ) {
    
        /* Let the first thread get a pair. */
        if ( threadID == 0 ) {
        
            /* Lock the mutex. */
            if ( cuda_mutex_lock_cond( &cuda_cell_mutex , &cuda_fifos_in[ blockID ].count ) ) {
            
                TIMER_TIC
            
                /* Run through the available pairs. */
                for ( i = cuda_pair_curr ; i < cuda_nr_pairs ; i++ ) {
                
                    /* Get the cell IDs of the ith pair. */
                    pid = cuda_pairIDs[i];
                    cid = cuda_pairs[ pid ].i;
                    cjd = cuda_pairs[ pid ].j;
                    oid = atomicCAS( &cuda_owner[ cid ] , 0 , 0 );
                    ojd = atomicCAS( &cuda_owner[ cjd ] , 0 , 0 );
                    
                    /* Is this pair free? */
                    if ( ( oid == 0 || oid == blockID+1 ) && ( ojd == 0 || ojd == blockID+1 ) ) {
                    
                        /* Own the cells. */
                        atomicAdd( &cuda_taboo[ cid ] , 1 );
                        atomicAdd( &cuda_taboo[ cjd ] , 1 );
                        atomicExch( &cuda_owner[ cid ] , blockID+1 );
                        if ( cid != cjd )
                            atomicExch( &cuda_owner[ cjd ] , blockID+1 );
                    
                        /* Push this pair onto the queue. */
                        cuda_fifo_push( &cuda_fifos_in[ blockID ] , pid );
                        
                        /* Swap back to curr if needed. */
                        if ( i != cuda_pair_curr )
                            cuda_pairIDs[i] = cuda_pairIDs[ cuda_pair_curr ];
                            
                        /* Increase the counters and stuff. */
                        cuda_pair_curr += 1;
                        cuda_pairIDs[ cuda_pair_count ] = pid;
                        cuda_pair_count += 1;
                        
                        /* Leave this loop. */
                        __threadfence();
                        break;
                    
                        }
                
                    /* Does this pair have a single owner? */
                    else if ( ( oid == 0 ) || ( ojd == 0 ) || ( oid == ojd ) ) {
                    
                        /* Get the unique owner. */
                        oid |= ojd;
                        
                        /* Does the owner's queue have any space left? */
                        if ( cuda_fifos_in[ oid-1 ].count < cuda_fifo_size ) {
                        
                            /* Set the ownership. */
                            atomicAdd( &cuda_taboo[ cid ] , 1 );
                            atomicAdd( &cuda_taboo[ cjd ] , 1 );
                            atomicExch( &cuda_owner[ cid ] , oid );
                            atomicExch( &cuda_owner[ cjd ] , oid );
                                
                            /* Put this pair on the owner's queue. */
                            cuda_fifo_push( &cuda_fifos_in[ oid-1 ] , pid );
                                
                            /* Swap back to curr if needed. */
                            if ( i != cuda_pair_curr )
                                cuda_pairIDs[i] = cuda_pairIDs[ cuda_pair_curr ];
                                
                            /* Increase the counters and stuff. */
                            cuda_pair_curr += 1;
                            cuda_pairIDs[ cuda_pair_count ] = pid;
                            cuda_pair_count += 1;
                        
                            } /* there is space in the queue. */
                    
                        } /* pair has single owner. */
                        
                    } /* run through the available pairs. */
                
                /* Un-lock the mutex. */
                cuda_mutex_unlock( &cuda_cell_mutex );
                
                /* Sync the memory. */
                __threadfence();
                
                TIMER_TOC(tid_queue)
                
                } /* got mutex? */
            
            } /* threadID=0 doing it's own thing. */
            

        /* Loop over the pairs in our own queue. */
        while ( cuda_fifos_in[blockID].count > 0 ) {
        
            /* Get a hold of the pair. */
            if ( threadID == 0 ) {
                spid = cuda_fifo_pop( &cuda_fifos_in[ blockID ] );
                __threadfence_block();
                }
            cid = cuda_pairs[ spid ].i;
            cjd = cuda_pairs[ spid ].j;
        
            /* Do the pair. */
            if ( cid != cjd ) {
            
                /* Get a local copy of the particle data. */
                #ifdef SHARED_BUFF
                    cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
                    cuda_memcpy( parts_j , &parts[ind[cjd]] , sizeof( struct part_cuda ) * counts[cjd] );
                    __threadfence_block();
                #else
                    parts_i = &parts[ ind[cid] ];
                    parts_j = &parts[ ind[cjd] ];
                #endif
                
                /* Compute the cell pair interactions. */
                if ( counts[cid] <= 2*cuda_frame || counts[cjd] <= 2*cuda_frame )
                    runner_dopair_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        cuda_pairs[spid].shift );
                else
                    runner_dopair_sorted_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        cuda_pairs[spid].shift );
                    
                /* Write the particle data back. */
                #ifdef SHARED_BUFF
                    cuda_memcpy( &parts[ind[cid]] , parts_i , sizeof( struct part_cuda ) * counts[cid] );
                    cuda_memcpy( &parts[ind[cjd]] , parts_j , sizeof( struct part_cuda ) * counts[cjd] );
                #endif
                    
                }
            else {
            
                /* Get a local copy of the particle data. */
                #ifdef SHARED_BUFF
                    cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
                    __threadfence_block();
                #else
                    parts_i = &parts[ ind[cid] ];
                #endif
                
                /* Compute the cell self interactions. */
                if ( counts[cid] <= cuda_frame )
                    runner_doself_cuda( parts_i , counts[cid] );
                else
                    runner_doself_diag_cuda( parts_i , counts[cid] );
                    
                /* Write the particle data back. */
                #ifdef SHARED_BUFF
                    cuda_memcpy( &parts[ind[cid]] , parts_i , sizeof( struct part_cuda ) * counts[cid] );
                #endif
                
                }
                
            /* Sync the memory before retuning the particle data. */
            __threadfence();

            /* Release the cells in the taboo list. */
            if ( threadID == 0 ) {
                if ( atomicSub( &cuda_taboo[ cid ] , 1 ) == 1 )
                    atomicExch( &cuda_owner[ cid ] , 0 );
                if ( atomicSub( &cuda_taboo[ cjd ] , 1 ) == 1 )
                    atomicExch( &cuda_owner[ cjd ] , 0 );
                }
                
            }
            
        } /* main loop. */
        
    /* Check out at the barrier. */
    if ( threadID == 0 )
        atomicSub( &cuda_barrier , 1 );
    
    /* The last one out cleans up the mess... */
    if ( threadID == 0 && blockID == 0 ) {
        while ( atomicCAS( &cuda_barrier , 0 , 0 ) != 0 );
        cuda_pair_count = 0;
        cuda_pair_curr = 0;
        __threadfence();
        }

    TIMER_TOC2(tid_total)

    }
    
    
/**
 * @brief Round-Robin dispatcher/client for cell pairs.
 *
 */
 
__global__ void runner_run_dispatcher_cuda ( struct part_cuda *parts , int *counts , int *ind ) {

    /* Local variables common to both dispatcher and client. */
    int blockID, threadID, nr_blocks;
    
    /* Get the shape of things. */
    threadID = threadIdx.x;
    blockID = blockIdx.x;
    nr_blocks = gridDim.x;
    
    /* Am I the dispatcher or a client? */
    if ( blockID == 0 ) {
    
        /* Some local variables for the dispatcher. */
        int pid, comm, max_comm, max_ind, ind;
        int i, k, cid, cjd, cpn, wrap;
        __shared__ int counter;
        struct cellpair_cuda temp;
        
        /* Clean-up the taboo list. */
        for ( k = threadID ; k < cuda_nr_cells ; k += cuda_frame )
            cuda_taboo[k] = 0;
            
        /* Reset the input and output fifos. */
        /* for ( k = threadID ; k < nr_blocks-1 ; k++ ) {
            cuda_fifos_in[k].first = 0;
            cuda_fifos_in[k].last = 0;
            cuda_fifos_in[k].count = 0;
            cuda_fifos_out[k].first = 0;
            cuda_fifos_out[k].last = 0;
            cuda_fifos_out[k].count = 0;
            } */
            
        /* Flush the memory just to be sure. */
        __threadfence();
    
        /* Main loop. */
        while ( cuda_pair_next < cuda_nr_pairs ) {
        
            /* Loop over the clients... */
            for ( k = threadID ; k < nr_blocks-1 ; k += cuda_frame ) {

                /* Only look for a new pair if the input buffer isn't full. */
                if ( cuda_fifos_in[k].count < cuda_fifo_size ) {

                    /* Loop over the cell pairs and find the one with the
                       highest number of common cells. */
                    max_comm = -1;
                    cpn = cuda_pair_next;
                    wrap = max( cuda_frame , cuda_nr_pairs - cpn );
                    for ( i = cpn ; max_comm < 2 && i < cpn+wrap ; i++ ) {

                        /* Shift and wrap so that every thread looks at a different pair. */
                        if ( ( pid = i + threadID ) >= cpn+wrap )
                            pid -= wrap;
                            
                        /* Is this a valid pair-ID? */
                        if ( pid < cuda_nr_pairs ) {

                            /* Check if this pair is free or already belongs to this client. */
                            cid = cuda_taboo[ cuda_pairs[ pid ].i ];
                            cjd = cuda_taboo[ cuda_pairs[ pid ].j ];
                            if ( ( ( cid == 0 ) || ( cid >> 16 == k ) ) &&
                                 ( ( cjd == 0 ) || ( cjd >> 16 == k ) ) ) {

                                /* Get the number of common threads. */
                                comm = ( ( cid > 0 ) && ( cid >> 16 ) == k ) +
                                       ( ( cjd > 0 ) && ( cjd >> 16 ) == k );

                                /* Store as new maximum? */
                                if ( comm > max_comm ) {
                                    max_comm = comm;
                                    max_ind = pid;
                                    }

                                } /* pair free? */
                                
                            } /* valid pair-ID? */
                            
                        } /* loop over cell pairs. */

                    /* Sequentially remove the pairs from the list. */
                    for ( i = 0 ; i < cuda_frame ; i++ )
                        if ( threadID == i && max_comm >= 0 ) {

                            /* Get the cells. */
                            cid = cuda_taboo[ cuda_pairs[ max_ind ].i ];
                            cjd = cuda_taboo[ cuda_pairs[ max_ind ].j ];

                            /* Is this the same pair I chose earlier? */
                            if ( ( ( cid == 0 ) || ( cid >> 16 == k ) ) &&
                                 ( ( cjd == 0 ) || ( cjd >> 16 == k ) ) &&
                                 ( max_comm <= ( ( cid >> 16 ) == k ) + ( ( cjd >> 16 ) == k ) ) ) {

                                /* Get a new local copy. */
                                cpn = cuda_pair_next;

                                /* Swap to the front of the queue .*/
                                temp = cuda_pairs[ max_ind ];
                                cuda_pairs[ max_ind ] = cuda_pairs[ cpn ];
                                cuda_pairs[ cpn ] = temp;

                                /* Update the taboo list. */
                                cuda_taboo[ temp.i ] = ( k << 16 ) | ( ( cuda_taboo[ temp.i ] & 0xffff ) + 1 );
                                cuda_taboo[ temp.j ] = ( k << 16 ) | ( ( cuda_taboo[ temp.j ] & 0xffff ) + 1 );

                                /* Update max_ind to the new position. */
                                max_ind = cpn;
                                cuda_pair_next += 1;

                                /* Make sure everybody is up-to-date. */
                                __threadfence_block();

                                }

                            /* Otherwise, invalidate this pair. */
                            else
                                max_comm = -1;

                            } /* remove pairs from list. */

                    /* Finally, add to the input buffer. */
                    if ( max_comm >= 0 ) {
                        cuda_fifo_push( &cuda_fifos_in[k] , max_ind );
                        __threadfence();
                        }


                    } /* there is room in the buffer. */

                /* Remove and free whatever is in the output buffers. */
                while ( cuda_fifos_out[k].count > 0 ) {

                    /* Get the pair that needs to be freed. */
                    ind = cuda_fifo_pop( &cuda_fifos_out[k] );
                    cid = cuda_pairs[ind].i;
                    cjd = cuda_pairs[ind].j;

                    /* Update the taboo list. */
                    if ( ( ( cuda_taboo[cid] -= 1 ) & 0xffff ) == 0 )
                        cuda_taboo[cid] = 0;;
                    if ( ( ( cuda_taboo[cjd] -= 1 ) & 0xffff ) == 0 )
                        cuda_taboo[cjd] = 0;;

                    /* Bring everybody up to speed. */
                    __threadfence_block();

                    }

                } /* loop over clients. */
                
            } /* Main loop. */
            
        /* Tell the clients it's over. */
        for ( k = threadID ; k < nr_blocks-1 ; k += cuda_frame ) {
            cuda_fifo_push( &cuda_fifos_in[k] , 0xffffffff );
            __threadfence();
            }
            
        /* Wait for the final acks. */
        if ( threadID == 0 )
            atomicExch( &counter , nr_blocks-1 );
        while ( counter > 0 )
            for ( k = threadID ; k < nr_blocks-1 ; k += cuda_frame )
                if ( ( cuda_fifos_out[k].count > 0 ) &&
                     ( cuda_fifo_pop( &cuda_fifos_out[k] ) == 0xffffffff ) )
                    atomicSub( &counter , 1 );
            
        }
        
    /* Nope, I'm a client. */
    else {
    
        /* Some local variables for this context. */
        struct fifo_cuda *f_in = &cuda_fifos_in[ blockID-1 ];
        struct fifo_cuda *f_out = &cuda_fifos_out[ blockID-1 ];
        struct cellpair_cuda pair;
        __shared__ unsigned int task;
        __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    
        /* Main loop. */
        while ( 1 ) {
        
            /* Wait for a task in the queue and get it. */
            if ( threadID == 0 ) {
                task = cuda_fifo_pop( f_in );
                __threadfence_block();
                }
            
            /* Decode the task, break if it's a quit. */
            if ( task == 0xffffffff ) {
                cuda_fifo_push( f_out , task );
                break;
                }
            pair = cuda_pairs[ task ];
        
            /* Do the pair. */
            if ( pair.i != pair.j ) {
            
                /* Get a local copy of the particle data. */
                cuda_memcpy( parts_i , &parts[ind[pair.i]] , sizeof( struct part_cuda ) * counts[pair.i] );
                cuda_memcpy( parts_j , &parts[ind[pair.j]] , sizeof( struct part_cuda ) * counts[pair.j] );
                __threadfence_block();
                
                /* Compute the cell pair interactions. */
                runner_dopair_sorted_cuda(
                    parts_i , counts[pair.i] ,
                    parts_j , counts[pair.j] ,
                    pair.shift );
                    
                /* Write the particle data back. */
                cuda_memcpy( &parts[ind[pair.i]] , parts_i , sizeof( struct part_cuda ) * counts[pair.i] );
                cuda_memcpy( &parts[ind[pair.j]] , parts_j , sizeof( struct part_cuda ) * counts[pair.j] ); 
                    
                }
            else {
            
                /* Get a local copy of the particle data. */
                cuda_memcpy( parts_i , &parts[ind[pair.i]] , sizeof( struct part_cuda ) * counts[pair.i] );
                __threadfence_block();
                
                /* Compute the cell self interactions. */
                runner_doself_diag_cuda( parts_i , counts[pair.i] );
                    
                /* Write the particle data back. */
                cuda_memcpy( &parts[ind[pair.i]] , parts_i , sizeof( struct part_cuda ) * counts[pair.i] );
                
                }
        
            /* Sync the memory before retuning the particle data. */
            __threadfence();
            
            /* Push the tsk to the outbound queue. */
            if ( threadID == 0 ) {
                cuda_fifo_push( f_out , task );
                __threadfence();
                }

            } /* main loop. */
    
        }

    }
    
    
/**
 * @brief Loop over the cell tuples and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_tuples_cuda ( struct part_cuda *parts , int *counts , int *ind ) {

    int threadID, blockID;
    int i, k, cid, cjd, ckd, ctn;
    __shared__ struct celltuple_cuda temp;
    __shared__ int finger;
    __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ], parts_k[ cuda_maxparts ];
    float shift[3];
    
    /* Get the block and thread ids. */
    blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Check in at the barrier. */
    if ( threadID == 0 )
        atomicAdd( &cuda_barrier , 1 );
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        

    /* Main loop... */
    while ( cuda_tuple_next < cuda_nr_tuples ) {
    
        /* Let the first thread get a pair. */
        if ( threadID == 0 ) {
        
            /* Lock the mutex. */
            cuda_mutex_lock( &cuda_cell_mutex );
            
            /* Get as many fingers as possible. */
            ctn = cuda_tuple_next;
            for ( i = ctn , finger = -1 ; finger < 0 && i < cuda_nr_tuples ; i++ ) {
                    
                /* Pick up this pair? */
                if ( cuda_taboo[ cuda_tuples[i].i ] == 0 &&
                     ( cuda_tuples[i].j < 0 || cuda_taboo[ cuda_tuples[i].j ] == 0 ) &&
                     ( cuda_tuples[i].k < 0 || cuda_taboo[ cuda_tuples[i].k ] == 0 ) ) {
                        
                    /* Swap entries in the pair list. */
                    temp = cuda_tuples[ i ];
                    cuda_tuples[ i ] = cuda_tuples[ ctn ];
                    cuda_tuples[ ctn ] = temp;
                    
                    /* Store this pair to the fingers. */
                    finger = ctn;
                    ctn += 1;
                    
                    }
                
                } /* get fingers. */
            
            /* Did we get anything? */
            if ( finger >= 0 ) {
            
                /* Mark the cells. */
                if ( temp.i >= 0 )
                    cuda_taboo[ temp.i ] += 1;
                if ( temp.j >= 0 )
                    cuda_taboo[ temp.j ] += 1;
                if ( temp.k >= 0 )
                    cuda_taboo[ temp.k ] += 1;
                    
                /* Store the modified cell_pair_next. */
                cuda_tuple_next = ctn;
                    
                /* Make sure everybody is on the same page. */
                __threadfence();
                
                }
                
            /* No, return empty-handed. */
            else
                __threadfence_block();
        
            /* Un-lock the mutex. */
            cuda_mutex_unlock( &cuda_cell_mutex );
            
            } /* threadID=0 doing it's own thing. */
            
            
            
        /* If we actually got a set of tuples, do them! */
        if ( finger >= 0 ) {
        
            /* Get the cell IDs. */
            cid = temp.i;
            cjd = temp.j;
            ckd = temp.k;
        
            /* Load the data first. */
            if ( cid >= 0 )  
                cuda_memcpy( parts_i , &parts[ind[cid]] , sizeof( struct part_cuda ) * counts[cid] );
            if ( cjd >= 0 )  
                cuda_memcpy( parts_j , &parts[ind[cjd]] , sizeof( struct part_cuda ) * counts[cjd] );
            if ( ckd >= 0 )  
                cuda_memcpy( parts_k , &parts[ind[ckd]] , sizeof( struct part_cuda ) * counts[ckd] );
            
            /* Make sure all the memory is in the right place. */
            __threadfence_block();
            
            /* Loop over the pairs in this tuple. */
            for ( k = 0 ; k < temp.nr_pairs ; k++ ) {
            
                switch ( temp.pairs[k] ) {
                
                    /* parts_i self-interaction. */
                    case 1:
                        if ( counts[cid] <= cuda_frame )
                            runner_doself_diag_cuda( parts_i , counts[cid] );
                        else
                            runner_doself_diag_cuda( parts_i , counts[cid] );
                        break;
                        
                    /* parts_j self-interaction. */
                    case 2:
                        if ( counts[cjd] <= cuda_frame )
                            runner_doself_cuda( parts_j , counts[cjd] );
                        else
                            runner_doself_diag_cuda( parts_j , counts[cjd] );
                        break;
                        
                    /* parts_i and parts_j interactions. */
                    case 3:
                        if ( counts[cid] <= 2*cuda_frame || counts[cjd] <= 2*cuda_frame )
                            runner_dopair_cuda(
                                parts_i , counts[cid] ,
                                parts_j , counts[cjd] ,
                                temp.shift_ij );
                        else
                            runner_dopair_sorted_cuda(
                                parts_i , counts[cid] ,
                                parts_j , counts[cjd] ,
                                temp.shift_ij );
                        break;
                        
                    /* parts_k self-interaction. */
                    case 4:
                        if ( counts[ckd] <= cuda_frame )
                            runner_doself_cuda( parts_k , counts[ckd] );
                        else
                            runner_doself_diag_cuda( parts_k , counts[ckd] );
                        break;
                        
                    /* parts_i and parts_k interactions. */
                    case 5:
                        if ( counts[cid] <= 2*cuda_frame || counts[ckd] <= 2*cuda_frame )
                            runner_dopair_cuda(
                                parts_i , counts[cid] ,
                                parts_k , counts[ckd] ,
                                temp.shift_ik );
                        else
                            runner_dopair_sorted_cuda(
                                parts_i , counts[cid] ,
                                parts_k , counts[ckd] ,
                                temp.shift_ik );
                        break;
                        
                    /* parts_j and parts_k interactions. */
                    case 6:
                        shift[0] = temp.shift_ik[0] - temp.shift_ij[0];
                        shift[1] = temp.shift_ik[1] - temp.shift_ij[1];
                        shift[2] = temp.shift_ik[2] - temp.shift_ij[2];
                        if ( counts[cjd] <= 2*cuda_frame || counts[ckd] <= 2*cuda_frame )
                            runner_dopair_cuda(
                                parts_j , counts[cjd] ,
                                parts_k , counts[ckd] ,
                                shift );
                        else
                            runner_dopair_sorted_cuda(
                                parts_j , counts[cjd] ,
                                parts_k , counts[ckd] ,
                                shift );
                        break;
                
                    }
            
                } /* loop over pairs in tuple. */
            
            /* Make sure all the memory is in the right place. */
            __threadfence_block();
            
            /* Write the data back. */
            if ( cid >= 0 )  
                cuda_memcpy( &parts[ind[cid]] , parts_i , sizeof( struct part_cuda ) * counts[cid] );
            if ( cjd >= 0 )  
                cuda_memcpy( &parts[ind[cjd]] , parts_j , sizeof( struct part_cuda ) * counts[cjd] );
            if ( ckd >= 0 )  
                cuda_memcpy( &parts[ind[ckd]] , parts_k , sizeof( struct part_cuda ) * counts[ckd] );
            
            /* Release the cells in the taboo list. */
            if ( threadID == 0 ) {
                if ( cid >= 0 )
                    cuda_taboo[ cid ] -= 1;
                if ( cjd >= 0 )
                    cuda_taboo[ cjd ] -= 1;                
                if ( ckd >= 0 )
                    cuda_taboo[ ckd ] -= 1;
                }
            
            /* Do we have to sync any memory? */
            __threadfence();
    
            }
            
        } /* main loop. */

    /* Check out at the barrier. */
    if ( threadID == 0 )
        atomicSub( &cuda_barrier , 1 );
    
    /* The last one out cleans up the mess... */
    if ( threadID == 0 && blockID == 0 ) {
        while ( atomicCAS( &cuda_barrier , 0 , 0 ) != 0 );
        cuda_tuple_next = 0;
        __threadfence();
        }

    }
    
    

