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
#define cuda_maxparts 160
#define cuda_frame 32
#define cuda_maxpots 100


/* the error macro. */
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )


/* Use textured or global potential data? */
#define USETEX 1


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
__constant__ int *cuda_pind;
__constant__ int cuda_maxtype = 0;
__constant__ struct potential *cuda_pots;

/* The potential coefficients, as a texture. */
texture< float , cudaTextureType2D > tex_coeffs;
texture< float , cudaTextureType2D > tex_alphas;
texture< int , cudaTextureType1D > tex_offsets;
cudaArray *cuda_coeffs, *cuda_alphas, *cuda_offsets;

/* Use a set of variables to communicate with the outside world. */
__device__ float cuda_fio[10];
__device__ int cuda_io[10];


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
    float x, ee, eff, r;
    struct potential *p = &cuda_pots[pid];
    
    /* Get r for the right type. */
    r = sqrtf(r2);
    
    /* compute the interval index */
    ind = fmaxf( 0.0f , tex2D( tex_alphas , 0 , pid ) + r * ( tex2D( tex_alphas , 1 , pid ) + r * tex2D( tex_alphas , 2 , pid ) ) );
    ind += tex1D( tex_offsets , pid );
    
    /* adjust x to the interval */
    x = (r - tex2D( tex_coeffs , 0 , ind ) ) * tex2D( tex_coeffs , 1 , ind );
    
    /* compute the potential and its derivative */
    ee = tex2D( tex_coeffs , 2 , ind ) * x + tex2D( tex_coeffs , 3 , ind );
    eff = tex2D( tex_coeffs , 2 , ind );
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + tex2D( tex_coeffs , k , ind );
        }

    /* store the result */
    *e = ee; *f = eff * tex2D( tex_coeffs , 1 , ind ) / r;
        
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
 * @param iparts_i Array of parts in the first cell.
 * @param count_i Number of parts in the first cell.
 * @param iparts_j Array of parts in the second cell.
 * @param count_j Number of parts in the second cell.
 * @param pshift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 *
 * @sa #runner_dopair.
 */
 
__device__ void runner_dopair_cuda ( struct part *iparts_i , int count_i , struct part *iparts_j , int count_j, float *pshift ) {

    int k, pid, pjd, ind, wrap, threadID;
    int pjoff;
    struct part_cuda *pi, *pj;
    struct part *temp;
    #ifdef USETEX
        int pot;
    #else
        struct potential *pot;
    #endif
    float epot = 0.0f, dx[3], pjx[3], pjf[3], shift[3], r2, w, ee, eff;
    __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Swap cells? cell_j loops in steps of frame... */
    if ( ( ( count_i + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_i < ( ( count_j + (cuda_frame-1) ) & ~(cuda_frame-1) ) - count_j ) {
        temp = iparts_i; iparts_i = iparts_j; iparts_j = temp;
        k = count_i; count_i = count_j; count_j = k;
        shift[0] = -pshift[0]; shift[1] = -pshift[1]; shift[2] = -pshift[2];
        }
    else {
        shift[0] = pshift[0]; shift[1] = pshift[1]; shift[2] = pshift[2];
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
        
    /* Get the wrap. */
    if ( ( wrap = count_i ) < cuda_frame )
        wrap = cuda_frame;
    
    /* Make sure everybody is in the same place. */
    __syncthreads();

    /* Loop over the particles in cell_j, frame-wise. */
    for ( pjd = threadID ; pjd < count_j ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        pj = &parts_j[pjd];
        pjoff = pj->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pjx[k] = pj->x[k] + shift[k];
            pjf[k] = 0.0f;
            }
        
        /* Loop over the particles in cell_i. */
        for ( pid = 0 ; pid < count_i ; pid++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( ind = pid + threadID ) >= wrap )
                ind -= wrap;
            
            /* Do we have a pair? */
            if ( ind < count_i ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                pi = &parts_i[ ind ];
                // printf( "runner_dopair_cuda: doing pair [%i,%i].\n" , pjd , ind );

                /* Compute the radius between pi and pj. */
                for ( r2 = 0.0f , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pi->x[k] - pjx[k];
                    r2 += dx[k] * dx[k];
                    }

                /* Set the null potential if anything is bad. */
                #ifdef USETEX
                if ( r2 < cuda_cutoff2 && ( pot = cuda_pind[ pjoff + pi->type ] ) != 0 ) {
                #else
                if ( r2 < cuda_cutoff2 && ( pot = cuda_p[ pjoff + pi->type ] ) != NULL ) {
                #endif

                    /* Interact particles pi and pj. */
                    #ifdef USETEX
                    potential_eval_cuda_tex( pot , r2 , &ee , &eff );
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

                } /* do we have a pair? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            pj->f[k] += pjf[k];
    
        /* Sync the shared memory values. */
        __threadfence_block();
        
        } /* loop over the particles in cell_j. */
    
    /* Make sure everybody is in the same place. */
    __syncthreads();

    /* Copy the particle data back from the local buffers */
    for ( k = threadID ; k < count_i ; k += cuda_frame ) {
        iparts_i[k].f[0] = parts_i[k].f[0];
        iparts_i[k].f[1] = parts_i[k].f[1];
        iparts_i[k].f[2] = parts_i[k].f[2];
        }
    for ( k = threadID ; k < count_j ; k += cuda_frame ) {
        iparts_j[k].f[0] = parts_j[k].f[0];
        iparts_j[k].f[1] = parts_j[k].f[1];
        iparts_j[k].f[2] = parts_j[k].f[2];
        }
        
    }


/**
 * @brief Compute the self interactions for the given cell on a CUDA device.
 *
 * @param iparts Array of parts in this cell.
 * @param count Number of parts in the cell.
 *
 * @sa #runner_dopair.
 */
 
__device__ void runner_doself_cuda ( struct part *iparts , int count ) {

    int k, ind, wrap, pid, pjd, threadID;
    int pjoff;
    struct part_cuda *pi, *pj;
    #ifdef USETEX
        int pot;
    #else
        struct potential *pot;
    #endif
    float epot = 0.0f, dx[3], pjx[3], pjf[3], r2, w, ee, eff;
    __shared__ struct part_cuda parts[ cuda_maxparts ];
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Copy the particle data to the local buffers */
    for ( k = threadID ; k < count ; k += cuda_frame ) {
        parts[k].x[0] = iparts[k].x[0];
        parts[k].x[1] = iparts[k].x[1];
        parts[k].x[2] = iparts[k].x[2];
        parts[k].f[0] = iparts[k].f[0];
        parts[k].f[1] = iparts[k].f[1];
        parts[k].f[2] = iparts[k].f[2];
        parts[k].type = iparts[k].type;
        }
    
    /* Make sure everybody is in the same place. */
    __syncthreads();

    /* Loop over the particles in the cell, frame-wise. */
    for ( pjd = threadID ; pjd < count ; pjd += cuda_frame ) {
    
        /* Get a direct pointer on the pjdth part in cell_j. */
        pj = &parts[pjd];
        pjoff = pj->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pjx[k] = pj->x[k];
            pjf[k] = 0.0f;
            }
            
        /* Set the wrapping. */
        wrap = (pjd + (cuda_frame - 1)) & ~(cuda_frame - 1);
        
        /* Loop over the particles in cell_i. */
        for ( pid = 0 ; pid < wrap ; pid++ ) {
        
            /* Get the correct wrapped id. */
            if ( ( ind = pid + threadID ) >= wrap )
                ind -= wrap;
                
            /* Valid particle pair? */
            if ( ind < pjd ) {
                
                // if ( threadID == 0 )
                // printf( "runner_doself_cuda: doing pair [%i,%i].\n" , pjd , ind );

                /* Get a handle on the wrapped particle pid in cell_i. */
                pi = &parts[ ind ];

                /* Compute the radius between pi and pj. */
                for ( r2 = 0.0f , k = 0 ; k < 3 ; k++ ) {
                    dx[k] = pi->x[k] - pjx[k];
                    r2 += dx[k] * dx[k];
                    }

                /* Set the null potential if anything is bad. */
                #ifdef USETEX
                if ( r2 < cuda_cutoff2 && ( pot = cuda_pind[ pjoff + pi->type ] ) != 0 ) {
                #else
                if ( r2 < cuda_cutoff2 && ( pot = cuda_p[ pjoff + pi->type ] ) != NULL ) {
                #endif

                    /* Interact particles pi and pj. */
                    #ifdef USETEX
                    potential_eval_cuda_tex( pot , r2 , &ee , &eff );
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

                } /* valid particle pair? */
        
            } /* loop over parts in cell_i. */
            
        /* Update the force on pj. */
        for ( k = 0 ; k < 3 ; k++ )
            pj->f[k] += pjf[k];
    
        /* Sync the shared memory values. */
        __threadfence_block();

        } /* loop over the particles in cell_j. */
    
    /* Make sure everybody is in the same place. */
    __syncthreads();

    /* Copy the particle data back from the local buffers */
    for ( k = threadID ; k < count ; k += cuda_frame ) {
        iparts[k].f[0] = parts[k].f[0];
        iparts[k].f[1] = parts[k].f[1];
        iparts[k].f[2] = parts[k].f[2];
        }
        
    }
    
    
/**
 * @brief Bind textures to the given cuda Arrays.
 *
 *
 * Hack to get around the fact that textures are static and can thus no
 * be externalized.
 */
 
int runner_bind ( cudaArray *cuArray_coeffs , cudaArray *cuArray_offsets , cudaArray *cuArray_alphas ) {

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
        
    /* Rock and roll. */
    return runner_err_ok;

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
    blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Check that we've got the correct warp size! */
    if ( warpSize != cuda_frame ) {
        /* if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame ); */
        return;
        }
    
    /* Greetings, earthling. */
    // if ( threadID == 0 )
    //     printf( "runner_run_cuda: thread %i of block %i says hi.\n" , threadID , blockID );
    
    /* Get some values from the textures to make sure they're ok. */
    if ( blockID == 0 && threadID == 0 ) {
        cuda_fio[0] = tex2D( tex_alphas , 0 , 0 );
        cuda_fio[1] = tex2D( tex_alphas , 1 , 0 );
        cuda_fio[2] = tex2D( tex_alphas , 2 , 0 );
        cuda_fio[3] = tex2D( tex_coeffs , 0 , 1 );
        cuda_fio[4] = tex2D( tex_coeffs , 1 , 1 );
        cuda_fio[5] = tex2D( tex_coeffs , 2 , 1 );
        cuda_io[0] = tex1D( tex_offsets , 0 );
        cuda_io[1] = tex1D( tex_offsets , 1 );
        cuda_io[2] = tex1D( tex_offsets , 2 );
        }
    
    /* If I'm the first thread in the first block, re-set the next pair. */
    if ( blockID == 0 && threadID == 0 )
        cuda_pair_next = 0;
        
    /* Make sure everybody is on the same page. */
    __threadfence();
            
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
                cuda_pair_next += 1;
                cuda_taboo[ cid ] = 1;
                cuda_taboo[ cjd ] = 1;
                }
            else
                finger = -1;
            
            /* Make sure everybody is on the same page. */
            __threadfence();
        
            /* Un-lock the mutex. */
            cuda_mutex_unlock( &cuda_cell_mutex );
            
            }
            
        /* Get everybody together. */
        __syncthreads();
            
        /* If we actually got a pair, do it! */
        if ( finger >= 0 ) {
        
            // if ( threadID == 0 )
            //     printf( "runner_run_cuda: block %i working on pair [%i,%i] (finger = %i).\n" , blockID , cid , cjd , finger );
        
            /* Do the pair. */
            if ( cid != cjd )
                runner_dopair_cuda( parts[cid] , counts[cid] , parts[cjd] , counts[cjd] , cuda_pairs[finger].shift );
            else
                runner_doself_cuda( parts[cid] , counts[cid] );
        
            /* Release the cells in the taboo list. */
            if ( threadID == 0 ) {
                cuda_taboo[ cid ] = 0;
                cuda_taboo[ cjd ] = 0;
                }
            
            /* Make sure everybody is on the same page. */
            __threadfence();
        
            }
    
        /* Get everybody together. */
        __syncthreads();
            
        } /* main loop. */

    }

