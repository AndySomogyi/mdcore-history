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
#include "runner_cuda.h"


/* the error macro. */
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )


/* The constant null potential. */
__constant__ struct potential *potential_null_cuda = NULL;

/* The number of cells and pairs. */
__constant__ int cuda_nr_pairs = 0;
__constant__ int cuda_nr_tuples = 0;
__constant__ int cuda_nr_cells = 0;

/* The mutex for accessing the cell pair list. */
__device__ int cuda_cell_mutex = 0;

/* The list of cell pairs. */
__device__ struct cellpair_cuda *cuda_pairs;
__device__ struct celltuple_cuda *cuda_tuples;
__device__ int *cuda_taboo;

/* The index of the next free cell pair. */
__device__ int cuda_pair_next = 0;
__device__ int cuda_tuple_next = 0;

/* Some constants. */
__constant__ float cuda_cutoff2 = 0.0f;
__constant__ float cuda_maxdist = 0.0f;
__constant__ struct potential **cuda_p;
__constant__ int cuda_maxtype = 0;
__constant__ struct potential *cuda_pots;

/* Sortlists for the Verlet algorithm. */
__device__ struct sortlist *cuda_sortlists = NULL;
__device__ int *cuda_sortlists_ind;

/* The potential coefficients, as a texture. */
texture< float , cudaTextureType2D > tex_coeffs;
texture< float , cudaTextureType2D > tex_alphas;
texture< int , cudaTextureType1D > tex_offsets;

/* Other textures. */
texture< int , cudaTextureType1D > tex_pind;
texture< int , cudaTextureType2D > tex_diags;

/* Arrays to hold the textures. */
cudaArray *cuda_coeffs, *cuda_alphas, *cuda_offsets, *cuda_pind, *cuda_diags;

/* The potential parameters (hard-wired size for now). */
__constant__ float cuda_eps[ 100 ];
__constant__ float cuda_rmin[ 100 ];

/* Use a set of variables to communicate with the outside world. */
__device__ float cuda_fio[32];
__device__ int cuda_io[32];


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
 * @brief Copy bulk memory in a strided way.
 *
 * @param dest Pointer to destination memory.
 * @param source Pointer to source memory.
 * @param count Number of bytes to copy, must be a multiple of sizeof(int).
 */
 
__device__ inline void cuda_memcpy ( void *dest , void *source , int count ) {

    int k, *idest = (int *)dest, *isource = (int *)source;
    
    /* Copy the data in chunks of sizeof(int). */
    for ( k = 0 + threadIdx.x ; k < count/sizeof(int) ; k += cuda_frame )
        idest[k] = isource[k];
        
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
    
    /* Get r for the right type. */
    ir = rsqrtf(r2);
    r = r2*ir;
    qir = q*ir;
    
    /* compute the interval index */
    // ind = fmaxf( 0.0f , tex2D( tex_alphas , 0 , pid ) + r * ( tex2D( tex_alphas , 1 , pid ) + r * tex2D( tex_alphas , 2 , pid ) ) );
    if ( ( ind = tex2D( tex_alphas , 0 , pid ) + r * ( tex2D( tex_alphas , 1 , pid ) + r * tex2D( tex_alphas , 2 , pid ) ) ) < 0 )
        ind = 0;
    ind += tex1D( tex_offsets , pid );
    
    /* pre-load the coefficients. */
    for ( k = 0 ; k < potential_chunk ; k++ )
        c[k] = tex2D( tex_coeffs , k , ind );
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    eff = c[2];
    ee = c[2] * x + c[3];
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        }

    /* store the result */
    *e = ee + qir;
    *f = ( eff * c[1] + qir ) * ir;
        
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
    
    /* Get r for the right type. */
    ir = rsqrtf(r2);
    r = r2*ir;
    
    /* compute the interval index */
    if ( ( ind = tex2D( tex_alphas , 0 , pid ) + r * ( tex2D( tex_alphas , 1 , pid ) + r * tex2D( tex_alphas , 2 , pid ) ) ) < 0 )
        ind = 0;
    ind += tex1D( tex_offsets , pid );
    
    /* pre-load the coefficients. */
    for ( k = 0 ; k < potential_chunk ; k++ )
        c[k] = tex2D( tex_coeffs , k , ind );
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    eff = c[2];
    ee = c[2] * x + c[3];
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        }

    /* store the result */
    *e = ee; *f = eff * c[1] * ir;
        
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
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        }

    /* store the result */
    *e = ee; *f = eff * c[1] * ir;
        
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

    int k, pid, pjd, ind, jnd, wrap_i, wrap_j, threadID;
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
    float epot = 0.0f, dx[3], pjx[3], pjf[3], shift[3], r2, w, ee = 0.0f, eff = 0.0f;
    
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
    wrap_j = (count_j < cuda_frame) ? cuda_frame : count_j;
    
    /* Make sure everybody is in the same place. */
    __threadfence_block();

    /* Loop over the particles in cell_j, frame-wise. */
    for ( jnd = 0 ; jnd < count_j ; jnd++ ) {
    
        /* Translate to a wrapped particle index. */
        if ( ( pjd = jnd + threadID ) >= wrap_j )
            pjd -= wrap_j;
        if ( pjd < count_j ) {
    
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
            for ( ind = 0 ; ind < count_i ; ind++ ) {
            
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
            
            } /* valid pjd? */
        
        } /* loop over the particles in cell_j. */
    
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
 
__device__ void runner_dopair_sorted_verlet_cuda ( struct part_cuda *parts_i , int count_i , struct part_cuda *parts_j , int count_j , float *pshift , int verlet_rebuild , struct sortlist *sortlist ) {

    int k, j, i, ind, jnd, pid, pjd, pjdid, threadID, wrap, cj;
    int pioff, swap_i;
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
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f, inshift, swap_f;
    float dx[3], pix[3], pif[3], shift[3];
    float maxdist = cuda_maxdist;
    __shared__ struct sortlist sort_i[ cuda_maxparts ], sort_j[ cuda_maxparts ];
    
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
    inshift = rsqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
       
    /* Re-build sorted pairs list? */
    if ( verlet_rebuild ) {
        
        /* Pack the parts of i and j into the sort arrays. */
        for ( k = threadID ; k < count_i ; k += cuda_frame ) {
            sort_i[k].d = inshift * (parts_i[k].x[0]*shift[0] + parts_i[k].x[1]*shift[1] + parts_i[k].x[2]*shift[2]);
            sort_i[k].ind = k;
            }
        for ( k = threadID ; k < count_j ; k += cuda_frame ) {
            sort_j[k].d = inshift * ((shift[0]+parts_j[k].x[0])*shift[0] + (shift[1]+parts_j[k].x[1])*shift[1] + (shift[2]+parts_j[k].x[2])*shift[2]);
            sort_j[k].ind = k;
            }
        /* for ( k = count_i + threadID ; k < cuda_maxparts ; k += cuda_frame )
            sort_i[k].d = -FLT_MAX;
        for ( k = count_j + threadID ; k < cuda_maxparts ; k += cuda_frame )
            sort_j[k].d = FLT_MAX; */
            
        /* Make sure all the memory is in the right place. */
        __threadfence_block();
        
        /* Sort using normalized bitonic sort. */
        for ( k = 1 ; k < count_i ; k *= 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_i ; i += cuda_frame ) {
                jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
                if ( jnd < count_i && sort_i[ind].d < sort_i[jnd].d ) {
                    swap_f = sort_i[ind].d; sort_i[ind].d = sort_i[jnd].d; sort_i[jnd].d = swap_f;
                    swap_i = sort_i[ind].ind; sort_i[ind].ind = sort_i[jnd].ind; sort_i[jnd].ind = swap_i;
                    }
                }
            __threadfence_block();
            for ( j = k/2 ; j > 0 ; j = j / 2 ) {
                for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_i ; i += cuda_frame ) {
                    jnd = ind + j;
                    if ( sort_i[ind].d < sort_i[jnd].d ) {
                        swap_f = sort_i[ind].d; sort_i[ind].d = sort_i[jnd].d; sort_i[jnd].d = swap_f;
                        swap_i = sort_i[ind].ind; sort_i[ind].ind = sort_i[jnd].ind; sort_i[jnd].ind = swap_i;
                        }
                    }
                __threadfence_block();
                }
            }
        for ( k = 1 ; k < count_j ; k *= 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_j ; i += cuda_frame ) {
                jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
                if ( jnd < count_j && sort_j[ind].d > sort_j[jnd].d ) {
                    swap_f = sort_j[ind].d; sort_j[ind].d = sort_j[jnd].d; sort_j[jnd].d = swap_f;
                    swap_i = sort_j[ind].ind; sort_j[ind].ind = sort_j[jnd].ind; sort_j[jnd].ind = swap_i;
                    }
                }
            __threadfence_block();
            for ( j = k/2 ; j > 0 ; j = j / 2 ) {
                for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_j ; i += cuda_frame ) {
                    jnd = ind + j;
                    if ( sort_j[ind].d > sort_j[jnd].d ) {
                        swap_f = sort_j[ind].d; sort_j[ind].d = sort_j[jnd].d; sort_j[jnd].d = swap_f;
                        swap_i = sort_j[ind].ind; sort_j[ind].ind = sort_j[jnd].ind; sort_j[jnd].ind = swap_i;
                        }
                    }
                __threadfence_block();
                }
            }

        /* Store the sorted list back to global memory. */
        cuda_memcpy( sortlist , sort_i , sizeof(struct sortlist) * count_i );
        cuda_memcpy( &sortlist[count_i] , sort_j , sizeof(struct sortlist) * count_j );
            
        } /* re-build sorted pairs list. */
        
    /* Otherwise, just read it from memory. */
    else {
        cuda_memcpy( sort_i , sortlist , sizeof(struct sortlist) * count_i );
        cuda_memcpy( sort_j , &sortlist[count_i] , sizeof(struct sortlist) * count_j );
        __threadfence_block();
        }
        
        
    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && sort_j[cj-1].d - sort_i[pid & ~(cuda_frame - 1)].d > maxdist )
            cj -= 1;
        wrap = ( cj < cuda_frame ) ? cuda_frame : cj;
            
        /* Get a direct pointer on the pjdth part in cell_j. */
        pi = &parts_i[ sort_i[pid].ind ];
        pioff = pi->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pix[k] = pi->x[k] - shift[k];
            pif[k] = 0.0f;
            }
        #if defined(USETEX_E) || defined(EXPLPOT)
        qi = pi->q;
        #endif
        
        /* Loop over the particles in cell_i. */
        for ( pjdid = 0 ; pjdid < cj ; pjdid++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pjd = pjdid + threadID ) >= wrap )
                pjd -= wrap;
            
            /* Do we have a pair? */
            if ( pjd < cj ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                pj = &parts_j[ sort_j[pjd].ind ];

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
    int pioff, swap_i;
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
    float epot = 0.0f, r2, w, ee = 0.0f, eff = 0.0f, inshift, swap_f;
    float dx[3], pix[3], pif[3], shift[3];
    float cutoff = sqrt(cuda_cutoff2);
    __shared__ struct sortlist sort_i[ cuda_maxparts ], sort_j[ cuda_maxparts ];
    
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
    inshift = rsqrtf( shift[0]*shift[0] + shift[1]*shift[1] + shift[2]*shift[2] );
       
    /* Pack the parts of i and j into the sort arrays. */
    for ( k = threadID ; k < count_i ; k += cuda_frame ) {
        sort_i[k].d = inshift * (parts_i[k].x[0]*shift[0] + parts_i[k].x[1]*shift[1] + parts_i[k].x[2]*shift[2]);
        sort_i[k].ind = k;
        }
    for ( k = threadID ; k < count_j ; k += cuda_frame ) {
        sort_j[k].d = inshift * ((shift[0]+parts_j[k].x[0])*shift[0] + (shift[1]+parts_j[k].x[1])*shift[1] + (shift[2]+parts_j[k].x[2])*shift[2]);
        sort_j[k].ind = k;
        }
        
    /* Make sure all the memory is in the right place. */
    __threadfence_block();
    
    /* Sort using normalized bitonic sort. */
    for ( k = 1 ; k < count_i ; k *= 2 ) {
        for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_i ; i += cuda_frame ) {
            jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
            if ( jnd < count_i && sort_i[ind].d < sort_i[jnd].d ) {
                swap_f = sort_i[ind].d; sort_i[ind].d = sort_i[jnd].d; sort_i[jnd].d = swap_f;
                swap_i = sort_i[ind].ind; sort_i[ind].ind = sort_i[jnd].ind; sort_i[jnd].ind = swap_i;
                }
            }
        __threadfence_block();
        for ( j = k/2 ; j > 0 ; j = j / 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_i ; i += cuda_frame ) {
                jnd = ind + j;
                if ( sort_i[ind].d < sort_i[jnd].d ) {
                    swap_f = sort_i[ind].d; sort_i[ind].d = sort_i[jnd].d; sort_i[jnd].d = swap_f;
                    swap_i = sort_i[ind].ind; sort_i[ind].ind = sort_i[jnd].ind; sort_i[jnd].ind = swap_i;
                    }
                }
            __threadfence_block();
            }
        }
    for ( k = 1 ; k < count_j ; k *= 2 ) {
        for ( i = threadID ; ( ind = ( i & ~(k - 1) ) * 2 + ( i & (k - 1) ) ) < count_j ; i += cuda_frame ) {
            jnd = ( i & ~(k - 1) ) * 2 + 2*k - ( i & (k - 1) ) - 1;
            if ( jnd < count_j && sort_j[ind].d > sort_j[jnd].d ) {
                swap_f = sort_j[ind].d; sort_j[ind].d = sort_j[jnd].d; sort_j[jnd].d = swap_f;
                swap_i = sort_j[ind].ind; sort_j[ind].ind = sort_j[jnd].ind; sort_j[jnd].ind = swap_i;
                }
            }
        __threadfence_block();
        for ( j = k/2 ; j > 0 ; j = j / 2 ) {
            for ( i = threadID ; ( ind = ( i & ~(j - 1) ) * 2 + ( i & (j - 1) ) ) + j < count_j ; i += cuda_frame ) {
                jnd = ind + j;
                if ( sort_j[ind].d > sort_j[jnd].d ) {
                    swap_f = sort_j[ind].d; sort_j[ind].d = sort_j[jnd].d; sort_j[jnd].d = swap_f;
                    swap_i = sort_j[ind].ind; sort_j[ind].ind = sort_j[jnd].ind; sort_j[jnd].ind = swap_i;
                    }
                }
            __threadfence_block();
            }
        }

    /* Loop over the particles in cell_j, frame-wise. */
    cj = count_j;
    for ( pid = threadID ; pid < count_i ; pid += cuda_frame ) {
    
        /* Get the wrap. */
        while ( cj > 0 && sort_j[cj-1].d - sort_i[pid & ~(cuda_frame - 1)].d > cutoff )
            cj -= 1;
        wrap = ( cj < cuda_frame ) ? cuda_frame : cj;
            
        /* Get a direct pointer on the pjdth part in cell_j. */
        pi = &parts_i[ sort_i[pid].ind ];
        pioff = pi->type * cuda_maxtype;
        for ( k = 0 ; k < 3 ; k++ ) {
            pix[k] = pi->x[k] - shift[k];
            pif[k] = 0.0f;
            }
        #if defined(USETEX_E) || defined(EXPLPOT)
        qi = pi->q;
        #endif
        
        /* Loop over the particles in cell_i. */
        for ( pjdid = 0 ; pjdid < cj ; pjdid++ ) {
        
            /* Wrap the particle index correctly. */
            if ( ( pjd = pjdid + threadID ) >= wrap )
                pjd -= wrap;
            
            /* Do we have a pair? */
            if ( pjd < cj ) {
            
                /* Get a handle on the wrapped particle pid in cell_i. */
                pj = &parts_j[ sort_j[pjd].ind ];

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
    
    /* Get the size of the frame, i.e. the number of threads in this block. */
    threadID = threadIdx.x % cuda_frame;
    
    /* Make sure everybody is in the same place. */
    __threadfence_block();
    
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
 
__global__ void runner_run_verlet_cuda ( struct part_cuda *parts , int *counts , int *ind , int verlet_rebuild ) {

    int threadID; //, blockID;
    int i, cpn, itemp;
    struct cellpair_cuda temp;
    int nr_fingers, finger[max_fingers], cid[max_fingers], cjd[max_fingers];
    __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    
    /* Get the block and thread ids. */
    // blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        

    /* Main loop... */
    while ( cuda_pair_next < cuda_nr_pairs ) {
    
        /* Lock the mutex. */
        if ( threadID == 0 )
            cuda_mutex_lock( &cuda_cell_mutex );
            
        /* Make sure everybody is in the same place. */
        __syncthreads();
        
        /* Get as many fingers as possible. */
        cpn = cuda_pair_next;
        for ( i = cpn , nr_fingers = 0 ; nr_fingers < max_fingers && i < cuda_nr_pairs ; i++ ) {
                
            /* Pick up this pair? */
            if ( cuda_taboo[ cuda_pairs[i].i ] == 0 &&
                 cuda_taboo[ cuda_pairs[i].j ] == 0 ) {
                    
                /* Swap entries in the pair list. */
                cuda_memcpy( &temp , &cuda_pairs[i] , sizeof(struct cellpair_cuda) );
                cuda_memcpy( &cuda_pairs[i] , &cuda_pairs[ cpn ] , sizeof(struct cellpair_cuda) );
                cuda_memcpy( &cuda_pairs[ cpn ] , &temp , sizeof(struct cellpair_cuda) );
                __threadfence_block();
                
                /* Swap sortlist index. */
                if ( threadID == 0 ) {
                    itemp = cuda_sortlists_ind[i];
                    cuda_sortlists_ind[i] = cuda_sortlists_ind[ cpn ];
                    cuda_sortlists_ind[ cpn ] = itemp;
                    }
                    
                /* Store this pair to the fingers. */
                finger[nr_fingers] = cpn;
                cid[nr_fingers] = cuda_pairs[ cpn ].i;
                cjd[nr_fingers] = cuda_pairs[ cpn ].j;
                nr_fingers += 1;
                cpn += 1;
                
                }
            
            } /* get fingers. */
            
        /* Only one thread needs to do the following */
        if ( threadID == 0 ) {
        
            /* Store the modified cell_pair_next. */
            cuda_pair_next = cpn;
                
            /* Mark the cells. */
            for ( i = 0 ; i < nr_fingers ; i++ ) {
                cuda_taboo[ cid[i] ] += 1;
                cuda_taboo[ cjd[i] ] += 1;
                }
            
            /* Make sure everybody is on the same page. */
            __threadfence();
        
            /* Un-lock the mutex. */
            cuda_mutex_unlock( &cuda_cell_mutex );
            
            }
            
            
        /* If we actually got a set of pairs, do them! */
        for ( i = 0 ; i < nr_fingers ; i++ ) {
        
            // if ( threadID == 0 )
            //     printf( "runner_run_cuda: block %i working on pair [%i,%i] (finger = %i).\n" , blockID , cid , cjd , finger );
        
            /* Do the pair. */
            if ( cid[i] != cjd[i] ) {
            
                /* Get a local copy of the particle data. */
                cuda_memcpy( parts_i , &parts[ind[cid[i]]] , sizeof( struct part_cuda ) * counts[cid[i]] );
                cuda_memcpy( parts_j , &parts[ind[cjd[i]]] , sizeof( struct part_cuda ) * counts[cjd[i]] );
                __threadfence_block();
                
                /* Compute the interactions. */
                runner_dopair_sorted_verlet_cuda(
                    parts_i , counts[cid[i]] ,
                    parts_j , counts[cjd[i]] ,
                    cuda_pairs[finger[i]].shift ,
                    verlet_rebuild , &cuda_sortlists[ cuda_sortlists_ind[ finger[i] ] ] );
                    
                /* Write the particle data back. */
                cuda_memcpy( &parts[ind[cid[i]]] , parts_i , sizeof( struct part_cuda ) * counts[cid[i]] );
                cuda_memcpy( &parts[ind[cjd[i]]] , parts_j , sizeof( struct part_cuda ) * counts[cjd[i]] );
                __threadfence_block();
                    
                }
            else {
                cuda_memcpy( parts_i , &parts[ind[cid[i]]] , sizeof( struct part_cuda ) * counts[cid[i]] );
                __threadfence_block();
                runner_doself_diag_cuda( parts_i , counts[cid[i]] );
                cuda_memcpy( &parts[ind[cid[i]]] , parts_i , sizeof( struct part_cuda ) * counts[cid[i]] );
                __threadfence_block();
                }

            /* Release the cells in the taboo list. */
            if ( threadID == 0 ) {
                cuda_taboo[ cid[i] ] -= 1;
                cuda_taboo[ cjd[i] ] -= 1;
                }
            
            }
            
        /* Do we have to sync any memory? */
        if ( nr_fingers > 0 )
            __threadfence();
    
        } /* main loop. */

    }
    
    
/**
 * @brief Loop over the cell pairs and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_cuda ( struct part_cuda *parts , int *counts , int *ind ) {

    int threadID; //, blockID;
    int i, cpn;
    struct cellpair_cuda temp;
    int nr_fingers, finger[max_fingers], cid[max_fingers], cjd[max_fingers];
    __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ];
    
    /* Get the block and thread ids. */
    // blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        

    /* Main loop... */
    while ( cuda_pair_next < cuda_nr_pairs ) {
    
        /* Lock the mutex. */
        if ( threadID == 0 )
            cuda_mutex_lock( &cuda_cell_mutex );
            
        /* Make sure everybody is in the same place. */
        __syncthreads();
        
        /* Get as many fingers as possible. */
        cpn = cuda_pair_next;
        for ( i = cpn , nr_fingers = 0 ; nr_fingers < max_fingers && i < cuda_nr_pairs ; i++ ) {
                
            /* Pick up this pair? */
            if ( cuda_taboo[ cuda_pairs[i].i ] == 0 &&
                 cuda_taboo[ cuda_pairs[i].j ] == 0 ) {
                    
                /* Swap entries in the pair list. */
                cuda_memcpy( &temp , &cuda_pairs[i] , sizeof(struct cellpair_cuda) );
                cuda_memcpy( &cuda_pairs[i] , &cuda_pairs[ cpn ] , sizeof(struct cellpair_cuda) );
                cuda_memcpy( &cuda_pairs[ cpn ] , &temp , sizeof(struct cellpair_cuda) );
                __threadfence_block();
                
                /* Store this pair to the fingers. */
                finger[nr_fingers] = cpn;
                cid[nr_fingers] = cuda_pairs[ cpn ].i;
                cjd[nr_fingers] = cuda_pairs[ cpn ].j;
                nr_fingers += 1;
                cpn += 1;
                
                }
            
            } /* get fingers. */
            
        /* Only one thread needs to do the following */
        if ( threadID == 0 ) {
        
            /* Store the modified cell_pair_next. */
            cuda_pair_next = cpn;
                
            /* Mark the cells. */
            for ( i = 0 ; i < nr_fingers ; i++ ) {
                cuda_taboo[ cid[i] ] += 1;
                cuda_taboo[ cjd[i] ] += 1;
                }
            
            /* Make sure everybody is on the same page. */
            __threadfence();
        
            /* Un-lock the mutex. */
            cuda_mutex_unlock( &cuda_cell_mutex );
            
            }
            
            
        /* If we actually got a set of pairs, do them! */
        for ( i = 0 ; i < nr_fingers ; i++ ) {
        
            /* Do the pair. */
            if ( cid[i] != cjd[i] ) {
            
                /* Get a local copy of the particle data. */
                cuda_memcpy( parts_i , &parts[ind[cid[i]]] , sizeof( struct part_cuda ) * counts[cid[i]] );
                cuda_memcpy( parts_j , &parts[ind[cjd[i]]] , sizeof( struct part_cuda ) * counts[cjd[i]] );
                __threadfence_block();
                
                /* Compute the cell pair interactions. */
                runner_dopair_sorted_cuda(
                    parts_i , counts[cid[i]] ,
                    parts_j , counts[cjd[i]] ,
                    cuda_pairs[finger[i]].shift );
                    
                /* Write the particle data back. */
                cuda_memcpy( &parts[ind[cid[i]]] , parts_i , sizeof( struct part_cuda ) * counts[cid[i]] );
                cuda_memcpy( &parts[ind[cjd[i]]] , parts_j , sizeof( struct part_cuda ) * counts[cjd[i]] );
                __threadfence_block();
                    
                }
            else {
                cuda_memcpy( parts_i , &parts[ind[cid[i]]] , sizeof( struct part_cuda ) * counts[cid[i]] );
                __threadfence_block();
                runner_doself_diag_cuda( parts_i , counts[cid[i]] );
                cuda_memcpy( &parts[ind[cid[i]]] , parts_i , sizeof( struct part_cuda ) * counts[cid[i]] );
                __threadfence_block();
                }
        
            /* Release the cells in the taboo list. */
            if ( threadID == 0 ) {
                cuda_taboo[ cid[i] ] -= 1;
                cuda_taboo[ cjd[i] ] -= 1;
                }
            
            }
            
        /* Do we have to sync any memory? */
        if ( nr_fingers > 0 )
            __threadfence();
    
        } /* main loop. */

    }
    
    
/**
 * @brief Loop over the cell tuples and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_tuples_cuda ( struct part_cuda *parts , int *counts , int *ind ) {

    int threadID; //, blockID;
    int i, k, ctn;
    struct celltuple_cuda temp;
    int finger;
    __shared__ struct part_cuda parts_i[ cuda_maxparts ], parts_j[ cuda_maxparts ], parts_k[ cuda_maxparts ];
    float shift[3];
    
    /* Get the block and thread ids. */
    // blockID = blockIdx.x;
    threadID = threadIdx.x;
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        

    /* Main loop... */
    while ( cuda_tuple_next < cuda_nr_tuples ) {
    
        /* Lock the mutex. */
        if ( threadID == 0 )
            cuda_mutex_lock( &cuda_cell_mutex );
            
        /* Make sure everybody is in the same place. */
        __syncthreads();
        
        /* Get as many fingers as possible. */
        ctn = cuda_tuple_next;
        for ( i = ctn , finger = -1 ; finger < 0 && i < cuda_nr_tuples ; i++ ) {
                
            /* Pick up this tuple? */
            if ( cuda_taboo[ cuda_tuples[i].i ] == 0 &&
                 ( cuda_tuples[i].j < 0 || cuda_taboo[ cuda_tuples[i].j ] == 0 ) &&
                 ( cuda_tuples[i].k < 0 || cuda_taboo[ cuda_tuples[i].k ] == 0 ) ) {
                    
                /* Swap entries in the tuple list. */
                cuda_memcpy( &temp , &cuda_tuples[i] , sizeof(struct celltuple_cuda) );
                cuda_memcpy( &cuda_tuples[i] , &cuda_tuples[ ctn ] , sizeof(struct celltuple_cuda) );
                cuda_memcpy( &cuda_tuples[ ctn ] , &temp , sizeof(struct celltuple_cuda) );
                
                /* Store this tuple to the fingers. */
                finger = ctn;
                ctn += 1;
                
                }
            
            } /* get fingers. */
            
        /* Only one thread needs to do the following */
        if ( threadID == 0 ) {
        
            /* Store the modified cell_tuple_next. */
            cuda_tuple_next = ctn;
                
            /* Mark the cells. */
            if ( finger >= 0 ) {
                temp = cuda_tuples[ finger ];
                if ( temp.i >= 0 )
                    cuda_taboo[ temp.i ] += 1;
                if ( temp.j >= 0 )
                    cuda_taboo[ temp.j ] += 1;
                if ( temp.k >= 0 )
                    cuda_taboo[ temp.k ] += 1;
                }
            
            /* Make sure everybody is on the same page. */
            __threadfence();
        
            /* Un-lock the mutex. */
            cuda_mutex_unlock( &cuda_cell_mutex );
            
            }
            
            
        /* If we actually got a set of tuples, do them! */
        if ( finger >= 0 ) {
        
            /* Get a local copy of this tuple. */
            temp = cuda_tuples[ finger ];
        
            /* Load the data first. */
            if ( temp.i >= 0 )  
                cuda_memcpy( parts_i , &parts[ind[temp.i]] , sizeof( struct part_cuda ) * counts[temp.i] );
            if ( temp.j >= 0 )  
                cuda_memcpy( parts_j , &parts[ind[temp.j]] , sizeof( struct part_cuda ) * counts[temp.j] );
            if ( temp.k >= 0 )  
                cuda_memcpy( parts_k , &parts[ind[temp.k]] , sizeof( struct part_cuda ) * counts[temp.k] );
            
            /* Make sure all the memory is in the right place. */
            __threadfence_block();
            
            /* Loop over the pairs in this tuple. */
            for ( k = 0 ; k < temp.nr_pairs ; k++ ) {
            
                switch ( temp.pairs[k] ) {
                
                    /* parts_i self-interaction. */
                    case 1:
                        runner_doself_diag_cuda( parts_i , counts[temp.i] );
                        break;
                        
                    /* parts_j self-interaction. */
                    case 2:
                        runner_doself_diag_cuda( parts_j , counts[temp.j] );
                        break;
                        
                    /* parts_i and parts_j interactions. */
                    case 3:
                        runner_dopair_sorted_cuda(
                            parts_i , counts[temp.i] ,
                            parts_j , counts[temp.j] ,
                            temp.shift_ij );
                        break;
                        
                    /* parts_k self-interaction. */
                    case 4:
                        runner_doself_diag_cuda( parts_k , counts[temp.k] );
                        break;
                        
                    /* parts_i and parts_k interactions. */
                    case 5:
                        runner_dopair_sorted_cuda(
                            parts_i , counts[temp.i] ,
                            parts_k , counts[temp.k] ,
                            temp.shift_ik );
                        break;
                        
                    /* parts_j and parts_k interactions. */
                    case 6:
                        shift[0] = temp.shift_ik[0] - temp.shift_ij[0];
                        shift[1] = temp.shift_ik[1] - temp.shift_ij[1];
                        shift[2] = temp.shift_ik[2] - temp.shift_ij[2];
                        runner_dopair_sorted_cuda(
                            parts_j , counts[temp.j] ,
                            parts_k , counts[temp.k] ,
                            shift );
                        break;
                
                    }
            
                } /* loop over pairs in tuple. */
            
            /* Make sure all the memory is in the right place. */
            __threadfence_block();
            
            /* Write the data back. */
            if ( temp.i >= 0 )  
                cuda_memcpy( &parts[ind[temp.i]] , parts_i , sizeof( struct part_cuda ) * counts[temp.i] );
            if ( temp.j >= 0 )  
                cuda_memcpy( &parts[ind[temp.j]] , parts_j , sizeof( struct part_cuda ) * counts[temp.j] );
            if ( temp.k >= 0 )  
                cuda_memcpy( &parts[ind[temp.k]] , parts_k , sizeof( struct part_cuda ) * counts[temp.k] );
            
            /* Release the cells in the taboo list. */
            if ( threadID == 0 ) {
                if ( temp.i >= 0 )
                    cuda_taboo[ temp.i ] -= 1;
                if ( temp.j >= 0 )
                    cuda_taboo[ temp.j ] -= 1;                
                if ( temp.k >= 0 )
                    cuda_taboo[ temp.k ] -= 1;
                }
            
            /* Do we have to sync any memory? */
            __threadfence();
    
            }
            
        } /* main loop. */

    }
    
    

