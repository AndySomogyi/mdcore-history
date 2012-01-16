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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>

/* Include conditional headers. */
#include "../config.h"
#ifdef HAVE_MPI
    #include <mpi.h>
#endif
#ifdef HAVE_OPENMP
    #include <omp.h>
#endif

/* include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "runner.h"
#include "bond.h"
#include "rigid.h"
#include "angle.h"
#include "dihedral.h"
#include "exclusion.h"
#include "reader.h"
#include "engine.h"


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )


/* As of here there is only CUDA-related stuff. */
#ifdef HAVE_CUDA


/* Forward declaration of runner kernel. */
__global__ void runner_run_cuda ( struct part *parts[] , int *counts );


/**
 * @brief Offload and compute the nonbonded interactions on a CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_nonbond_cuda ( struct engine *e ) {

    dim3 nr_threads( 32 , 1 );
    dim3 nr_blocks( e->nr_runners , 1 );

    /* Load the particle data onto the device. */
    if ( engine_cuda_load_parts( e ) < 0 )
        return error(engine_err);
        
    /* Start the kernel. */
    runner_run_cuda<<<nr_blocks,nr_threads>>>( e->s.parts_cuda , e->s.counts_cuda );
    
    /* Check for CUDA errors. */
    if ( cudaGetLastError() != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Unload the particle data from the device. */
    if ( engine_cuda_unload_parts( e ) < 0 )
        return error(engine_err);

    /* Go away. */
    return engine_err_ok;
    
    }



/**
 * @brief Load the cell data onto the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_cuda_load_parts ( struct engine *e ) {
    
    int counts[ e->s.nr_cells ];
    int k, cid;
    
    /* Clear the counts array. */
    bzero( counts , sizeof(int) * e->s.nr_cells );
    
    /* Loop over the marked cells. */
    for ( k = 0 ; k < e->s.nr_marked ; k++ ) {
    
        /* Get the cell id. */
        cid = e->s.cid_marked[k];
        
        /* Allocate memory on the device for the parts of this cell. */
        counts[cid] = e->s.cells[cid].count;
        if ( cudaMalloc( &e->s.parts_cuda_local[cid] , sizeof(struct part) * counts[cid] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Copy the particle data to the device. */
        if ( cudaMemcpy( e->s.parts_cuda_local[cid] , e->s.cells[cid].parts , sizeof(struct part) * counts[cid] , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    
        }
        
    /* Allocate and copy the counts onto the device. */
    if ( cudaMalloc( &e->s.counts_cuda , sizeof(int) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( e->s.counts_cuda , counts , sizeof(int) * e->s.nr_cells , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Finally, push the parts array onto the device. */
    if ( cudaMalloc( &e->s.parts_cuda , sizeof(struct part *) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( e->s.parts_cuda , e->s.parts_cuda_local , sizeof(struct part *) * e->s.nr_cells , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Our work is done here. */
    return engine_err_ok;

    }
    
    

/**
 * @brief Load the cell data from the CUDA device.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
extern "C" int engine_cuda_unload_parts ( struct engine *e ) {
    
    int k, cid;
    
    /* Loop over the marked cells. */
    for ( k = 0 ; k < e->s.nr_marked ; k++ ) {
    
        /* Get the cell id. */
        cid = e->s.cid_marked[k];
        
        /* Copy the particle data from the device. */
        if ( cudaMemcpy( e->s.cells[cid].parts , e->s.parts_cuda_local[cid] , sizeof(struct part) * e->s.cells[cid].count , cudaMemcpyDeviceToHost ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Deallocate the parts array on the device. */
        if ( cudaFree( e->s.parts_cuda_local[cid] ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    
        }
        
    /* Deallocate the pointer array and counts array. */
    if ( cudaFree( e->s.parts_cuda ) != cudaSuccess ||
         cudaFree( e->s.counts_cuda ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Our work is done here. */
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

    int i, j, nr_pots = 1, nr_coeffs = 1;
    struct potential *pots[ e->nr_types * (e->nr_types + 1) / 2 + 1 ];
    struct potential *p_cuda[ e->max_type * e->max_type ];
    struct potential *pots_cuda;
    struct cellpair_cuda *pairs_cuda;
    float *finger, *coeffs_cuda, cutoff2 = e->s.cutoff2;
    
    /* Init the null potential. */
    if ( ( pots[0] = (struct potential *)alloca( sizeof(struct potential) ) ) == NULL )
        return error(engine_err_malloc);
    pots[0]->alpha[0] = pots[0]->alpha[1] = pots[0]->alpha[2] = pots[0]->alpha[3] = 0.0f;
    pots[0]->a = 0.0; pots[0]->b = DBL_MAX;
    pots[0]->flags = potential_flag_none;
    pots[0]->n = 0;
    if ( ( pots[0]->c = (float *)alloca( sizeof(float) * potential_chunk ) ) == NULL )
        return error(engine_err_malloc);
    bzero( pots[0]->c , sizeof(float) * potential_chunk );
    
    /* Start by identifying the unique potentials in the engine. */
    for ( i = 0 ; i < e->nr_types * e->max_type ; i++ ) {
    
        /* Skip if there is no potential. */
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
    
        }
        
    /* Allocate space on the device for both the potential structures
       and the coefficient tables. */
    if ( cudaMalloc( &(e->p_cuda) , sizeof(struct potential *) * e->max_type * e->max_type ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMalloc( &(e->pots_cuda) , sizeof(struct potential) * nr_pots ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMalloc( &(e->coeffs_cuda) , sizeof(float) * nr_coeffs * potential_chunk ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Pack the potentials before shipping them off to the device. */
    if ( ( pots_cuda = (struct potential *)alloca( sizeof(struct potential) * nr_pots ) ) == NULL )
        return error(engine_err_malloc);
    for ( finger = e->coeffs_cuda , i = 0 ; i < nr_pots ; i++ ) {
        pots_cuda[i] = *pots[i];
        pots_cuda[i].c = finger;
        finger = &finger[ (pots_cuda[i].n + 1) * potential_chunk ];
        }
        
    /* Pack the potential matrix. */
    for ( i = 0 ; i < e->max_type * e->max_type ; i++ ) {
        if ( e->p[i] == NULL )
            p_cuda[i] = NULL;
        else {
            for ( j = 0 ; j < nr_pots && pots[j] != e->p[i] ; j++ );
            p_cuda[i] = &(e->pots_cuda[j]);
            }
        }
        
    /* Pack the coefficients before shipping them off to the device. */
    if ( ( coeffs_cuda = (float *)alloca( sizeof(float) * nr_coeffs * potential_chunk ) ) == NULL )
        return error(engine_err_malloc);
    for ( finger = coeffs_cuda , i = 0 ; i < nr_pots ; i++ ) {
        memcpy( finger , pots[i]->c , sizeof(float) * potential_chunk * (pots[i]->n + 1) );
        finger = &finger[ (pots[i]->n + 1) * potential_chunk ];
        }
        
    /* Copy the data to the device. */
    if ( cudaMemcpy( e->p_cuda , p_cuda , sizeof(struct potential *) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( e->pots_cuda , pots_cuda , sizeof(struct potential) * nr_pots , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( e->coeffs_cuda , coeffs_cuda , sizeof(float) * nr_coeffs * potential_chunk , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Bind the potential coefficients to a texture. */
    const textureReference* texRefPtr;
    cudaGetTextureReference( &texRefPtr , "tex_pots");
    cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<float>( );
    cudaBindTexture( NULL , texRefPtr , e->coeffs_cuda , &channelDesc , nr_coeffs * potential_chunk );
        
    /* Set the constant pointer to the null potential and other useful values. */
    if ( cudaMemcpyToSymbol( "potential_null_cuda" , &(e->pots_cuda) , sizeof(struct potential *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_p" , &(e->p_cuda) , sizeof(struct potential **) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_cutoff2" , &cutoff2 , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_maxtype" , &(e->max_type) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Allocate and fill the compact list of pairs. */
    if ( ( pairs_cuda = (struct cellpair_cuda *)alloca( sizeof(struct cellpair_cuda) * e->s.nr_pairs ) ) == NULL )
        return error(engine_err_malloc);
    for ( i = 0 ; i < e->s.nr_pairs ; i++ ) {
        pairs_cuda[i].i = e->s.pairs[i].i;
        pairs_cuda[i].j = e->s.pairs[i].j;
        pairs_cuda[i].shift[0] = e->s.pairs[i].shift[0];
        pairs_cuda[i].shift[1] = e->s.pairs[i].shift[1];
        pairs_cuda[i].shift[2] = e->s.pairs[i].shift[2];
        }
        
    /* Allocate and fill the pairs list on the device. */
    if ( cudaMalloc( &e->s.pairs_cuda , sizeof(struct cellpair_cuda) * e->s.nr_pairs ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( e->s.pairs_cuda , pairs_cuda , sizeof(struct cellpair_cuda) * e->s.nr_pairs , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_pairs" , &(e->s.pairs_cuda) , sizeof(struct cellpair_cuda *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);

    /* Set the number of pairs and cells. */
    if ( cudaMemcpyToSymbol( "cuda_nr_pairs" , &(e->s.nr_pairs) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_nr_cells" , &(e->s.nr_cells) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Allocate and init the taboo list on the device. */
    if ( cudaMalloc( &e->s.taboo_cuda , sizeof(int) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemset( e->s.taboo_cuda , 0 , sizeof(int) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_taboo" , &(e->s.taboo_cuda) , sizeof(int *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* He's done it! */
    return engine_err_ok;
    
    }
    
    
/* End CUDA-related stuff. */
#endif
