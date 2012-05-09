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

/* Disable vectorization for the nvcc compiler's sake. */
#undef __SSE__
#undef __SSE2__
#undef __ALTIVEC__
#undef __AVX__

/* include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "cell.h"
#include "fifo.h"
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
#include "runner_cuda.h"


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )
#define cuda_error(id)			( engine_err = errs_register( id , cudaGetErrorString(cudaGetLastError()) , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* As of here there is only CUDA-related stuff. */
#ifdef HAVE_CUDA


/* Forward declaration of runner kernel. */
__global__ void runner_run_verlet_cuda ( struct part_cuda *parts , int *counts , int *ind , int verlet_rebuild );
__global__ void runner_run_cuda ( struct part_cuda *parts , int *counts , int *ind );
__global__ void runner_run_cuda_new ( struct part_cuda *parts , int *counts , int *ind );
__global__ void runner_run_loose_cuda ( struct part_cuda *parts , int *counts , int *ind );
__global__ void runner_run_tuples_cuda ( struct part_cuda *parts , int *counts , int *ind );
__global__ void runner_run_dispatcher_cuda ( struct part_cuda *parts , int *counts , int *ind );
int runner_bind ( cudaArray *cuArray_coeffs , cudaArray *cuArray_offsets , cudaArray *cuArray_alphas , cudaArray *cuArray_pind , cudaArray *cuArray_diags );


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
    // int zero = 0;
    // int cuda_io[32];
    // float cuda_fio[32];
    #ifdef TIMERS
        float timers[ tid_count ];
        double icpms = 1000.0 / 1.4e9; 
    #endif

    /* Load the particle data onto the device. */
    if ( engine_cuda_load_parts( e ) < 0 )
        return error(engine_err);

    /* Init the pointer to the next entry. */    
    /* if ( cudaMemcpyToSymbol( "cuda_pair_next" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_tuple_next" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */
    /* if ( cudaMemcpyToSymbol( "cuda_pair_count" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_pair_curr" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */
    /* if ( cudaMemcpyToSymbol( "cuda_pairs_done" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */
    /* if ( cudaMemcpyToSymbol( "cuda_cell_mutex" , &zero , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda); */
        
    /* Re-set timers */
    #ifdef TIMERS
        for ( int k = 0 ; k < tid_count ; k++ )
            timers[k] = 0.0f;
        if ( cudaMemcpyToSymbol( "cuda_timers" , timers , sizeof(float) * tid_count , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    #endif
    
    /* Start the kernel. */
    if ( e->flags & engine_flag_dispatch )
        runner_run_dispatcher_cuda<<<nr_blocks,nr_threads>>>( e->s.parts_cuda , e->s.counts_cuda , e->s.ind_cuda );
    else if ( e->flags & engine_flag_verlet )
        runner_run_verlet_cuda<<<nr_blocks,nr_threads>>>( e->s.parts_cuda , e->s.counts_cuda , e->s.ind_cuda , e->s.verlet_rebuild );
    else if ( e->flags & engine_flag_tuples )
        runner_run_tuples_cuda<<<nr_blocks,nr_threads>>>( e->s.parts_cuda , e->s.counts_cuda , e->s.ind_cuda );
    else
        runner_run_cuda<<<nr_blocks,nr_threads>>>( e->s.parts_cuda , e->s.counts_cuda , e->s.ind_cuda );
    
    /* Check for CUDA errors. */
    if ( cudaPeekAtLastError() != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Get and dump timers. */
    #ifdef TIMERS
        if ( cudaMemcpyFromSymbol( timers , "cuda_timers" , sizeof(float) * tid_count , 0 , cudaMemcpyDeviceToHost ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        printf( "engine_nonbond_cuda: timers = [ %.2f " , icpms * timers[0] );
        for ( int k = 1 ; k < tid_count ; k++ )
            printf( "%.2f " , icpms * timers[k] );
        printf( "] ms\n" );
    #endif
        
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
    
    /* if ( cudaMemcpyFromSymbol( &zero , "cuda_pairs_done" , sizeof(int) , 0 , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    printf( "engine_nonbond_cuda: computed %i pairs.\n" , zero ); */

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
    
    int k, cid, pid;
    struct part *p;
    struct part_cuda *buff;
    struct space *s = &e->s;
    int *sortlists_ind, sortlists_count;
    struct cellpair_cuda *cellpairs;
    FPTYPE maxdist = s->cutoff + 2*s->verlet_maxdx;
    
    /* Clear the counts array. */
    bzero( s->counts_cuda_local , sizeof(int) * s->nr_cells );
    
    /* Load the counts. */
    for ( k = 0 ; k < s->nr_marked ; k++ )
        if ( ( s->counts_cuda_local[ s->cid_marked[k] ] = s->cells[ s->cid_marked[k] ].count ) > cuda_maxparts ) {
            printf( "engine_cuda_load_parts: cell %i has %i parts (max=%i).\n" , s->cid_marked[k] , s->cells[ s->cid_marked[k] ].count , cuda_maxparts );
            return error(engine_err_range);
            }
        
    /* Compute the indices. */
    s->ind_cuda_local[0] = 0;
    for ( k = 1 ; k < s->nr_cells ; k++ )
        s->ind_cuda_local[k] = s->ind_cuda_local[k-1] + s->counts_cuda_local[k-1];
        
    /* Are we using verlet lists? */
    if ( e->flags & engine_flag_verlet ) {
    
        /* Start by setting the maxdist on the device. */
        if ( cudaMemcpyToSymbol( "cuda_maxdist" , &maxdist , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
    
        /* Do we need to re-build the list? */
        if ( s->verlet_rebuild ) {
        
            /* Get a copy of the cellpairs from the device. */
            if ( ( cellpairs = (struct cellpair_cuda *)alloca( sizeof(struct cellpair_cuda) * s->nr_pairs ) ) == NULL )
                return engine_err_malloc;
            if ( cudaMemcpy( cellpairs , s->pairs_cuda , sizeof(struct cellpair_cuda) * s->nr_pairs , cudaMemcpyDeviceToHost ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
        
            /* Allocate and fill the sortlist. */
            if ( ( sortlists_ind = (int *)alloca( sizeof(int) * (s->nr_pairs + 1) ) ) == NULL )
                return error(engine_err_malloc);
            sortlists_ind[0] = 0;
            for ( k = 1 ; k <= s->nr_pairs ; k++ )
                if ( cellpairs[k-1].i != cellpairs[k-1].j )
                    sortlists_ind[k] = sortlists_ind[k-1] + s->counts_cuda_local[cellpairs[k-1].i] + s->counts_cuda_local[cellpairs[k-1].j];
                else
                    sortlists_ind[k] = sortlists_ind[k-1];
            sortlists_count = sortlists_ind[s->nr_pairs];
            
            /* Do we need to re-allocate the sortlists? */
            if ( e->sortlists_cuda == NULL || e->sortlists_size < sortlists_count ) {
                e->sortlists_size = sortlists_count * 1.2;
                if ( e->sortlists_cuda != NULL && cudaFree( e->sortlists_cuda ) != cudaSuccess )
                    return cuda_error(engine_err_cuda);
                if ( cudaMalloc( &e->sortlists_cuda , sizeof(int) * e->sortlists_size ) != cudaSuccess )
                    return cuda_error(engine_err_cuda);
                if ( cudaMemcpyToSymbol( "cuda_sortlists" , &e->sortlists_cuda , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
                    return cuda_error(engine_err_cuda);
                }
        
            /* Copy the indices over to the device. */
            if ( cudaMemcpy( e->sortlists_ind_cuda , sortlists_ind , sizeof(int) * s->nr_pairs , cudaMemcpyHostToDevice ) != cudaSuccess )
                return cuda_error(engine_err_cuda);
        
            }
    
        } /* are we using verlet lists? */
    
    /* Allocate the particle buffer. */
    if ( ( s->parts_cuda_local = (struct part_cuda *)malloc( sizeof( struct part_cuda ) * s->nr_parts ) ) == NULL )
        return error(engine_err_malloc);
    
    /* Loop over the marked cells. */
    for ( k = 0 ; k < s->nr_marked ; k++ ) {
    
        /* Get the cell id. */
        cid = s->cid_marked[k];
        
        /* Copy the particle data to the device. */
        #ifdef PACK_PARTS
            buff = (struct part_cuda *)&s->parts_cuda_local[ s->ind_cuda_local[cid] ];
            for ( pid = 0 ; pid < s->counts_cuda_local[cid] ; pid++ ) {
                p = &s->cells[cid].parts[pid];
                buff[ pid ].x[0] = p->x[0];
                buff[ pid ].x[1] = p->x[1];
                buff[ pid ].x[2] = p->x[2];
                buff[ pid ].f[0] = p->f[0];
                buff[ pid ].f[1] = p->f[1];
                buff[ pid ].f[2] = p->f[2];
                buff[ pid ].type = p->type;
                #if defined(USETEX_E) || defined(EXPLPOT)
                    buff[ pid ].q = p->q;
                #endif
                }
        #else
            memcpy( &buff[ s->ind_cuda_local[cid] ] , s->cells[cid].parts , sizeof( struct part ) * s->counts_cuda_local[cid] );
        #endif
    
        }
        
    /* Copy the counts onto the device. */
    if ( cudaMemcpy( s->counts_cuda , s->counts_cuda_local , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Copy the inds onto the device. */
    if ( cudaMemcpy( s->ind_cuda , s->ind_cuda_local , sizeof(int) * s->nr_cells , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Finally, push the parts array onto the device. */
    if ( cudaMalloc( &s->parts_cuda , sizeof( struct part_cuda ) * s->nr_parts ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( s->parts_cuda , s->parts_cuda_local , sizeof( struct part_cuda ) * s->nr_parts , cudaMemcpyHostToDevice ) != cudaSuccess )
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
    
    int k, cid, pid;
    struct part *p;
    struct part_cuda *buff;
    struct space *s = &e->s;
    
    /* Get the parts from the device. */
    if ( cudaMemcpy( s->parts_cuda_local , s->parts_cuda , sizeof(struct part_cuda) * s->nr_parts , cudaMemcpyDeviceToHost ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
                
    /* Loop over the marked cells. */
    for ( k = 0 ; k < s->nr_marked ; k++ ) {
    
        /* Get the cell id. */
        cid = s->cid_marked[k];
        
        /* Copy the particle data from the device. */
        #ifdef PACK_PARTS
            buff = (struct part_cuda *)&s->parts_cuda_local[ s->ind_cuda_local[cid] ];
            for ( pid = 0 ; pid < e->s.cells[cid].count ; pid++ ) {
                p = &e->s.cells[cid].parts[pid];
                p->f[0] = buff[ pid ].f[0];
                p->f[1] = buff[ pid ].f[1];
                p->f[2] = buff[ pid ].f[2];
                }
        #else
            memcpy( s->cells[cid].parts , &s->parts_cuda_local[ s->ind_cuda_local[cid] ] , sizeof( struct part ) * s->counts_cuda_local[cid] );            
        #endif
            
        }

    /* Deallocate the parts array and counts array. */
    free( s->parts_cuda_local );
    if ( cudaFree( e->s.parts_cuda ) != cudaSuccess )
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

    int i, j, k, nr_pots, nr_coeffs, *diags;
    int pind_cuda[ e->max_type * e->max_type ], *offsets_cuda;
    struct potential *pots[ e->nr_types * (e->nr_types + 1) / 2 + 1 ];
    struct potential *p_cuda[ e->max_type * e->max_type ];
    struct potential *pots_cuda;
    struct cellpair_cuda *pairs_cuda;
    struct celltuple_cuda *tuples_cuda;
    float *finger, *coeffs_cuda, *alphas_cuda;
    float cutoff = e->s.cutoff, cutoff2 = e->s.cutoff2, dscale, buff[ e->nr_types ];
    cudaArray *cuArray_coeffs, *cuArray_offsets, *cuArray_alphas, *cuArray_pind, *cuArray_diags;
    cudaChannelFormatDesc channelDesc_int = cudaCreateChannelDesc<int>();
    cudaChannelFormatDesc channelDesc_float = cudaCreateChannelDesc<float>();
    void *devptr;
    unsigned int *taboo_cuda, *owner_cuda, *pairIDs_cuda, pairIDs[ e->s.nr_pairs ];
    
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
    
        }
        
    /* Copy eps and rmin to the device. */
    for ( i = 0 ; i < e->nr_types ; i++ )
        buff[i] = sqrt( fabs( e->types[i].eps ) );
    if ( cudaMemcpyToSymbol( "cuda_eps" , buff , sizeof(float) * e->nr_types , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    for ( i = 0 ; i < e->nr_types ; i++ )
        buff[i] = e->types[i].rmin;
    if ( cudaMemcpyToSymbol( "cuda_rmin" , buff , sizeof(float) * e->nr_types , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
            
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
        if ( e->p[i] == NULL ) {
            p_cuda[i] = NULL;
            pind_cuda[i] = 0;
            }
        else {
            for ( j = 0 ; j < nr_pots && pots[j] != e->p[i] ; j++ );
            p_cuda[i] = &(e->pots_cuda[j]);
            pind_cuda[i] = j;
            }
        }
        
    /* Pack the coefficients before shipping them off to the device. */
    if ( ( coeffs_cuda = (float *)alloca( sizeof(float) * nr_coeffs * potential_chunk ) ) == NULL )
        return error(engine_err_malloc);
    for ( finger = coeffs_cuda , i = 0 ; i < nr_pots ; i++ ) {
        memcpy( finger , pots[i]->c , sizeof(float) * potential_chunk * (pots[i]->n + 1) );
        finger = &finger[ (pots[i]->n + 1) * potential_chunk ];
        }
    printf( "engine_cuda_load: packed %i potentials with %i coefficient chunks.\n" , nr_pots , nr_coeffs );
        
    /* Copy the data to the device. */
    if ( cudaMemcpy( e->p_cuda , p_cuda , sizeof(struct potential *) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( e->pots_cuda , pots_cuda , sizeof(struct potential) * nr_pots , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( e->coeffs_cuda , coeffs_cuda , sizeof(float) * nr_coeffs * potential_chunk , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
        
    /* Bind the potential coefficients to a texture. */
    if ( cudaMallocArray( &cuArray_coeffs , &channelDesc_float , potential_chunk , nr_coeffs ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToArray( cuArray_coeffs , 0 , 0 , coeffs_cuda , sizeof(float) * nr_coeffs * potential_chunk , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Pack the potential offsets into a newly allocated array and 
       copy to the device. */
    if ( ( offsets_cuda = (int *)alloca( sizeof(int) * nr_pots ) ) == NULL )
        return error(engine_err_malloc);
    offsets_cuda[0] = 0;
    for ( i = 1 ; i < nr_pots ; i++ )
        offsets_cuda[i] = offsets_cuda[i-1] + pots_cuda[i-1].n + 1;
    if ( cudaMallocArray( &cuArray_offsets , &channelDesc_int , nr_pots , 1 ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToArray( cuArray_offsets , 0 , 0 , offsets_cuda , sizeof(int) * nr_pots , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Pack the diagonal offsets into a newly allocated array and 
       copy to the device. */
    if ( ( diags = (int *)alloca( sizeof(int) * 2 * cuda_ndiags ) ) == NULL )
        return error(engine_err_malloc);
    for ( i = 0 ; i < cuda_ndiags ; i++ ) {
        k = ( sqrt( 8.0*i + 1 ) - 1 ) / 2;
        diags[ 2*i  + 0 ] = i - k*(k+1)/2;
        diags[ 2*i  + 1 ] = 1 + k - i + k*(k+1)/2;
        }
    if ( cudaMallocArray( &cuArray_diags , &channelDesc_int , 2 , cuda_ndiags ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToArray( cuArray_diags , 0 , 0 , diags , sizeof(int) * 2 * cuda_ndiags , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Copy the potential indices to the device. */
    if ( cudaMallocArray( &cuArray_pind , &channelDesc_int , e->max_type * e->max_type , 1 ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToArray( cuArray_pind , 0 , 0 , pind_cuda , sizeof(int) * e->max_type * e->max_type , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    
    /* Pack the potential alphas into a newly allocated array and copy
       to the device as a texture. */
    if ( ( alphas_cuda = (float *)alloca( sizeof(float) * nr_pots * 3 ) ) == NULL )
        return error(engine_err_malloc);
    for ( i = 0 ; i < nr_pots ; i++ ) {
        alphas_cuda[ 3*i ] = pots_cuda[i].alpha[0];
        alphas_cuda[ 3*i + 1 ] = pots_cuda[i].alpha[1];
        alphas_cuda[ 3*i + 2 ] = pots_cuda[i].alpha[2];
        }
    if ( cudaMallocArray( &cuArray_alphas , &channelDesc_float , 3 , nr_pots ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToArray( cuArray_alphas , 0 , 0 , alphas_cuda , sizeof(float) * nr_pots * 3 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Bind the textures on the device. */
    if ( runner_bind( cuArray_coeffs , cuArray_offsets , cuArray_alphas , cuArray_pind , cuArray_diags ) < 0 )
        return error(engine_err_runner);
        
        
    /* Set the constant pointer to the null potential and other useful values. */
    if ( cudaMemcpyToSymbol( "potential_null_cuda" , &(e->pots_cuda) , sizeof(struct potential *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_p" , &(e->p_cuda) , sizeof(struct potential **) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_pots" , &(e->pots_cuda) , sizeof(struct potential *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_cutoff2" , &cutoff2 , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_cutoff" , &cutoff , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_maxdist" , &cutoff , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_maxtype" , &(e->max_type) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    dscale = ((float)SHRT_MAX) / ( 3.0 * sqrt( e->s.h[0]*e->s.h[0] + e->s.h[1]*e->s.h[1] + e->s.h[2]*e->s.h[2] ) );
    if ( cudaMemcpyToSymbol( "cuda_dscale" , &dscale , sizeof(float) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
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
    if ( cudaGetSymbolAddress( &devptr , "cuda_fifos_in" ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemset( devptr , 0  , sizeof(struct fifo_cuda) * cuda_maxblocks ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaGetSymbolAddress( &devptr , "cuda_fifos_out" ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemset( devptr , 0  , sizeof(struct fifo_cuda) * cuda_maxblocks ) != cudaSuccess )
        return cuda_error(engine_err_cuda);

        
    /* Do we have tuples? */
    if ( e->flags & engine_flag_tuples ) {
    
        /* Allocate and fill the compact list of tuples. */
        if ( ( tuples_cuda = (struct celltuple_cuda *)malloc( sizeof(struct celltuple_cuda) * e->s.nr_pairs ) ) == NULL )
            return error(engine_err_malloc);
        for ( i = 0 ; i < e->s.nr_tuples ; i++ ) {
            tuples_cuda[i].i = ( e->s.tuples[i].n > 0 ) ? e->s.tuples[i].cellid[0] : -1;
            tuples_cuda[i].j = ( e->s.tuples[i].n > 1 ) ? e->s.tuples[i].cellid[1] : -1;
            tuples_cuda[i].k = ( e->s.tuples[i].n > 2 ) ? e->s.tuples[i].cellid[2] : -1;
            tuples_cuda[i].nr_pairs = 0;
            if ( e->s.tuples[i].pairid[ space_pairind(0,0) ] >= 0 )
                tuples_cuda[i].pairs[ tuples_cuda[i].nr_pairs++ ] = 1;
            if ( e->s.tuples[i].pairid[ space_pairind(0,1) ] >= 0 )
                tuples_cuda[i].pairs[ tuples_cuda[i].nr_pairs++ ] = 3;
            if ( e->s.tuples[i].pairid[ space_pairind(1,1) ] >= 0 )
                tuples_cuda[i].pairs[ tuples_cuda[i].nr_pairs++ ] = 2;
            if ( e->s.tuples[i].pairid[ space_pairind(0,2) ] >= 0 )
                tuples_cuda[i].pairs[ tuples_cuda[i].nr_pairs++ ] = 5;
            if ( e->s.tuples[i].pairid[ space_pairind(1,2) ] >= 0 )
                tuples_cuda[i].pairs[ tuples_cuda[i].nr_pairs++ ] = 6;
            if ( e->s.tuples[i].pairid[ space_pairind(2,2) ] >= 0 )
                tuples_cuda[i].pairs[ tuples_cuda[i].nr_pairs++ ] = 4;
            if ( tuples_cuda[i].i >= 0 && tuples_cuda[i].j >= 0 )
                for ( k = 0 ; k < 3 ; k++ ) {
                    tuples_cuda[i].shift_ij[k] = e->s.cells[ tuples_cuda[i].j ].origin[k] - e->s.cells[ tuples_cuda[i].i ].origin[k];
                    if ( tuples_cuda[i].shift_ij[k] * 2 > e->s.dim[k] )
                        tuples_cuda[i].shift_ij[k] -= e->s.dim[k];
                    else if ( tuples_cuda[i].shift_ij[k] * 2 < -e->s.dim[k] )
                        tuples_cuda[i].shift_ij[k] += e->s.dim[k];
                    }
            if ( tuples_cuda[i].i >= 0 && tuples_cuda[i].k >= 0 )
                for ( k = 0 ; k < 3 ; k++ ) {
                    tuples_cuda[i].shift_ik[k] = e->s.cells[ tuples_cuda[i].k ].origin[k] - e->s.cells[ tuples_cuda[i].i ].origin[k];
                    if ( tuples_cuda[i].shift_ik[k] * 2 > e->s.dim[k] )
                        tuples_cuda[i].shift_ik[k] -= e->s.dim[k];
                    else if ( tuples_cuda[i].shift_ik[k] * 2 < -e->s.dim[k] )
                        tuples_cuda[i].shift_ik[k] += e->s.dim[k];
                    }
            }
            
        /* Allocate and fill the tuples list on the device. */
        if ( cudaMalloc( &e->s.tuples_cuda , sizeof(struct celltuple_cuda) * e->s.nr_pairs ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpy( e->s.tuples_cuda , tuples_cuda , sizeof(struct celltuple_cuda) * e->s.nr_pairs , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( "cuda_tuples" , &(e->s.tuples_cuda) , sizeof(struct celltuple_cuda *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( "cuda_nr_tuples" , &(e->s.nr_tuples) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
            
        /* Clean up. */
        free(tuples_cuda);
            
        }

    /* Allocate the sortlists locally and on the device if needed. */
    if ( e->flags & engine_flag_verlet ) {
        e->sortlists_cuda = NULL;
        if ( cudaMalloc( &e->sortlists_ind_cuda , sizeof(int) * e->s.nr_pairs ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        if ( cudaMemcpyToSymbol( "cuda_sortlists_ind" , &e->sortlists_ind_cuda , sizeof(void *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
            return cuda_error(engine_err_cuda);
        }

    /* Set the number of pairs and cells. */
    if ( cudaMemcpyToSymbol( "cuda_nr_pairs" , &(e->s.nr_pairs) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_nr_cells" , &(e->s.nr_cells) , sizeof(int) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Allocate the cell counts and offsets. */
    if ( ( e->s.counts_cuda_local = (int *)malloc( sizeof(int) * e->s.nr_cells ) ) == NULL ||
         ( e->s.ind_cuda_local = (int *)malloc( sizeof(int) * e->s.nr_cells ) ) == NULL )
        return error(engine_err_malloc);
    if ( cudaMalloc( &e->s.counts_cuda , sizeof(int) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMalloc( &e->s.ind_cuda , sizeof(int) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Allocate and init the taboo list on the device. */
    if ( cudaMalloc( &taboo_cuda , sizeof(int) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemset( taboo_cuda , 0 , sizeof(int) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_taboo" , &taboo_cuda , sizeof(int *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMalloc( &owner_cuda , sizeof(int) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemset( owner_cuda , 0 , sizeof(int) * e->s.nr_cells ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_owner" , &owner_cuda , sizeof(int *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* Pack and send the pairIDs. */
    for ( k = 0 ; k < e->s.nr_pairs ; k++ )
        pairIDs[k] = k;
    if ( cudaMalloc( &pairIDs_cuda , sizeof(int) * e->s.nr_pairs ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpy( pairIDs_cuda , pairIDs , sizeof(int) * e->s.nr_pairs , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
    if ( cudaMemcpyToSymbol( "cuda_pairIDs" , &pairIDs_cuda , sizeof(int *) , 0 , cudaMemcpyHostToDevice ) != cudaSuccess )
        return cuda_error(engine_err_cuda);
        
    /* He's done it! */
    return engine_err_ok;
    
    }
    
    
/* End CUDA-related stuff. */
#endif
