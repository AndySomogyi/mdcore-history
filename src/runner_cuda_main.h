
/* Set the kernel names depending on cuda_nrparts. */
#define PASTE(x,y) x ## _ ## y
#define runner_run_verlet_cuda(N) PASTE(runner_run_verlet_cuda,N)
#define runner_run_cuda(N) PASTE(runner_run_cuda,N)


/**
 * @brief Loop over the cell pairs and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_verlet_cuda(cuda_nrparts) ( float *forces , int *counts , int *ind , int verlet_rebuild ) {

    int threadID, blockID;
    int k, cid, cjd;
    volatile __shared__ int pid;
    __shared__ float forces_i[ 3*cuda_nrparts ], forces_j[ 3*cuda_nrparts ];
    __shared__ unsigned int sort_i[ cuda_nrparts ], sort_j[ cuda_nrparts ];
    #if !defined(PARTS_TEX) && defined(PARTS_LOCAL)
        __shared__ float4 parts_i[ cuda_nrparts ], parts_j[ cuda_nrparts ];
    #endif
    float *forces_k;
    
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
            pid = atomicAdd( &cuda_pair_next , 1 );
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
        
            /* Clear the forces buffers. */
            for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                forces_i[k] = 0.0f;
            for ( k = threadID ; k < 3*counts[cjd] ; k += cuda_frame )
                forces_j[k] = 0.0f;
            __threadfence_block();
            
            /* Copy the particle data into the local buffers. */
            #ifndef PARTS_TEX
                #ifdef PARTS_LOCAL
                    cuda_memcpy( parts_i , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                    cuda_memcpy( parts_j , &cuda_parts[ ind[cjd] ] , sizeof(float4) * counts[cjd] );
                    __threadfence_block();
                #else
                    parts_i = &cuda_parts[ ind[cid] ];
                    parts_j = &cuda_parts[ ind[cjd] ];
                #endif
            #endif
            
            /* Compute the cell pair interactions. */
            #ifdef PARTS_TEX
                runner_dopair4_verlet_cuda(
                    cid , counts[cid] ,
                    cjd , counts[cjd] ,
                    forces_i , forces_j , 
                    sort_i , sort_j ,
                    cuda_pairs[pid].shift ,
                    verlet_rebuild , &cuda_sortlists[ cuda_sortlists_ind[ pid ] ] );
            #else
                runner_dopair4_verlet_cuda(
                    parts_i , counts[cid] ,
                    parts_j , counts[cjd] ,
                    forces_i , forces_j , 
                    sort_i , sort_j ,
                    cuda_pairs[pid].shift ,
                    verlet_rebuild , &cuda_sortlists[ cuda_sortlists_ind[ pid ] ] );
            #endif
                
            /* Write the particle forces back to cell_i. */
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cid] );
            forces_k = &forces[ 3*ind[cid] ];
            for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                forces_k[k] += forces_i[k];
            __threadfence();
            if ( threadID == 0 ) {
                cuda_mutex_unlock( &cuda_taboo[cid] );
                __threadfence();
                }
                
            /* Write the particle forces back to cell_j. */
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cjd] );
            forces_k = &forces[ 3*ind[cjd] ];
            for ( k = threadID ; k < 3*counts[cjd] ; k += cuda_frame )
                forces_k[k] += forces_j[k];
            __threadfence();
            if ( threadID == 0 ) {
                cuda_mutex_unlock( &cuda_taboo[cjd] );
                __threadfence();
                }
                
            }
        else {
        
            /* Clear the forces buffer. */
            for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                forces_i[k] = 0.0f;
            __threadfence_block();
            
            /* Copy the particle data into the local buffers. */
            #ifndef PARTS_TEX
                #ifdef PARTS_LOCAL
                    cuda_memcpy( parts_i , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                    __threadfence_block();
                #else
                    parts_i = &cuda_parts[ ind[cid] ];
                #endif
            #endif
            
            /* Compute the cell self interactions. */
            #ifdef PARTS_TEX
                if ( counts[cid] <= cuda_frame || counts[cid] > cuda_maxdiags )
                    runner_doself_cuda( ind[cid] , counts[cid] , forces_i );
                else
                    runner_doself_diag_cuda( ind[cid] , counts[cid] , forces_i );
            #else
                if ( counts[cid] <= cuda_frame || counts[cid] > cuda_maxdiags )
                    runner_doself_cuda( parts_i , counts[cid] , forces_i );
                else
                    runner_doself_diag_cuda( parts_i , counts[cid] , forces_i );
            #endif
                
            /* Write the particle forces back to cell_i. */
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cid] );
            forces_k = &forces[ 3*ind[cid] ];
            for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                forces_k[k] += forces_i[k];
            __threadfence();
            if ( threadID == 0 ) {
                cuda_mutex_unlock( &cuda_taboo[cid] );
                __threadfence();
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
 
__global__ void runner_run_cuda(cuda_nrparts) ( float *forces , int *counts , int *ind ) {

    int threadID, blockID;
    int k, cid, cjd;
    volatile __shared__ int pid;
    __shared__ float forces_i[ 3*cuda_nrparts ], forces_j[ 3*cuda_nrparts ];
    __shared__ unsigned int sort_i[ cuda_nrparts ], sort_j[ cuda_nrparts ];
    #if !defined(PARTS_TEX) && defined(PARTS_LOCAL)
        __shared__ float4 parts_i[ cuda_nrparts ], parts_j[ cuda_nrparts ];
    #endif
    float *forces_k;
    
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
            pid = atomicAdd( &cuda_pair_next , 1 );
            __threadfence_block();
            }
            
        /* Are we at the end of the list? */
        if ( pid >= cuda_nr_pairs )
            break;
        
        /* Get a hold of the pair cells. */
        cid = cuda_pairs[pid].i;
        cjd = cuda_pairs[pid].j;
    
        /* if ( threadID == 0 )
            printf( "runner_run_cuda: block %03i got pid=%i (%i/%i).\n" , blockID , pid , cid , cjd ); */
        
        /* Do the pair. */
        if ( cid != cjd ) {
        
            /* Clear the forces buffer. */
            for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                forces_i[k] = 0.0f;
            for ( k = threadID ; k < 3*counts[cjd] ; k += cuda_frame )
                forces_j[k] = 0.0f;
            __threadfence_block();
            
            /* Copy the particle data into the local buffers. */
            #ifndef PARTS_TEX
                #ifdef PARTS_LOCAL
                    cuda_memcpy( parts_i , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                    cuda_memcpy( parts_j , &cuda_parts[ ind[cjd] ] , sizeof(float4) * counts[cjd] );
                    __threadfence_block();
                #else
                    parts_i = &cuda_parts[ ind[cid] ];
                    parts_j = &cuda_parts[ ind[cjd] ];
                #endif
            #endif
            
            /* Compute the cell pair interactions. */
            #ifdef PARTS_TEX
                if ( counts[cid] <= 2*cuda_frame || counts[cjd] <= 2*cuda_frame )
                    runner_dopair4_cuda(
                        cid , counts[cid] ,
                        cjd , counts[cjd] ,
                        forces_i , forces_j , 
                        cuda_pairs[pid].shift );
                else
                    runner_dopair4_sorted_cuda(
                        cid , counts[cid] ,
                        cjd , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        cuda_pairs[pid].shift );
            #else
                if ( counts[cid] <= 2*cuda_frame || counts[cjd] <= 2*cuda_frame )
                    runner_dopair4_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        cuda_pairs[pid].shift );
                else
                    runner_dopair4_sorted_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        cuda_pairs[pid].shift );
            #endif
                
            /* if ( threadID == 0 )
                printf( "runner_run_cuda: block %03i waiting for cell %i...\n" , blockID , cid ); */
                
            /* Write the particle forces back to cell_i. */
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cid] );
            forces_k = &forces[ 3*ind[cid] ];
            for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                forces_k[k] += forces_i[k];
            __threadfence();
            if ( threadID == 0 ) {
                cuda_mutex_unlock( &cuda_taboo[cid] );
                __threadfence();
                }
                
            /* if ( threadID == 0 )
                printf( "runner_run_cuda: block %03i released cell %i.\n" , blockID , cid ); */
                
            /* if ( threadID == 0 )
                printf( "runner_run_cuda: block %03i waiting for cell %i...\n" , blockID , cjd ); */
                
            /* Write the particle forces back to cell_j. */
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cjd] );
            forces_k = &forces[ 3*ind[cjd] ];
            for ( k = threadID ; k < 3*counts[cjd] ; k += cuda_frame )
                forces_k[k] += forces_j[k];
            __threadfence();
            if ( threadID == 0 ) {
                cuda_mutex_unlock( &cuda_taboo[cjd] );
                __threadfence();
                }
                
            /* if ( threadID == 0 )
                printf( "runner_run_cuda: block %03i released cell %i.\n" , blockID , cjd ); */
                
            }
        else {
        
            /* Clear the forces buffer. */
            for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                forces_i[k] = 0.0f;
            __threadfence_block();
            
            /* Copy the particle data into the local buffers. */
            #ifndef PARTS_TEX
                #ifdef PARTS_LOCAL
                    cuda_memcpy( parts_i , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                    __threadfence_block();
                #else
                    parts_i = &cuda_parts[ ind[cid] ];
                #endif
            #endif
            
            /* Compute the cell self interactions. */
            #ifdef PARTS_TEX
                if ( counts[cid] <= cuda_frame || counts[cid] > cuda_maxdiags )
                    runner_doself_cuda( cid , counts[cid] , forces_i );
                else
                    runner_doself_diag_cuda( cid , counts[cid] , forces_i );
            #else
                if ( counts[cid] <= cuda_frame || counts[cid] > cuda_maxdiags )
                    runner_doself_cuda( parts_i , counts[cid] , forces_i );
                else
                    runner_doself_diag_cuda( parts_i , counts[cid] , forces_i );
            #endif
                
            /* if ( threadID == 0 )
                printf( "runner_run_cuda: block %03i waiting for cell %i...\n" , blockID , cid ); */
                
            /* Write the particle forces back to cell_i. */
            if ( threadID == 0 )
                cuda_mutex_lock( &cuda_taboo[cid] );
            forces_k = &forces[ 3*ind[cid] ];
            for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                forces_k[k] += forces_i[k];
            __threadfence();
            if ( threadID == 0 ) {
                cuda_mutex_unlock( &cuda_taboo[cid] );
                __threadfence();
                }
            
            /* if ( threadID == 0 )
                printf( "runner_run_cuda: block %03i released cell %i.\n" , blockID , cid ); */
                
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
        
    /* if ( threadID == 0 )
        printf( "runner_run_cuda: block %03i is done.\n" , blockID ); */
        
    TIMER_TOC2(tid_total)

    }

