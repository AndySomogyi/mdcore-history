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

    int threadID;
    int cid, cjd;
    float epot = 0.0f;
    volatile __shared__ int pid;
    #ifdef FORCES_LOCAL
        int k;
        __shared__ __align__(16) int buff[ 8*cuda_nrparts ];
        float *forces_i = (float *)&buff[ 0 ];
        float *forces_j = (float *)&buff[ 3*cuda_nrparts ];
        unsigned int *sort_i = (unsigned int *)&buff[ 6*cuda_nrparts ];
        unsigned int *sort_j = (unsigned int *)&buff[ 7*cuda_nrparts ];
    #else
        unsigned int seed = 6178 + blockIdx.x;
        float *forces_i, *forces_j;
        __shared__ unsigned int sort_i[ cuda_nrparts ];
        __shared__ unsigned int sort_j[ cuda_nrparts ];
        struct queue_cuda *myq , *queues[ cuda_maxqueues ];
        int naq = cuda_nrqueues, qid;
    #endif
    #if !defined(PARTS_TEX)
        #ifdef PARTS_LOCAL
            float4 *parts_i;
            __shared__ float4 parts_j[ cuda_nrparts ];
        #else
            float4 *parts_i, *parts_j;
        #endif
    #endif
    
    TIMER_TIC2
    
    /* Get the block and thread ids. */
    threadID = threadIdx.x;
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        

    /* Init the list of queues. */
    #ifndef FORCES_LOCAL
    if ( threadID == 0 ) {
        myq = &cuda_queues[ get_smid() ];
        for ( qid = 0 ; qid < cuda_nrqueues ; qid++ )
            queues[qid] = &cuda_queues[qid];
        naq = cuda_nrqueues - 1;
        queues[ get_smid() ] = queues[ naq ];
        }
    #endif
        

    /* Main loop... */
    while ( pid >= 0 && pid < cuda_nr_pairs ) {
    
        /* Let the first thread grab a pair. */
        if ( threadID == 0 ) {
            #ifdef FORCES_LOCAL
                if ( ( pid = atomicAdd( &cuda_pair_next , 1 ) ) >= cuda_nr_pairs )
                    pid = -1;
            #else
                TIMER_TIC
                while ( 1 ) {
                    if ( myq->rec_count >= myq->count || ( pid = runner_cuda_gettask( myq , 0 ) ) < 0 ) {
                        for ( qid = 0 ; qid < naq ; qid++ )
                            if ( queues[qid]->rec_count >= queues[qid]->count )
                                queues[ qid-- ] = queues[ --naq ];
                        if ( naq == 0 ) {
                            pid = -1;
                            break;
                            }
                        seed = 1103515245 * seed + 12345;
                        qid = seed % naq;
                        if ( ( pid = runner_cuda_gettask( queues[qid] , 1 ) ) >= 0 ) {
                            if ( atomicAdd( (int *)&myq->count , 1 ) < cuda_queue_size )
                                myq->rec_data[ atomicAdd( (int *)&myq->rec_count , 1 ) ] = pid;
                            else {
                                atomicSub( (int *)&myq->count , 1 );
                                atomicAdd( (int *)&queues[qid]->count , 1 );
                                queues[qid]->rec_data[ atomicAdd( (int *)&queues[qid]->rec_count , 1 ) ] = pid;
                                }
                            break;
                            }
                        }
                    else
                        break;
                    }
                TIMER_TOC(tid_gettask)
            #endif
            }
            
        /* Exit if we didn't get a valid pair. */
        if ( pid < 0 )
            break;

        /* Get a hold of the pair cells. */
        cid = cuda_pairs[pid].i;
        cjd = cuda_pairs[pid].j;
    
        /* Do the pair. */
        if ( cid != cjd ) {
        
            #ifdef FORCES_LOCAL
        
                /* Clear the forces buffers. */
                for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                    forces_i[k] = 0.0f;
                for ( k = threadID ; k < 3*counts[cjd] ; k += cuda_frame )
                    forces_j[k] = 0.0f;
                // __threadfence_block();
                
                /* Copy the particle data into the local buffers. */
                #ifndef PARTS_TEX
                    #ifdef PARTS_LOCAL
                        parts_i = &cuda_parts[ ind[cid] ];
                        cuda_memcpy( parts_j , &cuda_parts[ ind[cjd] ] , sizeof(float4) * counts[cjd] );
                        // __threadfence_block();
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
                        verlet_rebuild , &cuda_sortlists[ cuda_sortlists_ind[ pid ] ] ,
                        &epot );
                #else
                    runner_dopair4_verlet_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        cuda_pairs[pid].shift ,
                        verlet_rebuild , &cuda_sortlists[ cuda_sortlists_ind[ pid ] ] ,
                        &epot );
                #endif
                    
                /* Write the particle forces back to cell_i. */
                if ( threadID == 0 )
                    cuda_mutex_lock( &cuda_taboo[cid] );
                cuda_sum( &forces[ 3*ind[cid] ] , forces_i , 3*counts[cid] );
                __threadfence();
                if ( threadID == 0 ) {
                    cuda_mutex_unlock( &cuda_taboo[cid] );
                    __threadfence();
                    }
                    
                /* Write the particle forces back to cell_j. */
                if ( threadID == 0 )
                    cuda_mutex_lock( &cuda_taboo[cjd] );
                cuda_sum( &forces[ 3*ind[cjd] ] , forces_j , 3*counts[cjd] );
                __threadfence();
                if ( threadID == 0 ) {
                    cuda_mutex_unlock( &cuda_taboo[cjd] );
                    __threadfence();
                    }
                    
            #else
            
                /* Put a finger on the forces. */
                forces_i = &forces[ 3*ind[cid] ];
                forces_j = &forces[ 3*ind[cjd] ];
                
                /* Copy the particle data into the local buffers. */
                #ifndef PARTS_TEX
                    #ifdef PARTS_LOCAL
                        parts_i = &cuda_parts[ ind[cid] ];
                        cuda_memcpy( parts_j , &cuda_parts[ ind[cjd] ] , sizeof(float4) * counts[cjd] );
                        // __threadfence_block();
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
                        verlet_rebuild , &cuda_sortlists[ cuda_sortlists_ind[ pid ] ] ,
                        &epot );
                #else
                    runner_dopair4_verlet_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        cuda_pairs[pid].shift ,
                        verlet_rebuild , &cuda_sortlists[ cuda_sortlists_ind[ pid ] ] ,
                        &epot );
                #endif
                    
                /* Unlock these cells' mutexes. */
                if ( threadID == 0 ) {
                    cuda_mutex_unlock( &cuda_taboo[cid] );
                    cuda_mutex_unlock( &cuda_taboo[cjd] );
                    }
                       
            #endif
                
            }
        else {
        
            #ifdef FORCES_LOCAL
        
                /* Clear the forces buffer. */
                TIMER_TIC
                for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                    forces_i[k] = 0.0f;
                // __threadfence_block();
                TIMER_TOC(tid_update)
                
                /* Copy the particle data into the local buffers. */
                #ifndef PARTS_TEX
                    parts_j = (float4 *)forces_j;
                    cuda_memcpy( parts_j , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                #endif
                
                /* Compute the cell self interactions. */
                #ifdef PARTS_TEX
                    runner_doself4_diag_cuda( cid , counts[cid] , forces_i , &epot );
                #else
                    runner_doself4_diag_cuda( parts_j , counts[cid] , forces_i , &epot );
                #endif
                    
                /* Write the particle forces back to cell_i. */
                if ( threadID == 0 )
                    cuda_mutex_lock( &cuda_taboo[cid] );
                cuda_sum( &forces[ 3*ind[cid] ] , forces_i , 3*counts[cid] );
                __threadfence();
                if ( threadID == 0 ) {
                    cuda_mutex_unlock( &cuda_taboo[cid] );
                    __threadfence();
                    }
                    
            #else
                
                /* Put a finger on the forces. */
                forces_i = &forces[ 3*ind[cid] ];
                
                /* Copy the particle data into the local buffers. */
                #ifndef PARTS_TEX
                    parts_j = (float4 *)forces_j;
                    cuda_memcpy( parts_j , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                #endif
                
                /* Compute the cell self interactions. */
                #ifdef PARTS_TEX
                    runner_doself4_diag_cuda( cid , counts[cid] , forces_i , &epot );
                #else
                    runner_doself4_diag_cuda( parts_j , counts[cid] , forces_i , &epot );
                #endif
                
                /* Unlock this cell's mutex. */
                if ( threadID == 0 )
                    cuda_mutex_unlock( &cuda_taboo[cid] );
                       
            #endif
        
            }
            
        } /* main loop. */
        
    /* Accumulate the potential energy. */
    atomicAdd( &cuda_epot , epot );

    /* Make a notch on the barrier, last one out cleans up the mess... */
    if ( threadID == 0 && atomicAdd( &cuda_barrier , 1 ) == gridDim.x-1 ) {
        cuda_barrier = 0;
        cuda_epot_out = cuda_epot;
        cuda_epot = 0.0f;
        #ifdef FORCES_LOCAL
            cuda_pair_next = 0;
        #else
            for ( qid = 0 ; qid < cuda_nrqueues ; qid++ ) {
                volatile int *temp = cuda_queues[qid].data; cuda_queues[qid].data = cuda_queues[qid].rec_data; cuda_queues[qid].rec_data = temp;
                cuda_queues[qid].first = 0;
                cuda_queues[qid].last = cuda_queues[qid].count;
                cuda_queues[qid].rec_count = 0;
                }
        #endif
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

    int threadID;
    int cid, cjd;
    float epot = 0.0f;
    __shared__ volatile int pid;
    #ifdef FORCES_LOCAL
        int k;
        __shared__ __align__(16) int buff[ 8*cuda_nrparts ];
        float *forces_i = (float *)&buff[ 0 ];
        float *forces_j = (float *)&buff[ 3*cuda_nrparts ];
        unsigned int *sort_i = (unsigned int *)&buff[ 6*cuda_nrparts ];
        unsigned int *sort_j = (unsigned int *)&buff[ 7*cuda_nrparts ];
    #else
        unsigned int seed = 6178 + blockIdx.x;
        float *forces_i, *forces_j;
        __shared__ unsigned int sort_i[ cuda_nrparts ];
        __shared__ unsigned int sort_j[ cuda_nrparts ];
        struct queue_cuda *myq , *queues[ cuda_maxqueues ];
        int naq = cuda_nrqueues, qid;
    #endif
    #if !defined(PARTS_TEX)
        #ifdef PARTS_LOCAL
            float4 *parts_i;
            __shared__ float4 parts_j[ cuda_nrparts ];
        #else
            float4 *parts_i, *parts_j;
        #endif
    #endif
    
    TIMER_TIC2
    
    /* Get the block and thread ids. */
    threadID = threadIdx.x;
    
    /* Check that we've got the correct warp size! */
    /* if ( warpSize != cuda_frame ) {
        if ( blockID == 0 && threadID == 0 )
            printf( "runner_run_cuda: error: the warp size of the device (%i) does not match the warp size mdcore was compiled for (%i).\n" ,
                warpSize , cuda_frame );
        return;
        } */
        
    /* Init the list of queues. */
    #ifndef FORCES_LOCAL
    if ( threadID == 0 ) {
        myq = &cuda_queues[ get_smid() % cuda_nrqueues ];
        for ( qid = 0 ; qid < cuda_nrqueues ; qid++ )
            queues[qid] = &cuda_queues[qid];
        naq = cuda_nrqueues - 1;
        queues[ get_smid() ] = queues[ naq ];
        }
    #endif
        

    /* Main loop... */
    while ( 1 ) {
    
        /* Let the first thread grab a pair. */
        if ( threadID == 0 ) {
            #ifdef FORCES_LOCAL
                if ( ( pid = atomicAdd( &cuda_pair_next , 1 ) ) >= cuda_nr_pairs )
                    pid = -1;
            #else
                TIMER_TIC
                while ( 1 ) {
                    if ( myq->rec_count >= myq->count || ( pid = runner_cuda_gettask( myq , 0 ) ) < 0 ) {
                        for ( qid = 0 ; qid < naq ; qid++ )
                            if ( queues[qid]->rec_count >= queues[qid]->count )
                                queues[ qid-- ] = queues[ --naq ];
                        if ( naq == 0 ) {
                            pid = -1;
                            break;
                            }
                        seed = 1103515245 * seed + 12345;
                        qid = seed % naq;
                        if ( ( pid = runner_cuda_gettask( queues[qid] , 1 ) ) >= 0 ) {
                            if ( atomicAdd( (int *)&myq->count , 1 ) < cuda_queue_size )
                                myq->rec_data[ atomicAdd( (int *)&myq->rec_count , 1 ) ] = pid;
                            else {
                                atomicSub( (int *)&myq->count , 1 );
                                atomicAdd( (int *)&queues[qid]->count , 1 );
                                queues[qid]->rec_data[ atomicAdd( (int *)&queues[qid]->rec_count , 1 ) ] = pid;
                                }
                            break;
                            }
                        }
                    else
                        break;
                    }
                TIMER_TOC(tid_gettask)
            #endif
            }
            
        /* Exit if we didn't get a valid pair. */
        if ( pid < 0 )
            break;
            
        /* Get a hold of the pair cells. */
        cid = cuda_pairs[pid].i;
        cjd = cuda_pairs[pid].j;
    
        /* if ( threadID == 0 )
            printf( "runner_run_cuda: block %03i got pid=%i (%i/%i).\n" , blockID , pid , cid , cjd ); */
        
        /* Do the pair. */
        if ( cid != cjd ) {
        
            #ifdef FORCES_LOCAL
                /* Clear the forces buffer. */
                TIMER_TIC
                for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                    forces_i[k] = 0.0f;
                for ( k = threadID ; k < 3*counts[cjd] ; k += cuda_frame )
                    forces_j[k] = 0.0f;
                // __threadfence_block();
                TIMER_TOC(tid_update)
            
                /* Copy the particle data into the local buffers. */
                #ifndef PARTS_TEX
                    #ifdef PARTS_LOCAL
                        // cuda_memcpy( parts_i , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                        parts_i = &cuda_parts[ ind[cid] ];
                        cuda_memcpy( parts_j , &cuda_parts[ ind[cjd] ] , sizeof(float4) * counts[cjd] );
                        // __threadfence_block();
                    #else
                        parts_i = &cuda_parts[ ind[cid] ];
                        parts_j = &cuda_parts[ ind[cjd] ];
                    #endif
                #endif
            
                /* Compute the cell pair interactions. */
                #ifdef PARTS_TEX
                    runner_dopair4_sorted_cuda(
                        cid , counts[cid] ,
                        cjd , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        cuda_pairs[pid].shift ,
                        &epot );
                #else
                    runner_dopair4_sorted_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        cuda_pairs[pid].shift ,
                        &epot );
                #endif
                
                /* Write the particle forces back to cell_i. */
                if ( threadID == 0 )
                    cuda_mutex_lock( &cuda_taboo[cid] );
                cuda_sum( &forces[ 3*ind[cid] ] , forces_i , 3*counts[cid] );
                __threadfence();
                if ( threadID == 0 ) {
                    cuda_mutex_unlock( &cuda_taboo[cid] );
                    __threadfence();
                    }

                /* Write the particle forces back to cell_j. */
                if ( threadID == 0 )
                    cuda_mutex_lock( &cuda_taboo[cjd] );
                cuda_sum( &forces[ 3*ind[cjd] ] , forces_j , 3*counts[cjd] );
                __threadfence();
                if ( threadID == 0 ) {
                    cuda_mutex_unlock( &cuda_taboo[cjd] );
                    __threadfence();
                    }
                    
            #else
            
                /* Put a finger on the forces. */
                forces_i = &forces[ 3*ind[cid] ];
                forces_j = &forces[ 3*ind[cjd] ];
            
                /* Copy the particle data into the local buffers. */
                #ifndef PARTS_TEX
                    #ifdef PARTS_LOCAL
                        // cuda_memcpy( parts_i , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                        parts_i = &cuda_parts[ ind[cid] ];
                        cuda_memcpy( parts_j , &cuda_parts[ ind[cjd] ] , sizeof(float4) * counts[cjd] );
                        // __threadfence_block();
                    #else
                        parts_i = &cuda_parts[ ind[cid] ];
                        parts_j = &cuda_parts[ ind[cjd] ];
                    #endif
                #endif

                /* Compute the cell pair interactions. */
                #ifdef PARTS_TEX
                    runner_dopair4_sorted_cuda(
                        cid , counts[cid] ,
                        cjd , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        cuda_pairs[pid].shift ,
                        &epot );
                #else
                    runner_dopair4_sorted_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        cuda_pairs[pid].shift ,
                        &epot );
                #endif

                /* Unlock these cells' mutexes. */
                if ( threadID == 0 ) {
                    cuda_mutex_unlock( &cuda_taboo[cid] );
                    cuda_mutex_unlock( &cuda_taboo[cjd] );
                    }
                       
            #endif
                
            }
        else {
        
            #ifdef FORCES_LOCAL
                /* Clear the forces buffer. */
                TIMER_TIC
                for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                    forces_i[k] = 0.0f;
                // __threadfence_block();
                TIMER_TOC(tid_update)
            
                /* Copy the particle data into the local buffers. */
                #ifndef PARTS_TEX
                    #ifdef FORCES_LOCAL
                        parts_j = (float4 *)forces_j;
                        cuda_memcpy( parts_j , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                    #else
                        parts_j = &cuda_parts[ ind[cid] ];
                    #endif
                #endif
            
                /* Compute the cell self interactions. */
                #ifdef PARTS_TEX
                    runner_doself4_cuda( cid , counts[cid] , forces_i , &epot );
                #else
                    runner_doself4_cuda( parts_j , counts[cid] , forces_i , &epot );
                #endif
                
                /* Write the particle forces back to cell_i. */
                if ( threadID == 0 )
                    cuda_mutex_lock( &cuda_taboo[cid] );
                cuda_sum( &forces[ 3*ind[cid] ] , forces_i , 3*counts[cid] );
                __threadfence();
                if ( threadID == 0 ) {
                    cuda_mutex_unlock( &cuda_taboo[cid] );
                    __threadfence();
                    }
                    
            #else
                forces_i = &forces[ 3*ind[cid] ];
            
                /* Copy the particle data into the local buffers. */
                #ifndef PARTS_TEX
                    #ifdef FORCES_LOCAL
                        parts_j = (float4 *)forces_j;
                        cuda_memcpy( parts_j , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                    #else
                        parts_j = &cuda_parts[ ind[cid] ];
                    #endif
                #endif

                /* Compute the cell self interactions. */
                #ifdef PARTS_TEX
                    runner_doself4_cuda( cid , counts[cid] , forces_i , &epot );
                #else
                    runner_doself4_cuda( parts_j , counts[cid] , forces_i , &epot );
                #endif
                
                /* Unlock this cell's mutex. */
                if ( threadID == 0 )
                    cuda_mutex_unlock( &cuda_taboo[cid] );
                       
            #endif
            
            }
          
        } /* main loop. */

    /* Accumulate the potential energy. */
    atomicAdd( &cuda_epot , epot );

    /* Make a notch on the barrier, last one out cleans up the mess... */
    if ( threadID == 0 && atomicAdd( &cuda_barrier , 1 ) == gridDim.x-1 ) {
        cuda_barrier = 0;
        cuda_epot_out = cuda_epot;
        cuda_epot = 0.0f;
        #ifdef FORCES_LOCAL
            cuda_pair_next = 0;
        #else
            // int nr_tasks = 0;
            for ( qid = 0 ; qid < cuda_nrqueues ; qid++ ) {
                volatile int *temp = cuda_queues[qid].data; cuda_queues[qid].data = cuda_queues[qid].rec_data; cuda_queues[qid].rec_data = temp;
                cuda_queues[qid].first = 0;
                cuda_queues[qid].last = cuda_queues[qid].count;
                cuda_queues[qid].rec_count = 0;
                // printf( "runner_cuda: queue %i has %i elements.\n" , qid , cuda_queues[qid].count );
                // nr_tasks += cuda_queues[qid].count;
                }
            // printf( "runner_cuda: counted %i tasks in all queues.\n" , nr_tasks );
        #endif
        }
    
    /* if ( threadID == 0 )
        printf( "runner_run_cuda: block %03i is done.\n" , blockID ); */
        
    TIMER_TOC2(tid_total)

    }

