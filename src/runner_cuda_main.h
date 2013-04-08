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


/* Set the kernel names depending on cuda_nparts. */
#define PASTE(x,y) x ## _ ## y
#define runner_run_verlet_cuda(N) PASTE(runner_run_verlet_cuda,N)
#define runner_run_cuda(N) PASTE(runner_run_cuda,N)


/**
 * @brief Loop over the cell pairs and process them.
 *
 * @param cells Array of cells on the device.
 *
 */
 
__global__ void runner_run_cuda(cuda_nparts) ( float *forces , int *counts , int *ind , int verlet_rebuild ) {

    int k, threadID;
    int cid, cjd, sid;
    float epot = 0.0f;
    volatile __shared__ int tid;
    __shared__ float shift[3];
    __shared__ unsigned int dshift;
    #ifdef FORCES_LOCAL
        __shared__ __align__(16) int buff[ 8*cuda_nparts ];
        float *forces_i = (float *)&buff[ 0 ];
        float *forces_j = (float *)&buff[ 3*cuda_nparts ];
        unsigned int *sort_i = (unsigned int *)&buff[ 6*cuda_nparts ];
        unsigned int *sort_j = (unsigned int *)&buff[ 7*cuda_nparts ];
    #else
        float *forces_i, *forces_j;
        __shared__ unsigned int sort_i[ cuda_nparts ];
        __shared__ unsigned int sort_j[ cuda_nparts ];
    #endif
    #if !defined(PARTS_TEX)
        #ifdef PARTS_LOCAL
            float4 *parts_i;
            __shared__ float4 parts_j[ cuda_nparts ];
        #else
            float4 *parts_i, *parts_j;
        #endif
    #endif
    
    TIMER_TIC2
    
    /* Get the block and thread ids. */
    threadID = threadIdx.x;
    
    /* Main loop... */
    while ( 1 ) {
    
        /* Let the first thread grab a task. */
        if ( threadID == 0 ) {
            TIMER_TIC
            tid = runner_cuda_gettask( &cuda_queues[0] , 0 );
            TIMER_TOC(tid_gettask)
            }
            
        /* Sync if needed. */
        if ( cuda_frame > 32 )
            __syncthreads();
                
        /* Exit if we didn't get a valid task. */
        if ( tid < 0 )
            break;

        /* Switch task type. */
        if ( cuda_tasks[tid].type == task_type_pair ) {
        
            /* Get a hold of the pair cells. */
            cid = cuda_tasks[tid].i;
            cjd = cuda_tasks[tid].j;
            
            /* Get the shift and dshift vector for this pair. */
            if ( threadID == 0 ) {
                for ( k = 0 ; k < 3 ; k++ ) {
                    shift[k] = cuda_corig[ 3*cjd + k ] - cuda_corig[ 3*cid + k ];
                    if ( 2*shift[k] > cuda_dim[k] )
                        shift[k] -= cuda_dim[k];
                    else if ( 2*shift[k] < -cuda_dim[k] )
                        shift[k] += cuda_dim[k];
                    }
                dshift = cuda_dscale * ( shift[0]*cuda_shiftn[ 3*cuda_tasks[tid].flags + 0 ] +
                                         shift[1]*cuda_shiftn[ 3*cuda_tasks[tid].flags + 1 ] +
                                         shift[2]*cuda_shiftn[ 3*cuda_tasks[tid].flags + 2 ] );
                }
    
            /* Sync if needed. */
            if ( cuda_frame > 32 )
                __syncthreads();
                
            #ifdef FORCES_LOCAL
        
                /* Clear the forces buffers. */
                for ( k = threadID ; k < 3*counts[cid] ; k += cuda_frame )
                    forces_i[k] = 0.0f;
                for ( k = threadID ; k < 3*counts[cjd] ; k += cuda_frame )
                    forces_j[k] = 0.0f;
                // __threadfence_block();
                
                /* Load the sorted indices. */
                cuda_memcpy( sort_i , &cuda_sortlists[ 13*ind[cid] + counts[cid]*cuda_tasks[tid].flags ] , sizeof(int)*counts[cid] );
                cuda_memcpy( sort_j , &cuda_sortlists[ 13*ind[cjd] + counts[cjd]*cuda_tasks[tid].flags ] , sizeof(int)*counts[cjd] );
                
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
                    runner_dopair4_cuda(
                        cid , counts[cid] ,
                        cjd , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        shift , dshift ,
                        &epot );
                #else
                    runner_dopair4_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        shift , dshift ,
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
                
                /* Load the sorted indices. */
                cuda_memcpy( sort_i , &cuda_sortlists[ 13*ind[cid] + counts[cid]*cuda_tasks[tid].flags ] , sizeof(int)*counts[cid] );
                cuda_memcpy( sort_j , &cuda_sortlists[ 13*ind[cjd] + counts[cjd]*cuda_tasks[tid].flags ] , sizeof(int)*counts[cjd] );
                
                /* Sync if needed. */
                if ( cuda_frame > 32 )
                    __syncthreads();
                
                /* Copy the particle data into the local buffers. */
                #ifndef PARTS_TEX
                    #ifdef PARTS_LOCAL
                        parts_i = &cuda_parts[ ind[cid] ];
                        cuda_memcpy( parts_j , &cuda_parts[ ind[cjd] ] , sizeof(float4) * counts[cjd] );
                        if ( cuda_frame > 32 )
                            __syncthreads();
                    #else
                        parts_i = &cuda_parts[ ind[cid] ];
                        parts_j = &cuda_parts[ ind[cjd] ];
                    #endif
                #endif
                
                /* Compute the cell pair interactions. */
                #ifdef PARTS_TEX
                    runner_dopair_onesided1_cuda(
                        cid , counts[cid] ,
                        cjd , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        shift , dshift ,
                        &epot );
                    runner_dopair_onesided2_cuda(
                        cid , counts[cid] ,
                        cjd , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        shift , dshift ,
                        &epot );
                #else
                    runner_dopair_onesided1_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        shift , dshift ,
                        &epot );
                    runner_dopair_onesided2_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        shift , dshift ,
                        &epot );
                    /* runner_dopair_cuda(
                        parts_i , counts[cid] ,
                        parts_j , counts[cjd] ,
                        forces_i , forces_j , 
                        sort_i , sort_j ,
                        shift , dshift ,
                        &epot ); */
                #endif
                    
                /* Sync if needed. */
                if ( cuda_frame > 32 )
                    __syncthreads();
                
                /* Unlock these cells' mutexes. */
                /* if ( threadID == 0 ) {
                    cuda_mutex_unlock( &cuda_taboo[cid] );
                    cuda_mutex_unlock( &cuda_taboo[cjd] );
                    } */
                       
            #endif
                
            }
        else if ( cuda_tasks[tid].type == task_type_self ) {
        
            /* Get a hold of the cell id. */
            cid = cuda_tasks[tid].i;
            
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
                    runner_doself_cuda( cid , counts[cid] , forces_i , &epot );
                #else
                    runner_doself_cuda( parts_j , counts[cid] , forces_i , &epot );
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
                    #ifdef PARTS_LOCAL
                        cuda_memcpy( parts_j , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                        if ( cuda_frame > 32 )
                            __syncthreads();
                    #else
                        parts_j = &cuda_parts[ ind[cid] ];
                    #endif
                #endif
                
                /* Compute the cell self interactions. */
                #ifdef PARTS_TEX
                    runner_doself_onesided_cuda( cid , counts[cid] , forces_i , &epot );
                #else
                    runner_doself_onesided_cuda( parts_j , counts[cid] , forces_i , &epot );
                #endif
                
                /* Sync if needed. */
                if ( cuda_frame > 32 )
                    __syncthreads();
                
                /* Unlock this cell's mutex. */
                /* if ( threadID == 0 )
                    cuda_mutex_unlock( &cuda_taboo[cid] ); */
                       
            #endif
        
            }
            
        /* Only do sorts if we have to re-build the pseudo-verlet lists. */
        else if ( cuda_tasks[tid].type == task_type_sort && verlet_rebuild ) {
        
            /* Get a hold of the cell id. */
            cid = cuda_tasks[tid].i;
            
            /* Copy the particle data into the local buffers. */
            #ifndef PARTS_TEX
                #ifdef PARTS_LOCAL
                    cuda_memcpy( parts_j , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                    if ( cuda_frame > 32 )
                        __syncthreads();
                #elif defined(FORCES_LOCAL)
                    parts_j = (float4 *)forces_i;
                    cuda_memcpy( parts_j , &cuda_parts[ ind[cid] ] , sizeof(float4) * counts[cid] );
                    if ( cuda_frame > 32 )
                        __syncthreads();
                #else
                    parts_j = &cuda_parts[ ind[cid] ];
                #endif
            #endif
                
            /* Loop over the different sort IDs. */
            for ( sid = 0 ; sid < 13 ; sid++ ) {
            
                /* Is this sid selected? */
                if ( !( cuda_tasks[tid].flags & (1 << sid) ) )
                    continue;
                    
                /* Call the sorting function with the buffer. */
                #ifdef PARTS_TEX
                    runner_dosort_cuda( cid , counts[cid] , sort_i , sid );
                #else
                    runner_dosort_cuda( parts_j , counts[cid] , sort_i , sid );
                #endif
                
                /* Sync if needed. */
                if ( cuda_frame > 32 )
                    __syncthreads();
                
                /* Copy the local shared memory back to the global memory. */
                cuda_memcpy( &cuda_sortlists[ 13*ind[cid] + sid*counts[cid] ] , sort_i , sizeof(unsigned int) * counts[cid] );
            
                /* Sync if needed. */
                if ( cuda_frame > 32 )
                    __syncthreads();
                
                }
        
            }
            
        /* Unlock any follow-up tasks. */
        if ( threadID == 0 )
            for ( k = 0 ; k < cuda_tasks[tid].nr_unlock ; k++ )
                atomicSub( (int *)&cuda_tasks[ cuda_tasks[tid].unlock[k] ].wait , 1 );
            
        } /* main loop. */
        
    /* Accumulate the potential energy. */
    atomicAdd( &cuda_epot , epot );

    /* Make a notch on the barrier, last one out cleans up the mess... */
    if ( threadID == 0 )
        tid = ( atomicAdd( &cuda_barrier , 1 ) == gridDim.x-1 );
    if ( cuda_frame > 32 )
        __syncthreads();
    if ( tid ) {
        if ( threadID == 0 ) {
            cuda_barrier = 0;
            cuda_epot_out = cuda_epot;
            cuda_epot = 0.0f;
            volatile int *temp = cuda_queues[0].data; cuda_queues[0].data = cuda_queues[0].rec_data; cuda_queues[0].rec_data = temp;
            cuda_queues[0].first = 0;
            cuda_queues[0].last = cuda_queues[0].count;
            cuda_queues[0].rec_count = 0;
            }
        for ( int j = threadID ; j < cuda_nr_tasks ; j += cuda_frame )
            for ( k = 0 ; k < cuda_tasks[j].nr_unlock ; k++ )
                atomicAdd( (int *) &cuda_tasks[ cuda_tasks[j].unlock[k] ].wait , 1 );
        }
    
    TIMER_TOC2(tid_total)

    }
    
    

