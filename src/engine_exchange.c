/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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
#ifdef CELL
    #include <libspe2.h>
#endif

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



/** 
 * @brief Wait for an asynchronous data exchange to finalize.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
#ifdef HAVE_MPI
int engine_exchange_wait ( struct engine *e ) {

    /* Try to grab the xchg_mutex, which will only be free while
       the async routine is waiting on a condition. */
    if ( pthread_mutex_lock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);
        
    /* If the async exchange was started but is not running,
       wait for a signal. */
    while ( e->xchg_started && ~e->xchg_running )
        if ( pthread_cond_wait( &e->xchg_cond , &e->xchg_mutex ) != 0 )
            return error(engine_err_pthread);
        
    /* We don't actually need this, so release it again. */
    if ( pthread_mutex_unlock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);
        
    /* The end of the tunnel. */
    return engine_err_ok;

    }
#endif


/**
 * @brief Exchange data with other nodes asynchronously.
 *
 * @param e The #engine to work with.
 * @param comm The @c MPI_Comm over which to exchange data.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Starts a new thread which handles the particle exchange. At the
 * start of the exchange, ghost cells are marked in the taboo-list
 * and only freed once their data has been received.
 *
 * The function #engine_exchange_wait can be used to wait for
 * the asynchronous communication to finish.
 */

#ifdef HAVE_MPI 
int engine_exchange_async_run ( struct engine *e ) {

    int i, k, ind, res, cid;
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    MPI_Request reqs_send2[ e->nr_nodes ], reqs_recv2[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c;
    struct space *s;
    FPTYPE h[3];

    /* Check the input. */
    if ( e == NULL )
        return error(engine_err_null);

    /* Get local copies of some data. */
    s = &e->s;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Initialize the request queues. */
    for ( k = 0 ; k < e->nr_nodes ; k++ ) {
        reqs_recv[k] = MPI_REQUEST_NULL;
        reqs_recv2[k] = MPI_REQUEST_NULL;
        reqs_send[k] = MPI_REQUEST_NULL;
        reqs_send2[k] = MPI_REQUEST_NULL;
        }
        
    /* Start by acquiring the xchg_mutex. */
    if ( pthread_mutex_lock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);

    /* Main loop... */
    while ( 1 ) {

        /* Wait for a signal to start. */
        e->xchg_running = 0;
        if ( pthread_cond_wait( &e->xchg_cond , &e->xchg_mutex ) != 0 )
            return error(engine_err_pthread);
            
        /* Tell the world I'm alive! */
        e->xchg_started = 0; e->xchg_running = 1;
        if ( pthread_cond_signal( &e->xchg_cond ) != 0 )
            return error(engine_err_pthread);
        
        /* Start by packing and sending/receiving a counts array for each send queue. */
        for ( i = 0 ; i < e->nr_nodes ; i++ ) {

            /* Do we have anything to send? */
            if ( e->send[i].count > 0 ) {

                /* Allocate a new lengths array. */
                if ( ( counts_out[i] = (int *)malloc( sizeof(int) * e->send[i].count ) ) == NULL )
                    return error(engine_err_malloc);

                /* Pack the array with the counts. */
                totals_send[i] = 0;
                for ( k = 0 ; k < e->send[i].count ; k++ )
                    totals_send[i] += ( counts_out[i][k] = s->cells[ e->send[i].cellid[k] ].count );
                /* printf( "engine_exchange[%i]: totals_send[%i]=%i.\n" , e->nodeID , i , totals_send[i] ); */

                /* Ship it off to the correct node. */
                if ( ( res = MPI_Isend( counts_out[i] , e->send[i].count , MPI_INT , i , e->nodeID , e->comm , &reqs_send[i] ) ) != MPI_SUCCESS )
                    return error(engine_err_mpi);
                /* printf( "engine_exchange[%i]: sending %i counts to node %i.\n" , e->nodeID , e->send[i].count , i ); */

                }

            /* Are we expecting any parts? */
            if ( e->recv[i].count > 0 ) {

                /* Allocate a new lengths array for the incomming data. */
                if ( ( counts_in[i] = (int *)malloc( sizeof(int) * e->recv[i].count ) ) == NULL )
                    return error(engine_err_malloc);

                /* Dispatch a recv request. */
                if ( ( res = MPI_Irecv( counts_in[i] , e->recv[i].count , MPI_INT , i , i , e->comm , &reqs_recv[i] ) ) != MPI_SUCCESS )
                    return error(engine_err_mpi);
                /* printf( "engine_exchange[%i]: recving %i counts from node %i.\n" , e->nodeID , e->recv[i].count , i ); */

                }

            }

        /* Send and receive data. */
        while ( 1 ) {

            /* Wait for this recv to come in. */
            res = MPI_Waitany( e->nr_nodes , reqs_recv , &i , MPI_STATUS_IGNORE );
            if ( i == MPI_UNDEFINED )
                break;
            
            /* Do we have anything to send? */
            if ( e->send[i].count > 0 ) {

                /* Allocate a buffer for the send queue. */
                if ( ( buff_send[i] = (struct part *)malloc( sizeof(struct part) * totals_send[i] ) ) == NULL )
                    return error(engine_err_malloc);

                /* Fill the send buffer. */
                finger = buff_send[i];
                for ( k = 0 ; k < e->send[i].count ; k++ ) {
                    c = &( s->cells[e->send[i].cellid[k]] );
                    memcpy( finger , c->parts , sizeof(struct part) * c->count );
                    finger = &( finger[ c->count ] );
                    }

                /* File a send. */
                if ( ( res = MPI_Isend( buff_send[i] , totals_send[i]*sizeof(struct part) , MPI_BYTE , i , e->nodeID , e->comm , &reqs_send2[i] ) ) != MPI_SUCCESS )
                    return error(engine_err_mpi);
                /* printf( "engine_exchange[%i]: sending %i parts to node %i.\n" , e->nodeID , totals_send[i] , i ); */

                }

            /* Are we expecting any parts? */
            if ( e->recv[i].count > 0 ) {

                /* Count the nr of parts to recv. */
                totals_recv[i] = 0;
                for ( k = 0 ; k < e->recv[i].count ; k++ )
                    totals_recv[i] += counts_in[i][k];

                /* Allocate a buffer for the send and recv queues. */
                if ( ( buff_recv[i] = (struct part *)malloc( sizeof(struct part) * totals_recv[i] ) ) == NULL )
                    return error(engine_err_malloc);

                /* File a recv. */
                if ( ( res = MPI_Irecv( buff_recv[i] , totals_recv[i]*sizeof(struct part) , MPI_BYTE , i , i , e->comm , &reqs_recv2[i] ) ) != MPI_SUCCESS )
                    return error(engine_err_mpi);
                /* printf( "engine_exchange[%i]: recving %i parts from node %i.\n" , e->nodeID , totals_recv[i] , i ); */

                }

            }

        /* Wait for all the recvs to come in. */
        /* if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
            return error(engine_err_mpi); */

        /* Unpack the received data. */
        for ( i = 0 ; i < e->nr_nodes ; i++ ) {

            /* Wait for this recv to come in. */
            res = MPI_Waitany( e->nr_nodes , reqs_recv2 , &ind , MPI_STATUS_IGNORE );

            /* Did we get a propper index? */
            if ( ind != MPI_UNDEFINED ) {

                /* Loop over the data and pass it to the cells. */
                finger = buff_recv[ind];
                for ( k = 0 ; k < e->recv[ind].count ; k++ ) {
                    cid = e->recv[ind].cellid[k];
                    c = &( s->cells[cid] );
                    cell_load( c , finger , counts_in[ind][k] , s->partlist , s->celllist );
                    space_releasepair( &e->s , cid , cid );
                    finger = &( finger[ counts_in[ind][k] ] );
                    }

                }

            }

        /* Wait for all the sends to come in. */
        if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
            return error(engine_err_mpi);
        if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send2 , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
            return error(engine_err_mpi);
        /* printf( "engine_exchange[%i]: all send/recv completed.\n" , e->nodeID ); */

        /* Free the send and recv buffers. */
        for ( i = 0 ; i < e->nr_nodes ; i++ ) {
            if ( e->send[i].count > 0 ) {
                free( buff_send[i] );
                free( counts_out[i] );
                }
            if ( e->recv[i].count > 0 ) {
                free( buff_recv[i] );
                free( counts_in[i] );
                }
            }

        } /* main loop. */
        
    }
#endif
        
        
/**
 * @brief Exchange data with other nodes asynchronously.
 *
 * @param e The #engine to work with.
 * @param comm The @c MPI_Comm over which to exchange data.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Starts a new thread which handles the particle exchange. At the
 * start of the exchange, ghost cells are marked in the taboo-list
 * and only freed once their data has been received.
 *
 * The function #engine_exchange_wait can be used to wait for
 * the asynchronous communication to finish.
 */

#ifdef HAVE_MPI 
int engine_exchange_async ( struct engine *e ) {

    int k, cid;

    /* Check the input. */
    if ( e == NULL )
        return error(engine_err_null);

    /* Bail if not in parallel. */
    if ( !(e->flags & engine_flag_mpi) || e->nr_nodes <= 1 )
        return engine_err_ok;
        
    /* Mark all the ghost cells as taboo and flush them. */
    for ( k = 0 ; k < e->s.nr_ghost ; k++ ) {
        cid = e->s.cid_ghost[k];
        e->s.cells_taboo[ cid ] += 2;
        if ( cell_flush( &e->s.cells[cid] , e->s.partlist , e->s.celllist ) < 0 )
            return error(engine_err_cell);
        }
            
    /* Get a hold of the exchange mutex. */
    if ( pthread_mutex_lock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);
        
    /* Tell the async thread to get to work. */
    e->xchg_started = 1;
    if ( pthread_cond_signal( &e->xchg_cond ) != 0 )
        return error(engine_err_pthread);
        
    /* Release the exchange mutex and let the async run. */
    if ( pthread_mutex_unlock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);
        
    /* Done (for now). */
    return engine_err_ok;
        
    }
#endif
    
    
/**
 * @brief Exchange data with other nodes.
 *
 * @param e The #engine to work with.
 * @param comm The @c MPI_Comm over which to exchange data.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

#ifdef HAVE_MPI 
int engine_exchange ( struct engine *e ) {

    int i, k, ind, res, pid, cid, delta[3];
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    MPI_Request reqs_send2[ e->nr_nodes ], reqs_recv2[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c, *c_dest;
    struct part *p;
    struct space *s;
    FPTYPE h[3];
    
    /* Check the input. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Bail if not in parallel. */
    if ( !(e->flags & engine_flag_mpi) || e->nr_nodes <= 1 )
        return engine_err_ok;
        
    /* Get local copies of some data. */
    s = &e->s;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Wait for any asynchronous calls to finish. */
    if ( e->flags & engine_flag_async )
        if ( engine_exchange_wait( e ) < 0 )
            return error(engine_err);
            
    /* Initialize the request queues. */
    for ( k = 0 ; k < e->nr_nodes ; k++ ) {
        reqs_recv[k] = MPI_REQUEST_NULL;
        reqs_recv2[k] = MPI_REQUEST_NULL;
        reqs_send[k] = MPI_REQUEST_NULL;
        reqs_send2[k] = MPI_REQUEST_NULL;
        }
        
    /* Start by packing and sending/receiving a counts array for each send queue. */
    #pragma omp parallel for schedule(static), private(i,k,res)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Do we have anything to send? */
        if ( e->send[i].count > 0 ) {
        
            /* Allocate a new lengths array. */
            counts_out[i] = (int *)malloc( sizeof(int) * e->send[i].count );

            /* Pack the array with the counts. */
            totals_send[i] = 0;
            for ( k = 0 ; k < e->send[i].count ; k++ )
                totals_send[i] += ( counts_out[i][k] = s->cells[ e->send[i].cellid[k] ].count );
            /* printf( "engine_exchange[%i]: totals_send[%i]=%i.\n" , e->nodeID , i , totals_send[i] ); */

            /* Ship it off to the correct node. */
            /* printf( "engine_exchange[%i]: sending %i counts to node %i.\n" , e->nodeID , e->send[i].count , i ); */
            #pragma omp critical
            { res = MPI_Isend( counts_out[i] , e->send[i].count , MPI_INT , i , e->nodeID , e->comm , &reqs_send[i] ); }
            
            }
            
        /* Are we expecting any parts? */
        if ( e->recv[i].count > 0 ) {
    
            /* Allocate a new lengths array for the incomming data. */
            counts_in[i] = (int *)malloc( sizeof(int) * e->recv[i].count );

            /* Dispatch a recv request. */
            /* printf( "engine_exchange[%i]: recving %i counts from node %i.\n" , e->nodeID , e->recv[i].count , i ); */
            #pragma omp critical
            { res = MPI_Irecv( counts_in[i] , e->recv[i].count , MPI_INT , i , i , e->comm , &reqs_recv[i] ); }
            
            }
    
        }
        
    /* Send and receive data for each neighbour as the counts trickle in. */
    #pragma omp parallel for schedule(static), private(i,finger,k,c,res)
    for ( ind = 0 ; ind < e->nr_nodes ; ind++ ) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        { res = MPI_Waitany( e->nr_nodes , reqs_recv , &i , MPI_STATUS_IGNORE ); }
        if ( i == MPI_UNDEFINED )
            continue;
        
        /* Do we have anything to send? */
        if ( e->send[i].count > 0 ) {
            
            /* Allocate a buffer for the send queue. */
            buff_send[i] = (struct part *)malloc( sizeof(struct part) * totals_send[i] );

            /* Fill the send buffer. */
            finger = buff_send[i];
            for ( k = 0 ; k < e->send[i].count ; k++ ) {
                c = &( s->cells[e->send[i].cellid[k]] );
                memcpy( finger , c->parts , sizeof(struct part) * c->count );
                finger = &( finger[ c->count ] );
                }

            /* File a send. */
            /* printf( "engine_exchange[%i]: sending %i parts to node %i.\n" , e->nodeID , totals_send[i] , i ); */
            #pragma omp critical
            { res = MPI_Isend( buff_send[i] , totals_send[i]*sizeof(struct part) , MPI_BYTE , i , e->nodeID , e->comm , &reqs_send2[i] ); }
            
            }
            
        /* Are we expecting any parts? */
        if ( e->recv[i].count > 0 ) {
    
            /* Count the nr of parts to recv. */
            totals_recv[i] = 0;
            for ( k = 0 ; k < e->recv[i].count ; k++ )
                totals_recv[i] += counts_in[i][k];

            /* Allocate a buffer for the send and recv queues. */
            buff_recv[i] = (struct part *)malloc( sizeof(struct part) * totals_recv[i] );

            /* File a recv. */
            /* printf( "engine_exchange[%i]: recving %i parts from node %i.\n" , e->nodeID , totals_recv[i] , i ); */
            #pragma omp critical
            { res = MPI_Irecv( buff_recv[i] , totals_recv[i]*sizeof(struct part) , MPI_BYTE , i , i , e->comm , &reqs_recv2[i] ); }
            
            }
            
        }

    /* Wait for all the recvs to come in. */
    /* if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi); */
        
    /* Unpack the received data. */
    #pragma omp parallel for schedule(static), private(i,ind,res,finger,k,c)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        { res = MPI_Waitany( e->nr_nodes , reqs_recv2 , &ind , MPI_STATUS_IGNORE ); }
        
        /* Did we get a propper index? */
        if ( ind != MPI_UNDEFINED ) {

            /* Loop over the data and pass it to the cells. */
            finger = buff_recv[ind];
            for ( k = 0 ; k < e->recv[ind].count ; k++ ) {
                c = &( s->cells[e->recv[ind].cellid[k]] );
                cell_flush( c , s->partlist , s->celllist );
                cell_load( c , finger , counts_in[ind][k] , s->partlist , s->celllist );
                finger = &( finger[ counts_in[ind][k] ] );
                }
                
            }
                
        }
        
    /* Wait for all the sends to come in. */
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send2 , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    /* printf( "engine_exchange[%i]: all send/recv completed.\n" , e->nodeID ); */
        
    /* Free the send and recv buffers. */
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
        if ( e->send[i].count > 0 ) {
            free( buff_send[i] );
            free( counts_out[i] );
            }
        if ( e->recv[i].count > 0 ) {
            free( buff_recv[i] );
            free( counts_in[i] );
            }
        }
        
    /* Do we need to update cell locations? */
    if ( !( e->flags & engine_flag_verlet ) ) {
    
        /* Shuffle the particles to the correct cells. */
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,k,delta,c_dest)
        for ( cid = 0 ; cid < s->nr_marked ; cid++ ) {
            c = &(s->cells[s->cid_marked[cid]]);
            pid = 0;
            while ( pid < c->count ) {

                p = &( c->parts[pid] );
                for ( k = 0 ; k < 3 ; k++ )
                    delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );

                /* do we have to move this particle? */
                if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                    for ( k = 0 ; k < 3 ; k++ )
                        p->x[k] -= delta[k] * h[k];
                    c_dest = &( s->cells[ space_cellid( s ,
                        (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] , 
                        (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] , 
                        (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );

	                if ( c_dest->flags & cell_flag_marked ) {
                        pthread_mutex_lock(&c_dest->cell_mutex);
                        cell_add_incomming( c_dest , p );
	                    pthread_mutex_unlock(&c_dest->cell_mutex);
                        s->celllist[ p->id ] = c_dest;
                        }
                    else {
                        s->partlist[ p->id ] = NULL;
                        s->celllist[ p->id ] = NULL;
                        }

                    c->count -= 1;
                    if ( pid < c->count ) {
                        c->parts[pid] = c->parts[c->count];
                        s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
                        }
                    }
                else
                    pid += 1;
                }
            }

        /* Welcome the new particles in each cell. */
        #pragma omp parallel for schedule(static), private(c)
        for ( cid = 0 ; cid < s->nr_marked ; cid++ )
            cell_welcome( &(s->cells[s->cid_marked[cid]]) , s->partlist );
            
        }
        
    /* Call it a day. */
    return engine_err_ok;
        
    }
#endif


/**
 * @brief Exchange incomming particle data with other nodes.
 *
 * @param e The #engine to work with.
 * @param comm The @c MPI_Comm over which to exchange data.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

#ifdef HAVE_MPI 
int engine_exchange_incomming ( struct engine *e ) {

    int i, j, k, ind, res;
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    MPI_Request reqs_send2[ e->nr_nodes ], reqs_recv2[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c;
    struct space *s;
    FPTYPE h[3];
    
    /* Check the input. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Bail if not in parallel. */
    if ( !(e->flags & engine_flag_mpi) || e->nr_nodes <= 1 )
        return engine_err_ok;
        
    /* Get local copies of some data. */
    s = &e->s;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Initialize the request queues. */
    for ( k = 0 ; k < e->nr_nodes ; k++ ) {
        reqs_recv[k] = MPI_REQUEST_NULL;
        reqs_recv2[k] = MPI_REQUEST_NULL;
        reqs_send[k] = MPI_REQUEST_NULL;
        reqs_send2[k] = MPI_REQUEST_NULL;
        }
        
    /* As opposed to #engine_exchange, we are going to send the incomming
       particles on ghost cells that do not belong to us. We therefore invert
       the send/recv queues, i.e. we send the incommings for the cells
       from which we usually receive data. */
        
    /* Start by packing and sending/receiving a counts array for each send queue. */
    #pragma omp parallel for schedule(static), private(i,k,res)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Do we have anything to send? */
        if ( e->recv[i].count > 0 ) {
        
            /* Allocate a new lengths array. */
            counts_out[i] = (int *)malloc( sizeof(int) * e->recv[i].count );

            /* Pack the array with the counts. */
            totals_send[i] = 0;
            for ( k = 0 ; k < e->recv[i].count ; k++ )
                totals_send[i] += ( counts_out[i][k] = s->cells[ e->recv[i].cellid[k] ].incomming_count );
            /* printf( "engine_exchange[%i]: totals_send[%i]=%i.\n" , e->nodeID , i , totals_send[i] ); */

            /* Ship it off to the correct node. */
            /* printf( "engine_exchange[%i]: sending %i counts to node %i.\n" , e->nodeID , e->send[i].count , i ); */
            #pragma omp critical
            { res = MPI_Isend( counts_out[i] , e->recv[i].count , MPI_INT , i , e->nodeID , e->comm , &reqs_send[i] ); }
            
            }
            
        /* Are we expecting any parts? */
        if ( e->send[i].count > 0 ) {
    
            /* Allocate a new lengths array for the incomming data. */
            counts_in[i] = (int *)malloc( sizeof(int) * e->send[i].count );

            /* Dispatch a recv request. */
            /* printf( "engine_exchange[%i]: recving %i counts from node %i.\n" , e->nodeID , e->recv[i].count , i ); */
            #pragma omp critical
            { res = MPI_Irecv( counts_in[i] , e->send[i].count , MPI_INT , i , i , e->comm , &reqs_recv[i] ); }
            
            }
    
        }
        
    /* Send and receive data. */
    #pragma omp parallel for schedule(static), private(ind,i,finger,k,c,res)
    for ( ind = 0 ; ind < e->nr_nodes ; ind++ ) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        { res = MPI_Waitany( e->nr_nodes , reqs_recv , &i , MPI_STATUS_IGNORE ); }
        if ( i == MPI_UNDEFINED )
            continue;
        
        /* Do we have anything to send? */
        if ( e->recv[i].count > 0 ) {
            
            /* Allocate a buffer for the send queue. */
            buff_send[i] = (struct part *)malloc( sizeof(struct part) * totals_send[i] );

            /* Fill the send buffer. */
            finger = buff_send[i];
            for ( k = 0 ; k < e->recv[i].count ; k++ ) {
                c = &( s->cells[e->recv[i].cellid[k]] );
                memcpy( finger , c->incomming , sizeof(struct part) * c->incomming_count );
                finger = &( finger[ c->incomming_count ] );
                }

            /* File a send. */
            /* printf( "engine_exchange[%i]: sending %i parts to node %i.\n" , e->nodeID , totals_send[i] , i ); */
            #pragma omp critical
            { res = MPI_Isend( buff_send[i] , totals_send[i]*sizeof(struct part) , MPI_BYTE , i , e->nodeID , e->comm , &reqs_send2[i] ); }
            
            }
            
        /* Are we expecting any parts? */
        if ( e->send[i].count > 0 ) {
    
            /* Count the nr of parts to recv. */
            totals_recv[i] = 0;
            for ( k = 0 ; k < e->send[i].count ; k++ )
                totals_recv[i] += counts_in[i][k];

            /* Allocate a buffer for the send and recv queues. */
            buff_recv[i] = (struct part *)malloc( sizeof(struct part) * totals_recv[i] );

            /* File a recv. */
            /* printf( "engine_exchange[%i]: recving %i parts from node %i.\n" , e->nodeID , totals_recv[i] , i ); */
            #pragma omp critical
            { res = MPI_Irecv( buff_recv[i] , totals_recv[i]*sizeof(struct part) , MPI_BYTE , i , i , e->comm , &reqs_recv2[i] ); }
            
            }
            
        }

    /* Wait for all the recvs to come in. */
    /* if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi); */
        
    /* Unpack the received data. */
    #pragma omp parallel for schedule(static), private(i,j,ind,res,finger,k,c)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        { res = MPI_Waitany( e->nr_nodes , reqs_recv2 , &ind , MPI_STATUS_IGNORE ); }
        
        /* Did we get a propper index? */
        if ( ind != MPI_UNDEFINED ) {

            /* Loop over the data and pass it to the cells. */
            finger = buff_recv[ind];
            for ( k = 0 ; k < e->send[ind].count ; k++ ) {
                c = &( s->cells[e->send[ind].cellid[k]] );
                pthread_mutex_lock( &c->cell_mutex );
                cell_add_incomming_multiple( c , finger , counts_in[ind][k] );
                pthread_mutex_unlock( &c->cell_mutex );
                for ( j = 0 ; j < counts_in[ind][k] ; j++ )
                    e->s.celllist[ finger[j].id ] = c;
                finger = &( finger[ counts_in[ind][k] ] );
                }
                
            }
                
        }
        
    /* Wait for all the sends to come in. */
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send2 , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    /* printf( "engine_exchange[%i]: all send/recv completed.\n" , e->nodeID ); */
        
    /* Free the send and recv buffers. */
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
        if ( e->send[i].count > 0 ) {
            free( buff_send[i] );
            free( counts_out[i] );
            }
        if ( e->recv[i].count > 0 ) {
            free( buff_recv[i] );
            free( counts_in[i] );
            }
        }
        
    /* Call it a day. */
    return engine_err_ok;
        
    }
#endif


