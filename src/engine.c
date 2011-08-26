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
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "runner.h"
#include "bond.h"
#include "angle.h"
#include "engine.h"


/** ID of the last error. */
int engine_err = engine_err_ok;


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *engine_err_msg[13] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "An error occured when calling a space function.",
    "A call to a pthread routine failed.",
    "An error occured when calling a runner function.",
    "One or more values were outside of the allowed range.",
    "An error occured while calling a cell function.",
    "The computational domain is too small for the requested operation.",
    "mdcore was not compiled with MPI.",
    "An error occured while calling an MPI function.",
    "An error occured when calling a bond function.", 
    "An error occured when calling a, angle function."
	};


/**
 * @brief Dump the contents of the enginge to a PSF and PDB file.
 *
 * @param e The #engine.
 * @param psf A pointer to @c FILE to which to write the PSF file.
 * @param pdb A pointer to @c FILE to which to write the PDB file.
 *
 * If any of @c psf or @c pdb are @c NULL, the respective output will
 * not be generated.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_dump_PSF ( struct engine *e , FILE *psf , FILE *pdb ) {

    struct space *s;
    struct cell *c;
    struct part *p;
    int pid, bid, aid;

    /* Check inputs. */
    if ( e == NULL || ( psf == NULL && pdb == NULL ) )
        return error(engine_err_null);
        
    /* Get a hold of the space. */
    s = &e->s;
        
    /* Write the header for the psf file if needed. */
    if ( psf != NULL )
        fprintf( psf , "PSF\n0 !NTITLE\n%i !NATOM\n" , s->nr_parts );
        
    /* Loop over the cells and parts. */
    for ( pid = 0 ; pid < s->nr_parts ; pid++ ) {
        if ( ( p = s->partlist[pid] ) == NULL || ( c = s->celllist[pid] ) == NULL )
            continue;
        if ( pdb != NULL )
            fprintf( pdb , "ATOM  %5d %4s %-3s %5d%1s   %8.3f%8.3f%8.3f\n",
                p->id+1 , e->types[p->type].name2 , "TIP3" , p->vid+1 , "" ,
                10 * ( p->x[0] + c->origin[0] ) , 10 * ( p->x[1] + c->origin[1] ) , 10 * ( p->x[2] + c->origin[2] ) );
        if ( psf != NULL )
            fprintf( psf , "%8i %4s %4i %4s %4s %4s %15.6f %15.6f    0\n" ,
                p->id+1 , "WAT" , p->vid+1 , "TIP3" , e->types[p->type].name2 , e->types[p->type].name , e->types[p->type].charge , e->types[p->type].mass );
        }
        
    /* Dump bonds and angles to PSF? */
    if ( psf != NULL ) {
    
        /* Dump bonds. */
        fprintf( psf , "%i !NBOND\n" , e->nr_bonds + e->nr_angles );
        for ( bid = 0 ; bid < e->nr_bonds ; bid++ )
            if ( bid % 4 == 3 )
                fprintf( psf , " %i %i\n" , e->bonds[bid].i+1 , e->bonds[bid].j+1 );
            else
                fprintf( psf , " %i %i" , e->bonds[bid].i+1 , e->bonds[bid].j+1 );
        for ( aid = 0 ; aid < e->nr_angles ; aid++ )
            if ( aid % 4 == 3 )
                fprintf( psf , " %i %i\n" , e->angles[aid].i+1 , e->angles[aid].k+1 );
            else
                fprintf( psf , " %i %i" , e->angles[aid].i+1 , e->angles[aid].k+1 );
                
        /* Dump angles. */
        fprintf( psf , "%i !NTHETA\n" , e->nr_angles );
        for ( aid = 0 ; aid < e->nr_angles ; aid++ )
            if ( aid % 3 == 2 )
                fprintf( psf , " %i %i %i\n" , e->angles[aid].i+1 , e->angles[aid].j+1 , e->angles[aid].k+1 );
            else
                fprintf( psf , " %i %i %i" , e->angles[aid].i+1 , e->angles[aid].j+1 , e->angles[aid].k+1 );
                
        /* Dump remaining bogus headers. */
        fprintf( psf , "0 !NPHI\n" );
        fprintf( psf , "0 !NIMPHI\n" );
        fprintf( psf , "0 !NDON\n" );
        fprintf( psf , "0 !NACC\n" );
        fprintf( psf , "0 !NNB\n" );
        
        }
        
    /* We're on a road to nowhere... */
    return engine_err_ok;

    }


/**
 * @brief Add a angle interaction to the engine.
 *
 * @param e The #engine.
 * @param i The ID of the first #part.
 * @param i The ID of the second #part.
 * @param k The ID of the third #part.
 * @param pid Index of the #potential for this bond.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_angle_add ( struct engine *e , int i , int j , int k , int pid ) {

    struct angle *dummy;

    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( i > e->s.nr_parts || j > e->s.nr_parts )
        return error(engine_err_range);
    if ( pid > e->nr_anglepots )
        return error(engine_err_range);
        
    /* Do we need to grow the angles array? */
    if ( e->nr_angles == e->angles_size ) {
        e->angles_size += 100;
        if ( ( dummy = (struct angle *)malloc( sizeof(struct angle) * e->angles_size ) ) == NULL )
            return error(engine_err_malloc);
        memcpy( dummy , e->angles , sizeof(struct angle) * e->nr_angles );
        free( e->angles );
        e->angles = dummy;
        }
        
    /* Store this angle. */
    e->angles[ e->nr_angles ].i = i;
    e->angles[ e->nr_angles ].j = j;
    e->angles[ e->nr_angles ].k = k;
    e->angles[ e->nr_angles ].pid = pid;
    e->nr_angles += 1;
    
    /* It's the end of the world as we know it. */
    return engine_err_ok;

    }


/**
 * @brief Add a bonded interaction to the engine.
 *
 * @param e The #engine.
 * @param i The ID of the first #part.
 * @param i The ID of the second #part.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_bond_add ( struct engine *e , int i , int j ) {

    struct bond *dummy;

    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( i > e->s.nr_parts || j > e->s.nr_parts )
        return error(engine_err_range);
        
    /* Do we need to grow the bonds array? */
    if ( e->nr_bonds == e->bonds_size ) {
        e->bonds_size += 100;
        if ( ( dummy = (struct bond *)malloc( sizeof(struct bond) * e->bonds_size ) ) == NULL )
            return error(engine_err_malloc);
        memcpy( dummy , e->bonds , sizeof(struct bond) * e->nr_bonds );
        free( e->bonds );
        e->bonds = dummy;
        }
        
    /* Store this bond. */
    e->bonds[ e->nr_bonds ].i = i;
    e->bonds[ e->nr_bonds ].j = j;
    e->nr_bonds += 1;
    
    /* It's the end of the world as we know it. */
    return engine_err_ok;

    }


/**
 * @brief Compute the angleed interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_angle_eval ( struct engine *e ) {

    double epot = 0.0;
    struct space *s;
    #ifdef HAVE_OPENMP
        FPTYPE *eff;
        int finger_global = 0, finger, count, cid, pid, gpid, k;
        struct part *p;
        struct cell *c;
    #endif
    
    /* Get a handle on the space. */
    s = &e->s;

    #ifdef HAVE_OPENMP
    
        /* Is it worth parallelizing? */
        #pragma omp parallel private(eff,finger,count), reduction(+:epot)
        if ( omp_get_num_threads() > 1 && e->nr_angles > engine_angles_chunk ) {
    
            /* Allocate a buffer for the forces. */
            eff = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * s->nr_parts );
            bzero( eff , sizeof(FPTYPE) * 4 * s->nr_parts );

            /* Main loop. */
            while ( finger_global < e->nr_angles ) {

                /* Get a finger on the angles list. */
                #pragma omp critical
                {
                    if ( finger_global < e->nr_angles ) {
                        finger = finger_global;
                        count = engine_angles_chunk;
                        if ( finger + count > e->nr_angles )
                            count = e->nr_angles - finger;
                        finger_global += count;
                        }
                    else
                        count = 0;
                    }

                /* Compute the angleed interactions. */
                if ( count > 0 )
                    angle_evalf( &e->angles[finger] , count , e , eff , &epot );

                } /* main loop. */

            /* Write-back the forces. */
            #pragma omp critical
            for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
                c = &s->cells[ cid ];
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &c->parts[ pid ];
                    gpid = p->id;
                    for ( k = 0 ; k < 3 ; k++ )
                        p->f[k] += eff[ gpid*4 + k ];
                    }
                }
            free( eff );
                
            }
            
        /* Otherwise, evaluate directly. */
        else if ( omp_get_thread_num() == 0 )
            angle_eval( e->angles , e->nr_angles , e , &epot );
    #else
        if ( angle_eval( e->angles , e->nr_angles , e , &epot ) < 0 )
            return error(engine_err_angle);
    #endif
        
    /* Store the potential energy. */
    s->epot += epot;
    
    /* I'll be back... */
    return engine_err_ok;

    }


/**
 * @brief Compute the bonded interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_bond_eval ( struct engine *e ) {

    double epot = 0.0;
    struct space *s;
    #ifdef HAVE_OPENMP
        FPTYPE *eff;
        int finger_global = 0, finger, count, cid, pid, gpid, k;
        struct part *p;
        struct cell *c;
    #endif
    
    /* Get a handle on the space. */
    s = &e->s;

    #ifdef HAVE_OPENMP
    
        /* Is it worth parallelizing? */
        #pragma omp parallel private(eff,finger,count), reduction(+:epot)
        if ( omp_get_num_threads() > 1 && e->nr_bonds > engine_bonds_chunk ) {
    
            /* Allocate a buffer for the forces. */
            eff = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * s->nr_parts );
            bzero( eff , sizeof(FPTYPE) * 4 * s->nr_parts );

            /* Main loop. */
            while ( finger_global < e->nr_bonds ) {

                /* Get a finger on the bonds list. */
                #pragma omp critical
                {
                    if ( finger_global < e->nr_bonds ) {
                        finger = finger_global;
                        count = engine_bonds_chunk;
                        if ( finger + count > e->nr_bonds )
                            count = e->nr_bonds - finger;
                        finger_global += count;
                        }
                    else
                        count = 0;
                    }

                /* Compute the bonded interactions. */
                if ( count > 0 )
                    bond_evalf( &e->bonds[finger] , count , e , eff , &epot );

                } /* main loop. */

            /* Write-back the forces. */
            #pragma omp critical
            for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
                c = &s->cells[ cid ];
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &c->parts[ pid ];
                    gpid = p->id;
                    for ( k = 0 ; k < 3 ; k++ )
                        p->f[k] += eff[ gpid*4 + k ];
                    }
                }
            free( eff );
                
            }
            
        /* Otherwise, evaluate directly. */
        else if ( omp_get_thread_num() == 0 )
            bond_eval( e->bonds , e->nr_bonds , e , &epot );
    #else
        if ( bond_eval( e->bonds , e->nr_bonds , e , &epot ) < 0 )
            return error(engine_err_bond);
    #endif
        
    /* Store the potential energy. */
    s->epot += epot;
    
    /* I'll be back... */
    return engine_err_ok;

    }


/**
 * @brief Add a bond potential.
 *
 * @param e The #engine.
 * @param p The #potential to add to the #engine.
 * @param i ID of particle type for this interaction.
 * @param j ID of second particle type for this interaction.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Adds the given bonded potential for pairs of particles of type @c i and @c j,
 * where @c i and @c j may be the same type ID.
 */
 
int engine_bond_addpot ( struct engine *e , struct potential *p , int i , int j ) {

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( i < 0 || i >= e->max_type || j < 0 || j >= e->max_type )
        return error(engine_err_range);
        
    /* store the potential. */
    e->p_bond[ i * e->max_type + j ] = p;
    if ( i != j )
        e->p_bond[ j * e->max_type + i ] = p;
        
    /* end on a good note. */
    return engine_err_ok;

    }


/**
 * @brief Add a angle potential.
 *
 * @param e The #engine.
 * @param p The #potential to add to the #engine.
 *
 * @return The ID of the added angle potential or < 0 on error (see #engine_err).
 */
 
int engine_angle_addpot ( struct engine *e , struct potential *p ) {

    struct potential **dummy;

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Is there enough room in p_angle? */
    if ( e->nr_anglepots == e->anglepots_size ) {
        e->anglepots_size += 100;
        if ( ( dummy = (struct potential **)malloc( sizeof(struct potential *) * e->anglepots_size ) ) == NULL )
            return engine_err_malloc;
        memcpy( dummy , e->p_angle , sizeof(struct potential *) * e->nr_anglepots );
        free( e->p_angle );
        e->p_angle = dummy;
        }
        
    /* store the potential. */
    e->p_angle[ e->nr_anglepots ] = p;
    e->nr_anglepots += 1;
        
    /* end on a good note. */
    return e->nr_anglepots - 1;

    }


/**
 * @brief Exchange data with other nodes.
 *
 * @param e The #engine to work with.
 * @param comm The @c MPI_Comm over which to exchange data.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

#ifdef HAVE_MPI 
int engine_exchange ( struct engine *e , MPI_Comm comm ) {

    int i, k, ind, res, pid, cid, delta[3];
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c, *c_dest;
    struct part *p;
    struct space *s;
    FPTYPE h[3];
    pthread_mutex_t mpi_mutex = PTHREAD_MUTEX_INITIALIZER;
    
    /* Check the input. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Get local copies of some data. */
    s = &e->s;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Start by packing and sending/receiving a counts array for each send queue. */
    #pragma omp parallel for schedule(static), private(i,k,res)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Do we have anything to send? */
        if ( e->send[i].count > 0 ) {
        
            /* Allocate a new lengths array. */
            /* if ( ( counts[i] = (int *)alloca( sizeof(int) * e->send[i].count ) ) == NULL )
                return error(engine_err_malloc); */
            counts_out[i] = (int *)malloc( sizeof(int) * e->send[i].count );

            /* Pack the array with the counts. */
            totals_send[i] = 0;
            for ( k = 0 ; k < e->send[i].count ; k++ )
                totals_send[i] += ( counts_out[i][k] = s->cells[ e->send[i].cellid[k] ].count );
            /* printf( "engine_exchange[%i]: totals_send[%i]=%i.\n" , e->nodeID , i , totals_send[i] ); */

            /* Ship it off to the correct node. */
            /* if ( ( res = MPI_Isend( counts[i] , e->send[i].count , MPI_INT , i , e->nodeID , comm , &reqs_send[i] ) ) != MPI_SUCCESS )
                return error(engine_err_mpi); */
            /* printf( "engine_exchange[%i]: sending %i counts to node %i.\n" , e->nodeID , e->send[i].count , i ); */
            pthread_mutex_lock( &mpi_mutex );
            res = MPI_Isend( counts_out[i] , e->send[i].count , MPI_INT , i , e->nodeID , comm , &reqs_send[i] );
            pthread_mutex_unlock( &mpi_mutex );
            
            }
        else
            reqs_send[i] = MPI_REQUEST_NULL;
            
        /* Are we expecting any parts? */
        if ( e->recv[i].count > 0 ) {
    
            /* Allocate a new lengths array for the incomming data. */
            /* if ( ( counts[i] = (int *)alloca( sizeof(int) * e->recv[i].count ) ) == NULL )
                return error(engine_err_malloc); */
            counts_in[i] = (int *)malloc( sizeof(int) * e->recv[i].count );

            /* Dispatch a recv request. */
            /* if ( ( res = MPI_Irecv( counts[i] , e->recv[i].count , MPI_INT , i , i , comm , &reqs_recv[i] ) ) != MPI_SUCCESS )
                return error(engine_err_mpi); */
            /* printf( "engine_exchange[%i]: recving %i counts from node %i.\n" , e->nodeID , e->recv[i].count , i ); */
            pthread_mutex_lock( &mpi_mutex );
            res = MPI_Irecv( counts_in[i] , e->recv[i].count , MPI_INT , i , i , comm , &reqs_recv[i] );
            pthread_mutex_unlock( &mpi_mutex );
            
            }
        else
            reqs_recv[i] = MPI_REQUEST_NULL;
    
        }
        
    /* Wait for the counts to come in. */
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    /* printf( "engine_exchange[%i]: successfully exchanged counts.\n" , e->nodeID ); */
        
    /* Send and receive data. */
    #pragma omp parallel for schedule(static), private(i,finger,k,c,res)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Do we have anything to send? */
        if ( e->send[i].count > 0 ) {
            
            /* Allocate a buffer for the send queue. */
            /* if ( ( buff_send[i] = (struct part *)malloc( sizeof(struct part) * totals_send[i] ) ) == NULL )
                return error(engine_err_malloc); */
            buff_send[i] = (struct part *)malloc( sizeof(struct part) * totals_send[i] );

            /* Fill the send buffer. */
            finger = buff_send[i];
            for ( k = 0 ; k < e->send[i].count ; k++ ) {
                c = &( s->cells[e->send[i].cellid[k]] );
                memcpy( finger , c->parts , sizeof(struct part) * c->count );
                finger = &( finger[ c->count ] );
                }

            /* File a send. */
            /* if ( ( res = MPI_Isend( buff_send[i] , totals_send[i]*sizeof(struct part) , MPI_BYTE , i , e->nodeID , comm , &reqs_send[i] ) ) != MPI_SUCCESS )
                return error(engine_err_mpi); */
            /* printf( "engine_exchange[%i]: sending %i parts to node %i.\n" , e->nodeID , totals_send[i] , i ); */
            pthread_mutex_lock( &mpi_mutex );
            res = MPI_Isend( buff_send[i] , totals_send[i]*sizeof(struct part) , MPI_BYTE , i , e->nodeID , comm , &reqs_send[i] );
            pthread_mutex_unlock( &mpi_mutex );
            
            }
            
        /* Are we expecting any parts? */
        if ( e->recv[i].count > 0 ) {
    
            /* Count the nr of parts to recv. */
            totals_recv[i] = 0;
            for ( k = 0 ; k < e->recv[i].count ; k++ )
                totals_recv[i] += counts_in[i][k];

            /* Allocate a buffer for the send and recv queues. */
            /* if ( ( buff_recv[i] = (struct part *)malloc( sizeof(struct part) * totals_recv[i] ) ) == NULL )
                return error(engine_err_malloc); */
            buff_recv[i] = (struct part *)malloc( sizeof(struct part) * totals_recv[i] );

            /* File a recv. */
            /* if ( ( res = MPI_Irecv( buff_recv[i] , totals_recv[i]*sizeof(struct part) , MPI_BYTE , i , i , comm , &reqs_recv[i] ) ) != MPI_SUCCESS )
                return error(engine_err_mpi); */
            /* printf( "engine_exchange[%i]: recving %i parts from node %i.\n" , e->nodeID , totals_recv[i] , i ); */
            pthread_mutex_lock( &mpi_mutex );
            res = MPI_Irecv( buff_recv[i] , totals_recv[i]*sizeof(struct part) , MPI_BYTE , i , i , comm , &reqs_recv[i] );
            pthread_mutex_unlock( &mpi_mutex );
            
            }
            
        }

    /* Wait for all the recvs to come in. */
    /* if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi); */
        
    /* Unpack the received data. */
    #pragma omp parallel for schedule(static), private(i,ind,res,finger,k,c)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Wait for this recv to come in. */
        /* if ( ( res = MPI_Wait( &reqs_recv[i] , MPI_STATUS_IGNORE ) ) != MPI_SUCCESS )
            return error(engine_err_mpi); */
        pthread_mutex_lock( &mpi_mutex );
        res = MPI_Waitany( e->nr_nodes , reqs_recv , &ind , MPI_STATUS_IGNORE );
        pthread_mutex_unlock( &mpi_mutex );
        
        /* Did we get a propper index? */
        if ( ind != MPI_UNDEFINED ) {

            /* Loop over the data and pass it to the cells. */
            finger = buff_recv[ind];
            for ( k = 0 ; k < e->recv[ind].count ; k++ ) {
                c = &( s->cells[e->recv[ind].cellid[k]] );
                /* if ( cell_flush( c , s->partlist , s->celllist ) < 0 )
                    return error(engine_err_cell); */
                cell_flush( c , s->partlist , s->celllist );
                /* if ( cell_load( c , finger , counts[i][k] , s->partlist , s->celllist ) < 0 )
                    return error(engine_err_cell); */
                cell_load( c , finger , counts_in[ind][k] , s->partlist , s->celllist );
                finger = &( finger[ counts_in[ind][k] ] );
                }
                
            }
                
        }
        
    /* Wait for all the sends to come in. */
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
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
        
    /* Shuffle the particles to the correct cells. */
    #pragma omp parallel for schedule(static), private(cid,c,pid,p,k,delta,c_dest)
    for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
        c = &(s->cells[cid]);
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
    for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
        c = &(s->cells[cid]);
        if ( c->flags & cell_flag_marked )
            cell_welcome( c , s->partlist );
        }
        
    /* Call it a day. */
    return engine_err_ok;
        
    }
#endif


/**
 * @brief Set-up the engine for distributed-memory parallel operation.
 *
 * @param e The #engine to set-up.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * This function assumes that #engine_split_bisect or some similar
 * function has already been called and that #nodeID, #nr_nodes as
 * well as the #cell @c nodeIDs have been set.
 */
int engine_split ( struct engine *e ) {

    int i, k, cid, cjd;
    struct cell *ci, *cj, *ct;

    /* Check for nonsense inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Start by allocating and initializing the send/recv lists. */
    if ( ( e->send = (struct engine_comm *)malloc( sizeof(struct engine_comm) * e->nr_nodes ) ) == NULL ||
         ( e->recv = (struct engine_comm *)malloc( sizeof(struct engine_comm) * e->nr_nodes ) ) == NULL )
        return error(engine_err_malloc);
    for ( k = 0 ; k < e->nr_nodes ; k++ ) {
        if ( ( e->send[k].cellid = (int *)malloc( sizeof(int) * 100 ) ) == NULL )
            return error(engine_err_malloc);
        e->send[k].size = 100;
        e->send[k].count = 0;
        if ( ( e->recv[k].cellid = (int *)malloc( sizeof(int) * 100 ) ) == NULL )
            return error(engine_err_malloc);
        e->recv[k].size = 100;
        e->recv[k].count = 0;
        }
        
    /* Loop over each cell pair... */
    for ( i = 0 ; i < e->s.nr_pairs ; i++ ) {
    
        /* Get the cells in this pair. */
        cid = e->s.pairs[i].i;
        cjd = e->s.pairs[i].j;
        ci = &( e->s.cells[ cid ] );
        cj = &( e->s.cells[ cjd ] );
        
        /* If it is a ghost-ghost pair, skip it. */
        if ( (ci->flags & cell_flag_ghost) && (cj->flags & cell_flag_ghost) )
            continue;
            
        /* Mark the cells. */
        ci->flags |= cell_flag_marked;
        cj->flags |= cell_flag_marked;
            
        /* Make cj the ghost cell and bail if both are real. */
        if ( ci->flags & cell_flag_ghost ) {
            ct = ci; ci = cj; cj = ct;
            k = cid; cid = cjd; cjd = k;
            }
        else if ( !( cj->flags & cell_flag_ghost ) )
            continue;
        
        /* Store the communication between cid and cjd. */
        /* Store the send, if not already there... */
        for ( k = 0 ; k < e->send[cj->nodeID].count && e->send[cj->nodeID].cellid[k] != cid ; k++ );
        if ( k == e->send[cj->nodeID].count ) {
            if ( e->send[cj->nodeID].count == e->send[cj->nodeID].size ) {
                e->send[cj->nodeID].size += 100;
                if ( ( e->send[cj->nodeID].cellid = (int *)realloc( e->send[cj->nodeID].cellid , sizeof(int) * e->send[cj->nodeID].size ) ) == NULL )
                    return error(engine_err_malloc);
                }
            e->send[cj->nodeID].cellid[ e->send[cj->nodeID].count++ ] = cid;
            }
        /* Store the recv, if not already there... */
        for ( k = 0 ; k < e->recv[cj->nodeID].count && e->recv[cj->nodeID].cellid[k] != cjd ; k++ );
        if ( k == e->recv[cj->nodeID].count ) {
            if ( e->recv[cj->nodeID].count == e->recv[cj->nodeID].size ) {
                e->recv[cj->nodeID].size += 100;
                if ( ( e->recv[cj->nodeID].cellid = (int *)realloc( e->recv[cj->nodeID].cellid , sizeof(int) * e->recv[cj->nodeID].size ) ) == NULL )
                    return error(engine_err_malloc);
                }
            e->recv[cj->nodeID].cellid[ e->recv[cj->nodeID].count++ ] = cjd;
            }
            
        }
        
    /* Nuke all ghost-ghost pairs. */
    i = 0;
    while ( i < e->s.nr_pairs ) {
    
        /* Get the cells in this pair. */
        ci = &( e->s.cells[ e->s.pairs[i].i ] );
        cj = &( e->s.cells[ e->s.pairs[i].j ] );
        
        /* If it is a ghost-ghost pair, skip it. */
        if ( (ci->flags & cell_flag_ghost) && (cj->flags & cell_flag_ghost) )
            e->s.pairs[i] = e->s.pairs[ --(e->s.nr_pairs) ];
        else
            i += 1;
            
        }
        
    /* Empty unmarked cells. */
    for ( k = 0 ; k < e->s.nr_cells ; k++ )
        if ( !( e->s.cells[k].flags & cell_flag_marked ) )
            e->s.cells[k].count = 0;
        
    /* Re-build the tuples if needed. */
    if ( space_maketuples( &e->s ) < 0 )
        return error(engine_err_space);
        
    /* Done deal. */
    return engine_err_ok;

    }
        
    
/**
 * @brief Split the computational domain over a number of nodes using
 *      bisection.
 *
 * @param e The #engine to split up.
 * @param N The number of computational nodes.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_split_bisect ( struct engine *e , int N ) {

    /* Interior, recursive function that actually does the split. */
    int engine_split_bisect_rec( int N_min , int N_max , int x_min , int x_max , int y_min , int y_max , int z_min , int z_max ) {
    
        int i, j, k, m, Nm;
        int hx, hy, hz;
        unsigned int flag = 0;
        struct cell *c;
    
        /* Check inputs. */
        if ( x_max < x_min || y_max < y_min || z_max < z_min )
            return error(engine_err_domain);
            
        /* Is there nothing left to split? */
        if ( N_min == N_max ) {
        
            /* Flag as ghost or not? */
            if ( N_min != e->nodeID )
                flag = cell_flag_ghost;
                
            /* printf("engine_split_bisect: marking range [ %i..%i , %i..%i , %i..%i ] with flag %i.\n",
                x_min, x_max, y_min, y_max, z_min, z_max, flag ); */
        
            /* Run through the cells. */
            for ( i = x_min ; i < x_max ; i++ )
                for ( j = y_min ; j < y_max ; j++ )
                    for ( k = z_min ; k < z_max ; k++ ) {
                        c = &( e->s.cells[ space_cellid(&(e->s),i,j,k) ] );
                        c->flags |= flag;
                        c->nodeID = N_min;
                        }
                        
            }
            
        /* Otherwise, bisect. */
        else {
        
            hx = x_max - x_min;
            hy = y_max - y_min;
            hz = z_max - z_min;
            Nm = (N_min + N_max) / 2;
        
            /* Is the x-axis the largest? */
            if ( hx > hy && hx > hz ) {
                m = (x_min + x_max) / 2;
                if ( engine_split_bisect_rec( N_min , Nm , x_min , m , y_min , y_max , z_min , z_max ) < 0 ||
                     engine_split_bisect_rec( Nm+1 , N_max , m , x_max , y_min , y_max , z_min , z_max ) < 0 )
                    return error(engine_err);
                }
        
            /* Nope, maybe the y-axis? */
            else if ( hy > hz ) {
                m = (y_min + y_max) / 2;
                if ( engine_split_bisect_rec( N_min , Nm , x_min , x_max , y_min , m , z_min , z_max ) < 0 ||
                     engine_split_bisect_rec( Nm+1 , N_max , x_min , x_max , m , y_max , z_min , z_max ) < 0 )
                    return error(engine_err);
                }
        
            /* Then it has to be the z-axis. */
            else {
                m = (z_min + z_max) / 2;
                if ( engine_split_bisect_rec( N_min , Nm , x_min , x_max , y_min , y_max , z_min , m ) < 0 ||
                     engine_split_bisect_rec( Nm+1 , N_max , x_min , x_max , y_min , y_max , m , z_max ) < 0 )
                    return error(engine_err);
                }
        
            }
            
        /* So far, so good! */
        return engine_err_ok;
    
        }

    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Call the recursive bisection. */
    if ( engine_split_bisect_rec( 0 , N-1 , 0 , e->s.cdim[0] , 0 , e->s.cdim[1] , 0 , e->s.cdim[2] ) < 0 )
        return error(engine_err);
        
    /* Store the number of nodes. */
    e->nr_nodes = N;
        
    /* Call it a day. */
    return engine_err_ok;
    
    }
    
    
/**
 * @brief Clear all particles from this #engine.
 *
 * @param e The #engine to flush.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_flush ( struct engine *e ) {

    /* check input. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Clear the space. */
    if ( space_flush( &e->s ) < 0 )
        return error(engine_err_space);
        
    /* done for now. */
    return engine_err_ok;

    }


/**
 * @brief Clear all particles from this #engine's ghost cells.
 *
 * @param e The #engine to flush.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_flush_ghosts ( struct engine *e ) {

    /* check input. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Clear the space. */
    if ( space_flush_ghosts( &e->s ) < 0 )
        return error(engine_err_space);
        
    /* done for now. */
    return engine_err_ok;

    }


/** 
 * @brief Set the explicit electrostatic potential.
 *
 * @param e The #engine.
 * @param ep The electrostatic #potential.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * If @c ep is not @c NULL, the flag #engine_flag_explepot is set,
 * otherwise it is cleared.
 */
 
int engine_setexplepot ( struct engine *e , struct potential *ep ) {

    /* check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* was a potential supplied? */
    if ( ep != NULL ) {
    
        /* set the flag. */
        e->flags |= engine_flag_explepot;
        
        /* set the potential. */
        e->ep = ep;
        
        }
        
    /* otherwise, just clear the flag. */
    else
        e->flags &= ~engine_flag_explepot;
        
    /* done for now. */
    return engine_err_ok;

    }
    
    
/**
 * @brief Unload a set of particle data from the #engine.
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param vid A vector of length @c N of the particle vidtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param epot A pointer to a #double in which to store the total potential energy.
 * @param N the maximum number of particles.
 *
 * @return The number of particles unloaded or < 0 on
 *      error (see #engine_err).
 *
 * The fields @c x, @c v, @c type, @c vid, @c q, @c epot and/or @c flags may be NULL.
 */
 
int engine_unload ( struct engine *e , double *x , double *v , int *type , int *vid , double *q , unsigned int *flags , double *epot , int N ) {

    struct part *p;
    struct cell *c;
    int j, k, cid, pid, count = 0;
    double epot_acc = 0.0;
    
    /* check the inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Loop over each cell. */
    #pragma omp parallel for schedule(static,100), private(cid,c,k,p,pid,j), reduction(+:epot_acc,count)
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ ) {
    
        /* Get a hold of the cell. */
        c = &( e->s.cells[cid] );
    
        /* Collect the potential energy if requested. */
        epot_acc += c->epot;
            
        /* Loop over the parts in this cell. */
        __builtin_prefetch( &( c->parts[0] ) );
        __builtin_prefetch( &( c->parts[1] ) );
        __builtin_prefetch( &( c->parts[2] ) );
        __builtin_prefetch( &( c->parts[3] ) );
        for ( k = 0 ; k < c->count ; k++ ) {
        
            /* pre-fetch the next particle. */
            __builtin_prefetch( &( c->parts[k+4] ) );
        
            /* Get a hold of the particle. */
            p = &( c->parts[k] );
            if ( ( pid = p->id ) >= N )
                continue;
            count += 1;
        
            /* get this particle's data, where requested. */
            if ( x != NULL )
                for ( j = 0 ; j < 3 ; j++ )
                    x[pid*3+j] = c->origin[j] + p->x[j];
            if ( v != NULL)
                for ( j = 0 ; j < 3 ; j++ )
                    v[pid*3+j] = p->v[j];
            if ( type != NULL )
                type[pid] = p->type;
            if ( vid != NULL )
                vid[pid] = p->vid;
            if ( q != NULL )
                q[pid] = p->q;
            if ( flags != NULL )
                flags[pid] = p->flags;
                
            }
            
        }
        
    /* Write back the potential energy, if requested. */
    *epot += epot_acc;

    /* to the pub! */
    return count;

    }


/**
 * @brief Unload a set of particle data from the marked cells of an #engine
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param vid A vector of length @c N of the particle vidtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param epot A pointer to a #double in which to store the total potential energy.
 * @param N the maximum number of particles.
 *
 * @return The number of particles unloaded or < 0 on
 *      error (see #engine_err).
 *
 * The fields @c x, @c v, @c type, @c vid, @c q, @c epot and/or @c flags may be NULL.
 */
 
int engine_unload_marked ( struct engine *e , double *x , double *v , int *type , int *vid , double *q , unsigned int *flags , double *epot , int N ) {

    struct part *p;
    struct cell *c;
    int j, k, cid, count = 0;
    double epot_acc = 0.0;
    
    /* check the inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Loop over each cell. */
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ ) {
    
        /* Get a hold of the cell. */
        c = &( e->s.cells[cid] );
        
        /* Skip it? */
        if ( !(c->flags & cell_flag_marked) )
            continue;
    
        /* Collect the potential energy if requested. */
        epot_acc += c->epot;
            
        /* Loop over the parts in this cell. */
        __builtin_prefetch( &( c->parts[0] ) );
        __builtin_prefetch( &( c->parts[1] ) );
        __builtin_prefetch( &( c->parts[2] ) );
        __builtin_prefetch( &( c->parts[3] ) );
        for ( k = 0 ; k < c->count ; k++ ) {
        
            /* pre-fetch the next particle. */
            __builtin_prefetch( &( c->parts[k+4] ) );
        
            /* Get a hold of the particle. */
            p = &( c->parts[k] );
        
            /* get this particle's data, where requested. */
            if ( x != NULL )
                for ( j = 0 ; j < 3 ; j++ )
                    x[count*3+j] = c->origin[j] + p->x[j];
            if ( v != NULL)
                for ( j = 0 ; j < 3 ; j++ )
                    v[count*3+j] = p->v[j];
            if ( type != NULL )
                type[count] = p->type;
            if ( vid != NULL )
                vid[count] = p->vid;
            if ( q != NULL )
                q[count] = p->q;
            if ( flags != NULL )
                flags[count] = p->flags;
                
            /* increase the counter. */
            count += 1;
            
            }
            
        }
        
    /* Write back the potential energy, if requested. */
    if ( epot != NULL )
        *epot += epot_acc;

    /* to the pub! */
    return count;

    }


/**
 * @brief Unload real particles that may have wandered into a ghost cell.
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param vid A vector of length @c N of the particle vidtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param epot A pointer to a #double in which to store the total potential energy.
 * @param N the maximum number of particles.
 *
 * @return The number of particles unloaded or < 0 on
 *      error (see #engine_err).
 *
 * The fields @c x, @c v, @c type, @c vid, @c q, @c epot and/or @c flags may be NULL.
 */
 
int engine_unload_strays ( struct engine *e , double *x , double *v , int *type , int *vid , double *q , unsigned int *flags , double *epot , int N ) {

    struct part *p;
    struct cell *c;
    int j, k, cid, count = 0;
    double epot_acc = 0.0;
    
    /* check the inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Loop over each cell. */
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ ) {
    
        /* Get a hold of the cell. */
        c = &( e->s.cells[cid] );
        
        /* Skip it? */
        if ( !(c->flags & cell_flag_ghost) )
            continue;
    
        /* Collect the potential energy if requested. */
        epot_acc += c->epot;
            
        /* Loop over the parts in this cell. */
        for ( k = c->count-1 ; k >= 0 && !(c->parts[k].flags & part_flag_ghost) ; k-- ) {
        
            /* Get a hold of the particle. */
            p = &( c->parts[k] );
            if ( p->flags & part_flag_ghost )
                continue;
        
            /* get this particle's data, where requested. */
            if ( x != NULL )
                for ( j = 0 ; j < 3 ; j++ )
                    x[count*3+j] = c->origin[j] + p->x[j];
            if ( v != NULL)
                for ( j = 0 ; j < 3 ; j++ )
                    v[count*3+j] = p->v[j];
            if ( type != NULL )
                type[count] = p->type;
            if ( vid != NULL )
                vid[count] = p->vid;
            if ( q != NULL )
                q[count] = p->q;
            if ( flags != NULL )
                flags[count] = p->flags;
                
            /* increase the counter. */
            count += 1;
            
            }
            
        }
        
    /* Write back the potential energy, if requested. */
    if ( epot != NULL )
        *epot += epot_acc;

    /* to the pub! */
    return count;

    }


/**
 * @brief Load a set of particle data.
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param type A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param N the number of particles to load.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * If the parameters @c v, @c flags, @c vid or @c q are @c NULL, then
 * these values are set to zero.
 */
 
int engine_load ( struct engine *e , double *x , double *v , int *type , int *vid , double *q , unsigned int *flags , int N ) {

    struct part p;
    struct space *s;
    int k, pid;
    
    /* check the inputs. */
    if ( e == NULL || x == NULL || type == NULL )
        return error(engine_err_null);
        
    /* Get a handle on the space. */
    s = &(e->s);
        
    /* init the velocity and charge in case not specified. */
    p.v[0] = 0.0; p.v[1] = 0.0; p.v[2] = 0.0;
    p.f[0] = 0.0; p.f[1] = 0.0; p.f[2] = 0.0;
    p.q = 0.0;
    p.flags = part_flag_none;
        
    /* loop over the entries. */
    for ( pid = 0 ; pid < N ; pid++ ) {
    
        /* set the particle data. */
        p.id = pid;
        p.type = type[pid];
        if ( vid != NULL )
            p.vid = vid[pid];
        if ( flags != NULL )
            p.flags = flags[pid];
        if ( v != NULL )
            for ( k = 0 ; k < 3 ; k++ )
                p.v[k] = v[pid*3+k];
        if ( q != 0 )
            p.q = q[pid];
            
        /* add the part to the space. */
        if ( space_addpart( s , &p , &x[3*pid] ) < 0 )
            return error(engine_err_space);
    
        }
        
    /* to the pub! */
    return engine_err_ok;

    }


/**
 * @brief Load a set of particle data as ghosts
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param type A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param N the number of particles to load.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * If the parameters @c v, @c flags, @c vid or @c q are @c NULL, then
 * these values are set to zero.
 */
 
int engine_load_ghosts ( struct engine *e , double *x , double *v , int *type , int *vid , double *q , unsigned int *flags , int N ) {

    struct part p;
    struct space *s;
    int k, pid, nr_parts;
    
    /* check the inputs. */
    if ( e == NULL || x == NULL || type == NULL )
        return error(engine_err_null);
        
    /* Get a handle on the space. */
    s = &(e->s);
    nr_parts = s->nr_parts;
        
    /* init the velocity and charge in case not specified. */
    p.v[0] = 0.0; p.v[1] = 0.0; p.v[2] = 0.0;
    p.f[0] = 0.0; p.f[1] = 0.0; p.f[2] = 0.0;
    p.q = 0.0;
    p.flags = part_flag_ghost;
        
    /* loop over the entries. */
    for ( pid = 0 ; pid < N ; pid++ ) {
    
        /* set the particle data. */
        p.id = pid + nr_parts;
        p.type = type[pid];
        if ( vid != NULL )
            p.vid = vid[pid];
        if ( flags != NULL )
            p.flags = flags[pid] | part_flag_ghost;
        if ( v != NULL )
            for ( k = 0 ; k < 3 ; k++ )
                p.v[k] = v[pid*3+k];
        if ( q != 0 )
            p.q = q[pid];
            
        /* add the part to the space. */
        if ( space_addpart( s , &p , &x[3*pid] ) < 0 )
            return error(engine_err_space);
    
        }
        
    /* to the pub! */
    return engine_err_ok;

    }


/**
 * @brief Add a type definition.
 *
 * @param e The #engine.
 * @param id The particle type ID.
 * @param mass The particle type mass.
 * @param charge The particle type charge.
 * @param name Particle name, can be @c NULL.
 * @param name2 Particle second name, can be @c NULL.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * The particle type ID must be an integer greater or equal to 0
 * and less than the value @c max_type specified in #engine_init.
 */
 
int engine_addtype ( struct engine *e , int id , double mass , double charge , char *name , char *name2 ) {

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( id < 0 || id >= e->max_type )
        return error(engine_err_range);
    
    /* set the type. */
    e->types[id].mass = mass;
    e->types[id].imass = 1.0 / mass;
    e->types[id].charge = charge;
    if ( name != NULL )
        strcpy( e->types[id].name , name );
    else
        strcpy( e->types[id].name , "X" );
    if ( name2 != NULL )
        strcpy( e->types[id].name2 , name2 );
    else
        strcpy( e->types[id].name2 , "X" );
    
    /* bring good tidings. */
    return engine_err_ok;

    }


/**
 * @brief Add an interaction potential.
 *
 * @param e The #engine.
 * @param p The #potential to add to the #engine.
 * @param i ID of particle type for this interaction.
 * @param j ID of second particle type for this interaction.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Adds the given potential for pairs of particles of type @c i and @c j,
 * where @c i and @c j may be the same type ID.
 */
 
int engine_addpot ( struct engine *e , struct potential *p , int i , int j ) {

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( i < 0 || i >= e->max_type || j < 0 || j >= e->max_type )
        return error(engine_err_range);
        
    /* store the potential. */
    e->p[ i * e->max_type + j ] = p;
    if ( i != j )
        e->p[ j * e->max_type + i ] = p;
        
    /* end on a good note. */
    return engine_err_ok;

    }


/**
 * @brief Start the runners in the given #engine.
 *
 * @param e The #engine to start.
 * @param nr_runners The number of runners start.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Allocates and starts the specified number of #runner.
 */

int engine_start ( struct engine *e , int nr_runners ) {

    int i;
    struct runner *temp;

    /* (re)allocate the runners */
    if ( e->nr_runners == 0 ) {
        if ( ( e->runners = (struct runner *)malloc( sizeof(struct runner) * nr_runners )) == NULL )
            return error(engine_err_malloc);
        }
    else {
        if ( ( temp = (struct runner *)malloc( sizeof(struct runner) * (e->nr_runners + nr_runners) )) == NULL )
            return error(engine_err_malloc);
        memcpy( temp , e->runners , sizeof(struct runner) * e->nr_runners );
        e->runners = temp;
        }
        
    /* initialize the runners. */
    for ( i = 0 ; i < nr_runners ; i++ )
        if ( runner_init(&e->runners[e->nr_runners + i],e,e->nr_runners + i) < 0 )
            return error(engine_err_runner);
    e->nr_runners += nr_runners;
            
    /* wait for the runners to be in place */
    while (e->barrier_count != e->nr_runners)
        if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
            return error(engine_err_pthread);
        
    /* all is well... */
    return engine_err_ok;
    
    }


/**
 * @brief Start the SPU-associated runners in the given #engine.
 *
 * @param e The #engine to start.
 * @param nr_runners The number of runners start.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Allocates and starts the specified number of #runner.
 */

int engine_start_SPU ( struct engine *e , int nr_runners ) {

    int i;
    struct runner *temp;

    /* (re)allocate the runners */
    if ( e->nr_runners == 0 ) {
        if ( ( e->runners = (struct runner *)malloc( sizeof(struct runner) * nr_runners )) == NULL )
            return error(engine_err_malloc);
        }
    else {
        if ( ( temp = (struct runner *)malloc( sizeof(struct runner) * (e->nr_runners + nr_runners) )) == NULL )
            return error(engine_err_malloc);
        memcpy( temp , e->runners , sizeof(struct runner) * e->nr_runners );
        free( e->runners );
        e->runners = temp;
        }
        
    /* initialize the runners. */
    for ( i = 0 ; i < nr_runners ; i++ )
        if ( runner_init_SPU(&e->runners[e->nr_runners + i],e,e->nr_runners + i) < 0 )
            return error(engine_err_runner);
    e->nr_runners += nr_runners;
            
    /* wait for the runners to be in place */
    while (e->barrier_count != e->nr_runners)
        if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
            return error(engine_err_pthread);
        
    /* all is well... */
    return engine_err_ok;
    
    }


/**
 * @brief Run the engine for a single time step.
 *
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * This routine advances the timestep counter by one, prepares the #space
 * for a timestep, releases the #runner's associated with the #engine
 * and waits for them to finnish.
 *
 * Once all the #runner's are done, the particle velocities and positions
 * are updated and the particles are re-sorted in the #space.
 */
/* TODO: Should the velocities and positions really be updated here? */

int engine_step ( struct engine *e ) {

    int cid, pid, k, delta[3];
    struct cell *c, *c_dest;
    struct part *p;
    struct space *s;
    FPTYPE dt, w, h[3];
    double epot = 0.0;
    
    /* Get a grip on the space. */
    s = &(e->s);
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];

    /* increase the time stepper */
    e->time += 1;
    /* printf("engine_step: running time step %i...\n",e->time); */
    
    /* prepare the space */
    if ( space_prepare( s ) != space_err_ok )
        return error(engine_err_space);

    /* Do we need to set up a Verlet list? */
    if ( e->flags & engine_flag_verlet )
        if ( space_verlet_init( s , !(e->flags & engine_flag_verlet_pairwise) ) != space_err_ok )
            return error(engine_err_space);
    
    /* open the door for the runners */
    e->barrier_count *= -1;
    if (pthread_cond_broadcast(&e->barrier_cond) != 0)
        return error(engine_err_pthread);

    /* wait for the runners to come home */
    while (e->barrier_count < e->nr_runners)
        if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
            return error(engine_err_pthread);
            
    /* Do bonds? */
    if ( e->nr_bonds > 0 )
        if ( engine_bond_eval( e ) < 0 )
            return error(engine_err);

    /* Do angles? */
    if ( e->nr_bonds > 0 )
        if ( engine_angle_eval( e ) < 0 )
            return error(engine_err);

    /* update the particle velocities and positions */
    dt = e->dt;
    if ( e->flags & engine_flag_verlet || e->flags & engine_flag_mpi )
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,w), reduction(+:epot)
        for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
            c = &(s->cells[cid]);
            epot += c->epot;
            if ( c->flags & cell_flag_ghost )
                continue;
            for ( pid = 0 ; pid < c->count ; pid++ ) {
                p = &( c->parts[pid] );
                w = dt * e->types[p->type].imass;
                for ( k = 0 ; k < 3 ; k++ ) {
                    p->v[k] += p->f[k] * w;
                    p->x[k] += dt * p->v[k];
                    }
                }
            }
    else {
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,w,k,delta,c_dest), reduction(+:epot)
        for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
            c = &(s->cells[cid]);
            epot += c->epot;
            if ( c->flags & cell_flag_ghost )
                continue;
            pid = 0;
            while ( pid < c->count ) {
            
                p = &( c->parts[pid] );
                w = dt * e->types[p->type].imass;
                for ( k = 0 ; k < 3 ; k++ ) {
                    p->v[k] += p->f[k] * w;
                    p->x[k] += dt * p->v[k];
                    delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
                    }
                    
                /* do we have to move this particle? */
                if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                    for ( k = 0 ; k < 3 ; k++ )
                        p->x[k] -= delta[k] * h[k];
                    c_dest = &( s->cells[ space_cellid( s ,
                        (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] , 
                        (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] , 
                        (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );
                    
	                pthread_mutex_lock(&c_dest->cell_mutex);
                    cell_add_incomming( c_dest , p );
	                pthread_mutex_unlock(&c_dest->cell_mutex);
        
                    s->celllist[ p->id ] = c_dest;
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
        for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
            c = &(s->cells[cid]);
            cell_welcome( c , s->partlist );
            }
        }
            
    /* Store the accumulated potential energy. */
    s->epot += epot;
        
    /* return quietly */
    return engine_err_ok;
    
    }


/**
 * @brief Barrier routine to hold the @c runners back.
 *
 * @param e The #engine to wait on.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * After being initialized, and after every timestep, every #runner
 * calls this routine which blocks until all the runners have returned
 * and the #engine signals the next timestep.
 */

int engine_barrier ( struct engine *e ) {

    /* lock the barrier mutex */
	if (pthread_mutex_lock(&e->barrier_mutex) != 0)
		return error(engine_err_pthread);
	
    /* wait for the barrier to close */
	while (e->barrier_count < 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return error(engine_err_pthread);
	
    /* if i'm the last thread in, signal that the barrier is full */
	if (++e->barrier_count == e->nr_runners) {
		if (pthread_cond_signal(&e->done_cond) != 0)
			return error(engine_err_pthread);
		}

    /* wait for the barrier to re-open */
	while (e->barrier_count > 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return error(engine_err_pthread);
				
    /* if i'm the last thread out, signal to those waiting to get back in */
	if (++e->barrier_count == 0)
		if (pthread_cond_broadcast(&e->barrier_cond) != 0)
			return error(engine_err_pthread);
			
    /* free the barrier mutex */
	if (pthread_mutex_unlock(&e->barrier_mutex) != 0)
		return error(engine_err_pthread);
		
    /* all is well... */
	return engine_err_ok;
	
	}
	
	
/**
 * @brief Initialize an #engine with the given data.
 *
 * @param e The #engine to initialize.
 * @param origin An array of three doubles containing the cartesian origin
 *      of the space.
 * @param dim An array of three doubles containing the size of the space.
 * @param cutoff The maximum interaction cutoff to use.
 * @param period A bitmask describing the periodicity of the domain
 *      (see #space_periodic_full).
 * @param max_type The maximum number of particle types that will be used
 *      by this engine.
 * @param flags Bit-mask containing the flags for this engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_init ( struct engine *e , const double *origin , const double *dim , double cutoff , unsigned int period , int max_type , unsigned int flags ) {

    /* make sure the inputs are ok */
    if ( e == NULL || origin == NULL || dim == NULL )
        return error(engine_err_null);
        
    /* init the space with the given parameters */
    if ( space_init( &(e->s) , origin , dim , cutoff , period ) < 0 )
        return error(engine_err_space);
        
    /* Set some flag implications. */
    if ( flags & engine_flag_verlet_pairwise2 )
        flags |= engine_flag_verlet_pairwise;
    if ( flags & engine_flag_verlet_pairwise )
        flags |= engine_flag_verlet;
    if ( flags & engine_flag_verlet )
        flags |= engine_flag_tuples;
        
    /* Set the flags. */
    e->flags = flags;
    
    /* Init the runners to 0. */
    e->runners = NULL;
    e->nr_runners = 0;
    
    /* Init the bonds array. */
    e->bonds_size = 100;
    if ( ( e->bonds = (struct bond *)malloc( sizeof( struct bond ) * e->bonds_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_bonds = 0;
    
    /* Init the angles array. */
    e->angles_size = 100;
    if ( ( e->angles = (struct angle *)malloc( sizeof( struct angle ) * e->angles_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_angles = 0;
    
    /* set the maximum nr of types */
    e->max_type = max_type;
    if ( ( e->types = (struct part_type *)malloc( sizeof(struct part_type) * max_type ) ) == NULL )
        return error(engine_err_malloc);
    
    /* allocate the interaction matrices */
    if ( (e->p = (struct potential **)malloc( sizeof(struct potential *) * max_type * max_type )) == NULL)
        return error(engine_err_malloc);
    bzero( e->p , sizeof(struct potential *) * max_type * max_type );
    if ( (e->p_bond = (struct potential **)malloc( sizeof(struct potential *) * max_type * max_type )) == NULL)
        return error(engine_err_malloc);
    bzero( e->p_bond , sizeof(struct potential *) * max_type * max_type );
    e->anglepots_size = 100;
    if ( (e->p_angle = (struct potential **)malloc( sizeof(struct potential *) * e->anglepots_size )) == NULL)
        return error(engine_err_malloc);
    bzero( e->p_angle , sizeof(struct potential *) * e->anglepots_size );
    e->nr_anglepots = 0;
        
    /* init the barrier variables */
    e->barrier_count = 0;
	if ( pthread_mutex_init( &e->barrier_mutex , NULL ) != 0 ||
		 pthread_cond_init( &e->barrier_cond , NULL ) != 0 ||
		 pthread_cond_init( &e->done_cond , NULL ) != 0)
		return error(engine_err_pthread);
        
    /* init the barrier */
    if (pthread_mutex_lock(&e->barrier_mutex) != 0)
        return error(engine_err_pthread);
    e->barrier_count = 0;
    
    /* Init the comm arrays. */
    e->send = NULL;
    e->recv = NULL;
        
    /* all is well... */
    return engine_err_ok;
    
    }
