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
#include "../config.h"
#ifdef HAVE_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fifo.h"


/* Global variables. */
/** The ID of the last error. */
int fifo_err = fifo_err_ok;

/* the error macro. */
#define error(id)				( fifo_err = errs_register( id , fifo_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *fifo_err_msg[6] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "A call to a pthread routine failed.",
    "Tried to push onto a full FIFO-queue." ,
    "Tried to pop from an empty FIFO-queue." 
	};
    
    

/**
 * @brief Add an element to the fifo, non-blocking.
 * 
 * @param f The #fifo
 * @param e The entry to add.
 *
 * @return The new number of entries or < 0 on error (see #fifo_err).
 */
 
int fifo_push_nb ( struct fifo *f , int e ) {

    /* Is there any space left? */
    if ( f->count == f->size )
        return fifo_err_fifo_full;
        
    /* Store the entry in the fifo. */
    f->data[ f->last ] = e;
    
    /* Increase the last pointer. */
    f->last = ( f->last + 1 ) % f->size;
    
    /* Atomically increase the count. */
    __sync_fetch_and_add( &f->count , 1 );
    
    /* Return the new counter. */
    return f->count;

    }
    

/**
 * @brief Remove an element from the fifo, non-blocking.
 * 
 * @param f The #fifo
 * @param e Pointer to the popped element.
 *
 * @return The new number of entries or < 0 on error (see #fifo_err).
 */
 
int fifo_pop_nb ( struct fifo *f , int *e ) {

    /* Are there any elements in the queue? */
    if ( f->count == 0 )
        return fifo_err_fifo_empty;
        
    /* Get the first element in the queue. */
    *e = f->data[ f->first ];
    
    /* Increase the first pointer. */
    f->first = ( f->first + 1 ) % f->size;
    
    /* Atomically decrease the counter. */
    __sync_fetch_and_sub( &f->count , 1 );

    /* Return the new counter. */
    return f->count;

    }
    

/**
 * @brief Add an element to the fifo, blocking.
 * 
 * @param f The #fifo
 * @param e The entry to add.
 *
 * @return The new number of entries or < 0 on error (see #fifo_err).
 */
 
int fifo_push ( struct fifo *f , int e ) {

    /* Get the FIFO mutex. */
    if ( pthread_mutex_lock( &f->mutex ) != 0 )
        return error(fifo_err_pthread);

    /* Wait for space on the fifo. */
    while ( f->count == f->size )
        if ( pthread_cond_wait( &f->cond , &f->mutex ) != 0 )
            return error(fifo_err_pthread);
        
    /* Store the entry in the fifo. */
    f->data[ f->last ] = e;
    
    /* Increase the last pointer. */
    f->last = ( f->last + 1 ) % f->size;
    
    /* Send a signal if the queue was empty. */
    if ( f->count == 0 )
        if ( pthread_cond_broadcast( &f->cond ) != 0 )
            return error(fifo_err_pthread);
    
    /* Increase the count. */
    f->count += 1;
    
    /* Release the FIFO mutex. */
    if ( pthread_mutex_unlock( &f->mutex ) != 0 )
        return error(fifo_err_pthread);

    /* Return the new counter. */
    return f->count;

    }
    

/**
 * @brief Remove an element from the fifo, blocking.
 * 
 * @param f The #fifo
 * @param e Pointer to the popped element.
 *
 * @return The new number of entries or < 0 on error (see #fifo_err).
 */
 
int fifo_pop ( struct fifo *f , int *e ) {

    /* Get the FIFO mutex. */
    if ( pthread_mutex_lock( &f->mutex ) != 0 )
        return error(fifo_err_pthread);

    /* Wait for an entry on the fifo. */
    while ( f->count == 0 )
        if ( pthread_cond_wait( &f->cond , &f->mutex ) != 0 )
            return error(fifo_err_pthread);
        
    /* Get the first element in the queue. */
    *e = f->data[ f->first ];
    
    /* Increase the first pointer. */
    f->first = ( f->first + 1 ) % f->size;
    
    /* Send a signal if the queue was full. */
    if ( f->count == f->size )
        if ( pthread_cond_broadcast( &f->cond ) != 0 )
            return error(fifo_err_pthread);
    
    /* Decrease the count. */
    f->count -= 1;
    
    /* Release the FIFO mutex. */
    if ( pthread_mutex_unlock( &f->mutex ) != 0 )
        return error(fifo_err_pthread);

    /* Return the new counter. */
    return f->count;

    }
    

/**
 * @brief Initialize the given fifo.
 * 
 * @param f The #fifo
 * @param size The number of entries
 *
 * @return #fifo_err_ok or < 0 on error (see #fifo_err).
 */
 
int fifo_init ( struct fifo *f , int size ) {

    /* Init the mutex and condition variable. */
	if ( pthread_mutex_init( &f->mutex , NULL ) != 0 ||
		 pthread_cond_init( &f->cond , NULL ) != 0 )
		return error(fifo_err_pthread);
        
    /* Allocate the data. */
    if ( ( f->data = (int *)malloc( sizeof(int) * size ) ) == NULL )
        return error(fifo_err_malloc);
        
    /* Set the indices to zero. */
    f->first = 0;
    f->last = 0;
    f->count = 0;
    f->size = size;
    
    /* Good times. */
    return fifo_err_ok;

    }
    
    
