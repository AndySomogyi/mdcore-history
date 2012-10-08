/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include "part.h"
#include "fifo.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "engine.h"
#include "queue.h"


/* Global variables. */
/** The ID of the last error. */
int queue_err = queue_err_ok;
unsigned int queue_rcount = 0;

/* the error macro. */
#define error(id)				( queue_err = errs_register( id , queue_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *queue_err_msg[5] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "Attempted to insert into a full queue.",
    "An error occured in a lock function."
	};
    
    

/**
 * @brief Get a task from the queue.
 * 
 * @param q The #queue.
 * @param rid #runner ID for ownership issues.
 * @param keep If true, remove the returned index from the queue.
 *
 * @return A task (pair or tuple) with no unresolved conflicts
 *      or @c NULL if none could be found.
 */
 
void *queue_get_old ( struct queue *q , int rid , int keep ) {

    int j, k, tid = -1, qflags = q->flags;
    struct cellpair *p;
    struct celltuple *t;
    struct space *s = q->space;
    char *cells_taboo = s->cells_taboo, *cells_owner = s->cells_owner;

    /* Check if the queue is empty first. */
    if ( q->next >= q->count )
        return NULL;

    /* Lock the queue. */
    if ( lock_lock( &q->lock ) != 0 ) {
        error(queue_err_lock);
        return NULL;
        }
        
    /* Loop over the entries. */
    for ( k = q->next ; k < q->count ; k++ ) {
    
        /* Pairs or tuples? */
        if ( qflags & queue_flag_pairs ) {
        
            /* Put a finger on the kth pair. */
            p = &q->data.pairs[ q->ind[k] ];
            
            /* Is this pair ok? */
            if ( __sync_val_compare_and_swap( &cells_taboo[ p->i ] , 0 , 1 ) == 0 ) {
                if ( p->i == p->j || __sync_val_compare_and_swap( &cells_taboo[ p->j ] , 0 , 1 ) == 0 )
                    break;
                else
                    s->cells_taboo[ p->i ] = 0;
                }
        
            }
            
        else {
        
            /* Put a finger on the kth tuple. */
            t = &q->data.tuples[ q->ind[k] ];
            
            /* Is this tuple ok? */
            for ( j = 0 ; j < t->n ; j++ )
                if ( __sync_val_compare_and_swap( &cells_taboo[ t->cellid[ j ] ] , 0 , 1 ) != 0 )
                    break;
            if ( j == t->n )
                break;
            else
                while ( j >= 0 )
                    cells_taboo[ t->cellid[ j-- ] ] = 0;
            
            }
    
        } /* loop over the entries. */
        
    /* Did we get an entry? */
    if ( k < q->count ) {
    
        /* Keep an eye on this index. */
        tid = q->ind[k];
        
        /* Own this pair/tuple. */
        if ( qflags & queue_flag_pairs ) {
            p = &q->data.pairs[ q->ind[k] ];
            cells_owner[ p->i ] = rid;
            cells_owner[ p->j ] = rid;
            }
        else {
            t = &q->data.tuples[ q->ind[k] ];
            for ( j = 0 ; j < t->n ; j++ )
                cells_owner[ t->cellid[ j ] ] = rid;
            }
    
        /* Remove this entry from the queue? */
        if ( keep ) {
        
            /* Shuffle all the indices down. */
            q->count -= 1;
            for ( j = k ; j < q->count ; j++ )
                q->ind[j] = q->ind[j+1];
        
            }
            
        /* Otherwise, just shuffle it to the front. */
        else {
        
            /* Bubble down... */
            while ( k > q->next ) {
                q->ind[k] = q->ind[k-1];
                k -= 1;
                }
                
            /* Write the original index back to the list. */
            q->ind[ k ] = tid;
            
            /* Move the next pointer up a notch. */
            q->next += 1;
        
            }
    
        } /* did we get an entry? */

    /* Unlock the queue. */
    if ( lock_unlock( &q->lock ) != 0 ) {
        error(queue_err_lock);
        return NULL;
        }
        
    /* Return whatever we've got. */
    if ( tid == -1 )
        return NULL;
    else {
        if ( qflags & queue_flag_pairs )
            return &q->data.pairs[ tid ];
        else
            return &q->data.tuples[ tid ];
        }
        
    }


void *queue_get ( struct queue *q , int rid , int keep ) {

    int j, k, tid = -1, ind_best = -1, score, score_best = -1, qflags = q->flags;
    struct cellpair *p;
    struct celltuple *t;
    struct space *s = q->space;
    char *cells_taboo = s->cells_taboo, *cells_owner = s->cells_owner;

    /* Check if the queue is empty first. */
    if ( q->next >= q->count )
        return NULL;

    /* Lock the queue. */
    if ( lock_lock( &q->lock ) != 0 ) {
        error(queue_err_lock);
        return NULL;
        }
        
    /* Loop over the entries. */
    for ( k = q->next ; k < q->count ; k++ ) {
    
        /* Pairs or tuples? */
        if ( qflags & queue_flag_pairs ) {
        
            /* Put a finger on the kth pair. */
            p = &q->data.pairs[ q->ind[k] ];
            
            /* Get this pair's score. */
            score = ( cells_owner[ p->i ] == rid ) + ( cells_owner[ p->j ] == rid );
            
            /* Is this better than what we've seen so far? */
            if ( score <= score_best )
                continue;
            
            /* Is this pair ok? */
            if ( __sync_val_compare_and_swap( &cells_taboo[ p->i ] , 0 , 1 ) == 0 ) {
                if ( p->i == p->j || __sync_val_compare_and_swap( &cells_taboo[ p->j ] , 0 , 1 ) == 0 ) {
                    if ( ind_best >= 0 ) {
                        p = &q->data.pairs[ q->ind[ ind_best ] ];
                        cells_taboo[ p->i ] = 0;
                        cells_taboo[ p->j ] = 0;
                        }
                    score_best = score;
                    ind_best = k;
                    }
                else
                    cells_taboo[ p->i ] = 0;
                }
        
            }
            
        else {
        
            /* Put a finger on the kth tuple. */
            t = &q->data.tuples[ q->ind[k] ];
            
            /* Get this tuple's score. */
            score = 0;
            for ( j = 0 ; j < t->n ; j++ )
                score += ( cells_owner[ t->cellid[j] ] == rid );
            
            /* Is this better than what we've seen so far? */
            if ( score <= score_best )
                continue;
            
            /* Is this tuple ok? */
            for ( j = 0 ; j < t->n ; j++ )
                if ( __sync_val_compare_and_swap( &cells_taboo[ t->cellid[ j ] ] , 0 , 1 ) != 0 )
                    break;
            if ( j == t->n ) {
                if ( ind_best >= 0 ) {
                    t = &q->data.tuples[ q->ind[ ind_best ] ];
                    for ( j = 0 ; j < t->n ; j++ )
                        cells_taboo[ t->cellid[ j ] ] = 0;
                    }
                score_best = score;
                ind_best = k;
                }
            else
                while ( j >= 0 )
                    cells_taboo[ t->cellid[ j-- ] ] = 0;
            
            }
            
        /* If we have the maximum score, break. */
        if ( qflags & queue_flag_pairs ) {
            if ( score_best == 2 )
                break;
            }
        else {
            if ( score_best == space_maxtuples )
                break;
            }
    
        } /* loop over the entries. */
        
    /* Did we get an entry? */
    if ( ind_best >= 0 ) {
    
        /* Keep an eye on this index. */
        tid = q->ind[ ind_best ];
        
        /* Own this pair/tuple. */
        if ( qflags & queue_flag_pairs ) {
            p = &q->data.pairs[ tid ];
            cells_owner[ p->i ] = rid;
            cells_owner[ p->j ] = rid;
            }
        else {
            t = &q->data.tuples[ tid ];
            for ( j = 0 ; j < t->n ; j++ )
                cells_owner[ t->cellid[ j ] ] = rid;
            }
    
        /* Remove this entry from the queue? */
        if ( keep ) {
        
            /* Shuffle all the indices down. */
            q->count -= 1;
            for ( j = ind_best ; j < q->count ; j++ )
                q->ind[j] = q->ind[j+1];
        
            }
            
        /* Otherwise, just shuffle it to the front. */
        else {
        
            /* Bubble down... */
            for ( k = ind_best ; k > q->next ; k-- )
                q->ind[k] = q->ind[k-1];
                
            /* Write the original index back to the list. */
            q->ind[ q->next ] = tid;
            
            /* Move the next pointer up a notch. */
            q->next += 1;
        
            }
    
        } /* did we get an entry? */

    /* Unlock the queue. */
    if ( lock_unlock( &q->lock ) != 0 ) {
        error(queue_err_lock);
        return NULL;
        }
        
    /* Return whatever we've got. */
    if ( tid == -1 )
        return NULL;
    else {
        if ( qflags & queue_flag_pairs )
            return &q->data.pairs[ tid ];
        else
            return &q->data.tuples[ tid ];
        }
        
    }


/**
 * @brief Reset the queue.
 * 
 * @param q The #queue.
 */
 
void queue_reset ( struct queue *q ) {

    /* Set the next index to the start of the queue. */
    q->next = 0;

    }


/**
 * @brief Add an index to the given queue.
 * 
 * @param q The #queue.
 * @param thing The thing to be inserted.
 *
 * Inserts a task into the queue at the location of the next pointer
 * and moves all remaining tasks up by one. Thus, if the queue is executing,
 * the inserted task is considered to already have been taken.
 *
 * @return 1 on success, 0 if the queue is full and <0 on error (see #queue_err).
 */
 
int queue_insert ( struct queue *q , void *thing ) {

    int k;

    /* Should we even try? */
    if ( q->count == q->size )
        return 0;
        
    /* Lock the queue. */
    if ( lock_lock( &q->lock ) != 0 )
        return error(queue_err_lock);
        
    /* Is there space left? */
    if ( q->count == q->size ) {
        if ( lock_unlock( &q->lock ) != 0 )
            return error(queue_err_lock);
        return 0;
        }
        
    /* Add the new index to the end of the queue. */
    for ( k = q->count ; k > q->next ; k-- )
        q->ind[ k ] = q->ind[ k-1 ];
    if ( q->flags & queue_flag_pairs )
        q->ind[ q->next ] = (struct cellpair *)thing - q->data.pairs;
    else
        q->ind[ q->next ] = (struct celltuple *)thing - q->data.tuples;
    q->count += 1;
    q->next += 1;
        
    /* Unlock the queue. */
    if ( lock_unlock( &q->lock ) != 0 )
        return error(queue_err_lock);
        
    /* No news is good news. */
    return 1;

    }


/**
 * @brief Initialize a task queue with pairs.
 *
 * @param q The #queue to initialize.
 * @param size The maximum number of cellpairs in this queue.
 * @param s The space with which this queue is associated.
 * @param pairs An array containing the pairs to which the queue
 *        indices will refer to.
 *
 * @return #queue_err_ok or <0 on error (see #queue_err).
 *
 * Initializes a queue of the maximum given size. The initial queue
 * is empty and can be filled with pair ids.
 *
 * @sa #queue_tuples_init
 */
 
int queue_pairs_init ( struct queue *q , int size , struct space *s , struct cellpair *pairs ) {

    /* Sanity check. */
    if ( q == NULL || s == NULL || pairs == NULL )
        return error(queue_err_null);
        
    /* Allocate the indices. */
    if ( ( q->ind = malloc( sizeof(int) * size ) ) == NULL )
        return error(queue_err_malloc);
        
    /* Init the queue data. */
    q->flags = queue_flag_pairs;
    q->space = s;
    q->size = size;
    q->next = 0;
    q->count = 0;
    q->data.pairs = pairs;
    
    /* Init the lock. */
    if ( lock_init( &q->lock ) != 0 )
        return error(queue_err_lock);

    /* Nothing to see here. */
    return queue_err_ok;

    }
    
    
/**
 * @brief Initialize a task queue with tuples.
 *
 * @param q The #queue to initialize.
 * @param size The maximum number of cellpairs in this queue.
 * @param s The space with which this queue is associated.
 * @param tuples An array containing the tuples to which the queue
 *        indices will refer to.
 *
 * @return #queue_err_ok or <0 on error (see #queue_err).
 *
 * Initializes a queue of the maximum given size. The initial queue
 * is empty and can be filled with tuple ids.
 *
 * @sa #queue_tuples_init
 */
 
int queue_tuples_init ( struct queue *q , int size , struct space *s , struct celltuple *tuples ) {

    /* Sanity check. */
    if ( q == NULL || s == NULL || tuples == NULL )
        return error(queue_err_null);
        
    /* Allocate the indices. */
    if ( ( q->ind = malloc( sizeof(int) * size ) ) == NULL )
        return error(queue_err_malloc);
        
    /* Init the queue data. */
    q->flags = queue_flag_tuples;
    q->space = s;
    q->size = size;
    q->next = 0;
    q->count = 0;
    q->data.tuples = tuples;

    /* Init the lock. */
    if ( lock_init( &q->lock ) != 0 )
        return error(queue_err_lock);

    /* Nothing to see here. */
    return queue_err_ok;

    }


