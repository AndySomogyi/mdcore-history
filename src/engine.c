/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 ******************************************************************************/


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#ifdef CELL
    #include <libspe2.h>
#endif

/* include local headers */
#include "fptype.h"
#include "part.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "runner.h"
#include "engine.h"


/** ID of the last error. */
int engine_err = engine_err_ok;


/**
 * @brief Init the data for the pair finding algorithm.
 *
 * @param The #engine on which to do this.
 *
 * Fills the matrices @c M and @c cellpairs and initializes the @c nneigh,
 * @c cell_count, @c runner_count and @c owner indices.
 *
 * This function must be called at the beginning of every time step if
 * #engine_getpairs is used.
 */

int engine_initpairs ( struct engine *e ) {

    struct space *s = &( e->s );
    int nr_cells = s->nr_cells;
    int i;
    
    /* set the nr of pairs */
    e->nr_pairs = s->nr_pairs;
    
    /* init owner, cell_count and nneigh */
    for ( i = 0 ; i < nr_cells ; i++ ) {
        e->nneigh[i] = 0;
        e->cell_count[i] = 0;
        e->owner[i] = -1;
        }
        
    /* init runner_count */
    for ( i = 0 ; i < e->nr_runners ; i++ )
        e->runner_count[i] = 0;

    /* run throught the cell list and fill M, cellpairs */
    for ( i = 0 ; i < e->nr_pairs ; i++ ) {
        e->M[ s->pairs[i].i * nr_cells + s->pairs[i].j ] = 1;
        e->cellpairs[ s->pairs[i].i * 27 + e->nneigh[s->pairs[i].i]++ ] = s->pairs[i].j;
        if ( s->pairs[i].i != s->pairs[i].j ) {
            e->M[ s->pairs[i].j * nr_cells + s->pairs[i].i ] = 1;
            e->cellpairs[ s->pairs[i].j * 27 + e->nneigh[s->pairs[i].j]++ ] = s->pairs[i].i;
            }
        }
        
    /* nothgin bad can happen here... */
    return engine_err_ok;
        
    }


/*////////////////////////////////////////////////////////////////////////////// */
/* int engine_releasepair */
//
/* release the cells in a given pair */
/*////////////////////////////////////////////////////////////////////////////// */

int engine_releasepairs ( struct engine *e , struct runner *r ) {

    int ci, cj, k;
    int *buff = &( e->owns[ r->id * runner_qlen * 2 ] );

    /* try to grab the mutex */
    if ( pthread_mutex_trylock( &(e->getpairs_mutex) ) == 0 ) {

        /* loop over the freed pairs */
        while ( r->count_free > 0 ) {
        
            /* get the cell indices */
            ci = r->queue[r->free].i;
            cj = r->queue[r->free].j;

            /* release the cells involved */
            if ( --e->cell_count[ ci ] == 0 ) {
                e->owner[ ci ] = -1;
                for ( k = 0 ; buff[k] != ci ; k++ );
                buff[k] = buff[ --e->runner_count[r->id] ];
                }
            if ( --e->cell_count[ cj ] == 0 ) {
                e->owner[ cj ] = -1;
                for ( k = 0 ; buff[k] != cj ; k++ );
                buff[k] = buff[ --e->runner_count[r->id] ];
                }

            /* signal any other runner waiting */
            if ( e->cell_count[ ci ] == 0 ||
                 e->cell_count[ cj ] == 0 )
                if ( pthread_cond_signal( &e->getpairs_avail ) != 0 )
                    return engine_err_pthread;

            /* move the free pointer */
            r->free = ( r->free + 1 ) % runner_qlen;
            r->count_free -= 1;

            }
            
        /* unlock the mutex */
	    if ( pthread_mutex_unlock( &e->getpairs_mutex ) != 0 )
		    return engine_err_pthread;
	
        }

    /* nothing bad can happen here... */
    return engine_err_ok;
        
    }


/*////////////////////////////////////////////////////////////////////////////// */
/* int engine_getpairs */
//
/* feed the queues for the runners */
/*////////////////////////////////////////////////////////////////////////////// */

int engine_getpairs ( struct engine *e , struct runner *r , int wait ) {

    struct space *s = &( e->s );
    int nr_cells = s->nr_cells;
    int w[ nr_cells ], w_max, w_max_last = -1;
    int found = 0, hasfree;
    int ci, cj, i, j, k;
    int *buff = &( e->owns[ r->id * runner_qlen * 2 ] );
    int rid = r->id;

    /* lock the entry mutex */
	if ( pthread_mutex_lock(&e->getpairs_mutex) != 0 )
		return engine_err_pthread;
        
    /* loop over the freed pairs */
    while ( r->count_free > 0 ) {

        /* get the cell ids */
        ci = r->queue[r->free].i;
        cj = r->queue[r->free].j;

        /* release the cells involved */
        if ( --e->cell_count[ ci ] == 0 ) {
            e->owner[ ci ] = -1;
            for ( k = 0 ; buff[k] != ci ; k++ );
            buff[k] = buff[ --e->runner_count[rid] ];
            }
        if ( --e->cell_count[ cj ] == 0 ) {
            e->owner[ cj ] = -1;
            for ( k = 0 ; buff[k] != cj ; k++ );
            buff[k] = buff[ --e->runner_count[rid] ];
            }

        /* move the free pointer */
        r->free = ( r->free + 1 ) % runner_qlen;
        r->count_free -= 1;

        /* signal any other runner waiting */
        if (pthread_cond_signal(&e->getpairs_avail) != 0)
            return engine_err_pthread;

        }

    /* main loop */
    while ( e->nr_pairs > 0 ) {
	
        /* find all pairs within the cells owned by this runner */
        for ( i = 0 ; i < e->runner_count[rid] ; i++ ) {
            ci = buff[i];
            for ( j = i ; j < e->runner_count[rid] ; j++ ) {
                cj = buff[j];
                if ( e->M[ ci * nr_cells + cj ] == 1 ) {

                    /* add pair to queue */
                    r->queue[ r->next ].i = ci;
                    r->queue[ r->next ].j = cj;
                    r->next = ( r->next + 1 ) % runner_qlen;

                    /* remove from M and cellpairs */
                    e->M[ ci * nr_cells + cj ] = 0;
                    e->M[ cj * nr_cells + ci ] = 0;
                    for ( k = 0 ; k < e->nneigh[i] ; k++ )
                        if ( e->cellpairs[ ci * 27 + k ] == cj ) {
                            e->nneigh[ci] -= 1;
                            e->cellpairs[ ci * 27 + k ] = e->cellpairs[ ci * 27 + e->nneigh[ci] ];
                            break;
                            }
                    if ( ci != cj )
                        for ( k = 0 ; k < e->nneigh[cj] ; k++ )
                            if ( e->cellpairs[ cj * 27 + k ] == ci ) {
                                e->nneigh[cj] -= 1;
                                e->cellpairs[ cj * 27 + k ] = e->cellpairs[ cj * 27 + e->nneigh[cj] ];
                                break;
                                }
                                
                    /* add ci and cj to this runner */
                    for ( k = 0 ; k < e->runner_count[rid] && buff[k] != ci ; k++ );
                    if ( k == e->runner_count[rid] )
                        buff[ e->runner_count[rid]++ ] = ci;
                    for ( k = 0 ; k < e->runner_count[rid] && buff[k] != cj ; k++ );
                    if ( k == e->runner_count[rid] )
                        buff[ e->runner_count[rid]++ ] = cj;
                                
                    /* increase the counters on these two cells */
                    e->cell_count[ci] += 1;
                    e->cell_count[cj] += 1;
                    found += 1;
                    /* printf("engine_getpairs: added pair [%i,%i]\n",i,j); */

                    /* lose one pair */
                    e->nr_pairs -= 1;

                    /* signal the runner */
                    if ( pthread_cond_signal( &r->queue_avail ) != 0 )
                        return engine_err_pthread;

                    /* break? */
                    r->count += 1;
                    if ( r->count == runner_qlen )
                        break;

                    }
                }

            /* break? */
            if ( r->count == runner_qlen )
                break;

            }

        /* break? */
        if ( r->count == runner_qlen )
            break;

        /* compute the weight (benefit) of each cell */
        w_max = -1;
        for ( i = 0 ; i < nr_cells ; i++ ) {
            if ( e->owner[i] != -1 || e->nneigh[i] == 0 ) {
                w[i] = -1;
                continue;
                }
            w[i] = 0;
            for ( hasfree = 0, k = 0 ; k < e->nneigh[i] ; k++ ) {
                w[i] += ( e->owner[ e->cellpairs[ i * 27 + k ] ] == rid );
                hasfree += ( e->owner[ e->cellpairs[ i * 27 + k ] ] == -1 );
                }
            if ( ( w_max < 0 || w[i] > w[w_max] ) && ( w[i] > 0 || hasfree > 0 ) )
                w_max = i;
            }

        /* loop while we've got something */
        while ( w_max >= 0 ) {

            /* own the cell with the largest weight */
            e->owner[ w_max ] = rid;

            /* add any pairs resulting from adding this cell */
            for ( k = 0 ; k < e->nneigh[w_max] ; k++ ) {
                if ( e->owner[ e->cellpairs[ w_max * 27 + k ] ] == rid ) {
                
                    /* get the cell ids */
                    ci = w_max;
                    cj = e->cellpairs[ w_max * 27 + k ];

                    /* add pair to queue */
                    r->queue[ r->next ].i = ci;
                    r->queue[ r->next ].j = cj;
                    r->next = ( r->next + 1 ) % runner_qlen;

                    /* remove from M and cellpairs */
                    e->M[ ci * nr_cells + cj ] = 0;
                    e->M[ cj * nr_cells + ci ] = 0;
                    e->nneigh[ci] -= 1;
                    e->cellpairs[ ci * 27 + k ] = e->cellpairs[ ci * 27 + e->nneigh[ci] ];
                    if ( ci != cj )
                        for ( i = 0 ; i < e->nneigh[cj] ; i++ )
                            if ( e->cellpairs[ cj * 27 + i ] == ci ) {
                                e->nneigh[cj] -= 1;
                                e->cellpairs[ cj * 27 + i ] = e->cellpairs[ cj * 27 + e->nneigh[cj] ];
                                break;
                                }
                                
                    /* add ci and cj to this runner */
                    for ( i = 0 ; i < e->runner_count[rid] && buff[i] != ci ; i++ );
                    if ( i == e->runner_count[rid] )
                        buff[ e->runner_count[rid]++ ] = ci;
                    for ( i = 0 ; i < e->runner_count[rid] && buff[i] != cj ; i++ );
                    if ( i == e->runner_count[rid] )
                        buff[ e->runner_count[rid]++ ] = cj;
                                
                    /* increase the counters on the two cells */
                    e->cell_count[ci] += 1;
                    e->cell_count[cj] += 1;
                    found += 1;
                    /* printf("engine_getpairs: added pair [%i,%i]\n",w_max,j); */

                    /* lose one pair */
                    e->nr_pairs -= 1;

                    /* signal the runner */
                    if ( pthread_cond_signal(&r->queue_avail) != 0 )
                        return engine_err_pthread;

                    /* break? */
                    r->count += 1;
                    if ( r->count == runner_qlen )
                        break;

                    }
                }

            /* break? */
            if ( r->count == runner_qlen )
                break;

            /* update the weights and w_max */
            w_max_last = w_max; w_max = -1;
            for ( i = 0 ; i < nr_cells ; i++ )
                if ( e->owner[i] == -1 && e->M[ w_max_last * nr_cells + i ] == 1 ) {
                    w[i] += 1;
                    if ( w_max < 0 || w[i] > w[w_max] )
                        w_max = i;
                    }
            /* printf("engine_getpairs: w_max=%i\n",w_max); fflush(stdout); */

            }
            
        /* wait or break? */
        if ( wait && found == 0 && e->nr_pairs > 0 ) {
            s->nr_stalls += 1;
            if ( pthread_cond_wait( &e->getpairs_avail , &e->getpairs_mutex ) != 0 )
                return engine_err = engine_err_pthread;
            }
        else
            break;
            
        }
        
    /* if there are no pairs left, wake any thread still waiting */
    if ( e->nr_pairs == 0 )
        if ( pthread_cond_broadcast( &e->getpairs_avail ) != 0 )
            return engine_err_pthread;
        
                    
    /* unlock the entry mutex */
	if (pthread_mutex_unlock(&e->getpairs_mutex) != 0)
		return engine_err_pthread;
	
    /* all is well... */
    return found;
    
    }


/*////////////////////////////////////////////////////////////////////////////// */
/* int engine_start */
//
/* init and start the runners */
/*////////////////////////////////////////////////////////////////////////////// */

int engine_start ( struct engine *e , int nr_runners ) {

    int i;

    /* allocate data for the improved pair search */
    if ( ( e->runner_count = (int *)malloc( sizeof(int) * nr_runners ) ) == NULL ||
         ( e->owns = (int *)malloc( sizeof(int) * nr_runners * runner_qlen * 2 ) ) == NULL )
        return engine_err = engine_err_malloc;

    /* allocate and initialize the runners */
    e->nr_runners = nr_runners;
    if ( (e->runners = (struct runner *)malloc( sizeof(struct runner) * nr_runners )) == NULL )
        return engine_err = engine_err_malloc;
    for ( i = 0 ; i < nr_runners ; i++ )
        if ( runner_init(&e->runners[i],e,i) < 0 )
            return engine_err = engine_err_runner;
            
    /* wait for the runners to be in place */
    while (e->barrier_count != e->nr_runners)
        if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
            return engine_err = engine_err_pthread;
        
    /* all is well... */
    return engine_err_ok;
    
    }


/*////////////////////////////////////////////////////////////////////////////// */
/* int engine_step */
//
/* wait at the barrier in the given engine */
/*////////////////////////////////////////////////////////////////////////////// */

int engine_step ( struct engine *e ) {

    int cid, pid, k;
    struct cell *c;
    struct part *p;

    /* increase the time stepper */
    e->time += 1;
    /* printf("engine_step: running time step %i...\n",e->time); */
    
    /* prepare the space */
    if ( space_prepare( &(e->s) ) != space_err_ok )
        return engine_err_space;
        
    /* prepare the pairs */
    if ( engine_initpairs( e ) < 0 )
        return engine_err;
    
    /* open the door for the runners */
    e->barrier_count *= -1;
    if (pthread_cond_broadcast(&e->barrier_cond) != 0)
        return engine_err_pthread;
        
    /* wait for the runners to come home */
    while (e->barrier_count < e->nr_runners)
        if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
            return engine_err_pthread;
    
    /* update the particle velocities and positions */
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ ) {
        c = &(e->s.cells[cid]);
        for ( pid = 0 ; pid < c->count ; pid++ ) {
            p = &(c->parts[pid]);
            for ( k = 0 ; k < 3 ; k++ ) {
                p->v[k] += p->f[k] * e->dt * e->types[p->type].imass;
                p->x[k] += e->dt * p->v[k];
                }
            }
        }
        
    /* re-shuffle the space (every particle in its box) */
    if ( space_shuffle( &(e->s) ) != space_err_ok )
        return engine_err_space;
    
    /* return quietly */
    return engine_err_ok;
    
    }


/*////////////////////////////////////////////////////////////////////////////// */
/* int engine_barrier */
//
/* wait at the barrier in the given engine */
/*////////////////////////////////////////////////////////////////////////////// */

int engine_barrier ( struct engine *e ) {

    /* lock the barrier mutex */
	if (pthread_mutex_lock(&e->barrier_mutex) != 0)
		return engine_err_pthread;
	
    /* wait for the barrier to close */
	while (e->barrier_count < 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return engine_err_pthread;
	
    /* if i'm the last thread in, signal that the barrier is full */
	if (++e->barrier_count == e->nr_runners) {
		if (pthread_cond_signal(&e->done_cond) != 0)
			return engine_err_pthread;
		}

    /* wait for the barrier to re-open */
	while (e->barrier_count > 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return engine_err_pthread;
				
    /* if i'm the last thread out, signal to those waiting to get back in */
	if (++e->barrier_count == 0)
		if (pthread_cond_broadcast(&e->barrier_cond) != 0)
			return engine_err_pthread;
			
    /* free the barrier mutex */
	if (pthread_mutex_unlock(&e->barrier_mutex) != 0)
		return engine_err_pthread;
		
    /* all is well... */
	return engine_err_ok;
	
	}
	
	
/*////////////////////////////////////////////////////////////////////////////// */
/* int engine_init */
//
/* initialize the engine with the given dimensions. */
/*////////////////////////////////////////////////////////////////////////////// */

int engine_init ( struct engine *e , const double *origin , const double *dim , double cutoff , unsigned int period , int max_type ) {

    /* make sure the inputs are ok */
    if ( e == NULL || origin == NULL || dim == NULL )
        return engine_err = engine_err_null;
        
    /* init the space with the given parameters */
    if ( space_init( &(e->s) ,origin , dim , cutoff , period ) < 0 )
        return engine_err_space;
        
    /* init the data for the pair finding algorithm */
    if ( ( e->M = (char *)malloc( sizeof(char) * e->s.nr_cells * e->s.nr_cells ) ) == NULL ||
         ( e->cellpairs = (int *)malloc( sizeof(int) * e->s.nr_cells * 27 ) ) == NULL ||
         ( e->nneigh = (int *)malloc( sizeof(int) * e->s.nr_cells ) ) == NULL ||
         ( e->cell_count = (int *)malloc( sizeof(int) * e->s.nr_cells ) ) == NULL ||
         ( e->owner = (int *)malloc( sizeof(int) * e->s.nr_cells ) ) == NULL )
        return engine_err = engine_err_malloc;
    if ( pthread_mutex_init( &e->getpairs_mutex , NULL ) != 0 ||
         pthread_cond_init( &e->getpairs_avail , NULL ) != 0 )
        return engine_err = engine_err_pthread;
        
    /* set the maximum nr of types */
    e->max_type = max_type;
    if ( ( e->types = (struct part_type *)malloc( sizeof(struct part_type) * max_type ) ) == NULL )
        return engine_err_malloc;
    
    /* allocate the interaction matrix */
    if ( (e->p = (struct potential **)malloc( sizeof(struct potential *) * max_type * max_type )) == NULL)
        return engine_err_malloc;
        
    /* init the barrier variables */
    e->barrier_count = 0;
	if (pthread_mutex_init(&e->barrier_mutex,NULL) != 0 ||
		pthread_cond_init(&e->barrier_cond,NULL) != 0 ||
		pthread_cond_init(&e->done_cond,NULL) != 0)
		return engine_err = engine_err_pthread;
        
    /* init the barrier */
    if (pthread_mutex_lock(&e->barrier_mutex) != 0)
        return engine_err = engine_err_pthread;
    e->barrier_count = 0;
        
    /* all is well... */
    return engine_err_ok;
    
    }
