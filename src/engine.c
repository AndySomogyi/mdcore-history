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
