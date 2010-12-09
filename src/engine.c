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
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "runner.h"
#include "engine.h"


/** ID of the last error. */
int engine_err = engine_err_ok;


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *engine_err_msg[7] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "An error occured when calling a space function.",
    "A call to a pthread routine failed.",
    "An error occured when calling a runner function.",
    "One or more values were outside of the allowed range."
	};


/**
 * @brief Add a type definition.
 *
 * @param e The #engine.
 * @param id The particle type ID.
 * @param mass The particle type mass.
 * @param charge The particle type charge.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * The particle type ID must be an integer greater or equal to 0
 * and less than the value @c max_type specified in #engine_init.
 */
 
int engine_addtype ( struct engine *e , int id , double mass , double charge ) {

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( id < 0 || id >= e->max_type )
        return error(engine_err_range);
    
    /* set the type. */
    e->types[id].mass = mass;
    e->types[id].imass = 1.0 / mass;
    e->types[id].charge = charge;
    
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

    int cid, pid, k;
    struct cell *c;
    struct part *p;

    /* increase the time stepper */
    e->time += 1;
    /* printf("engine_step: running time step %i...\n",e->time); */
    
    /* prepare the space */
    if ( space_prepare( &(e->s) ) != space_err_ok )
        return engine_err_space;
        
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
        return engine_err = engine_err_null;
        
    /* init the space with the given parameters */
    if ( space_init( &(e->s) ,origin , dim , cutoff , period ) < 0 )
        return engine_err_space;
        
    /* Set the flags. */
    e->flags = flags;
        
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
