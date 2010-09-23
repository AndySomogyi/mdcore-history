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


/* engine error codes */
#define engine_err_ok                    0
#define engine_err_null                  -1
#define engine_err_malloc                -2
#define engine_err_space                 -3
#define engine_err_pthread               -4
#define engine_err_runner                -5


/* some constants */


/* the last error */
extern int engine_err;


/* the engine structure */
struct engine {

    /* the space on which to work */
    struct space s;
    
    /* time variables */
    int time;
    double dt;
    
    /* what is the maximum nr of types? */
    int max_type;
    struct part_type *types;
    
    /* the interaction matrix */
    struct potential **p;
    
    /* mutexes, conditions and counters for the barrier */
	pthread_mutex_t barrier_mutex;
	pthread_cond_t barrier_cond;
	pthread_cond_t done_cond;
    int barrier_count;
    
    /* nr of runners */
    int nr_runners;
    
    /* the runners */
    struct runner *runners;
    
    /* the data buffers */
    double *xbuff, *vbuff, *fbuff;
    
    /* data for the improved pair search */
    char *M;
    int *nneigh, *cellpairs, *cell_count, *runner_count;
    int *owner, *owns, nr_pairs;
    pthread_mutex_t getpairs_mutex;
    pthread_cond_t getpairs_avail;
    
    };
    

/* associated functions */
int engine_init ( struct engine *e , const double *origin , const double *dim , double cutoff , unsigned int period , int max_type );
int engine_start ( struct engine *e , int nr_runners );
int engine_barrier ( struct engine *e );
int engine_step ( struct engine *e );
int engine_initpairs ( struct engine *e );
int engine_releasepairs ( struct engine *e , struct runner *r );
int engine_getpairs ( struct engine *e , struct runner *r , int wait );
