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


/* Local includes. */
#include "fptype.h"


/* engine error codes */
#define engine_err_ok                    0
#define engine_err_null                  -1
#define engine_err_malloc                -2
#define engine_err_space                 -3
#define engine_err_pthread               -4
#define engine_err_runner                -5


/* some constants */
#define engine_flag_none                 0
#define engine_flag_tuples               1
#define engine_flag_static               2
#define engine_flag_localparts           4
#define engine_flag_cell                 8
#define engine_flag_usePPU               16
#define engine_flag_GPU                  32


/** ID of the last error. */
extern int engine_err;


/** The #engine structure. */
struct engine {

    /** Some flags controlling how this engine works. */
    unsigned int flags;

    /** The space on which to work */
    struct space s;
    
    /** Time variables */
    int time;
    double dt;
    
    /** What is the maximum nr of types? */
    int max_type;
    
    /** The particle types. */
    struct part_type *types;
    
    /** The interaction matrix */
    struct potential **p;
    
    /** Mutexes, conditions and counters for the barrier */
	pthread_mutex_t barrier_mutex;
	pthread_cond_t barrier_cond;
	pthread_cond_t done_cond;
    int barrier_count;
    
    /** Nr of runners */
    int nr_runners;
    
    /** The runners */
    struct runner *runners;
    
    /** The data buffers */
    FPTYPE *xbuff, *vbuff, *fbuff;
    
    /** Data for the improved pair search */
    char *M;
    int *nneigh, *cellpairs, *cell_count, *runner_count;
    int *owner, *owns, nr_pairs;
    pthread_mutex_t getpairs_mutex;
    pthread_cond_t getpairs_avail;
    
    };
    

/* associated functions */
int engine_init ( struct engine *e , const double *origin , const double *dim , double cutoff , unsigned int period , int max_type , unsigned int flags );
int engine_start ( struct engine *e , int nr_runners );
int engine_barrier ( struct engine *e );
int engine_step ( struct engine *e );
