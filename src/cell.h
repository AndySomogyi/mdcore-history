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


/* cell error codes */
#define cell_err_ok                     0
#define cell_err_null                   -1
#define cell_err_malloc                 -2
#define cell_err_pthread                -3


/* some constants */
#define cell_default_size               100
#define cell_incr                       10

/** Alignment when allocating parts. */
#ifdef CELL
    #define cell_partalign                  128
#else
    #define cell_partalign                  64
#endif

/** Cell flags */
#define cell_flag_none                  0
#define cell_flag_ghost                 1
#define cell_flag_wait                  2
#define cell_flag_waited                4
#define cell_flag_marked                8


/* the last error */
extern int cell_err;


/* the cell structure */
struct cell {

    /* some flags */
    unsigned int flags;

    /* relative cell location */
    int loc[3];
    
    /* absolute cell origin */
    double origin[3];
    
    /* cell dimensions */
    double dim[3];
    
    /* size and count of particle buffer */
    int size, count;
    
    /* the particle buffer */
    struct part *parts;
    
    /* buffer to store the potential energy */
    double epot;
    
    /* a buffer to store incomming parts. */
    struct part *incomming;
    int incomming_size, incomming_count;
    
    /* Mutex for synchronized cell access. */
    pthread_mutex_t cell_mutex;
	pthread_cond_t cell_cond;
    
    };
    

/* associated functions */
int cell_init ( struct cell *c , int *loc , double *origin , double *dim );
struct part *cell_add ( struct cell *c , struct part *p , struct part **partlist );
struct part *cell_add_incomming ( struct cell *c , struct part *p );
int cell_welcome ( struct cell *c , struct part **partlist );
