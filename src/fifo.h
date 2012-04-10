/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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

/* Local includes. */
#include "fptype.h"

/* fifo error codes */
#define fifo_err_ok                    0
#define fifo_err_null                  -1
#define fifo_err_malloc                -2
#define fifo_err_pthread               -3
#define fifo_err_fifo_full             -4
#define fifo_err_fifo_empty            -5


/* the last error */
extern int fifo_err;


/* The fifo-queue for dispatching. */
struct fifo {

    /* Access mutex and condition signal for blocking use. */
	pthread_mutex_t mutex;
	pthread_cond_t cond;
    
    /* Counters. */
    int first, last, size, count;
    
    /* The FIFO data. */
    int *data;
    
    };

    
/* associated functions */
int fifo_init ( struct fifo *f , int size );
int fifo_pop_nb ( struct fifo *f , int *e );
int fifo_pop ( struct fifo *f , int *e );
int fifo_push_nb ( struct fifo *f , int e );
int fifo_push ( struct fifo *f , int e );
