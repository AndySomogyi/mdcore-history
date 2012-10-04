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

/* queue error codes */
#define queue_err_ok                    0
#define queue_err_null                  -1
#define queue_err_malloc                -2
#define queue_err_full                  -3
#define queue_err_lock                  -4


/* some constants */
#define queue_flag_pairs                1
#define queue_flag_tuples               2


/** ID of the last error */
extern int queue_err;


/** The queue structure */
struct queue {

    /* Queue flags. */
    unsigned int flags;
    
    /* Allocated size. */
    int size;
    
    /* The queue data. */
    union {
        struct cellpair *pairs;
        struct celltuple *tuples;
        } data;
        
    /* The space in which this queue lives. */
    struct space *space;
        
    /* The queue indices. */
    int *ind;
        
    /* Index of next entry. */
    int next;
    
    /* Index of last entry. */
    int count;
    
    /* Lock for this queue. */
    lock_type lock;
    
    };


/* Associated functions */
int queue_pairs_init ( struct queue *q , int size , struct space *s , struct cellpair *pairs );
int queue_tuples_init ( struct queue *q , int size , struct space *s , struct celltuple *tuples );
void queue_reset ( struct queue *q );
int queue_insert ( struct queue *q , void *thing );
void *queue_get ( struct queue *q , int keep );
