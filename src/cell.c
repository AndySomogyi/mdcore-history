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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <string.h>
#include <math.h>
#ifdef CELL
    #include <libspe2.h>
    #include <libmisc.h>
    #define mfc_ceil128(v) (((v) + 127) & ~127)
#endif

/* macro to algin memory sizes to a multiple of cell_partalign. */
#define align_ceil(v) (((v) + (cell_partalign-1) ) & ~(cell_partalign-1))

/* include local headers */
#include "errs.h"
#include "part.h"
#include "cell.h"


/* the error macro. */
#define error(id)				( cell_err = errs_register( id , cell_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *cell_err_msg[] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "A call to a pthread routine failed."
	};


/* the last error */
int cell_err = cell_err_ok;


/**
 * @brief Move particles from the incomming buffer to the cell.
 *
 * @param c The #cell.
 * @param partlist A pointer to the partlist to set the part indices.
 *
 * @return #cell_err_ok or < 0 on error (see #cell_err).
 */
 
int cell_welcome ( struct cell *c , struct part **partlist ) {

    int k;

    /* Check inputs. */
    if ( c == NULL )
        return error(cell_err_null);
        
    /* Loop over the incomming parts. */
    for ( k = 0 ; k < c->incomming_count ; k++ )
        if ( cell_add( c , &c->incomming[k] , partlist ) < 0 )
            return error(cell_err);
        
        
    /* Clear the incomming particles list. */
    c->incomming_count = 0;
    
    /* All done! */
    return cell_err_ok;
        
    }


/*////////////////////////////////////////////////////////////////////////////// */
/* int cell_add */
//
/* add the given particle to the given space and return a pointer to its */
/* location. assume the position has already been adjusted. */
/*////////////////////////////////////////////////////////////////////////////// */

struct part *cell_add_incomming ( struct cell *c , struct part *p ) {

    struct part *temp;

    /* check inputs */
    if ( c == NULL || p == NULL ) {
        error(cell_err_null);
        return NULL;
        }
        
    /* is there room for this particle? */
    if ( c->incomming_count == c->incomming_size ) {
        if ( posix_memalign( (void **)&temp , cell_partalign , align_ceil( sizeof(struct part) * ( c->incomming_size + cell_incr ) ) ) != 0 ) {
            error(cell_err_malloc);
            return NULL;
            }
        memcpy( temp , c->incomming , sizeof(struct part) * c->incomming_count );
        free( c->incomming );
        c->incomming = temp;
        c->incomming_size += cell_incr;
        }
        
    /* store this particle */
    c->incomming[c->incomming_count] = *p;
        
    /* all is well */
    return &( c->incomming[ c->incomming_count++ ] );

    }


/*////////////////////////////////////////////////////////////////////////////// */
/* int cell_add */
//
/* add the given particle to the given space and return a pointer to its */
/* location. assume the position has already been adjusted. */
/*////////////////////////////////////////////////////////////////////////////// */

struct part *cell_add ( struct cell *c , struct part *p , struct part **partlist ) {

    struct part *temp;
    int k;

    /* check inputs */
    if ( c == NULL || p == NULL ) {
        error(cell_err_null);
        return NULL;
        }
        
    /* is there room for this particle? */
    if ( c->count == c->size ) {
        if ( posix_memalign( (void **)&temp , cell_partalign , align_ceil( sizeof(struct part) * ( c->size + cell_incr ) ) ) != 0 ) {
            error(cell_err_malloc);
            return NULL;
            }
        memcpy( temp , c->parts , sizeof(struct part) * c->count );
        free( c->parts );
        c->parts = temp;
        c->size += cell_incr;
        if ( partlist != NULL )
            for ( k = 0 ; k < c->count ; k++ )
                partlist[ c->parts[k].id ] = &( c->parts[k] );
        }
        
    /* store this particle */
    c->parts[c->count] = *p;
    if ( partlist != NULL )
        partlist[ p->id ] = &c->parts[ c->count ];
        
    /* all is well */
    return &( c->parts[ c->count++ ] );

    }


/*////////////////////////////////////////////////////////////////////////////// */
/* int cell_init */
//
/* initialize the cell with the given dimensions. */
/*////////////////////////////////////////////////////////////////////////////// */

int cell_init ( struct cell *c , int *loc , double *origin , double *dim ) {

    int i;

    /* check inputs */
    if ( c == NULL || loc == NULL || origin == NULL || dim == NULL )
        return error(cell_err_null);
        
    /* default flags. */
    c->flags = cell_flag_none;
        
    /* Init this cell's mutex. */
    if ( pthread_mutex_init( &c->cell_mutex , NULL ) != 0 )
        return error(cell_err_pthread);
    if ( pthread_cond_init( &c->cell_cond , NULL ) != 0 )
        return error(cell_err_pthread);
        
    /* store values */
    for ( i = 0 ; i < 3 ; i++ ) {
        c->loc[i] = loc[i];
        c->origin[i] = origin[i];
        c->dim[i] = dim[i];
        }
        
    /* allocate the particle pointers */
    if ( posix_memalign( (void **)&(c->parts) , cell_partalign , align_ceil( sizeof(struct part) * cell_default_size ) ) != 0 )
        return error(cell_err_malloc);
    c->size = cell_default_size;
    c->count = 0;
    c->oldx_size = 0;
    c->oldx = NULL;
    
    /* allocate the incomming part buffer. */
    if ( posix_memalign( (void **)&(c->incomming) , cell_partalign , align_ceil( sizeof(struct part) * cell_incr ) ) != 0 )
        return error(cell_err_malloc);
    c->incomming_size = cell_incr;
    c->incomming_count = 0;
        
    /* all is well... */
    return cell_err_ok;

    }
