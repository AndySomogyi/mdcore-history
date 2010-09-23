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
#include <string.h>
#include <math.h>
#ifdef CELL
    #include <libspe2.h>
    #include <libmisc.h>
    #define mfc_ceil128(v) (((v) + 127) & ~127)
#endif

/* include local headers */
#include "part.h"
#include "cell.h"


/* the last error */
int cell_err = cell_err_ok;


/*////////////////////////////////////////////////////////////////////////////// */
/* int cell_add */
//
/* add the given particle to the given space and return a pointer to its */
/* location. assume the position has already been adjusted. */
/*////////////////////////////////////////////////////////////////////////////// */

struct part *cell_add ( struct cell *c , struct part *p ) {

    #ifdef CELL
        struct part *temp;
    #endif

    /* check inputs */
    if ( c == NULL || p == NULL ) {
        cell_err = cell_err_null;
        return NULL;
        }
        
    /* is there room for this particle? */
    if ( c->count == c->size ) {
        #ifdef CELL
            if ( ( temp = (struct part *)malloc_align( mfc_ceil128( sizeof(struct part) * ( c->size + cell_incr ) ) , 7 ) ) == NULL ) {
                cell_err = cell_err_malloc;
                return NULL;
                }
            memcpy( temp , c->parts , sizeof(struct part) * c->count );
            free_align( c->parts );
            c->parts = temp;
            c->size += cell_incr;
        #else
            c->size += cell_incr;
            if ( ( c->parts = (struct part *)realloc( c->parts , sizeof(struct part) * c->size ) ) == NULL ) {
                cell_err = cell_err_malloc;
                return NULL;
                }
        #endif
        }
        
    /* store this particle */
    c->parts[c->count] = *p;
        
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
        return cell_err = cell_err_null;
        
    /* store values */
    for ( i = 0 ; i < 3 ; i++ ) {
        c->loc[i] = loc[i];
        c->origin[i] = origin[i];
        c->dim[i] = dim[i];
        }
        
    /* allocate the particle pointers */
    #if CELL
    if ( (c->parts = (struct part *)malloc_align( mfc_ceil128( sizeof(struct part) * cell_default_size ) , 7 ) ) == NULL )
    #else
    if ( (c->parts = (struct part *)malloc( sizeof(struct part) * cell_default_size ) ) == NULL )
    #endif
        return cell_err_malloc;
    c->size = cell_default_size;
    c->count = 0;
        
    /* all is well... */
    return cell_err_ok;

    }
