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
#include <math.h>

/* include local headers */
#include "part.h"


/* the last error */
int part_err = part_err_ok;


/*////////////////////////////////////////////////////////////////////////////// */
/*  */
//
/* initialize the part with the given data. */
/*////////////////////////////////////////////////////////////////////////////// */

int part_init ( struct part *p ) {

    /* check inputs */
    if ( p == NULL )
        return part_err = part_err_null;
        
    /* all is well... */
    return part_err_ok;

    }
