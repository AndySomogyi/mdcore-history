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


/* error codes */
#define part_err_ok                     0
#define part_err_null                   -1
#define part_err_malloc                 -2


/* particle flags */
#define part_flag_none                  0
#define part_flag_frozen                1
#define part_flag_ghost                 2


/* default values */


/** ID of the last error. */
extern int part_err;


/**
 * The #part data structure.
 *
 * Note that the arrays for @c x, @c v and @c f are 4 entries long for
 * propper alignment.
 */
struct part {

    #ifdef CELL
    
        /** Particle position */
        vector float x;

        /** Particle velocity */
        vector float v;

        /** Particle force */
        vector float f;
    
    #else
    
        /** Particle position */
        FPTYPE x[4];

        /** Particle velocity */
        FPTYPE v[4];

        /** Particle force */
        FPTYPE f[4];
    
    #endif
    
    /** individual particle charge, if needed. */
    float q;
    
    /** Particle id and type */
    int id, vid;
    
    /** particle type. */
    short int type;
    
    /** Particle flags */
    unsigned short int flags;
    
    };
    
    
/** Structure containing information on each particle species. */
struct part_type {

    /** ID of this type */
    int id;
    
    /** Constant physical characteristics */
    double mass, imass, charge;
    
    };


/* associated functions */
int part_init ( struct part *p , int vid , int type , unsigned int flags );
