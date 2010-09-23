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


/* error codes */
#define part_err_ok                     0
#define part_err_null                   -1
#define part_err_malloc                 -2


/* particle flags */
#define part_flag_none                  0
#define part_flag_frozen                1
#define part_falg_ghost                 2


/* default values */


/* the last error */
extern int part_err;


/* the data structure */
struct part {

    /* particle id and type */
    int id, type;
    
    /* particle flags */
    unsigned int flags;
    
    #ifdef VECTORIZE
    
        /* buffer value for correct alignment */
        unsigned int buff;
    
        /* particle position */
        vector float x;

        /* particle velocity */
        vector float v;

        /* particle force */
        vector float f;
    
    #else
    
        #ifdef USE_DOUBLES
            /* particle position */
            double x[3];

            /* particle velocity */
            double v[3];

            /* particle force */
            double f[3];
        #else
            /* particle position */
            float x[3];

            /* particle velocity */
            float v[3];

            /* particle force */
            float f[3];
        #endif
    
    #endif
    
    };
    
    
struct part_type {

    /* id of this type */
    int id;
    
    /* constant physical characteristics */
    double mass, imass, charge;
    
    };


/* associated functions */
