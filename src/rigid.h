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

/* Local includes. */
#include "fptype.h"


/* rigid error codes */
#define rigid_err_ok                    0
#define rigid_err_null                  -1
#define rigid_err_malloc                -2


/* Some constants. */
#define rigid_maxparts                  10
#define rigid_maxconstr                 (3*rigid_maxparts)
#define rigid_maxiter                   50


/** ID of the last error */
extern int rigid_err;


/** The rigid structure */
struct rigid {

    /** Nr of parts involved. */
    int nr_parts;

    /** ids of particles involved */
    int parts[ rigid_maxparts ];
    
    /** Nr of constraints involved. */
    int nr_constr;
    
    /** The constraints themselves. */
    struct {
        int i, j;
        double d2;
        } constr[ rigid_maxconstr ];
    
    };
    

/* associated functions */
int rigid_eval_shake ( struct rigid *r , int N , struct engine *e );
