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


/* angle error codes */
#define angle_err_ok                    0
#define angle_err_null                  -1
#define angle_err_malloc                -2


#define angle_partlist_incr             100

/** Maximum number of cells per tuple. */
#define angle_maxtuples                 4

/** Maximum number of interactions per particle in the Verlet list. */
#define angle_verlet_maxpairs           750


/** ID of the last error */
extern int angle_err;


/** The angle structure */
struct angle {

    /* ids of particles involved */
    int i, j, k;
    
    /* id of the potential. */
    int pid;
    
    };
    

/* associated functions */
int angle_eval ( struct angle *a , int N , struct engine *e , double *epot_out );
int angle_evalf ( struct angle *a , int N , struct engine *e , FPTYPE *f , double *epot_out );