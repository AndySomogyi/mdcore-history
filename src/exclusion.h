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


/* exclusion error codes */
#define exclusion_err_ok                    0
#define exclusion_err_null                  -1
#define exclusion_err_malloc                -2


/** ID of the last error */
extern int exclusion_err;


/** The exclusion structure */
struct exclusion {

    /* ids of particles involved */
    int i, j;
    
    };
    

/* associated functions */
int exclusion_eval ( struct exclusion *b , int N , struct engine *e , double *epot_out );
int exclusion_eval_mod ( struct exclusion *b , int N , int nr_threads , int cid_mod , struct engine *e , double *epot_out );
int exclusion_eval_div ( struct exclusion *b , int N , int nr_threads , int cid_div , struct engine *e , double *epot_out );