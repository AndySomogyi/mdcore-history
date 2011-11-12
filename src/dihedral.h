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


/* dihedral error codes */
#define dihedral_err_ok                    0
#define dihedral_err_null                  -1
#define dihedral_err_malloc                -2


/** ID of the last error */
extern int dihedral_err;


/** The dihedral structure */
struct dihedral {

    /* ids of particles involved */
    int i, j, k, l;
    
    /* id of the potential. */
    int pid;
    
    };
    

/* associated functions */
int dihedral_eval ( struct dihedral *d , int N , struct engine *e , double *epot_out );
int dihedral_evalf ( struct dihedral *d , int N , struct engine *e , FPTYPE *f , double *epot_out );
int dihedral_eval_mod ( struct dihedral *d , int N , int nr_threads , int cid_mod , struct engine *e , double *epot_out );
int dihedral_eval_div ( struct dihedral *d , int N , int nr_threads , int cid_div , struct engine *e , double *epot_out );