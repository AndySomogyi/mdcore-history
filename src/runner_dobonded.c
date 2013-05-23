/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2013 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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

/* Include configuration header */
#include "../config.h"

/* Include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <limits.h>

/* Include some conditional headers. */
#include "../config.h"
#ifdef CELL
    #include <libspe2.h>
    #include <libmisc.h>
    #define ceil128(v) (((v) + 127) & ~127)
#endif
#ifdef HAVE_SETAFFINITY
    #include <sched.h>
#endif
#ifdef WITH_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "lock.h"
#include "part.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "potential_eval.h"
#include "engine.h"
#include "bond.h"
#include "angle.h"
#include "dihedral.h"
#include "exclusion.h"
#include "runner.h"



/* the error macro. */
#define error(id)				( runner_err = errs_register( id , runner_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
extern char *runner_err_msg[];
extern unsigned int runner_rcount;


/**
 * @brief Compute the pairwise interactions for the given pair.
 *
 * @param r The #runner computing the pair.
 * @param cell_i The first cell.
 * @param cell_j The second cell.
 * @param shift A pointer to an array of three floating point values containing
 *      the vector separating the centers of @c cell_i and @c cell_j.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err)
 *
 * Computes the interactions between all the particles in @c cell_i and all
 * the paritcles in @c cell_j. @c cell_i and @c cell_j may be the same cell.
 *
 * @sa #runner_sortedpair.
 */

__attribute__ ((flatten)) int runner_dobonded ( struct runner *r , struct engine_set *set ) {

    int j, k, count, sid = set->id;
    struct engine *e = r->e;
    struct space *s = &e->s;
    struct angle *a;
    struct bond *b;
    struct dihedral *d;
    struct exclusion *ex;
    struct cell **celllist = s->celllist;
    struct cell *ci, *cj, *ck, *cl;
    double epot_bond = 0.0, epot_angle = 0.0, epot_dihedral = 0.0, epot_exclusion = 0.0;
    ticks tic;
    
    /* printf( "runner_dobonded[%i]: working on bonded set %i with cells [ " , r->id , sid );
    for ( k = 0 ; k < set->nr_cells ; k++ )
        printf( "%i " , set->cells[k] );
    printf( "].\n" ); fflush(stdout); */
    
    /* Do we need to re-assess our list of bonded interactions? */
    if ( s->verlet_rebuild ) {
    
        /* Partition bonds. */
        count = e->nr_bonds;
        for ( j = 0 , k = 0 ; k < count ; k++ ) {
            b = &e->bonds[k];
            ci = celllist[ b->i ]; cj = celllist[ b->j ];
            if ( ( ci != NULL && ci->setID == sid ) ||
                 ( cj != NULL && cj->setID == sid ) )
                set->bonds[j++] = b;
            }
        set->nr_bonds = j;
    
        /* Partition angles. */
        count = e->nr_angles;
        for ( j = 0 , k = 0 ; k < count ; k++ ) {
            a = &e->angles[k];
            ci = celllist[ a->i ]; cj = celllist[ a->j ]; ck = celllist[ a->k ];
            if ( ( ci != NULL && ci->setID == sid ) ||
                 ( cj != NULL && cj->setID == sid ) ||
                 ( ck != NULL && ck->setID == sid ) )
                set->angles[j++] = a;
            }
        set->nr_angles = j;
    
        /* Partition dihedrals. */
        count = e->nr_dihedrals;
        for ( j = 0 , k = 0 ; k < count ; k++ ) {
            d = &e->dihedrals[k];
            ci = celllist[ d->i ]; cj = celllist[ d->j ]; ck = celllist[ d->k ]; cl = celllist[ d->l ];
            if ( ( ci != NULL && ci->setID == sid ) ||
                 ( cj != NULL && cj->setID == sid ) ||
                 ( ck != NULL && ck->setID == sid ) ||
                 ( cl != NULL && cl->setID == sid ) )
                set->dihedrals[j++] = d;
            }
        set->nr_dihedrals = j;
    
        /* Partition exclusions. */
        count = e->nr_exclusions;
        for ( j = 0 , k = 0 ; k < count ; k++ ) {
            ex = &e->exclusions[k];
            ci = celllist[ ex->i ]; cj = celllist[ ex->j ];
            if ( ( ci != NULL && ci->setID == sid ) ||
                 ( cj != NULL && cj->setID == sid ) )
                set->exclusions[j++] = ex;
            }
        set->nr_exclusions = j;
    
        }
        
    /* Do the bonds. */
    tic = getticks();
    bond_eval_set( set->bonds , set->nr_bonds , e , set->id , &epot_bond );
    e->timers[engine_timer_bonds] += getticks() - tic;
    
    /* Do the angles. */
    tic = getticks();
    angle_eval_set( set->angles , set->nr_angles , e , set->id , &epot_angle );
    e->timers[engine_timer_angles] += getticks() - tic;
    
    /* Do the dihedrals. */
    tic = getticks();
    dihedral_eval_set( set->dihedrals , set->nr_dihedrals , e , set->id , &epot_dihedral );
    e->timers[engine_timer_dihedrals] += getticks() - tic;
    
    /* Do the exclusions. */
    tic = getticks();
    exclusion_eval_set( set->exclusions , set->nr_exclusions , e , set->id , &epot_exclusion );
    e->timers[engine_timer_exclusions] += getticks() - tic;
    
    /* Store the potential energy. */
    if ( lock_lock( &s->lock ) != 0 )
        return error(runner_err_pthread);
    s->epot += epot_bond + epot_angle + epot_dihedral + epot_exclusion;
    s->epot_bond += epot_bond;
    s->epot_angle += epot_angle;
    s->epot_dihedral += epot_dihedral;
    s->epot_exclusion += epot_exclusion;
    if ( lock_unlock( &s->lock ) != 0 )
        return error(runner_err_pthread);
    
    /* The bear, the bear... */
    return runner_err_ok;

    }



