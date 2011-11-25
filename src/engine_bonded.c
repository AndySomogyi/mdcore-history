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
#include <math.h>
#include <float.h>
#include <string.h>
#ifdef CELL
    #include <libspe2.h>
#endif

/* Include conditional headers. */
#include "../config.h"
#ifdef HAVE_MPI
    #include <mpi.h>
#endif
#ifdef HAVE_OPENMP
    #include <omp.h>
#endif

/* include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "runner.h"
#include "bond.h"
#include "rigid.h"
#include "angle.h"
#include "dihedral.h"
#include "exclusion.h"
#include "reader.h"
#include "engine.h"


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )



/**
 * @brief Assemble non-conflicting sets of bonded interactions.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_bonded_sets ( struct engine *e ) {

    struct {
        int i, j;
        } *confl;
    int confl_size, confl_count = 0;
    int *nconfl, *weight;
    int *setid_bonds, *setid_angles, *setid_dihedras, *setid_exclusions;
    int nr_sets;
    int sid = 0, k;
    
    /* Start with one set per bonded interaction. */
    nr_sets = e->nr_bonds + e->nr_angles + e->nr_dihedrals + e->nr_exclusions;
    if ( ( weight = (int *)malloc( sizeof(int) * nr_sets ) ) == NULL ||
         ( nconfl = (int *)calloc( nr_sets , sizeof(int) ) ) == NULL )
        return error(engine_err_malloc);
        
    /* Fill the initial setids and weights. */
    if ( ( setid_bonds = (int *)malloc( sizeof(int) * e->nr_bonds ) ) == NULL ||
         ( setid_angles = (int *)malloc( sizeof(int) * e->nr_angles ) ) == NULL ||
         ( setid_dihedrals = (int *)malloc( sizeof(int) * e->nr_dihedrals ) ) == NULL ||
         ( setid_exclusions = (int *)malloc( sizeof(int) * e->nr_exclusions ) ) == NULL )
        return error(engine_err_malloc);
    for ( k = 0 ; k < e->nr_bonds ; k++ ) {
        weight[ sid ] = 1;
        setid_bonds[k] = sid;
        sid += 1;
        }
    for ( k = 0 ; k < e->nr_angles ; k++ ) {
        weight[ sid ] = 2;
        setid_angles[k] = sid;
        sid += 1;
        }
    for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {
        weight[ sid ] = 3;
        setid_dihedrals[k] = sid;
        sid += 1;
        }
    for ( k = 0 ; k < e->nr_exclusions ; k++ ) {
        weight[ sid ] = 1;
        setid_exclusions[k] = sid;
        sid += 1;
        }
        
    /* Generate the set of conflicts. */
    confl_size = nr_sets;
    if ( ( confl = malloc( sizeof(int) * 2 * confl_size ) ) == NULL )
        return error(engine_err_malloc);
        
    /* Loop over all bonds. */
    for ( k = 0 ; k < e->nr_bonds ; k++ ) {
    
        /* Loop over other bonds... */
        for ( j = k+1 ; j < e->nr_bonds ; j++ )
            if ( e->bonds[k].i == e->bonds[j].i || e->bonds[k].i == e->bonds[j].j ||
                 e->bonds[k].j == e->bonds[j].i || e->bonds[k].j == e->bonds[j].j ) {
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_bonds[k];
                confl[confl_count].j = setid_bonds[j];
                confl_count += 1;
                }
    
        /* Loop over angles... */
        for ( j = 0 ; j < e->nr_angles ; j++ )
            if ( e->bonds[k].i == e->angles[j].i || e->bonds[k].i == e->angles[j].j || e->bonds[k].i == e->angles[j].k ||
                 e->bonds[k].j == e->angles[j].i || e->bonds[k].j == e->angles[j].j || e->bonds[k].j == e->angles[j].k ) {
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_bonds[k];
                confl[confl_count].j = setid_angles[j];
                confl_count += 1;
                }
    
        /* Loop over dihedrals... */
        for ( j = 0 ; j < e->nr_dihedrals ; j++ )
            if ( e->bonds[k].i == e->dihedrals[j].i || e->bonds[k].i == e->dihedrals[j].j || e->bonds[k].i == e->dihedrals[j].k || e->bonds[k].i == e->dihedrals[j].l ||
                 e->bonds[k].j == e->dihedrals[j].i || e->bonds[k].j == e->dihedrals[j].j || e->bonds[k].j == e->dihedrals[j].k || e->bonds[k].j == e->dihedrals[j].l ) {
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_bonds[k];
                confl[confl_count].j = setid_dihedrals[j];
                confl_count += 1;
                }
    
        /* Loop over exclusions... */
        for ( j = 0 ; j < e->nr_exclusions ; j++ )
            if ( e->bonds[k].i == e->exclusions[j].i || e->bonds[k].i == e->exclusions[j].j ||
                 e->bonds[k].j == e->exclusions[j].i || e->bonds[k].j == e->exclusions[j].j ) {
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_bonds[k];
                confl[confl_count].j = setid_exclusions[j];
                confl_count += 1;
                }
    
        } /* Loop over bonds. */

    /* Loop over all angles. */
    for ( k = 0 ; k < e->nr_angles ; k++ ) {
    
        /* Loop over other angles... */
        for ( j = k+1 ; j < e->nr_angles ; j++ )
            if ( e->angles[k].i == e->angles[j].i || e->angles[k].i == e->angles[j].j || e->angles[k].i == e->angles[j].k ||
                 e->angles[k].j == e->angles[j].i || e->angles[k].j == e->angles[j].j || e->angles[k].j == e->angles[j].k ||
                 e->angles[k].k == e->angles[j].i || e->angles[k].k == e->angles[j].j || e->angles[k].k == e->angles[j].k ) {
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_angles[k];
                confl[confl_count].j = setid_angles[j];
                confl_count += 1;
                }
    
        /* Loop over dihedrals... */
        for ( j = 0 ; j < e->nr_dihedrals ; j++ )
            if ( e->angles[k].i == e->dihedrals[j].i || e->angles[k].i == e->dihedrals[j].j || e->angles[k].i == e->dihedrals[j].k || e->angles[k].i == e->dihedrals[j].l ||
                 e->angles[k].j == e->dihedrals[j].i || e->angles[k].j == e->dihedrals[j].j || e->angles[k].j == e->dihedrals[j].k || e->angles[k].j == e->dihedrals[j].l ||
                 e->angles[k].k == e->dihedrals[j].i || e->angles[k].k == e->dihedrals[j].j || e->angles[k].k == e->dihedrals[j].k || e->angles[k].k == e->dihedrals[j].l ) {
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_angles[k];
                confl[confl_count].j = setid_dihedrals[j];
                confl_count += 1;
                }
    
        /* Loop over exclusions... */
        for ( j = 0 ; j < e->nr_exclusions ; j++ )
            if ( e->angles[k].i == e->exclusions[j].i || e->angles[k].i == e->exclusions[j].j ||
                 e->angles[k].j == e->exclusions[j].i || e->angles[k].j == e->exclusions[j].j ||
                 e->angles[k].k == e->exclusions[j].i || e->angles[k].k == e->exclusions[j].j ) {
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_angles[k];
                confl[confl_count].j = setid_exclusions[j];
                confl_count += 1;
                }
    
        } /* Loop over bonds. */

    /* Loop over all dihedrals. */
    for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {
    
        /* Loop over other dihedrals... */
        for ( j = k+1 ; j < e->nr_dihedrals ; j++ )
            if ( e->dihedrals[k].i == e->dihedrals[j].i || e->dihedrals[k].i == e->dihedrals[j].j || e->dihedrals[k].i == e->dihedrals[j].k || e->dihedrals[k].i == e->dihedrals[j].l ||
                 e->dihedrals[k].j == e->dihedrals[j].i || e->dihedrals[k].j == e->dihedrals[j].j || e->dihedrals[k].j == e->dihedrals[j].k || e->dihedrals[k].j == e->dihedrals[j].l ||
                 e->dihedrals[k].k == e->dihedrals[j].i || e->dihedrals[k].k == e->dihedrals[j].j || e->dihedrals[k].k == e->dihedrals[j].k || e->dihedrals[k].k == e->dihedrals[j].l ||
                 e->dihedrals[k].l == e->dihedrals[j].i || e->dihedrals[k].l == e->dihedrals[j].j || e->dihedrals[k].l == e->dihedrals[j].k || e->dihedrals[k].l == e->dihedrals[j].l ) {
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_dihedrals[k];
                confl[confl_count].j = setid_dihedrals[j];
                confl_count += 1;
                }
    
        /* Loop over exclusions... */
        for ( j = 0 ; j < e->nr_exclusions ; j++ )
            if ( e->dihedrals[k].i == e->exclusions[j].i || e->dihedrals[k].i == e->exclusions[j].j ||
                 e->dihedrals[k].j == e->exclusions[j].i || e->dihedrals[k].j == e->exclusions[j].j ||
                 e->dihedrals[k].k == e->exclusions[j].i || e->dihedrals[k].k == e->exclusions[j].j ||
                 e->dihedrals[k].l == e->exclusions[j].i || e->dihedrals[k].l == e->exclusions[j].j ) {
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_dihedrals[k];
                confl[confl_count].j = setid_exclusions[j];
                confl_count += 1;
                }
    
        } /* Loop over bonds. */

    /* Loop over all exclusions. */
    for ( k = 0 ; k < e->nr_exclusions ; k++ ) {
    
        /* Loop over other exclusions... */
        for ( j = k+1 ; j < e->nr_exclusions ; j++ )
            if ( e->exclusions[k].i == e->exclusions[j].i || e->exclusions[k].i == e->exclusions[j].j ||
                 e->exclusions[k].j == e->exclusions[j].i || e->exclusions[k].j == e->exclusions[j].j ||
                if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
                    return error(engine_err_malloc);
                confl[confl_count].i = setid_exclusions[k];
                confl[confl_count].j = setid_exclusions[j];
                confl_count += 1;
                }
    
        } /* Loop over bonds. */

    }
    
    

/**
 * @brief Add a dihedral interaction to the engine.
 *
 * @param e The #engine.
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 * @param k The ID of the third #part.
 * @param l The ID of the fourth #part.
 * @param pid Index of the #potential for this bond.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_dihedral_add ( struct engine *e , int i , int j , int k , int l , int pid ) {

    struct dihedral *dummy;

    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
    /* if ( i > e->s.nr_parts || j > e->s.nr_parts )
        return error(engine_err_range);
    if ( pid > e->nr_dihedralpots )
        return error(engine_err_range); */
        
    /* Do we need to grow the dihedrals array? */
    if ( e->nr_dihedrals == e->dihedrals_size ) {
        e->dihedrals_size *= 1.414;
        if ( ( dummy = (struct dihedral *)malloc( sizeof(struct dihedral) * e->dihedrals_size ) ) == NULL )
            return error(engine_err_malloc);
        memcpy( dummy , e->dihedrals , sizeof(struct dihedral) * e->nr_dihedrals );
        free( e->dihedrals );
        e->dihedrals = dummy;
        }
        
    /* Store this dihedral. */
    e->dihedrals[ e->nr_dihedrals ].i = i;
    e->dihedrals[ e->nr_dihedrals ].j = j;
    e->dihedrals[ e->nr_dihedrals ].k = k;
    e->dihedrals[ e->nr_dihedrals ].l = l;
    e->dihedrals[ e->nr_dihedrals ].pid = pid;
    e->nr_dihedrals += 1;
    
    /* It's the end of the world as we know it. */
    return engine_err_ok;

    }


/**
 * @brief Add a angle interaction to the engine.
 *
 * @param e The #engine.
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 * @param k The ID of the third #part.
 * @param pid Index of the #potential for this bond.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_angle_add ( struct engine *e , int i , int j , int k , int pid ) {

    struct angle *dummy;

    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
    /* if ( i > e->s.nr_parts || j > e->s.nr_parts )
        return error(engine_err_range);
    if ( pid > e->nr_anglepots )
        return error(engine_err_range); */
        
    /* Do we need to grow the angles array? */
    if ( e->nr_angles == e->angles_size ) {
        e->angles_size *= 1.414;
        if ( ( dummy = (struct angle *)malloc( sizeof(struct angle) * e->angles_size ) ) == NULL )
            return error(engine_err_malloc);
        memcpy( dummy , e->angles , sizeof(struct angle) * e->nr_angles );
        free( e->angles );
        e->angles = dummy;
        }
        
    /* Store this angle. */
    e->angles[ e->nr_angles ].i = i;
    e->angles[ e->nr_angles ].j = j;
    e->angles[ e->nr_angles ].k = k;
    e->angles[ e->nr_angles ].pid = pid;
    e->nr_angles += 1;
    
    /* It's the end of the world as we know it. */
    return engine_err_ok;

    }


/**
 * @brief Remove duplicate exclusions.
 *
 * @param e The #engine.
 *
 * @return The number of unique exclusions or < 0 on error (see #engine_err).
 */
 
int engine_exclusion_shrink ( struct engine *e ) {

    int j, k;

    /* Recursive quicksort for the exclusions. */
    void qsort ( int l , int r ) {
        
        int i = l, j = r;
        int pivot_i = e->exclusions[ (l + r)/2 ].i;
        int pivot_j = e->exclusions[ (l + r)/2 ].j;
        struct exclusion temp;
        
        /* Too small? */
        if ( r - l < 10 ) {
        
            /* Use Insertion Sort. */
            for ( i = l+1 ; i <= r ; i++ ) {
                pivot_i = e->exclusions[i].i;
                pivot_j = e->exclusions[i].j;
                for ( j = i-1 ; j >= l ; j-- )
                    if ( e->exclusions[j].i < pivot_i ||
                         ( e->exclusions[j].i == pivot_i && e->exclusions[j].j < pivot_j ) ) {
                        temp = e->exclusions[j];
                        e->exclusions[j] = e->exclusions[j+1];
                        e->exclusions[j+1] = temp;
                        }
                    else
                        break;
                }
        
            }
            
        else {
        
            /* Partition. */
            while ( i <= j ) {
                while ( e->exclusions[i].i < pivot_i ||
                       ( e->exclusions[i].i == pivot_i && e->exclusions[i].j < pivot_j ) )
                    i += 1;
                while ( e->exclusions[j].i > pivot_i ||
                       ( e->exclusions[j].i == pivot_i && e->exclusions[j].j > pivot_j ) )
                    j -= 1;
                if ( i <= j ) {
                    temp = e->exclusions[i];
                    e->exclusions[i] = e->exclusions[j];
                    e->exclusions[j] = temp;
                    i += 1;
                    j -= 1;
                    }
                }

            /* Recurse. */
            if ( l < j )
                qsort( l , j );
            if ( i < r )
                qsort( i , r );
                
            }
        
        }
        
    /* Sort the exclusions. */
    qsort( 0 , e->nr_exclusions-1 );
    
    /* Run through the exclusions and skip duplicates. */
    for ( j = 1 , k = 1 ; k < e->nr_exclusions ; k++ )
        if ( e->exclusions[k].j != e->exclusions[k-1].j ||
             e->exclusions[k].i != e->exclusions[k-1].i ) {
            e->exclusions[j] = e->exclusions[k];
            j += 1;
            }
            
    /* Set the number of exclusions to j. */
    e->nr_exclusions = j;
    
    /* Go home. */
    return engine_err_ok;

    }


/**
 * @brief Add a exclusioned interaction to the engine.
 *
 * @param e The #engine.
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_exclusion_add ( struct engine *e , int i , int j ) {

    struct exclusion *dummy;

    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
    /* if ( i > e->s.nr_parts || j > e->s.nr_parts )
        return error(engine_err_range); */
        
    /* Do we need to grow the exclusions array? */
    if ( e->nr_exclusions == e->exclusions_size ) {
        e->exclusions_size *= 1.414;
        if ( ( dummy = (struct exclusion *)malloc( sizeof(struct exclusion) * e->exclusions_size ) ) == NULL )
            return error(engine_err_malloc);
        memcpy( dummy , e->exclusions , sizeof(struct exclusion) * e->nr_exclusions );
        free( e->exclusions );
        e->exclusions = dummy;
        }
        
    /* Store this exclusion. */
    if ( i <= j ) {
        e->exclusions[ e->nr_exclusions ].i = i;
        e->exclusions[ e->nr_exclusions ].j = j;
        }
    else {
        e->exclusions[ e->nr_exclusions ].i = j;
        e->exclusions[ e->nr_exclusions ].j = i;
        }
    e->nr_exclusions += 1;
    
    /* It's the end of the world as we know it. */
    return engine_err_ok;

    }


/**
 * @brief Add a bonded interaction to the engine.
 *
 * @param e The #engine.
 * @param i The ID of the first #part.
 * @param j The ID of the second #part.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_bond_add ( struct engine *e , int i , int j ) {

    struct bond *dummy;

    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
    /* if ( i > e->s.nr_parts || j > e->s.nr_parts )
        return error(engine_err_range); */
        
    /* Do we need to grow the bonds array? */
    if ( e->nr_bonds == e->bonds_size ) {
        e->bonds_size  *= 1.414;
        if ( ( dummy = (struct bond *)malloc( sizeof(struct bond) * e->bonds_size ) ) == NULL )
            return error(engine_err_malloc);
        memcpy( dummy , e->bonds , sizeof(struct bond) * e->nr_bonds );
        free( e->bonds );
        e->bonds = dummy;
        }
        
    /* Store this bond. */
    e->bonds[ e->nr_bonds ].i = i;
    e->bonds[ e->nr_bonds ].j = j;
    e->nr_bonds += 1;
    
    /* It's the end of the world as we know it. */
    return engine_err_ok;

    }


/**
 * @brief Compute all bonded interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Does the same as #engine_bond_eval, #engine_angle_eval and
 * #engine_dihedral eval, yet all in one go to avoid excessive
 * updates of the particle forces.
 */
 
int engine_bonded_eval ( struct engine *e ) {

    double epot = 0.0;
    struct space *s;
    struct dihedral dtemp;
    struct angle atemp;
    struct bond btemp;
    struct exclusion etemp;
    int nr_dihedrals = e->nr_dihedrals, nr_bonds = e->nr_bonds;
    int nr_angles = e->nr_angles, nr_exclusions = e->nr_exclusions;
    int i, j, k;
    #ifdef HAVE_OPENMP
        int nr_threads, thread_id;
        double epot_local;
    #endif
    ticks tic;
    
    /* Bail if there are no bonded interaction. */
    if ( nr_bonds == 0 && nr_angles == 0 && nr_dihedrals == 0 && nr_exclusions == 0 )
        return engine_err_ok;
    
    /* Get a handle on the space. */
    s = &e->s;

    /* If in parallel... */
    if ( e->nr_nodes > 1 ) {
    
        tic = getticks();
    
        #pragma omp parallel for schedule(static), private(i,j,dtemp,atemp,btemp,etemp)
        for ( k = 0 ; k < 4 ; k++ ) {
    
            if ( k == 0 ) {
                /* Sort the dihedrals. */
                i = 0; j = nr_dihedrals-1;
                while ( i < j ) {
                    while ( i < nr_dihedrals &&
                            s->partlist[e->dihedrals[i].i] != NULL &&
                            s->partlist[e->dihedrals[i].j] != NULL &&
                            s->partlist[e->dihedrals[i].k] != NULL &&
                            s->partlist[e->dihedrals[i].l] != NULL )
                        i += 1;
                    while ( j >= 0 &&
                            ( s->partlist[e->dihedrals[j].i] == NULL ||
                              s->partlist[e->dihedrals[j].j] == NULL ||
                              s->partlist[e->dihedrals[j].k] == NULL ||
                              s->partlist[e->dihedrals[j].l] == NULL ) )
                        j -= 1;
                    if ( i < j ) {
                        dtemp = e->dihedrals[i];
                        e->dihedrals[i] = e->dihedrals[j];
                        e->dihedrals[j] = dtemp;
                        }
                    }
                nr_dihedrals = i;
                }

            else if ( k == 1 ) {
                /* Sort the angles. */
                i = 0; j = nr_angles-1;
                while ( i < j ) {
                    while ( i < nr_angles &&
                            s->partlist[e->angles[i].i] != NULL &&
                            s->partlist[e->angles[i].j] != NULL &&
                            s->partlist[e->angles[i].k] != NULL )
                        i += 1;
                    while ( j >= 0 &&
                            ( s->partlist[e->angles[j].i] == NULL ||
                              s->partlist[e->angles[j].j] == NULL ||
                              s->partlist[e->angles[j].k] == NULL ) )
                        j -= 1;
                    if ( i < j ) {
                        atemp = e->angles[i];
                        e->angles[i] = e->angles[j];
                        e->angles[j] = atemp;
                        }
                    }
                nr_angles = i;
                }

            else if ( k == 2 ) {
                /* Sort the bonds. */
                i = 0; j = nr_bonds-1;
                while ( i < j ) {
                    while ( i < nr_bonds &&
                            s->partlist[e->bonds[i].i] != NULL &&
                            s->partlist[e->bonds[i].j] != NULL )
                        i += 1;
                    while ( j >= 0 &&
                            ( s->partlist[e->bonds[j].i] == NULL ||
                              s->partlist[e->bonds[j].j] == NULL ) )
                        j -= 1;
                    if ( i < j ) {
                        btemp = e->bonds[i];
                        e->bonds[i] = e->bonds[j];
                        e->bonds[j] = btemp;
                        }
                    }
                nr_bonds = i;
                }

            else if ( k == 3 ) {
                /* Sort the exclusions. */
                i = 0; j = nr_exclusions-1;
                while ( i < j ) {
                    while ( i < nr_exclusions &&
                            s->partlist[e->exclusions[i].i] != NULL &&
                            s->partlist[e->exclusions[i].j] != NULL )
                        i += 1;
                    while ( j >= 0 &&
                            ( s->partlist[e->exclusions[j].i] == NULL ||
                              s->partlist[e->exclusions[j].j] == NULL ) )
                        j -= 1;
                    if ( i < j ) {
                        etemp = e->exclusions[i];
                        e->exclusions[i] = e->exclusions[j];
                        e->exclusions[j] = etemp;
                        }
                    }
                nr_exclusions = i;
                }
        
            }
            
        /* Stop the clock. */
        e->timers[engine_timer_bonded_sort] += getticks() - tic;
        
        }
        

    #ifdef HAVE_OPENMP
    
        /* Is it worth parallelizing? */
        #pragma omp parallel private(thread_id,nr_threads,epot_local)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) &&
             ( nr_bonds + nr_angles + nr_dihedrals ) > 0 ) {
             
            /* Init the local potential energy. */
            epot_local = 0;
             
            /* Get the thread ID. */
            thread_id = omp_get_thread_num();

            /* Compute the bonded interactions. */
            bond_eval_div( e->bonds , nr_bonds , nr_threads , thread_id , e , &epot_local );
                    
            /* Compute the angle interactions. */
            angle_eval_div( e->angles , nr_angles , nr_threads , thread_id , e , &epot_local );
                    
            /* Compute the dihedral interactions. */
            dihedral_eval_div( e->dihedrals , nr_dihedrals , nr_threads , thread_id , e , &epot_local );
                    
            /* Correct for excluded interactons. */
            exclusion_eval_div( e->exclusions , nr_exclusions , nr_threads , thread_id , e , &epot_local );
              
            /* Aggregate the global potential energy. */
            #pragma omp atomic
            epot += epot_local;
                    
            }
            
        /* Otherwise, evaluate directly. */
        else if ( omp_get_thread_num() == 0 ) {
        
            /* Do bonds. */
            tic = getticks();
            bond_eval( e->bonds , nr_bonds , e , &epot );
            e->timers[engine_timer_bonds] += getticks() - tic;
            
            /* Do angles. */
            tic = getticks();
            angle_eval( e->angles , nr_angles , e , &epot );
            e->timers[engine_timer_angles] += getticks() - tic;
            
            /* Do dihedrals. */
            tic = getticks();
            dihedral_eval( e->dihedrals , nr_dihedrals , e , &epot );
            e->timers[engine_timer_dihedrals] += getticks() - tic;
            
            /* Do exclusions. */
            tic = getticks();
            exclusion_eval( e->exclusions , nr_exclusions , e , &epot );
            e->timers[engine_timer_exclusions] += getticks() - tic;
            
            }
            
    #else
    
        /* Do bonds. */
        tic = getticks();
        if ( bond_eval( e->bonds , nr_bonds , e , &epot ) < 0 )
            return error(engine_err_bond);
        e->timers[engine_timer_bonds] += getticks() - tic;
            
        /* Do angles. */
        tic = getticks();
        if ( angle_eval( e->angles , nr_angles , e , &epot ) < 0 )
            return error(engine_err_angle);
        e->timers[engine_timer_angles] += getticks() - tic;
            
        /* Do dihedrals. */
        tic = getticks();
        if ( dihedral_eval( e->dihedrals , nr_dihedrals , e , &epot ) < 0 )
            return error(engine_err_dihedral);
        e->timers[engine_timer_dihedrals] += getticks() - tic;
            
        /* Do exclusions. */
        tic = getticks();
        if ( exclusion_eval( e->exclusions , nr_exclusions , e , &epot ) < 0 )
            return error(engine_err_exclusion);
        e->timers[engine_timer_exclusions] += getticks() - tic;
            
    #endif
        
    /* Store the potential energy. */
    s->epot += epot;
    
    /* I'll be back... */
    return engine_err_ok;

    }


/**
 * @brief Compute all bonded interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Does the same as #engine_bond_eval, #engine_angle_eval and
 * #engine_dihedral eval, yet all in one go to avoid excessive
 * updates of the particle forces.
 */
 
int engine_bonded_eval_alt ( struct engine *e ) {

    double epot = 0.0;
    struct space *s;
    struct dihedral dtemp, *dihedrals;
    struct angle atemp, *angles;
    struct bond btemp, *bonds;
    struct exclusion etemp, *exclusions;
    int nr_dihedrals = e->nr_dihedrals, nr_bonds = e->nr_bonds;
    int nr_angles = e->nr_angles, nr_exclusions = e->nr_exclusions;
    int i, j, k;
    #ifdef HAVE_OPENMP
        int count, nr_threads, thread_id;
        double scale, epot_local;
    #endif
    ticks tic;
    
    /* Bail if there are no bonded interaction. */
    if ( nr_bonds == 0 && nr_angles == 0 && nr_dihedrals == 0 && nr_exclusions == 0 )
        return engine_err_ok;
    
    /* Get a handle on the space. */
    s = &e->s;

    /* If in parallel... */
    if ( e->nr_nodes > 1 ) {
    
        tic = getticks();
    
        #pragma omp parallel for schedule(static), private(i,j,dtemp,atemp,btemp,etemp)
        for ( k = 0 ; k < 4 ; k++ ) {
    
            if ( k == 0 ) {
                /* Sort the dihedrals. */
                i = 0; j = nr_dihedrals-1;
                while ( i < j ) {
                    while ( i < nr_dihedrals &&
                            s->partlist[e->dihedrals[i].i] != NULL &&
                            s->partlist[e->dihedrals[i].j] != NULL &&
                            s->partlist[e->dihedrals[i].k] != NULL &&
                            s->partlist[e->dihedrals[i].l] != NULL )
                        i += 1;
                    while ( j >= 0 &&
                            ( s->partlist[e->dihedrals[j].i] == NULL ||
                              s->partlist[e->dihedrals[j].j] == NULL ||
                              s->partlist[e->dihedrals[j].k] == NULL ||
                              s->partlist[e->dihedrals[j].l] == NULL ) )
                        j -= 1;
                    if ( i < j ) {
                        dtemp = e->dihedrals[i];
                        e->dihedrals[i] = e->dihedrals[j];
                        e->dihedrals[j] = dtemp;
                        }
                    }
                nr_dihedrals = i;
                }

            else if ( k == 1 ) {
                /* Sort the angles. */
                i = 0; j = nr_angles-1;
                while ( i < j ) {
                    while ( i < nr_angles &&
                            s->partlist[e->angles[i].i] != NULL &&
                            s->partlist[e->angles[i].j] != NULL &&
                            s->partlist[e->angles[i].k] != NULL )
                        i += 1;
                    while ( j >= 0 &&
                            ( s->partlist[e->angles[j].i] == NULL ||
                              s->partlist[e->angles[j].j] == NULL ||
                              s->partlist[e->angles[j].k] == NULL ) )
                        j -= 1;
                    if ( i < j ) {
                        atemp = e->angles[i];
                        e->angles[i] = e->angles[j];
                        e->angles[j] = atemp;
                        }
                    }
                nr_angles = i;
                }

            else if ( k == 2 ) {
                /* Sort the bonds. */
                i = 0; j = nr_bonds-1;
                while ( i < j ) {
                    while ( i < nr_bonds &&
                            s->partlist[e->bonds[i].i] != NULL &&
                            s->partlist[e->bonds[i].j] != NULL )
                        i += 1;
                    while ( j >= 0 &&
                            ( s->partlist[e->bonds[j].i] == NULL ||
                              s->partlist[e->bonds[j].j] == NULL ) )
                        j -= 1;
                    if ( i < j ) {
                        btemp = e->bonds[i];
                        e->bonds[i] = e->bonds[j];
                        e->bonds[j] = btemp;
                        }
                    }
                nr_bonds = i;
                }

            else if ( k == 3 ) {
                /* Sort the exclusions. */
                i = 0; j = nr_exclusions-1;
                while ( i < j ) {
                    while ( i < nr_exclusions &&
                            s->partlist[e->exclusions[i].i] != NULL &&
                            s->partlist[e->exclusions[i].j] != NULL )
                        i += 1;
                    while ( j >= 0 &&
                            ( s->partlist[e->exclusions[j].i] == NULL ||
                              s->partlist[e->exclusions[j].j] == NULL ) )
                        j -= 1;
                    if ( i < j ) {
                        etemp = e->exclusions[i];
                        e->exclusions[i] = e->exclusions[j];
                        e->exclusions[j] = etemp;
                        }
                    }
                nr_exclusions = i;
                }
        
            }
            
        /* Stop the clock. */
        e->timers[engine_timer_bonded_sort] += getticks() - tic;
        
        } /* If in parallel... */
        

    #ifdef HAVE_OPENMP
    
        /* Is it worth parallelizing? */
        #pragma omp parallel private(scale,thread_id,nr_threads,epot_local,count,k,bonds,angles,dihedrals,exclusions)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) &&
             ( nr_bonds + nr_angles + nr_dihedrals ) > 0 ) {
             
            /* Init the local potential energy. */
            epot_local = 0;
             
            /* Get the thread ID. */
            thread_id = omp_get_thread_num();
            scale = ((double)nr_threads) / s->nr_real;
            
            /* Allocate and fill a buffer with the local bonds. */
            bonds = (struct bond *)malloc( sizeof(struct bond) * nr_bonds );
            for ( count = 0 , k = 0 ; k < nr_bonds ; k++ )
                if ( (int)(s->celllist[e->bonds[k].i]->id * scale) == thread_id ||
                     (int)(s->celllist[e->bonds[k].j]->id * scale) == thread_id )
                    bonds[ count++ ] = e->bonds[k];

            /* Compute the bonded interactions. */
            bond_eval_div( bonds , count , nr_threads , thread_id , e , &epot_local );
            
            /* Free the local bonds list. */
            free(bonds);
                    
            /* Allocate and fill a buffer with the local angles. */
            angles = (struct angle *)malloc( sizeof(struct angle) * nr_angles );
            for ( count = 0 , k = 0 ; k < nr_angles ; k++ )
                if ( (int)(s->celllist[e->angles[k].i]->id * scale) == thread_id ||
                     (int)(s->celllist[e->angles[k].j]->id * scale) == thread_id ||
                     (int)(s->celllist[e->angles[k].k]->id * scale) == thread_id )
                    angles[ count++ ] = e->angles[k];

            /* Compute the angle interactions. */
            angle_eval_div( angles , count , nr_threads , thread_id , e , &epot_local );
                    
            /* Free the local angles list. */
            free(angles);
                    
            /* Allocate and fill a buffer with the local dihedrals. */
            dihedrals = (struct dihedral *)malloc( sizeof(struct dihedral) * nr_dihedrals );
            for ( count = 0 , k = 0 ; k < nr_dihedrals ; k++ )
                if ( (int)(s->celllist[e->dihedrals[k].i]->id * scale) == thread_id ||
                     (int)(s->celllist[e->dihedrals[k].j]->id * scale) == thread_id ||
                     (int)(s->celllist[e->dihedrals[k].k]->id * scale) == thread_id ||
                     (int)(s->celllist[e->dihedrals[k].l]->id * scale) == thread_id )
                    dihedrals[ count++ ] = e->dihedrals[k];

            /* Compute the dihedral interactions. */
            dihedral_eval_div( dihedrals , count , nr_threads , thread_id , e , &epot_local );
                    
            /* Free the local dihedrals list. */
            free(dihedrals);
                    
            /* Allocate and fill a buffer with the local exclusions. */
            exclusions = (struct exclusion *)malloc( sizeof(struct exclusion) * nr_exclusions );
            for ( count = 0 , k = 0 ; k < nr_exclusions ; k++ )
                if ( (int)(s->celllist[e->exclusions[k].i]->id * scale) == thread_id ||
                     (int)(s->celllist[e->exclusions[k].j]->id * scale) == thread_id )
                    exclusions[ count++ ] = e->exclusions[k];

            /* Correct for excluded interactons. */
            exclusion_eval_div( exclusions , count , nr_threads , thread_id , e , &epot_local );
              
            /* Free the local exclusions list. */
            free(exclusions);
                    
            /* Aggregate the global potential energy. */
            #pragma omp atomic
            epot += epot_local;
                    
            }
            
        /* Otherwise, evaluate directly. */
        else if ( omp_get_thread_num() == 0 ) {
        
            /* Do bonds. */
            tic = getticks();
            bond_eval( e->bonds , nr_bonds , e , &epot );
            e->timers[engine_timer_bonds] += getticks() - tic;
            
            /* Do angles. */
            tic = getticks();
            angle_eval( e->angles , nr_angles , e , &epot );
            e->timers[engine_timer_angles] += getticks() - tic;
            
            /* Do dihedrals. */
            tic = getticks();
            dihedral_eval( e->dihedrals , nr_dihedrals , e , &epot );
            e->timers[engine_timer_dihedrals] += getticks() - tic;
            
            /* Do exclusions. */
            tic = getticks();
            exclusion_eval( e->exclusions , nr_exclusions , e , &epot );
            e->timers[engine_timer_exclusions] += getticks() - tic;
            
            }
            
    #else
    
        /* Do bonds. */
        tic = getticks();
        if ( bond_eval( e->bonds , nr_bonds , e , &epot ) < 0 )
            return error(engine_err_bond);
        e->timers[engine_timer_bonds] += getticks() - tic;
            
        /* Do angles. */
        tic = getticks();
        if ( angle_eval( e->angles , nr_angles , e , &epot ) < 0 )
            return error(engine_err_angle);
        e->timers[engine_timer_angles] += getticks() - tic;
            
        /* Do dihedrals. */
        tic = getticks();
        if ( dihedral_eval( e->dihedrals , nr_dihedrals , e , &epot ) < 0 )
            return error(engine_err_dihedral);
        e->timers[engine_timer_dihedrals] += getticks() - tic;
            
        /* Do exclusions. */
        tic = getticks();
        if ( exclusion_eval( e->exclusions , nr_exclusions , e , &epot ) < 0 )
            return error(engine_err_exclusion);
        e->timers[engine_timer_exclusions] += getticks() - tic;
            
    #endif
        
    /* Store the potential energy. */
    s->epot += epot;
    
    /* I'll be back... */
    return engine_err_ok;

    }


/**
 * @brief Compute the dihedral interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_dihedral_eval ( struct engine *e ) {

    double epot = 0.0;
    struct space *s;
    struct dihedral temp;
    int nr_dihedrals = e->nr_dihedrals, i, j;
    #ifdef HAVE_OPENMP
        FPTYPE *eff;
        int nr_threads, cid, pid, gpid, k;
        struct part *p;
        struct cell *c;
        double epot_local;
    #endif
    
    /* Get a handle on the space. */
    s = &e->s;

    /* Sort the dihedrals (if in parallel). */
    if ( e->nr_nodes > 1 ) {
        i = 0; j = nr_dihedrals-1;
        while ( i < j ) {
            while ( i < nr_dihedrals &&
                    s->partlist[e->dihedrals[i].i] != NULL &&
                    s->partlist[e->dihedrals[i].j] != NULL &&
                    s->partlist[e->dihedrals[i].k] != NULL &&
                    s->partlist[e->dihedrals[i].l] != NULL )
                i += 1;
            while ( j >= 0 &&
                    ( s->partlist[e->dihedrals[j].i] == NULL ||
                      s->partlist[e->dihedrals[j].j] == NULL ||
                      s->partlist[e->dihedrals[j].k] == NULL ||
                      s->partlist[e->dihedrals[j].l] == NULL ) )
                j -= 1;
            if ( i < j ) {
                temp = e->dihedrals[i];
                e->dihedrals[i] = e->dihedrals[j];
                e->dihedrals[j] = temp;
                }
            }
        nr_dihedrals = i;
        }

    #ifdef HAVE_OPENMP
    
        /* Is it worth parallelizing? */
        #pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff,epot_local)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) && 
             ( nr_dihedrals > engine_dihedrals_chunk ) ) {
    
            /* Init the local potential energy. */
            epot_local = 0.0;
            
            /* Allocate a buffer for the forces. */
            eff = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * s->nr_parts );
            bzero( eff , sizeof(FPTYPE) * 4 * s->nr_parts );

            /* Compute the dihedral interactions. */
            k = omp_get_thread_num();
            dihedral_evalf( &e->dihedrals[k*nr_dihedrals/nr_threads] , (k+1)*nr_dihedrals/nr_threads - k*nr_dihedrals/nr_threads , e , eff , &epot_local );
                    
            /* Write-back the forces (if anything was done). */
            for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
                c = &s->cells[ s->cid_real[cid] ];
                pthread_mutex_lock( &c->cell_mutex );
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &c->parts[ pid ];
                    gpid = p->id;
                    for ( k = 0 ; k < 3 ; k++ )
                        p->f[k] += eff[ gpid*4 + k ];
                    }
                pthread_mutex_unlock( &c->cell_mutex );
                }
            free( eff );
            
            /* Aggregate the global potential energy. */
            #pragma omp atomic
            epot += epot_local;
                
            }
            
        /* Otherwise, evaluate directly. */
        else if ( omp_get_thread_num() == 0 )
            dihedral_eval( e->dihedrals , nr_dihedrals , e , &epot );
    #else
        if ( dihedral_eval( e->dihedrals , nr_dihedrals , e , &epot ) < 0 )
            return error(engine_err_dihedral);
    #endif
        
    /* Store the potential energy. */
    s->epot += epot;
    
    /* I'll be back... */
    return engine_err_ok;

    }


/**
 * @brief Compute the angled interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_angle_eval ( struct engine *e ) {

    double epot = 0.0;
    struct space *s;
    struct angle temp;
    int nr_angles = e->nr_angles, i, j;
    #ifdef HAVE_OPENMP
        FPTYPE *eff;
        int nr_threads, cid, pid, gpid, k;
        struct part *p;
        struct cell *c;
        double epot_local;
    #endif
    
    /* Get a handle on the space. */
    s = &e->s;

    /* Sort the angles (if in parallel). */
    if ( e->nr_nodes > 1 ) {
        i = 0; j = nr_angles-1;
        while ( i < j ) {
            while ( i < nr_angles &&
                    s->partlist[e->angles[i].i] != NULL &&
                    s->partlist[e->angles[i].j] != NULL &&
                    s->partlist[e->angles[i].k] != NULL )
                i += 1;
            while ( j >= 0 &&
                    ( s->partlist[e->angles[j].i] == NULL ||
                      s->partlist[e->angles[j].j] == NULL ||
                      s->partlist[e->angles[j].k] == NULL ) )
                j -= 1;
            if ( i < j ) {
                temp = e->angles[i];
                e->angles[i] = e->angles[j];
                e->angles[j] = temp;
                }
            }
        nr_angles = i;
        }

    #ifdef HAVE_OPENMP
    
        /* Is it worth parallelizing? */
        #pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff,epot_local)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) && 
             ( nr_angles > engine_angles_chunk ) ) {
    
            /* Init the local potential energy. */
            epot_local = 0.0;
            
            /* Allocate a buffer for the forces. */
            eff = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * s->nr_parts );
            bzero( eff , sizeof(FPTYPE) * 4 * s->nr_parts );

            /* Compute the angle interactions. */
            k = omp_get_thread_num();
            angle_evalf( &e->angles[k*nr_angles/nr_threads] , (k+1)*nr_angles/nr_threads - k*nr_angles/nr_threads , e , eff , &epot_local );
                    
            /* Write-back the forces (if anything was done). */
            for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
                c = &s->cells[ s->cid_real[cid] ];
                pthread_mutex_lock( &c->cell_mutex );
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &c->parts[ pid ];
                    gpid = p->id;
                    for ( k = 0 ; k < 3 ; k++ )
                        p->f[k] += eff[ gpid*4 + k ];
                    }
                pthread_mutex_unlock( &c->cell_mutex );
                }
            free( eff );
                
            /* Aggregate the global potential energy. */
            #pragma omp atomic
            epot += epot_local;
                
            }
            
        /* Otherwise, evaluate directly. */
        else if ( omp_get_thread_num() == 0 )
            angle_eval( e->angles , nr_angles , e , &epot );
    #else
        if ( angle_eval( e->angles , nr_angles , e , &epot ) < 0 )
            return error(engine_err_angle);
    #endif
        
    /* Store the potential energy. */
    s->epot += epot;
    
    /* I'll be back... */
    return engine_err_ok;

    }


/**
 * @brief Correct for the excluded interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_exclusion_eval ( struct engine *e ) {

    double epot = 0.0;
    struct space *s;
    int nr_exclusions = e->nr_exclusions, i, j;
    struct exclusion temp;
    #ifdef HAVE_OPENMP
        int nr_threads, thread_id;
        double epot_local;
    #endif
    
    /* Get a handle on the space. */
    s = &e->s;
    
    /* Sort the exclusions (if in parallel). */
    if ( e->nr_nodes > 1 ) {
        i = 0; j = nr_exclusions-1;
        while ( i < j ) {
            while ( i < nr_exclusions &&
                    s->partlist[e->exclusions[i].i] != NULL &&
                    s->partlist[e->exclusions[i].j] != NULL )
                i += 1;
            while ( j >= 0 &&
                    ( s->partlist[e->exclusions[j].i] == NULL ||
                      s->partlist[e->exclusions[j].j] == NULL ) )
                j -= 1;
            if ( i < j ) {
                temp = e->exclusions[i];
                e->exclusions[i] = e->exclusions[j];
                e->exclusions[j] = temp;
                }
            }
        nr_exclusions = i;
        }

    #ifdef HAVE_OPENMP
    
        /* Is it worth parallelizing? */
        #pragma omp parallel private(thread_id,nr_threads,epot_local)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) &&
             ( nr_exclusions > 0 ) ) {
             
            /* Init the local potential energy. */
            epot_local = 0.0;
            
            /* Get the thread ID. */
            thread_id = omp_get_thread_num();

            /* Correct for excluded interactons. */
            exclusion_eval_mod( e->exclusions , nr_exclusions , nr_threads , thread_id , e , &epot_local );
                    
            /* Aggregate the global potential energy. */
            #pragma omp atomic
            epot += epot_local;
                
            }
            
        /* Otherwise, evaluate directly. */
        else if ( omp_get_thread_num() == 0 )
            exclusion_eval( e->exclusions , nr_exclusions , e , &epot );
    #else
        if ( exclusion_eval( e->exclusions , nr_exclusions , e , &epot ) < 0 )
            return error(engine_err_exclusion);
    #endif
        
    /* Store the potential energy. */
    s->epot += epot;
    
    /* I'll be back... */
    return engine_err_ok;

    }


/**
 * @brief Compute the bonded interactions stored in this engine.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_bond_eval ( struct engine *e ) {

    double epot = 0.0;
    struct space *s;
    int nr_bonds = e->nr_bonds, i, j;
    struct bond temp;
    #ifdef HAVE_OPENMP
        FPTYPE *eff;
        int nr_threads, cid, pid, gpid, k;
        struct part *p;
        struct cell *c;
        double epot_local;
    #endif
    
    /* Get a handle on the space. */
    s = &e->s;
    
    /* Sort the bonds (if in parallel). */
    if ( e->nr_nodes > 1 ) {
        i = 0; j = nr_bonds-1;
        while ( i < j ) {
            while ( i < nr_bonds &&
                    s->partlist[e->bonds[i].i] != NULL &&
                    s->partlist[e->bonds[i].j] != NULL )
                i += 1;
            while ( j >= 0 &&
                    ( s->partlist[e->bonds[j].i] == NULL ||
                      s->partlist[e->bonds[j].j] == NULL ) )
                j -= 1;
            if ( i < j ) {
                temp = e->bonds[i];
                e->bonds[i] = e->bonds[j];
                e->bonds[j] = temp;
                }
            }
        nr_bonds = i;
        }

    #ifdef HAVE_OPENMP
    
        /* Is it worth parallelizing? */
        #pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff,epot_local)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) && 
             ( nr_bonds > engine_bonds_chunk ) ) {
    
            /* Init the local potential energy. */
            epot_local = 0.0;
            
            /* Allocate a buffer for the forces. */
            eff = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * s->nr_parts );
            bzero( eff , sizeof(FPTYPE) * 4 * s->nr_parts );

            /* Compute the bonded interactions. */
            k = omp_get_thread_num();
            bond_evalf( &e->bonds[k*nr_bonds/nr_threads] , (k+1)*nr_bonds/nr_threads - k*nr_bonds/nr_threads , e , eff , &epot_local );
                    
            /* Write-back the forces (if anything was done). */
            for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
                c = &s->cells[ s->cid_real[cid] ];
                pthread_mutex_lock( &c->cell_mutex );
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &c->parts[ pid ];
                    gpid = p->id;
                    for ( k = 0 ; k < 3 ; k++ )
                        p->f[k] += eff[ gpid*4 + k ];
                    }
                pthread_mutex_unlock( &c->cell_mutex );
                }
            free( eff );
                
            /* Aggregate the global potential energy. */
            #pragma omp atomic
            epot += epot_local;
                
            }
            
        /* Otherwise, evaluate directly. */
        else if ( omp_get_thread_num() == 0 )
            bond_eval( e->bonds , nr_bonds , e , &epot );
    #else
        if ( bond_eval( e->bonds , nr_bonds , e , &epot ) < 0 )
            return error(engine_err_bond);
    #endif
        
    /* Store the potential energy. */
    s->epot += epot;
    
    /* I'll be back... */
    return engine_err_ok;

    }


/**
 * @brief Add a bond potential.
 *
 * @param e The #engine.
 * @param p The #potential to add to the #engine.
 * @param i ID of particle type for this interaction.
 * @param j ID of second particle type for this interaction.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Adds the given bonded potential for pairs of particles of type @c i and @c j,
 * where @c i and @c j may be the same type ID.
 */
 
int engine_bond_addpot ( struct engine *e , struct potential *p , int i , int j ) {

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( i < 0 || i >= e->max_type || j < 0 || j >= e->max_type )
        return error(engine_err_range);
        
    /* store the potential. */
    e->p_bond[ i * e->max_type + j ] = p;
    if ( i != j )
        e->p_bond[ j * e->max_type + i ] = p;
        
    /* end on a good note. */
    return engine_err_ok;

    }


/**
 * @brief Add a dihedral potential.
 *
 * @param e The #engine.
 * @param p The #potential to add to the #engine.
 *
 * @return The ID of the added dihedral potential or < 0 on error (see #engine_err).
 */
 
int engine_dihedral_addpot ( struct engine *e , struct potential *p ) {

    struct potential **dummy;

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Is there enough room in p_dihedral? */
    if ( e->nr_dihedralpots == e->dihedralpots_size ) {
        e->dihedralpots_size += 100;
        if ( ( dummy = (struct potential **)malloc( sizeof(struct potential *) * e->dihedralpots_size ) ) == NULL )
            return engine_err_malloc;
        memcpy( dummy , e->p_dihedral , sizeof(struct potential *) * e->nr_dihedralpots );
        free( e->p_dihedral );
        e->p_dihedral = dummy;
        }
        
    /* store the potential. */
    e->p_dihedral[ e->nr_dihedralpots ] = p;
    e->nr_dihedralpots += 1;
        
    /* end on a good note. */
    return e->nr_dihedralpots - 1;

    }


/**
 * @brief Add a angle potential.
 *
 * @param e The #engine.
 * @param p The #potential to add to the #engine.
 *
 * @return The ID of the added angle potential or < 0 on error (see #engine_err).
 */
 
int engine_angle_addpot ( struct engine *e , struct potential *p ) {

    struct potential **dummy;

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Is there enough room in p_angle? */
    if ( e->nr_anglepots == e->anglepots_size ) {
        e->anglepots_size += 100;
        if ( ( dummy = (struct potential **)malloc( sizeof(struct potential *) * e->anglepots_size ) ) == NULL )
            return engine_err_malloc;
        memcpy( dummy , e->p_angle , sizeof(struct potential *) * e->nr_anglepots );
        free( e->p_angle );
        e->p_angle = dummy;
        }
        
    /* store the potential. */
    e->p_angle[ e->nr_anglepots ] = p;
    e->nr_anglepots += 1;
        
    /* end on a good note. */
    return e->nr_anglepots - 1;

    }
    
    
