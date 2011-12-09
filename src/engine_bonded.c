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
 
int engine_bonded_eval_sets ( struct engine *e ) {

    double epot = 0.0;
    #ifdef HAVE_OPENMP
        int sets_taboo[ e->nr_sets];
        int k, j, set_curr, sets_next = 0, sets_ind[ e->nr_sets ];
        double epot_local;
        ticks toc_bonds, toc_angles, toc_dihedrals, toc_exclusions;
    #endif
    ticks tic;
    
    #ifdef HAVE_OPENMP
    
        /* Fill the indices. */
        for ( k = 0 ; k < e->nr_sets ; k++ ) {
            sets_ind[k] = k;
            sets_taboo[k] = 0;
            }
    
        #pragma omp parallel private(k,j,set_curr,epot_local,toc_bonds,toc_angles,toc_dihedrals,toc_exclusions)
        if ( e->nr_sets > 0 && omp_get_num_threads() > 1 ) {
        
            /* Init local counters. */
            toc_bonds = 0; toc_angles = 0; toc_dihedrals = 0; toc_exclusions = 0;
            epot_local = 0.0; set_curr = -1;
            
            /* Main loop. */
            while ( sets_next < e->nr_sets ) {
            
                /* Try to grab a set. */
                set_curr = -1;
                #pragma omp critical (setlist)
                while ( sets_next < e->nr_sets ) {
                
                    /* Find the next free set. */
                    for ( k = sets_next ; k < e->nr_sets && sets_taboo[ sets_ind[k] ] ; k++ );
                    
                    /* If a set was found... */
                    if ( k < e->nr_sets ) {
                    
                        /* Swap it to the top and put a finger on it. */
                        set_curr = sets_ind[k];
                        sets_ind[k] = sets_ind[sets_next];
                        sets_ind[sets_next] = set_curr;
                        sets_next += 1;
                        
                        /* Mark conflicting sets as taboo. */
                        #pragma omp critical (taboo)
                        for ( j = 0 ; j < e->sets[set_curr].nr_confl ; j++ )
                            sets_taboo[ e->sets[set_curr].confl[j] ] += 1;
                            
                        /* And exit the loop. */
                        break;
                        
                        }
                        
                    }
                    
                /* Did we even get a set? */
                if ( set_curr < 0 )
                    break;
                    
                /* Evaluate the bonded interaction in the set. */
                /* Do bonds. */
                tic = getticks();
                bond_eval( e->sets[set_curr].bonds , e->sets[set_curr].nr_bonds , e , &epot_local );
                toc_bonds += getticks() - tic;

                /* Do angles. */
                tic = getticks();
                angle_eval( e->sets[set_curr].angles , e->sets[set_curr].nr_angles , e , &epot_local );
                toc_angles += getticks() - tic;

                /* Do dihedrals. */
                tic = getticks();
                dihedral_eval( e->sets[set_curr].dihedrals , e->sets[set_curr].nr_dihedrals , e , &epot_local );
                toc_dihedrals += getticks() - tic;

                /* Do exclusions. */
                tic = getticks();
                exclusion_eval( e->sets[set_curr].exclusions , e->sets[set_curr].nr_exclusions , e , &epot_local );
                toc_exclusions += getticks() - tic;
                
                /* Un-mark conflicting sets. */
                #pragma omp critical (taboo)
                for ( k = 0 ; k < e->sets[set_curr].nr_confl ; k++ )
                    sets_taboo[ e->sets[set_curr].confl[k] ] -= 1;
                   
                } /* main loop. */
        
            /* Write-back global data. */
            #pragma omp critical (writeback)
            {
                e->timers[engine_timer_bonds] += toc_bonds;
                e->timers[engine_timer_angles] += toc_angles;
                e->timers[engine_timer_dihedrals] += toc_dihedrals;
                e->timers[engine_timer_exclusions] += toc_exclusions;
                epot += epot_local;
                }
            
            }
    
        /* Otherwise, just do the sequential thing. */
        else {
        
            tic = getticks();
            bond_eval( e->bonds , e->nr_bonds , e , &epot );
            e->timers[engine_timer_bonds] += getticks() - tic;

            /* Do angles. */
            tic = getticks();
            angle_eval( e->angles , e->nr_angles , e , &epot );
            e->timers[engine_timer_angles] += getticks() - tic;

            /* Do dihedrals. */
            tic = getticks();
            dihedral_eval( e->dihedrals , e->nr_dihedrals , e , &epot );
            e->timers[engine_timer_dihedrals] += getticks() - tic;

            /* Do exclusions. */
            tic = getticks();
            exclusion_eval( e->exclusions , e->nr_exclusions , e , &epot );
            e->timers[engine_timer_exclusions] += getticks() - tic;
        
            }
    #else

        /* Do bonds. */
        tic = getticks();
        if ( bond_eval( e->bonds , e->nr_bonds , e , &epot ) < 0 )
            return error(engine_err_bond);
        e->timers[engine_timer_bonds] += getticks() - tic;
            
        /* Do angles. */
        tic = getticks();
        if ( angle_eval( e->angles , e->nr_angles , e , &epot ) < 0 )
            return error(engine_err_angle);
        e->timers[engine_timer_angles] += getticks() - tic;
            
        /* Do dihedrals. */
        tic = getticks();
        if ( dihedral_eval( e->dihedrals , e->nr_dihedrals , e , &epot ) < 0 )
            return error(engine_err_dihedral);
        e->timers[engine_timer_dihedrals] += getticks() - tic;
            
        /* Do exclusions. */
        tic = getticks();
        if ( exclusion_eval( e->exclusions , e->nr_exclusions , e , &epot ) < 0 )
            return error(engine_err_exclusion);
        e->timers[engine_timer_exclusions] += getticks() - tic;
            
    #endif
        
    /* Store the potential energy. */
    e->s.epot += epot;
    
    /* I'll be back... */
    return engine_err_ok;

    }


/**
 * @brief Assemble non-conflicting sets of bonded interactions.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_bonded_sets ( struct engine *e , int max_sets ) {

    struct {
        int i, j;
        } *confl, *confl_sorted, temp;
    int *confl_index, confl_size, confl_count = 0;
    int *nconfl, *weight;
    int *setid_bonds, *setid_angles, *setid_dihedrals, *setid_exclusions, *setid_rigids;
    int nr_sets;
    int i, jj , j , k, min_i, min_j, min_weight, max_weight, max_confl, nr_confl;
    double avg_nconfl, avg_weight, tot_weight;
    char *confl_counts;
    
    /* Function to add a conflict. */
    int confl_add ( int i , int j ) {
        if ( confl_count == confl_size && ( confl = realloc( confl , sizeof(int) * 2 * (confl_size *= 2) ) ) == NULL )
            return error(engine_err_malloc);
        confl[confl_count].i = i; confl[confl_count].j = j;
        nconfl[i] += 1; nconfl[j] += 1;
        confl_count += 1;
        return engine_err_ok;
        }
        
    /* Recursive quicksort for the conflicts. */
    void confl_qsort ( int l , int r ) {
        
        int i = l, j = r;
        int pivot_i = confl_sorted[ (l + r)/2 ].i;
        
        /* Too small? */
        if ( r - l < 10 ) {
        
            /* Use Insertion Sort. */
            for ( i = l+1 ; i <= r ; i++ ) {
                pivot_i = confl_sorted[i].i;
                for ( j = i-1 ; j >= l ; j-- )
                    if ( confl_sorted[j].i > pivot_i ) {
                        temp = confl_sorted[j];
                        confl_sorted[j] = confl_sorted[j+1];
                        confl_sorted[j+1] = temp;
                        }
                    else
                        break;
                }
        
            }
            
        else {
        
            /* Partition. */
            while ( i <= j ) {
                while ( confl_sorted[i].i < pivot_i )
                    i += 1;
                while ( confl_sorted[j].i > pivot_i )
                    j -= 1;
                if ( i <= j ) {
                    temp = confl_sorted[i];
                    confl_sorted[i] = confl_sorted[j];
                    confl_sorted[j] = temp;
                    i += 1;
                    j -= 1;
                    }
                }

            /* Recurse. */
            if ( l < j )
                confl_qsort( l , j );
            if ( i < r )
                confl_qsort( i , r );
                
            }
        
        }
        
        
    /* Start with one set per bonded interaction. */
    nr_sets = e->nr_bonds + e->nr_angles + e->nr_dihedrals + e->nr_exclusions + e->nr_rigids;
    tot_weight = e->nr_bonds + 2*e->nr_angles + 3*e->nr_dihedrals + e->nr_exclusions;
    if ( ( weight = (int *)malloc( sizeof(int) * nr_sets ) ) == NULL ||
         ( nconfl = (int *)calloc( nr_sets , sizeof(int) ) ) == NULL ||
         ( confl_counts = (char *)malloc( sizeof(char) * nr_sets ) ) == NULL )
        return error(engine_err_malloc);
        
    /* Allocate the initial setids. */
    if ( ( setid_bonds = (int *)malloc( sizeof(int) * e->nr_bonds ) ) == NULL ||
         ( setid_angles = (int *)malloc( sizeof(int) * e->nr_angles ) ) == NULL ||
         ( setid_dihedrals = (int *)malloc( sizeof(int) * e->nr_dihedrals ) ) == NULL ||
         ( setid_exclusions = (int *)malloc( sizeof(int) * e->nr_exclusions ) ) == NULL ||
         ( setid_rigids = (int *)malloc( sizeof(int) * e->nr_rigids ) ) == NULL )
        return error(engine_err_malloc);
        
    /* Generate the set of conflicts. */
    confl_size = nr_sets;
    if ( ( confl = malloc( sizeof(int) * 2 * confl_size ) ) == NULL )
        return error(engine_err_malloc);
    nr_sets = 0;
        
    /* Loop over all dihedrals. */
    for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {
    
        /* This dihedral gets its own id. */
        weight[ nr_sets ] = 3;
        setid_dihedrals[k] = nr_sets++;
                 
        /* Loop over other dihedrals... */
        for ( j = 0 ; j < k ; j++ )
            if ( e->dihedrals[k].i == e->dihedrals[j].i || e->dihedrals[k].i == e->dihedrals[j].j || e->dihedrals[k].i == e->dihedrals[j].k || e->dihedrals[k].i == e->dihedrals[j].l ||
                 e->dihedrals[k].j == e->dihedrals[j].i || e->dihedrals[k].j == e->dihedrals[j].j || e->dihedrals[k].j == e->dihedrals[j].k || e->dihedrals[k].j == e->dihedrals[j].l ||
                 e->dihedrals[k].k == e->dihedrals[j].i || e->dihedrals[k].k == e->dihedrals[j].j || e->dihedrals[k].k == e->dihedrals[j].k || e->dihedrals[k].k == e->dihedrals[j].l ||
                 e->dihedrals[k].l == e->dihedrals[j].i || e->dihedrals[k].l == e->dihedrals[j].j || e->dihedrals[k].l == e->dihedrals[j].k || e->dihedrals[k].l == e->dihedrals[j].l )
                if ( confl_add( setid_dihedrals[k] , setid_dihedrals[j] ) < 0 )
                    return error(engine_err);
    
        } /* Loop over dihedrals. */

    /* Loop over all angles. */
    for ( k = 0 ; k < e->nr_angles ; k++ ) {
    
        /* Loop over dihedrals, looking for matches... */
        for ( j = 0 ; j < e->nr_dihedrals ; j++ )
            if ( ( e->angles[k].i == e->dihedrals[j].i && e->angles[k].j == e->dihedrals[j].j && e->angles[k].k == e->dihedrals[j].k ) ||
                 ( e->angles[k].i == e->dihedrals[j].j && e->angles[k].j == e->dihedrals[j].k && e->angles[k].k == e->dihedrals[j].l ) ||
                 ( e->angles[k].k == e->dihedrals[j].j && e->angles[k].j == e->dihedrals[j].k && e->angles[k].i == e->dihedrals[j].l ) ||
                 ( e->angles[k].k == e->dihedrals[j].i && e->angles[k].j == e->dihedrals[j].j && e->angles[k].i == e->dihedrals[j].k ) ) {
                setid_angles[k] = -setid_dihedrals[j];
                weight[ setid_dihedrals[j] ] += 2;
                break;
                }
                 
        /* Does this angle get its own id? */
        if ( j < e->nr_dihedrals)
            continue;
        else {
            weight[ nr_sets ] = 2;
            setid_angles[k] = nr_sets++;
            }
                 
        /* Loop over dihedrals, looking for conflicts... */
        for ( j = 0 ; j < e->nr_dihedrals ; j++ )
            if ( e->angles[k].i == e->dihedrals[j].i || e->angles[k].i == e->dihedrals[j].j || e->angles[k].i == e->dihedrals[j].k || e->angles[k].i == e->dihedrals[j].l ||
                 e->angles[k].j == e->dihedrals[j].i || e->angles[k].j == e->dihedrals[j].j || e->angles[k].j == e->dihedrals[j].k || e->angles[k].j == e->dihedrals[j].l ||
                 e->angles[k].k == e->dihedrals[j].i || e->angles[k].k == e->dihedrals[j].j || e->angles[k].k == e->dihedrals[j].k || e->angles[k].k == e->dihedrals[j].l )
                if ( confl_add( setid_angles[k] , setid_dihedrals[j] ) < 0 )
                    return error(engine_err);
    
        /* Loop over previous angles... */
        for ( j = 0 ; j < k ; j++ )
            if ( setid_angles[j] >= 0 &&
                 ( e->angles[k].i == e->angles[j].i || e->angles[k].i == e->angles[j].j || e->angles[k].i == e->angles[j].k ||
                   e->angles[k].j == e->angles[j].i || e->angles[k].j == e->angles[j].j || e->angles[k].j == e->angles[j].k ||
                   e->angles[k].k == e->angles[j].i || e->angles[k].k == e->angles[j].j || e->angles[k].k == e->angles[j].k ) )
                if ( confl_add( setid_angles[k] , setid_angles[j] ) < 0 )
                    return error(engine_err);
    
        } /* Loop over angles. */

    /* Loop over all bonds. */
    for ( k = 0 ; k < e->nr_bonds ; k++ ) {
    
        /* Loop over dihedrals, looking for overlap... */
        for ( j = 0 ; j < e->nr_dihedrals ; j++ )
            if ( ( e->bonds[k].i == e->dihedrals[j].i && e->bonds[k].j == e->dihedrals[j].j ) ||
                 ( e->bonds[k].j == e->dihedrals[j].i && e->bonds[k].i == e->dihedrals[j].j ) ||
                 ( e->bonds[k].i == e->dihedrals[j].j && e->bonds[k].j == e->dihedrals[j].k ) ||
                 ( e->bonds[k].j == e->dihedrals[j].j && e->bonds[k].i == e->dihedrals[j].k ) ||
                 ( e->bonds[k].i == e->dihedrals[j].k && e->bonds[k].j == e->dihedrals[j].l ) ||
                 ( e->bonds[k].j == e->dihedrals[j].k && e->bonds[k].i == e->dihedrals[j].l ) ) {
                setid_bonds[k] = -setid_dihedrals[j];
                weight[ setid_dihedrals[j] ] += 1;
                break;
                }
    
        /* Does this bond get its own id? */
        if ( j < e->nr_dihedrals)
            continue;
                 
        /* Loop over angles, looking for overlap... */
        for ( j = 0 ; j < e->nr_angles ; j++ )
            if ( setid_angles[j] >= 0 &&
                 ( ( e->bonds[k].i == e->angles[j].i && e->bonds[k].j == e->angles[j].j ) ||
                   ( e->bonds[k].j == e->angles[j].i && e->bonds[k].i == e->angles[j].j ) ||
                   ( e->bonds[k].i == e->angles[j].j && e->bonds[k].j == e->angles[j].k ) ||
                   ( e->bonds[k].j == e->angles[j].j && e->bonds[k].i == e->angles[j].k ) ) ) {
                setid_bonds[k] = -setid_angles[j];
                weight[ setid_angles[j] ] += 1;
                break;
                }

        /* Does this bond get its own id? */
        if ( j < e->nr_angles)
            continue;
        else {
            weight[ nr_sets ] = 1;
            setid_bonds[k] = nr_sets++;
            }
                 
        /* Loop over dihedrals... */
        for ( j = 0 ; j < e->nr_dihedrals ; j++ )
            if ( e->bonds[k].i == e->dihedrals[j].i || e->bonds[k].i == e->dihedrals[j].j || e->bonds[k].i == e->dihedrals[j].k || e->bonds[k].i == e->dihedrals[j].l ||
                 e->bonds[k].j == e->dihedrals[j].i || e->bonds[k].j == e->dihedrals[j].j || e->bonds[k].j == e->dihedrals[j].k || e->bonds[k].j == e->dihedrals[j].l )
                if ( confl_add( setid_bonds[k] , setid_dihedrals[j] ) < 0 )
                    return error(engine_err);

        /* Loop over angles... */
        for ( j = 0 ; j < e->nr_angles ; j++ )
            if ( setid_angles[j] >= 0 &&
                 ( e->bonds[k].i == e->angles[j].i || e->bonds[k].i == e->angles[j].j || e->bonds[k].i == e->angles[j].k ||
                   e->bonds[k].j == e->angles[j].i || e->bonds[k].j == e->angles[j].j || e->bonds[k].j == e->angles[j].k ) )
                if ( confl_add( setid_bonds[k] , setid_angles[j] ) < 0 )
                    return error(engine_err);
                    
        /* Loop over previous bonds... */
        for ( j = 0 ; j < k ; j++ )
            if ( setid_bonds[j] >= 0 &&
                 ( e->bonds[k].i == e->bonds[j].i || e->bonds[k].i == e->bonds[j].j ||
                   e->bonds[k].j == e->bonds[j].i || e->bonds[k].j == e->bonds[j].j ) )
                if ( confl_add( setid_bonds[k] , setid_bonds[j] ) < 0 )
                    return error(engine_err);
    
        } /* Loop over bonds. */
        
    /* Blindly add all the rigids as sets. */
    for ( k = 0 ; k < e->nr_rigids ; k++ ) {
    
        /* Add this rigid as a set. */
        weight[ nr_sets ] = 0;
        setid_rigids[k] = nr_sets++;
        
        /* Loop over dihedrals, looking for overlap. */
        for ( j = 0 ; j < e->nr_dihedrals ; j++ ) {
            for ( i = 0 ; i < e->rigids[k].nr_parts; i ++ )
                if ( e->rigids[k].parts[i] == e->dihedrals[j].i || e->rigids[k].parts[i] == e->dihedrals[j].j || e->rigids[k].parts[i] == e->dihedrals[j].k || e->rigids[k].parts[i] == e->dihedrals[j].l )
                    break;
            if ( i < e->rigids[k].nr_parts && confl_add( setid_rigids[k] , setid_dihedrals[j] ) )
                return error(engine_err);
            }
        
        /* Loop over angles, looking for overlap. */
        for ( j = 0 ; j < e->nr_angles ; j++ ) {
            if ( setid_angles[j] < 0 )
                continue;
            for ( i = 0 ; i < e->rigids[k].nr_parts; i ++ )
                if ( e->rigids[k].parts[i] == e->angles[j].i || e->rigids[k].parts[i] == e->angles[j].j || e->rigids[k].parts[i] == e->angles[j].k )
                    break;
            if ( i < e->rigids[k].nr_parts && confl_add( setid_rigids[k] , setid_angles[j] ) )
                return error(engine_err);
            }
        
        /* Loop over bonds, looking for overlap. */
        for ( j = 0 ; j < e->nr_bonds ; j++ ) {
            if ( setid_bonds[j] < 0 )
                continue;
            for ( i = 0 ; i < e->rigids[k].nr_parts; i ++ )
                if ( e->rigids[k].parts[i] == e->bonds[j].i || e->rigids[k].parts[i] == e->bonds[j].j )
                    break;
            if ( i < e->rigids[k].nr_parts && confl_add( setid_rigids[k] , setid_bonds[j] ) )
                return error(engine_err);
            }
        
        }

    /* Loop over all exclusions. */
    for ( k = 0 ; k < e->nr_exclusions ; k++ ) {
    
        /* Loop over rigids, looking for overlap. */
        for ( j = 0 ; j < e->nr_rigids ; j++ ) {
            for ( i = 0 ; i < e->rigids[j].nr_constr ; i++ )
                if ( ( e->exclusions[k].i == e->rigids[j].parts[ e->rigids[j].constr[i].i ] && e->exclusions[k].j == e->rigids[j].parts[ e->rigids[j].constr[i].j ] ) ||
                     ( e->exclusions[k].j == e->rigids[j].parts[ e->rigids[j].constr[i].i ] && e->exclusions[k].i == e->rigids[j].parts[ e->rigids[j].constr[i].j ] ) )
                    break;
            if ( i < e->rigids[j].nr_constr ) {
                setid_exclusions[k] = -setid_rigids[j];
                weight[ setid_rigids[j] ] += 1;
                break;
                }
            }
  
        /* Does this bond get its own id? */
        if ( j < e->nr_rigids )
            continue;
                 
        /* Loop over dihedrals, looking for overlap... */
        for ( j = 0 ; j < e->nr_dihedrals ; j++ )
            if ( ( e->exclusions[k].i == e->dihedrals[j].i && e->exclusions[k].j == e->dihedrals[j].j ) ||
                 ( e->exclusions[k].j == e->dihedrals[j].i && e->exclusions[k].i == e->dihedrals[j].j ) ||
                 ( e->exclusions[k].i == e->dihedrals[j].j && e->exclusions[k].j == e->dihedrals[j].k ) ||
                 ( e->exclusions[k].j == e->dihedrals[j].j && e->exclusions[k].i == e->dihedrals[j].k ) ||
                 ( e->exclusions[k].i == e->dihedrals[j].k && e->exclusions[k].j == e->dihedrals[j].l ) ||
                 ( e->exclusions[k].j == e->dihedrals[j].k && e->exclusions[k].i == e->dihedrals[j].l ) ||
                 ( e->exclusions[k].i == e->dihedrals[j].i && e->exclusions[k].j == e->dihedrals[j].k ) ||
                 ( e->exclusions[k].j == e->dihedrals[j].i && e->exclusions[k].i == e->dihedrals[j].k ) ||
                 ( e->exclusions[k].i == e->dihedrals[j].j && e->exclusions[k].j == e->dihedrals[j].l ) ||
                 ( e->exclusions[k].j == e->dihedrals[j].j && e->exclusions[k].i == e->dihedrals[j].l ) ||
                 ( e->exclusions[k].i == e->dihedrals[j].i && e->exclusions[k].j == e->dihedrals[j].l ) ||
                 ( e->exclusions[k].j == e->dihedrals[j].i && e->exclusions[k].i == e->dihedrals[j].l ) ) {
                setid_exclusions[k] = -setid_dihedrals[j];
                weight[ setid_dihedrals[j] ] += 1;
                break;
                }
    
        /* Does this bond get its own id? */
        if ( j < e->nr_dihedrals )
            continue;
                 
        /* Loop over angles, looking for overlap... */
        for ( j = 0 ; j < e->nr_angles ; j++ )
            if ( setid_angles[j] >= 0 &&
                 ( ( e->exclusions[k].i == e->angles[j].i && e->exclusions[k].j == e->angles[j].j ) ||
                   ( e->exclusions[k].j == e->angles[j].i && e->exclusions[k].i == e->angles[j].j ) ||
                   ( e->exclusions[k].i == e->angles[j].j && e->exclusions[k].j == e->angles[j].k ) ||
                   ( e->exclusions[k].j == e->angles[j].j && e->exclusions[k].i == e->angles[j].k ) ||
                   ( e->exclusions[k].i == e->angles[j].i && e->exclusions[k].j == e->angles[j].k ) ||
                   ( e->exclusions[k].j == e->angles[j].i && e->exclusions[k].i == e->angles[j].k ) ) ) {
                setid_exclusions[k] = -setid_angles[j];
                weight[ setid_angles[j] ] += 1;
                break;
                }

        /* Does this bond get its own id? */
        if ( j < e->nr_angles)
            continue;
                 
        /* Loop over bonds, looking for overlap... */
        for ( j = 0 ; j < e->nr_bonds ; j++ )
            if ( setid_bonds[j] >= 0 &&
                 ( ( e->exclusions[k].i == e->bonds[j].i && e->exclusions[k].j == e->bonds[j].j ) ||
                   ( e->exclusions[k].j == e->bonds[j].i && e->exclusions[k].i == e->bonds[j].j ) ) ) {
                setid_exclusions[k] = -setid_bonds[j];
                weight[ setid_bonds[j] ] += 1;
                break;
                }

        /* Does this bond get its own id? */
        if ( j < e->nr_bonds )
            continue;
        else {
            weight[ nr_sets ] = 1;
            setid_exclusions[k] = nr_sets++;
            }
                 
        /* Loop over dihedrals... */
        for ( j = 0 ; j < e->nr_dihedrals ; j++ )
            if ( e->exclusions[k].i == e->dihedrals[j].i || e->exclusions[k].i == e->dihedrals[j].j || e->exclusions[k].i == e->dihedrals[j].k || e->exclusions[k].i == e->dihedrals[j].l ||
                 e->exclusions[k].j == e->dihedrals[j].i || e->exclusions[k].j == e->dihedrals[j].j || e->exclusions[k].j == e->dihedrals[j].k || e->exclusions[k].j == e->dihedrals[j].l )
                if ( confl_add( setid_exclusions[k] , setid_dihedrals[j] ) < 0 )
                    return error(engine_err);

        /* Loop over angles... */
        for ( j = 0 ; j < e->nr_angles ; j++ )
            if ( setid_angles[j] >= 0 && 
                 ( e->exclusions[k].i == e->angles[j].i || e->exclusions[k].i == e->angles[j].j || e->exclusions[k].i == e->angles[j].k ||
                   e->exclusions[k].j == e->angles[j].i || e->exclusions[k].j == e->angles[j].j || e->exclusions[k].j == e->angles[j].k ) )
                if ( confl_add( setid_exclusions[k] , setid_angles[j] ) < 0 )
                    return error(engine_err);
                    
        /* Loop over  bonds... */
        for ( j = 0 ; j < e->nr_bonds ; j++ )
            if ( setid_bonds[j] >= 0 &&
                 ( e->exclusions[k].i == e->bonds[j].i || e->exclusions[k].i == e->bonds[j].j ||
                   e->exclusions[k].j == e->bonds[j].i || e->exclusions[k].j == e->bonds[j].j ) )
                if ( confl_add( setid_exclusions[k] , setid_bonds[j] ) < 0 )
                    return error(engine_err);
                    
        /* Loop over previous exclusions... */
        for ( j = 0 ; j < k ; j++ )
            if ( setid_exclusions[j] >= 0 &&
                 ( e->exclusions[k].i == e->exclusions[j].i || e->exclusions[k].i == e->exclusions[j].j ||
                   e->exclusions[k].j == e->exclusions[j].i || e->exclusions[k].j == e->exclusions[j].j ) )
                if ( confl_add( setid_exclusions[k] , setid_exclusions[j] ) < 0 )
                    return error(engine_err);
    
        } /* Loop over exclusions. */
        
    /* Make the setids positive again. */
    for ( k = 0 ; k < e->nr_angles ; k++ )
        setid_angles[k] = abs(setid_angles[k]);
    for ( k = 0 ; k < e->nr_bonds ; k++ )
        setid_bonds[k] = abs(setid_bonds[k]);
    for ( k = 0 ; k < e->nr_exclusions ; k++ )
        setid_exclusions[k] = abs(setid_exclusions[k]);
        
    /* Allocate the sorted conflict data. */
    if ( ( confl_sorted = malloc( sizeof(int) * 4 * confl_size ) ) == NULL ||
         ( confl_index = (int *)malloc( sizeof(int) * (nr_sets + 1) ) ) == NULL )
        return error(engine_err_malloc);


    /* As of here, the data structure has been set-up! */
    
    
    /* Main loop... */
    while ( nr_sets > max_sets ) {
    
        /* Get the average number of conflicts. */
        avg_nconfl = (2.0 * confl_count) / nr_sets;
        min_weight = weight[0]; max_weight = weight[0];
        for ( k = 1 ; k < nr_sets ; k++ )
            if ( weight[k] < min_weight )
                min_weight = weight[k];
            else if ( weight[k] > max_weight )
                max_weight = weight[k];
        avg_weight = ( 2.0*min_weight + max_weight ) / 3;
        /* printf( "engine_bonded_sets: nr_sets=%i, confl_count=%i, avg_weight=%f, avg_nconfl=%f.\n" ,
            nr_sets, confl_count, avg_weight , avg_nconfl );
        fflush(stdout); */
        
        /* First try to do the cheap thing: find a pair with
           zero conflicts each. */
        for ( min_i = 0 ; min_i < nr_sets && ( weight[min_i] >= avg_weight || nconfl[min_i] > 0 ) ; min_i++ );
        for ( min_j = min_i+1 ; min_j < nr_sets && ( weight[min_j] >= avg_weight || nconfl[min_j] > 0 ) ; min_j++ );
                    
        /* Did we find a mergeable pair? */
        if ( min_i < nr_sets && min_j < nr_sets ) {
        
            /* printf( "engine_bonded_sets: found disjoint sets %i and %i, %i confl.\n" ,
                min_i , min_j , nconfl[min_i] + nconfl[min_j] ); */
        
            }
        
        /* Otherwise, look for a pair sharing a conflict. */
        else {
    
            /* Assemble and sort the conflicts array. */
            for ( k = 0 ; k < confl_count ; k++ ) {
                confl_sorted[k] = confl[k];
                confl_sorted[confl_count+k].i = confl[k].j;
                confl_sorted[confl_count+k].j = confl[k].i;
                }
            confl_qsort( 0 , 2*confl_count - 1 );
            confl_index[0] = 0;
            for ( j = 0 , k = 0 ; k < 2*confl_count ; k++ )
                while ( confl_sorted[k].i > j )
                    confl_index[++j] = k;
            while ( j < nr_sets )
                confl_index[ ++j ] = 2*confl_count;
            bzero( confl_counts , sizeof(char) * nr_sets );

            /* Verify a few things... */
            /* if ( j != nr_sets )
                printf( "engine_bonded_sets: indexing is botched (j=%i)!\n" , j );
            for ( k = 0 ; k < confl_count ; k++ )
                if ( confl[k].i < 0 || confl[k].i >= nr_sets || confl[k].j < 0 || confl[k].j >= nr_sets || confl[k].i == confl[k].j )
                    printf( "engine_bonded_sets: invalid %ith conflict [%i,%i].\n" ,
                        k , confl[k].i , confl[k].j );
            for ( avg_nconfl = 0 , k = 0 ; k < nr_sets ; k++ )
                avg_nconfl += nconfl[k];
            if ( avg_nconfl/2 != confl_count )
                printf( "engine_bonded_sets: inconsistent nconfl (%f != %i)!\n" ,
                    avg_nconfl/2 , confl_count );
            for ( k = 1 ; k < 2*confl_count ; k++ )
                if ( confl_sorted[k].i < confl_sorted[k-1].i )
                    printf( "engine_bonded_sets: sorting is botched!\n" );
            for ( k = 0 ; k < nr_sets ; k++ )
                if ( confl_index[k+1]-confl_index[k] != nconfl[k] ) {
                    printf( "engine_bonded_sets: nconfl and confl inconsistent (%i:%i-%i != %i)!\n" ,
                        k , confl_index[k+1] , confl_index[k] , nconfl[k] );
                    printf( "engine_bonded_sets: conflicts are" );
                    for ( j = confl_index[k] ; j < confl_index[k+1] ; j++ )
                        printf( " [%i,%i]" , confl_sorted[j].i , confl_sorted[j].j );
                    printf( ".\n" );
                    } */

            /* Init min_i, min_j and min_confl. */
            min_i = -1;
            min_j = -1;
            max_confl = -1;

            /* For every pair of sets i and j... */
            for ( i = 0; i < nr_sets ; i++ ) {

                /* Skip i? */
                if ( weight[i] > avg_weight || nconfl[i] <= max_confl )
                    continue;

                /* Mark the conflicts in the ith set. */
                for ( k = confl_index[i] ; k < confl_index[i+1] ; k++ )
                    confl_counts[ confl_sorted[k].j ] = 1;
                confl_counts[i] = 1;

                /* Loop over all following sets. */
                for ( jj = confl_index[i] ; jj < confl_index[i+1] ; jj++ ) {

                    /* Skip j? */
                    j = confl_sorted[jj].j;
                    if ( weight[j] > avg_weight || nconfl[j] <= max_confl )
                        continue;

                    /* Get the number of conflicts in the combined set of i and j. */
                    for ( nr_confl = 0 , k = confl_index[j] ; k < confl_index[j+1] ; k++ )
                        if ( confl_counts[ confl_sorted[k].j ] )
                            nr_confl += 1;

                    /* Is this value larger than the current maximum? */
                    if ( nr_confl > max_confl ) {
                        max_confl = nr_confl; min_i = i; min_j = j;
                        }

                    } /* loop over following sets. */

                /* Un-mark the conflicts in the ith set. */
                for ( k = confl_index[i] ; k < confl_index[i+1] ; k++ )
                    confl_counts[ confl_sorted[k].j ] = 0;
                confl_counts[i] = 0;

                } /* for every pair of sets i and j. */
                
                
            /* If we didn't find anything, look for non-related set pairs (more expensive). */
            if ( min_i < 0 || min_j < 0 ) {
            
                /* For every pair of sets i and j... */
                for ( i = 0; i < nr_sets ; i++ ) {

                    /* Skip i? */
                    if ( weight[i] > avg_weight || nconfl[i] <= max_confl )
                        continue;

                    /* Mark the conflicts in the ith set. */
                    for ( k = confl_index[i] ; k < confl_index[i+1] ; k++ )
                        confl_counts[ confl_sorted[k].j ] = 1;
                    confl_counts[i] = 1;

                    /* Loop over all following sets. */
                    for ( j = i+1 ; j < nr_sets ; j++ ) {

                        /* Skip j? */
                        if ( weight[j] > avg_weight || nconfl[j] <= max_confl )
                            continue;

                        /* Get the number of conflicts in the combined set of i and j. */
                        for ( nr_confl = 0 , k = confl_index[j] ; k < confl_index[j+1] ; k++ )
                            if ( confl_counts[ confl_sorted[k].j ] )
                                nr_confl += 1;

                        /* Is this value larger than the current maximum? */
                        if ( nr_confl > max_confl ) {
                            max_confl = nr_confl; min_i = i; min_j = j;
                            }

                        } /* loop over following sets. */

                    /* Un-mark the conflicts in the ith set. */
                    for ( k = confl_index[i] ; k < confl_index[i+1] ; k++ )
                        confl_counts[ confl_sorted[k].j ] = 0;
                    confl_counts[i] = 0;

                    } /* for every pair of sets i and j. */
                
                }
            
            /* If we didn't find anything, merge the pairs with the lowest weight. */
            if ( min_i < 0 || min_j < 0 ) {
            
                /* Find the set with the minimum weight. */
                for (  min_i = 0 , i = 1 ; i < nr_sets ; i++ )
                    if ( weight[i] < weight[min_i] )
                        min_i = i;
                      
                /* Find the set with the second-minimum weight. */
                min_j = ( min_i == 0 ? 1 : 0 );
                for ( j = 0 ; j < nr_sets ; j++ )
                    if ( j != min_i && weight[j] < weight[min_j] )
                        min_j = j;
                    
                }
                
            /* Did we catch any pair? */
            if ( min_i < 0 || min_j < 0 ) {
                printf( "engine_bonded_sets: could not find a pair to merge!\n" );
                return error(engine_err_sets);
                }
                
            /* Mark the sets with which min_i conflicts. */
            for ( k = confl_index[min_i] ; k < confl_index[min_i+1] ; k++ )
                confl_counts[ confl_sorted[k].j ] = 1;
            confl_counts[ min_i ] = 1;

            /* Re-label or remove conflicts with min_j. */
            for ( k = 0 ; k < confl_count ; k++ )
                if ( confl[k].i == min_j ) {
                    if ( confl_counts[ confl[k].j ] ) {
                        nconfl[ confl[k].j ] -= 1;
                        confl[ k-- ] = confl[ --confl_count ];
                        }
                    else {
                        confl[k].i = min_i;
                        nconfl[min_i] += 1;
                        }
                    }
                else if ( confl[k].j == min_j ) {
                    if ( confl_counts[ confl[k].i ] ) {
                        nconfl[ confl[k].i ] -= 1;
                        confl[ k-- ] = confl[ --confl_count ];
                        }
                    else {
                        confl[k].j = min_i;
                        nconfl[min_i] += 1;
                        }
                    }
                
            }
            
        /* Otherwise, say something. */
        /* else    
            printf( "engine_bonded_sets: found pair of sets %i and %i with %i less confl.\n" ,
                min_i , min_j , max_confl ); */
            
        /* Dump both sets. */
        /* printf( "engine_bonded_sets: set %i has conflicts" , min_i );
        for ( k = 0 ; k < confl_count ; k++ )
            if ( confl[k].i == min_i )
                printf( " %i" , confl[k].j );
            else if ( confl[k].j == min_i )
                printf( " %i" ,  confl[k].i );
        printf(".\n");
        printf( "engine_bonded_sets: set %i has conflicts" , min_j );
        for ( k = 0 ; k < confl_count ; k++ )
            if ( confl[k].i == min_j )
                printf( " %i" , confl[k].j );
            else if ( confl[k].j == min_j )
                printf( " %i" , confl[k].i );
        printf(".\n"); */
    
        /* Merge the sets min_i and min_j. */
        for ( k = 0 ; k < e->nr_bonds ; k++ )
            if ( setid_bonds[k] == min_j )
                setid_bonds[k] = min_i;
        for ( k = 0 ; k < e->nr_angles ; k++ )
            if ( setid_angles[k] == min_j )
                setid_angles[k] = min_i;
        for ( k = 0 ; k < e->nr_dihedrals ; k++ )
            if ( setid_dihedrals[k] == min_j )
                setid_dihedrals[k] = min_i;
        for ( k = 0 ; k < e->nr_exclusions ; k++ )
            if ( setid_exclusions[k] == min_j )
                setid_exclusions[k] = min_i;
                
        /* Remove the set min_j (replace by last). */
        weight[min_i] += weight[min_j];
        nr_sets -= 1;
        weight[min_j] = weight[nr_sets];
        nconfl[min_j] = nconfl[nr_sets];
        for ( k = 0 ; k < e->nr_bonds ; k++ )
            if ( setid_bonds[k] == nr_sets )
                setid_bonds[k] = min_j;
        for ( k = 0 ; k < e->nr_angles ; k++ )
            if ( setid_angles[k] == nr_sets )
                setid_angles[k] = min_j;
        for ( k = 0 ; k < e->nr_dihedrals ; k++ )
            if ( setid_dihedrals[k] == nr_sets )
                setid_dihedrals[k] = min_j;
        for ( k = 0 ; k < e->nr_exclusions ; k++ )
            if ( setid_exclusions[k] == nr_sets )
                setid_exclusions[k] = min_j;
        for ( k = 0 ; k < confl_count ; k++ )
            if ( confl[k].i == nr_sets )
                confl[k].i = min_j;
            else if ( confl[k].j == nr_sets )
                confl[k].j = min_j;
            
        /* printf( "engine_bonded_sets: merged sets %i and %i, weight %i and %i confl.\n" ,
            min_i , min_j , weight[min_i] , nconfl[min_i] ); fflush(stdout);
        getchar(); */
            
    
        } /* main loop. */
        
    
    /* Allocate the sets. */
    e->nr_sets = nr_sets;
    if ( ( e->sets = (struct engine_set *)malloc( sizeof(struct engine_set) * nr_sets ) ) == NULL )
        return error(engine_err_malloc);
    bzero( e->sets , sizeof(struct engine_set) * nr_sets );
    
    /* Fill in the counts. */
    for ( k = 0 ; k < e->nr_bonds ; k++ )
        e->sets[ setid_bonds[k] ].nr_bonds += 1;
    for ( k = 0 ; k < e->nr_angles ; k++ )
        e->sets[ setid_angles[k] ].nr_angles += 1;
    for ( k = 0 ; k < e->nr_dihedrals ; k++ )
        e->sets[ setid_dihedrals[k] ].nr_dihedrals += 1;
    for ( k = 0 ; k < e->nr_exclusions ; k++ )
        e->sets[ setid_exclusions[k] ].nr_exclusions += 1;
        
    /* Allocate the index lists. */
    for ( k = 0 ; k < nr_sets ; k++ ) {
        if ( ( e->sets[k].bonds = (struct bond *)malloc( sizeof(struct bond) * e->sets[k].nr_bonds ) ) == NULL ||
             ( e->sets[k].angles = (struct angle *)malloc( sizeof(struct angle) * e->sets[k].nr_angles ) ) == NULL ||
             ( e->sets[k].dihedrals = (struct dihedral *)malloc( sizeof(struct dihedral) * e->sets[k].nr_dihedrals ) ) == NULL ||
             ( e->sets[k].exclusions = (struct exclusion *)malloc( sizeof(struct exclusion) * e->sets[k].nr_exclusions ) ) == NULL ||
             ( e->sets[k].confl = (int *)malloc( sizeof(int) * nconfl[k] ) ) == NULL )
            return error(engine_err_malloc);
        e->sets[k].weight = e->sets[k].nr_bonds + e->sets[k].nr_exclusions + 2*e->sets[k].nr_angles + 3*e->sets[k].nr_dihedrals;
        e->sets[k].nr_bonds = 0;
        e->sets[k].nr_angles = 0;
        e->sets[k].nr_dihedrals = 0;
        e->sets[k].nr_exclusions = 0;
        }
    
    /* Fill in the indices. */
    for ( k = 0 ; k < e->nr_bonds ; k++ ) {
        j = setid_bonds[k];
        e->sets[j].bonds[ e->sets[j].nr_bonds++ ] = e->bonds[ k ];
        }
    for ( k = 0 ; k < e->nr_angles ; k++ ) {
        j = setid_angles[k];
        e->sets[j].angles[ e->sets[j].nr_angles++ ] = e->angles[ k ];
        }
    for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {
        j = setid_dihedrals[k];
        e->sets[j].dihedrals[ e->sets[j].nr_dihedrals++ ] = e->dihedrals[ k ];
        }
    for ( k = 0 ; k < e->nr_exclusions ; k++ ) {
        j = setid_exclusions[k];
        e->sets[j].exclusions[ e->sets[j].nr_exclusions++ ] = e->exclusions[ k ];
        }
        
    /* Fill in the conflicts. */
    for ( k = 0 ; k < confl_count ; k++ ) {
        i = confl[k].i; j = confl[k].j;
        e->sets[i].confl[ e->sets[i].nr_confl++ ] = j;
        e->sets[j].confl[ e->sets[j].nr_confl++ ] = i;
        }
        
        
    /* Dump the sets. */
    /* for ( k = 0 ; k < nr_sets ; k++ ) {
        printf( "engine_bonded_sets: set %i:\n" , k );
        printf( "engine_bonded_sets:    bonds[%i] = [ " , e->sets[k].nr_bonds );
        for ( j = 0 ; j < e->sets[k].nr_bonds ; j++ )
            printf( "%i " , e->sets[k].bonds[j] );
        printf( "]\n" );
        printf( "engine_bonded_sets:    angles[%i] = [ " , e->sets[k].nr_angles );
        for ( j = 0 ; j < e->sets[k].nr_angles ; j++ )
            printf( "%i " , e->sets[k].angles[j] );
        printf( "]\n" );
        printf( "engine_bonded_sets:    dihedrals[%i] = [ " , e->sets[k].nr_dihedrals );
        for ( j = 0 ; j < e->sets[k].nr_dihedrals ; j++ )
            printf( "%i " , e->sets[k].dihedrals[j] );
        printf( "]\n" );
        printf( "engine_bonded_sets:    exclusions[%i] = [ " , e->sets[k].nr_exclusions );
        for ( j = 0 ; j < e->sets[k].nr_exclusions ; j++ )
            printf( "%i " , e->sets[k].exclusions[j] );
        printf( "]\n" );
        printf( "engine_bonded_sets:    conflicts[%i] = [ " , e->sets[k].nr_confl );
        for ( j = 0 ; j < e->sets[k].nr_confl ; j++ )
            printf( "%i " , e->sets[k].confl[j] );
        printf( "]\n" );
        printf( "engine_bonded_sets:    weight = %i\n" , e->sets[k].nr_bonds + e->sets[k].nr_exclusions + 2*e->sets[k].nr_angles + 3*e->sets[k].nr_dihedrals );
        } */
        
    /* Clean up the allocated memory. */
    free( nconfl );
    free( weight );
    free( confl );
    free( confl_sorted );
    free( setid_bonds ); free( setid_angles ); free( setid_dihedrals );
    free( setid_rigids ); free( setid_exclusions );
        
        
    /* It's the end of the world as we know it... */
    return engine_err_ok;

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
                    if ( e->exclusions[j].i > pivot_i ||
                         ( e->exclusions[j].i == pivot_i && e->exclusions[j].j > pivot_j ) ) {
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
    
    /* Verify the sort. */
    /* for ( k = 1 ; k < e->nr_exclusions ; k++ )
        if ( e->exclusions[k].i < e->exclusions[k-1].i ||
             ( e->exclusions[k].i == e->exclusions[k-1].i && e->exclusions[k].j < e->exclusions[k-1].j ) )
            printf( "engine_exclusion_shrink: sorting failed!\n" ); */
    
    /* Run through the exclusions and skip duplicates. */
    for ( j = 0 , k = 1 ; k < e->nr_exclusions ; k++ )
        if ( e->exclusions[k].j != e->exclusions[j].j ||
             e->exclusions[k].i != e->exclusions[j].i ) {
            j += 1;
            e->exclusions[j] = e->exclusions[k];
            }
            
    /* Set the number of exclusions to j. */
    e->nr_exclusions = j+1;
    if ( ( e->exclusions = (struct exclusion *)realloc( e->exclusions , sizeof(struct exclusion) * e->nr_exclusions ) ) == NULL )
        return error(engine_err_malloc);
    
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
    
    
