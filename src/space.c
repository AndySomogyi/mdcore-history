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
#include <stdio.h>
#include <string.h>
#include <strings.h>
#include <alloca.h>
#include <pthread.h>
#include <math.h>

/* include local headers */
#include "errs.h"
#include "part.h"
#include "cell.h"
#include "space.h"


/* the last error */
int space_err = space_err_ok;


/* the error macro. */
#define error(id)				( space_err = errs_register( id , space_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *space_err_msg[7] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "An error occured when calling a cell function.",
    "A call to a pthread routine failed.",
    "One or more values were outside of the allowed range.",
    "Too many pairs associated with a single particle in Verlet list.",
	};
    
    
/**
 * @brief Collect forces and potential energies
 *
 * @param s The #space.
 * @param maxcount The maximum number of entries.
 * @param from Pointer to an integer which will contain the index to the
 *        first entry on success.
 * @param to Pointer to an integer which will contain the index to the
 *        last entry on success.
 *
 * @return The number of entries returned or < 0 on error (see #space_err).
 */
 
int space_verlet_force ( struct space *s , FPTYPE *f , double epot ) {

    int cid, pid, k, ind;
    struct cell *c;
    struct part *p;
    int nr_cells = s->nr_cells, *scells;
    
    /* Allocate a buffer to mix-up the cells. */
    if ( ( scells = (int *)alloca( sizeof(int) * nr_cells ) ) == NULL )
        return error(space_err_malloc);
        
    /* Mix-up the order of the cells. */
    for ( k = 0 ; k < nr_cells ; k++ )
        scells[k] = k;
    for ( k = 0 ; k < nr_cells ; k++ ) {
        cid = rand() % nr_cells;
        pid = scells[k]; scells[k] = scells[cid]; scells[cid] = pid;
        }

    /* Loop over the cells. */
    for ( cid = 0 ; cid < nr_cells ; cid++ ) {
    
        /* Get a pointer on the cell. */
        c = &(s->cells[scells[cid]]);
        
        /* Get a lock on the cell. */
	    if ( pthread_mutex_lock( &c->verlet_force_mutex ) != 0 )
		    return error(space_err_pthread);
        
        for ( pid = 0 ; pid < c->count ; pid++ ) {
            p = &(c->parts[pid]);
            ind = 4 * p->id;
            for ( k = 0 ; k < 3 ; k++ )
                p->f[k] += f[ ind + k ];
            }
            
        /* Release the cells mutex */
	    if ( pthread_mutex_unlock( &c->verlet_force_mutex ) != 0 )
		    return error(space_err_pthread);
        
        }
        
    /* Add the potential energy to the space's potential energy. */
	if ( pthread_mutex_lock( &s->verlet_force_mutex ) != 0 )
		return error(space_err_pthread);
    s->epot += epot;
	if ( pthread_mutex_unlock( &s->verlet_force_mutex ) != 0 )
		return error(space_err_pthread);
    
    /* Relax. */
    return space_err_ok;
        
    }


/**
 * @brief Get a chunk of Verlet list entries.
 *
 * @param s The #space.
 * @param maxcount The maximum number of entries.
 * @param from Pointer to an integer which will contain the index to the
 *        first entry on success.
 * @param to Pointer to an integer which will contain the index to the
 *        last entry on success.
 *
 * @return The number of entries returned or < 0 on error (see #space_err).
 */
 
int space_verlet_get ( struct space *s , int maxcount , int *from ) {

    int count = 0;

    /* Try to get a hold of the cells mutex */
	if ( pthread_mutex_lock( &s->cellpairs_mutex ) != 0 )
		return error(space_err_pthread);
        
    /* Are there any entries left? */
    if ( s->verlet_next < s->nr_parts ) {
    
        /* Set count and from. */
        *from = s->verlet_next;
        count = s->nr_parts - s->verlet_next;
        if ( count > maxcount )
            count = maxcount;
        
        /* Increase verlet_next. */
        s->verlet_next += count;
    
        }
        
    /* Release the cells mutex */
	if ( pthread_mutex_unlock( &s->cellpairs_mutex ) != 0 )
		return error(space_err_pthread);
        
    /* Bring good tidings. */
    return count;
        
    }


/**
 * @brief Initialize the Verlet-list data structures.
 *
 * @param s The #space.
 *
 * @return #space_err_ok or < 0 on error (see #space_err).
 */
 
int space_verlet_init ( struct space *s , int list_global ) {

    int cid, pid, ind, k;
    double dx, w, maxdx = 0.0, skin;
    struct cell *c;
    struct part *p;

    /* Check input for nonsense. */
    if ( s == NULL )
        return error(space_err_null);
    
    /* Get the skin width. */    
    skin = fmin( s->h[0] , fmin( s->h[1] , s->h[2] ) ) - s->cutoff;
    
    /* Allocate the parts and nrpairs lists if necessary. */
    if ( s->verlet_oldx == NULL || s->verlet_size < s->nr_parts ) {
    
        printf("space_verlet_init: (re)allocating verlet lists...\n");
    
        /* Free old lists if necessary. */
        if ( s->verlet_list != NULL )
            free( s->verlet_list );
        if ( s->verlet_oldx != NULL )
            free( s->verlet_oldx );
        if ( s->verlet_nrpairs != NULL )
            free( s->verlet_nrpairs );
            
        /* Allocate new arrays. */
        s->verlet_size = 1.1 * s->nr_parts;
        if ( list_global ) {
            if ( ( s->verlet_list = (struct verlet_entry *)malloc( sizeof(struct verlet_entry) * s->verlet_size * space_verlet_maxpairs ) ) == NULL )
                return error(space_err_malloc);
            if ( ( s->verlet_nrpairs = (int *)malloc( sizeof(int) * s->verlet_size ) ) == NULL )
                return error(space_err_malloc);
            }
        if ( ( s->verlet_oldx = (FPTYPE *)malloc( sizeof(FPTYPE) * s->verlet_size * 4 ) ) == NULL )
            return error(space_err_malloc);
            
        /* We have to re-build the list now. */
        s->verlet_rebuild = 1;
            
        }
        
    else {
        
        /* Check if we need to re-shuffle the particles. */
        for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
            c = &(s->cells[cid]);
            for ( pid = 0 ; pid < c->count ; pid++ ) {
                p = &(c->parts[pid]);
                ind = 4 * p->id;
                for ( dx = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    w = p->x[k] - s->verlet_oldx[ ind + k ];
                    dx += w*w;
                    }
                maxdx = fmax( dx , maxdx );
                }
            }
            
        /* Are we still in the green? */
        s->verlet_rebuild = ( 2.0*sqrt(maxdx) > skin );
        
        }
        
    /* Do we have to rebuild the Verlet list? */
    if ( s->verlet_rebuild ) {
    
        /* printf("space_verlet_init: (re)building verlet lists...\n"); */
        /* printf("space_verlet_init: maxdx=%e, skin=%e.\n",sqrt(maxdx),skin); */
        
        /* Shuffle the domain. */
        if ( space_shuffle( s ) < 0 )
            return error(space_err);
            
        /* Store the current positions as a reference. */
        for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
            c = &(s->cells[cid]);
            for ( pid = 0 ; pid < c->count ; pid++ ) {
                p = &(c->parts[pid]);
                ind = 4 * p->id;
                for ( k = 0 ; k < 3 ; k++ )
                    s->verlet_oldx[ ind + k ] = p->x[k];
                }
            }
            
        /* Set the nrpairs to zero. */
        if ( list_global )
            bzero( s->verlet_nrpairs , sizeof(int) * s->nr_parts );

        }
        
    /* re-set the Verlet list index. */
    s->verlet_next = 0;
        
    /* All done! */
    return space_err_ok;

    }


/**
 * @brief Clear all particles from this #space.
 *
 * @param s The #space to flush.
 *
 * @return #space_err_ok or < 0 on error (see #space_err).
 */
 
int space_flush ( struct space *s ) {

    int cid;

    /* check input. */
    if ( s == NULL )
        return error(space_err_null);
        
    /* loop through the cells. */
    for ( cid = 0 ; cid < s->nr_cells ; cid++ )
        s->cells[cid].count = 0;
        
    /* done for now. */
    return space_err_ok;

    }


/**
 * @brief Get the next free #celltuple from the space.
 * 
 * @param s The #space in which to look for tuples.
 * @param out A pointer to a #celltuple in which to copy the result.
 *
 * @return The number of #celltuple found or 0 if the list is empty and
 *      < 0 on error (see #space_err).
 */
 
int space_gettuple ( struct space *s , struct celltuple **out ) {

    int i, j, k;
    struct celltuple *t, temp;

    /* Try to get a hold of the cells mutex */
	if ( pthread_mutex_lock( &s->cellpairs_mutex ) != 0 )
		return error(space_err_pthread);
        
    /* Main loop, while there are still tuples left. */
    while ( s->next_tuple < s->nr_tuples ) {
    
        /* Loop over all tuples. */
        for ( k = s->next_tuple ; k < s->nr_tuples ; k++ ) {
        
            /* Put a t on this tuple. */
            t = &( s->tuples[k] );
            
            /* Check if all the cells of this tuple are free. */
            for ( i = 0 ; i < t->n ; i++ )
                if ( s->cells_taboo[ t->cellid[i] ] != 0 )
                    break;
            if ( i < t->n )
                continue;
                
            /* If so, mark-off the cells pair by pair. */
            for ( i = 0 ; i < t->n ; i++ )
                for ( j = i ; j < t->n ; j++ )
                    if ( t->pairs & ( 1 << ( i * space_maxtuples + j ) ) ) {
                        s->cells_taboo[ t->cellid[i] ] += 1;
                        s->cells_taboo[ t->cellid[j] ] += 1;
                        }
                        
            /* Swap this tuple to the top of the list. */
            if ( k != s->next_tuple ) {
                temp = s->tuples[ k ];
                s->tuples[ k ] = s->tuples[ s->next_tuple ];
                s->tuples[ s->next_tuple ] = temp;
                s->nr_swaps += 1;
                }
                
            /* Copy this tuple out. */
            *out = &( s->tuples[ s->next_tuple ] );
            
            /* Increase the top of the list. */
            s->next_tuple += 1;
            
            /* If this was the last tuple, broadcast to all waiting
                runners to go home. */
            if ( s->next_tuple == s->nr_tuples )
                if (pthread_cond_broadcast(&s->cellpairs_avail) != 0)
                    return error(space_err_pthread);
            
            /* And leave. */
	        if ( pthread_mutex_unlock( &s->cellpairs_mutex ) != 0 )
		        return error(space_err_pthread);
            return 1;
        
            }
            
        /* If we got here without catching anything, wait for a sign. */
        s->nr_stalls += 1;
        if ( pthread_cond_wait( &s->cellpairs_avail , &s->cellpairs_mutex ) != 0 )
            return error(space_err_pthread);
    
        }
        
    /* Release the cells mutex */
	if ( pthread_mutex_unlock( &s->cellpairs_mutex ) != 0 )
		return error(space_err_pthread);
        
    /* Bring good tidings. */
    return space_err_ok;
        
    }


/**
 * @brief Generate the list of #celltuple.
 * 
 * @param s Pointer to the #space to make tuples for.
 *
 * @return #space_err_ok or < 0 on error (see #space_err).
 */
 
int space_maketuples ( struct space *s ) {

    int size, *w, w_max;
    int i, j, k;
    struct celltuple *t;
    struct cellpair *p, temppair;
    
    /* Check for bad input. */
    if ( s == NULL )
        return error(space_err_null);
        
    /* Guess the size of the tuple array and allocate it. */
    size = s->nr_pairs / space_maxtuples;
    if ( ( s->tuples = (struct celltuple *)malloc( sizeof(struct celltuple) * size ) ) == NULL )
        return error(space_err_malloc);
    bzero( s->tuples , sizeof(struct celltuple) * size );
    s->nr_tuples = 0;
        
    /* Allocate the vector w. */
    if ( ( w = (int *)alloca( sizeof(int) * s->nr_cells ) ) == NULL )
        return error(space_err_malloc);
    s->next_pair = 0;
        
    /* While there are still pairs that are not part of a tuple... */
    while ( s->next_pair < s->nr_pairs ) {
    
        /* Is the array of tuples long enough? */
        if ( s->nr_tuples >= size ) {
            if ( ( t = (struct celltuple *)malloc( sizeof(struct celltuple) * (size + 20) ) ) == NULL )
                return error(space_err_malloc);
            memcpy( t , s->tuples , sizeof(struct celltuple) * size );
            bzero( &t[size] , sizeof(struct celltuple) * 20 );
            size += 20;
            free( s->tuples );
            s->tuples = t;
            }
            
        /* Get a pointer on the next free tuple. */
        t = &( s->tuples[ s->nr_tuples++ ] );
            
        /* Just put the next pair into this tuple. */
        p = &( s->pairs[ s->next_pair++ ] );
        t->cellid[0] = p->i; t->n = 1;
        if ( p->j != p->i ) {
            t->cellid[ t->n++ ] = p->j;
            t->pairs = ( 1 << ( 0 * space_maxtuples + 1 ) ) | ( 1 << ( 1 * space_maxtuples + 0 ) );
            }
        else
            t->pairs = 1;
        /* printf("space_maketuples: starting tuple %i with pair [%i,%i].\n",
            s->nr_tuples-1 , p->i , p->j ); */
            
        /* Fill the weights for the cells. */
        bzero( w , sizeof(int) * s->nr_cells );
        for ( k = 0 ; k < t->n ; k++ )
            w[ t->cellid[k] ] = -1;
        for ( k = s->next_pair ; k < s->nr_pairs ; k++ ) {
            for ( i = 0 ; i < t->n ; i++ ) {
                if ( s->pairs[k].i == t->cellid[i] && w[ s->pairs[k].j ] >= 0 )
                    w[ s->pairs[k].j ] += 1;
                if ( s->pairs[k].j == t->cellid[i] && w[ s->pairs[k].i ] >= 0 )
                    w[ s->pairs[k].i ] += 1;
                }
            }
            
        /* Find the cell with the maximum weight. */
        w_max = 0;
        for ( k = 1 ; k < s->nr_cells ; k++ )
            if ( w[k] > w[w_max] )
                w_max = k;
            
        /* While there is still another cell that can be added... */
        while ( w[w_max] > 0 && t->n < space_maxtuples && s->next_pair < s->nr_pairs ) {
        
            /* printf("space_maketuples: adding cell %i to tuple %i (w[%i]=%i).\n",
                w_max, s->nr_tuples-1, w_max, w[w_max] ); */
            
            /* Add this cell to the tuple. */
            t->cellid[ t->n++ ] = w_max;
        
            /* Look for pairs that contain w_max and someone from the tuple. */
            for ( k = s->next_pair ; k < s->nr_pairs ; k++ ) {
            
                /* Get this pair. */
                p = &( s->pairs[ k ] );
                
                /* Get the tuple indices of the cells in this pair. */
                for ( i = 0 ; i < t->n && t->cellid[i] != p->i ; i++ );
                for ( j = 0 ; j < t->n && t->cellid[j] != p->j ; j++ );
                
                /* If this pair is not in the tuple, skip it. */
                if ( i == t->n || j == t->n )
                    continue;

                /* Add this pair to the tuple. */
                t->pairs |= ( 1 << ( i * space_maxtuples + j ) ) | ( 1 << ( j * space_maxtuples + i ) );
                /* printf("space_maketuples: adding pair [%i,%i] to tuple %i (w[%i]=%i).\n",
                    p->i, p->j, s->nr_tuples-1 , w_max , w[w_max] ); */

                /* If this pair was not at the top of the list, swap it there. */
                if ( k != s->next_pair ) {
                    temppair = s->pairs[ s->next_pair ];
                    s->pairs[ s->next_pair ] = s->pairs[ k ];
                    s->pairs[ k ] = temppair;
                    }

                /* Move the next_pair pointer. */
                s->next_pair += 1;
                
                }
            
            /* Update the weights and get the ID of the new max. */
            w[ w_max ] = -1;
            for ( k = s->next_pair ; k < s->nr_pairs ; k++ ) {
                if ( s->pairs[k].i == w_max && w[ s->pairs[k].j ] >= 0 )
                    w[ s->pairs[k].j ] += 1;
                if ( s->pairs[k].j == w_max && w[ s->pairs[k].i ] >= 0 )
                    w[ s->pairs[k].i ] += 1;
                }
        
            /* Find the cell with the maximum weight. */
            w_max = 0;
            for ( k = 1 ; k < s->nr_cells ; k++ )
                if ( w[k] > w[w_max] )
                    w_max = k;
            
            }
    
        }
        
    /* Dump the list of tuples. */
    /* for ( i = 0 ; i < s->nr_tuples ; i++ ) {
        t = &( s->tuples[i] );
        printf("space_maketuples: tuple %i has pairs:",i);
        for ( k = 0 ; k < t->n ; k++ )
            for ( j = k ; j < t->n ; j++ )
                if ( t->pairs & ( 1 << ( k * space_maxtuples + j ) ) )
                    printf(" [%i,%i]", t->cellid[j], t->cellid[k] );
        printf("\n");
        } */
        
    /* If we made it up to here, we're done! */
    return space_err_ok;

    }


/**
 * @brief Prepare the space before a time step.
 *
 * @param s A pointer to the #space to prepare.
 *
 * @return #space_err_ok or < 0 on error (see #space_err)
 *
 * Initializes a #space for a single time step. This routine runs
 * through the particles and sets their forces to zero.
 */

int space_prepare ( struct space *s ) {

    int pid, cid, k;

    /* re-set next_pair */
    s->next_pair = 0;
    s->next_tuple = 0;
    s->nr_swaps = 0;
    s->nr_stalls = 0;
    s->epot = 0.0;
    
    /* run through the cells and re-set the potential energy and forces */
    for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
        s->cells[cid].epot = 0.0;
        for ( pid = 0 ; pid < s->cells[cid].count ; pid++ )
            for ( k = 0 ; k < 3 ; k++ )
                s->cells[cid].parts[pid].f[k] = 0.0;
        }
        
    /* what else could happen? */
    return space_err_ok;

    }


/**
 * @brief Run through the cells of a #space and make sure every particle is in
 * its place.
 *
 * @param s The #space on which to operate.
 *
 * @returns #space_err_ok or < 0 on error.
 *
 * Runs through the cells of @c s and if a particle has stepped outside the
 * cell bounds, moves it to the correct cell.
 */
/* TODO: Check non-periodicity and ghost cells. */

int space_shuffle ( struct space *s ) {

    int k, cid, pid, delta[3];
    struct cell *c, *c_dest;
    struct part *p;

    /* loop over all cells */
    for ( cid = 0 ; cid < s->nr_cells ; cid++ ) {
    
        /* get the cell */
        c = &(s->cells[cid]);
    
        /* loop over all particles in this cell */
        pid = 0;
        while ( pid < c->count ) {
        
            /* get a handle on the particle */
            p = &(c->parts[pid]);
            
            /* check if this particle is out of bounds */
            for ( k = 0 ; k < 3 ; k++ ) {
                if ( p->x[k] < 0.0 )
                    delta[k] = -1;
                else if ( p->x[k] >= s->h[k] )
                    delta[k] = 1;
                else
                    delta[k] = 0;
                }
                
            /* do we have to move this particle? */
            if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                for ( k = 0 ; k < 3 ; k++ )
                    p->x[k] -= delta[k] * s->h[k];
                c_dest = &( s->cells[ space_cellid( s ,
                    (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] , 
                    (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] , 
                    (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );
                cell_add( c_dest , p );
                c->count -= 1;
                if ( pid < c->count )
                    c->parts[pid] = c->parts[c->count];
                /* printf("space_shuffle: moving particle %i from cell [%i,%i,%i] to cell [%i,%i,%i].\n", */
                /*     p->id, c->loc[0], c->loc[1], c->loc[2], */
                /*     c_dest->loc[0], c_dest->loc[1], c_dest->loc[2]); */
                /* printf("space_shuffle: particle coords are [%e,%e,%e].\n", */
                /*     p->x[0], p->x[1], p->x[2]); */
                }
            else
                pid++;
        
            } /* loop over all particles */
    
        } /* loop over all cells */
        
    /* run through the cells again and reconstruct the partlist */
    for ( cid = 0 ; cid < s->nr_cells ; cid++ )
        for ( pid = 0 ; pid < s->cells[cid].count ; pid++ ) {
            s->partlist[ s->cells[cid].parts[pid].id ] = &( s->cells[cid].parts[pid] );
            s->celllist[ s->cells[cid].parts[pid].id ] = &( s->cells[cid] );
            }
            
    /* If we've got a Verlet list, reset the counts. */
    if ( s->verlet_nrpairs != NULL )
        bzero( s->verlet_nrpairs , sizeof(int) * s->nr_parts );
    
    /* all is well... */
    return space_err_ok;

    }


/**
 * @brief Add a #part to a #space at the given coordinates.
 *
 * @param s The space to which @c p should be added.
 * @param p The #part to be added.
 * @param x A pointer to an array of three doubles containing the particle
 *      position.
 *
 * @returns #space_err_ok or < 0 on error (see #space_err).
 *
 * Inserts a #part @c p into the #space @c s at the position @c x.
 * Note that since particle positions in #part are relative to the cell, that
 * data in @c p is overwritten and @c x is used.
 */

int space_addpart ( struct space *s , struct part *p , double *x ) {

    int k, cid, ind[3];
    struct part **temp;
    struct cell **tempc;

    /* check input */
    if ( s == NULL || p == NULL || x == NULL )
        return error(space_err_null);
        
    /* do we need to extend the partlist? */
    if ( s->nr_parts == s->size_parts ) {
        s->size_parts += space_partlist_incr;
        if ( ( temp = (struct part **)malloc( sizeof(struct part *) * s->size_parts ) ) == NULL )
            return error(space_err_malloc);
        if ( ( tempc = (struct cell **)malloc( sizeof(struct cell *) * s->size_parts ) ) == NULL )
            return error(space_err_malloc);
        memcpy( temp , s->partlist , sizeof(struct part *) * s->nr_parts );
        memcpy( tempc , s->celllist , sizeof(struct cell *) * s->nr_parts );
        free( s->partlist );
        free( s->celllist );
        s->partlist = temp;
        s->celllist = tempc;
        }
        
    /* Create the local id of this part. */
    p->id = s->nr_parts++;
        
    /* get the hypothetical cell coordinate */
    for ( k = 0 ; k < 3 ; k++ )
        ind[k] = (x[k] - s->origin[k]) * s->ih[k];
        
    /* is this particle within the space? */
    for ( k = 0 ; k < 3 ; k++ )
        if ( ind[k] < 0 || ind[k] >= s->cdim[k] )
            return error(space_err_range);
            
    /* get the appropriate cell */
    cid = space_cellid(s,ind[0],ind[1],ind[2]);
    
    /* make the particle position local */
    for ( k = 0 ; k < 3 ; k++ )
        p->x[k] = x[k] - s->cells[cid].origin[k];
        
    /* delegate the particle to the cell */
    if ( ( s->partlist[p->id] = cell_add(&s->cells[cid],p) ) == NULL )
        return error(space_err_cell);
    s->celllist[p->id] = &( s->cells[cid] );
    
    /* end well */
    return space_err_ok;

    }
    
    
/**
 * @brief Get the absolute position of a particle
 *
 * @param s The #space in which the particle resides.
 * @param id The local id of the #part.
 * @param x A pointer to a vector of at least three @c doubles in
 *      which to store the particle position.
 *
 */
 
int space_getpos ( struct space *s , int id , double *x ) {

    int k;

    /* Sanity check. */
    if ( s == NULL || x == NULL )
        return error(space_err_null);
    if ( id >= s->nr_parts )
        return error(space_err_range);
        
    /* Copy the position to x. */
    for ( k = 0 ; k < 3 ; k++ )
        x[k] = s->partlist[id]->x[k] + s->celllist[id]->origin[k];
        
    /* All is well... */
    return space_err_ok;
    
    }


/**
 * @brief Free the cells involved in the current pair.
 *
 * @param s The #space to operate on.
 * @param ci ID of the first cell.
 * @param cj ID of the second cell.
 *
 * @returns #space_err_ok or < 0 on error (see #space_err).
 *
 * Decreases the taboo-counter of the cells involved in the pair
 * and signals any #runner that might be waiting.
 * Note that only a single waiting #runner is released per released cell
 * and therefore, if two different cells become free, the condition
 * @c cellpairs_avail is signaled twice.
 */

int space_releasepair ( struct space *s , int ci , int cj ) {

    int count = 0;

    /* release the cells in the given pair */
    count += ( --(s->cells_taboo[ ci ]) == 0 );
    count += ( --(s->cells_taboo[ cj ]) == 0 );
    
    /* send a strong signal to anybody waiting on pairs... */
    if ( count ) {
	    /* if (pthread_mutex_lock(&s->cellpairs_mutex) != 0) 
            return error(space_err_pthread); */
        while ( count-- )
            if (pthread_cond_signal(&s->cellpairs_avail) != 0)
                return error(space_err_pthread);
	    /* if (pthread_mutex_unlock(&s->cellpairs_mutex) != 0)
            return error(space_err_pthread); */
        }
    
    /* all is well... */
    return space_err_ok;
        
    }
    

/**
 * @brief Get a set of #cellpair from the space.
 *
 * @param s The #space from which to get pairs.
 * @param owner The id of the calling #runner.
 * @param count The maximum number of #cellpair to return.
 * @param old A list of #cellpair that have been processed and may be released,
 *      can also be @c NULL.
 * @param err A pointer to an integer in which to store the error code.
 * @param wait A boolean integer specifying if to wait or not if no pairs are
 *      available.
 * 
 * @return A pointer to a linked list of #cellpair. If none were available,
 *      @c NULL is returned. If an error occurs, @c NULL is returned and
 *      @c err is set to the respective error code.
 *
 * The returned #cellpair are linked through the field @c next. The value
 * of @c next in the last #cellpair is @c NULL.
 *
 * The routine starts by blocking the @c cellpair_mutex of the #space @c s
 * and searches for available pairs in the #space pair-list until either
 * @c count pairs have been found or the list has been exhausted.
 *
 * For each pair found, the #space @c s counters in @c cells_taboo are
 * incremented.
 *
 * If no pairs are
 * available and @c wait is not @c 0, the routines waits for a signal on
 * the #space @c s condition variable @c cellpairs_avail. If @c wait is @c 0
 * and no #cellpair have been found, the routine returns @c NULL and sets
 * @c err to @c 0.
 *
 * If the last #cellpair has been taken, the routine broadcasts a signal
 * on the #space @c s condition variable @c cellpairs_avail to release any
 * other @c runners waiting for pairs.
 * 
 */

struct cellpair *space_getpair ( struct space *s , int owner , int count , struct cellpair *old , int *err , int wait ) {

    struct cellpair *res = NULL;
    struct cellpair temp;
    int i;
    
    /* try to get a hold of the cells mutex */
	if (pthread_mutex_lock(&s->cellpairs_mutex) != 0) {
		*err = space_err_pthread;
        return NULL;
        }
        
    /* did the user return any old pairs */
    if ( old != NULL ) {
    
        /* release any old pairs */
        while ( old != NULL ) {
            s->cells_taboo[ old->i ] -= 1;
            s->cells_taboo[ old->j ] -= 1;
            old = old->next;
            }

        }
        
    /* set i */
    i = s->next_pair;
    
    /* try to find an unused pair */
    while ( s->next_pair < s->nr_pairs && count > 0 ) {
     
        /* run through the list of pairs */
        for (  ; i < s->nr_pairs ; i++ )
            if ( ( s->cells_taboo[s->pairs[i].i] == 0  || s->cells_owner[s->pairs[i].i] == owner ) &&
                ( s->cells_taboo[s->pairs[i].j] == 0  || s->cells_owner[s->pairs[i].j] == owner ) )
                break;
                
        /* did we actually find a pair? */
        if ( i < s->nr_pairs ) {
        
            /* do we need to swap this pair up? */
            if ( i > s->next_pair ) {
                temp = s->pairs[i];
                s->pairs[i] = s->pairs[s->next_pair];
                s->pairs[s->next_pair] = temp;
                s->nr_swaps += 1;
                }
                
            /* does this pair actually require any work? */
            if ( s->cells[ s->pairs[ s->next_pair ].i ].count > 0 &&
                s->cells[ s->pairs[ s->next_pair ].j ].count > 0 ) {
                
                /* set the result */
                s->pairs[s->next_pair].next = res;
                res = &s->pairs[s->next_pair++];

                /* mark the cells as taken */
                s->cells_taboo[res->i] += 1;
                s->cells_taboo[res->j] += 1;
                s->cells_owner[res->i] = owner;
                s->cells_owner[res->j] = owner;
        
                /* adjust the counters */
                count -= 1;
                }
                
            /* otherwise, just skip this pair */
            else
                s->next_pair += 1;
                
            /* move the finger one up */
            i += 1;
            
            }
        
        /* otherwise, if all cells are blocked and we have nothing... */
        else if ( wait && res == NULL ) {

            /* count this as a stall */
            s->nr_stalls += 1;

            /* wait for a signal to try looking again */
            if ( pthread_cond_wait(&s->cellpairs_avail,&s->cellpairs_mutex) != 0 ) {
                *err = space_err_pthread;
                return NULL;
                }

            /* re-set i */
            i = s->next_pair;

            }
            
        /* otherwise, if we've got at least one pair, leave. */
        else
            break;
     
        }
     
    /* just in case we totalled the queue and other */
    /* threads are still waiting on it... */
    if ( s->next_pair == s->nr_pairs )
        if ( pthread_cond_broadcast( &s->cellpairs_avail ) != 0 ) {
            *err = space_err_pthread;
            return NULL;
            }
    
    /* let go of the cells mutex */
	if (pthread_mutex_unlock(&s->cellpairs_mutex) != 0) {
        *err = space_err_pthread;
        return NULL;
        }
    
    /* return the pair (if any) */
    *err = space_err_ok;
    return res;
    
    }


/**
 * @brief Initialize the space with the given dimensions.
 *
 * @param s The #space to initialize.
 * @param origin Pointer to an array of three doubles specifying the origin
 *      of the rectangular domain.
 * @param dim Pointer to an array of three doubles specifying the length
 *      of the rectangular domain along each dimension.
 * @param cutoff A double-precision value containing the maximum cutoff lenght
 *      that will be used in the potentials.
 * @param period Unsigned integer containing the flags #space_periodic_x,
 *      #space_periodic_y and/or #space_periodic_z or #space_periodic_full.
 *
 * @return #space_err_ok or <0 on error (see #space_err).
 * 
 * This routine initializes the fields of the #space @c s, creates the cells and
 * generates the cell-pair list.
 */

int space_init ( struct space *s , const double *origin , const double *dim , double cutoff , unsigned int period ) {

    int i, j, k, l[3], ii, jj, kk;
    int id1, id2;
    double o[3];
    float shift[3];

    /* check inputs */
    if ( s == NULL || origin == NULL || dim == NULL )
        return error(space_err_null);
        
    /* set origin and compute the dimensions */
    for ( i = 0 ; i < 3 ; i++ ) {
        s->origin[i] = origin[i];
        s->dim[i] = dim[i];
        s->cdim[i] = floor( dim[i] / cutoff );
        }
        
    /* remember the cutoff */
    s->cutoff = cutoff;
    s->cutoff2 = cutoff*cutoff;
        
    /* set the periodicity */
    s->period = period;
        
    /* allocate the cells */
    s->nr_cells = s->cdim[0] * s->cdim[1] * s->cdim[2];
    s->cells = (struct cell *)malloc( sizeof(struct cell) * s->nr_cells );
    if ( s->cells == NULL )
        return error(space_err_malloc);
        
    /* get the dimensions of each cell */
    for ( i = 0 ; i < 3 ; i++ ) {
        s->h[i] = s->dim[i] / s->cdim[i];
        s->ih[i] = 1.0 / s->h[i];
        }
        
    /* initialize the cells  */
    for ( l[0] = 0 ; l[0] < s->cdim[0] ; l[0]++ ) {
        o[0] = origin[0] + l[0] * s->h[0];
        for ( l[1] = 0 ; l[1] < s->cdim[1] ; l[1]++ ) {
            o[1] = origin[1] + l[1] * s->h[1];
            for ( l[2] = 0 ; l[2] < s->cdim[2] ; l[2]++ ) {
                o[2] = origin[2] + l[2] * s->h[2];
                if ( cell_init( &(s->cells[space_cellid(s,l[0],l[1],l[2])]) , l , o , s->h ) < 0 )
                    return error(space_err_cell);
                }
            }
        }
        
    /* allocate the cell pairs array (pessimistic guess) */
    if ( (s->pairs = (struct cellpair *)malloc( sizeof(struct cellpair) * s->nr_cells * 14 )) == NULL )
        return error(space_err_malloc);
    
    /* fill the cell pairs array */
    s->nr_pairs = 0;
    /* for every cell */
    for ( i = 0 ; i < s->cdim[0] ; i++ ) {
        for ( j = 0 ; j < s->cdim[1] ; j++ ) {
            for ( k = 0 ; k < s->cdim[2] ; k++ ) {
            
                /* get this cell's id */
                id1 = space_cellid(s,i,j,k);
                
                /* if this cell is a ghost cell, skip it. */
                if ( s->cells[id1].flags & cell_flag_ghost )
                    continue;
            
                /* for every neighbouring cell in the x-axis... */
                for ( l[0] = -1 ; l[0] < 2 ; l[0]++ ) {
                
                    /* get coords of neighbour */
                    ii = i + l[0];

                    /* wrap or abort if not periodic */
                    if ( ii < 0 ) {
                        if (s->period & space_periodic_x)
                            ii = s->cdim[0] - 1;
                        else
                            continue;
                        }
                    else if ( ii >= s->cdim[0] ) {
                        if (s->period & space_periodic_x)
                            ii = 0;
                        else
                            continue;
                        }
                        
                    /* set the shift in x */
                    shift[0] = l[0] * s->h[0];
                        
                    /* for every neighbouring cell in the y-axis... */
                    for ( l[1] = -1 ; l[1] < 2 ; l[1]++ ) {
                    
                        /* get coords of neighbour */
                        jj = j + l[1];

                        /* wrap or abort if not periodic */
                        if ( jj < 0 ) {
                            if (s->period & space_periodic_y)
                                jj = s->cdim[1] - 1;
                            else
                                continue;
                            }
                        else if ( jj >= s->cdim[1] ) {
                            if (s->period & space_periodic_y)
                                jj = 0;
                            else
                                continue;
                            }
                            
                        /* set the shift in y */
                        shift[1] = l[1] * s->h[1];

                        /* for every neighbouring cell in the z-axis... */
                        for ( l[2] = -1 ; l[2] < 2 ; l[2]++ ) {

                            /* get coords of neighbour */
                            kk = k + l[2];

                            /* wrap or abort if not periodic */
                            if ( kk < 0 ) {
                                if (s->period & space_periodic_z)
                                    kk = s->cdim[2] - 1;
                                else
                                    continue;
                                }
                            else if ( kk >= s->cdim[0] ) {
                                if (s->period & space_periodic_z)
                                    kk = 0;
                                else
                                    continue;
                                }
                                
                            /* set the shift in z */
                            shift[2] = l[2] * s->h[2];
                                
                            /* get the neighbour's id */
                            id2 = space_cellid(s,ii,jj,kk);
                            
                            /* store this pair? */
                            if ( id1 <= id2 || (s->cells[id2].flags & cell_flag_ghost ) ) {
                                s->pairs[s->nr_pairs].i = id1;
                                s->pairs[s->nr_pairs].j = id2;
                                s->pairs[s->nr_pairs].shift[0] = shift[0];
                                s->pairs[s->nr_pairs].shift[1] = shift[1];
                                s->pairs[s->nr_pairs].shift[2] = shift[2];
                                #ifdef VECTORIZE
                                    s->pairs[s->nr_pairs].shift[3] = 0.0;
                                #endif
                                s->nr_pairs++;
                                }

                            } /* for every neighbouring cell in the z-axis... */
                        } /* for every neighbouring cell in the y-axis... */
                    } /* for every neighbouring cell in the x-axis... */
            
                }
            }
        }
        
    /* allocate and init the taboo-list */
    if ( (s->cells_taboo = (char *)malloc( sizeof(char) * s->nr_pairs )) == NULL )
        return error(space_err_malloc);
    bzero( s->cells_taboo , sizeof(char) * s->nr_pairs );
    if ( (s->cells_owner = (char *)malloc( sizeof(char) * s->nr_pairs )) == NULL )
        return error(space_err_malloc);
    bzero( s->cells_owner , sizeof(char) * s->nr_pairs );
    
    /* Make the list of celltuples. */
    if ( space_maketuples( s ) < 0 )
        return space_err;
    
    /* allocate the initial partlist */
    if ( ( s->partlist = (struct part **)malloc( sizeof(struct part *) * space_partlist_incr ) ) == NULL )
        return error(space_err_malloc);
    if ( ( s->celllist = (struct cell **)malloc( sizeof(struct cell *) * space_partlist_incr ) ) == NULL )
        return error(space_err_malloc);
    s->nr_parts = 0;
    s->size_parts = space_partlist_incr;
    
    /* init the cellpair mutexes */
    if ( pthread_mutex_init( &s->cellpairs_mutex , NULL ) != 0 ||
        pthread_cond_init( &s->cellpairs_avail , NULL ) != 0 ||
        pthread_mutex_init( &s->verlet_force_mutex , NULL ) != 0 )
        return error(space_err_pthread);
    
        
    /* Init the Verlet table (NULL for now). */
    s->verlet_list = NULL;
    s->verlet_nrpairs = NULL;
    s->verlet_oldx = NULL;
    s->verlet_size = 0;
    s->verlet_rebuild = 0;
        
    /* all is well that ends well... */
    return space_err_ok;

    }
