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
#include <pthread.h>
#include <math.h>

/* include local headers */
#include "part.h"
#include "cell.h"
#include "space.h"


/* the last error */
int space_err = space_err_ok;


/*/*//////////////////////////////////////////////////////////////////////////// */ */
/* int space_prepare */
//
/* get this space ready for a time-step */
/*/*//////////////////////////////////////////////////////////////////////////// */ */

int space_prepare ( struct space *s ) {

    int pid, cid, k;

    /* re-set next_pair */
    s->next_pair = 0;
    s->nr_swaps = 0;
    s->nr_stalls = 0;
    
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


/*/*//////////////////////////////////////////////////////////////////////////// */ */
/* int space_shuffle */
//
/* run through the cells and make sure every particle is in its place. */
/*/*//////////////////////////////////////////////////////////////////////////// */ */

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
    
    /* all is well... */
    return space_err_ok;

    }


/*/*//////////////////////////////////////////////////////////////////////////// */ */
/* int space_addpart */
//
/* adds the given particle to the given space at the position given */
/* by the double-precision values. */
/*/*//////////////////////////////////////////////////////////////////////////// */ */

int space_addpart ( struct space *s , struct part *p , double *x ) {

    int k, cid, ind[3];
    struct part **temp;
    struct cell **tempc;

    /* check input */
    if ( s == NULL || p == NULL || x == NULL )
        return space_err = space_err_null;
        
    /* do we need to extend the partlist? */
    if ( s->nr_parts == s->size_parts ) {
        s->size_parts += space_partlist_incr;
        if ( ( temp = (struct part **)malloc( sizeof(struct part *) * s->size_parts ) ) == NULL )
            return space_err = space_err_malloc;
        if ( ( tempc = (struct cell **)malloc( sizeof(struct cell *) * s->size_parts ) ) == NULL )
            return space_err = space_err_malloc;
        memcpy( temp , s->partlist , sizeof(struct part *) * s->nr_parts );
        memcpy( tempc , s->celllist , sizeof(struct cell *) * s->nr_parts );
        free( s->partlist );
        free( s->celllist );
        s->partlist = temp;
        s->celllist = tempc;
        }
    p->id = s->nr_parts++;
        
    /* get the hypothetical cell coordinate */
    for ( k = 0 ; k < 3 ; k++ )
        ind[k] = (x[k] - s->origin[k]) * s->ih[k];
        
    /* is this particle within the space? */
    for ( k = 0 ; k < 3 ; k++ )
        if ( ind[k] < 0 || ind[k] >= s->cdim[k] )
            return space_err = space_err_range;
            
    /* get the appropriate cell */
    cid = space_cellid(s,ind[0],ind[1],ind[2]);
    
    /* make the particle position local */
    for ( k = 0 ; k < 3 ; k++ )
        p->x[k] = x[k] - s->cells[cid].origin[k];
        
    /* delegate the particle to the cell */
    if ( ( s->partlist[p->id] = cell_add(&s->cells[cid],p) ) == NULL )
        return space_err = space_err_cell;
    s->celllist[p->id] = &( s->cells[cid] );
    
    /* end well */
    return space_err_ok;

    }


/*/*//////////////////////////////////////////////////////////////////////////// */ */
/* int space_releasepair */
//
/* free the cells involved in the current pair */
/*/*//////////////////////////////////////////////////////////////////////////// */ */

int space_releasepair ( struct space *s , struct cellpair *p ) {

    /* try to get a hold of the cells mutex */
	/* if (pthread_mutex_lock(&s->cellpairs_mutex) != 0) */
    /*     return space_err_pthread; */
        
    /* release the given pair */
    s->cells_taboo[ p->i ] -= 1;
    s->cells_taboo[ p->j ] -= 1;
    
    /* send a strong signal to anybody waiting on pairs... */
    if (pthread_cond_signal(&s->cellpairs_avail) != 0)
        return space_err_pthread;
    
    /* let go of the cells mutex */
	/* if (pthread_mutex_unlock(&s->cellpairs_mutex) != 0) */
    /*     return space_err_pthread; */
    
    /* all is well... */
    return space_err_ok;
        
    }
    

/*/*//////////////////////////////////////////////////////////////////////////// */ */
/* struct cellpair *space_getpair */
//
/* returns a pointer to the next pair or NULL if the list is empty. */
/* if anything goes wrong, the error code is returned in 'err'. */
/*/*//////////////////////////////////////////////////////////////////////////// */ */

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


/*/*//////////////////////////////////////////////////////////////////////////// */ */
/* int space_init */
//
/* initialize the space with the given dimensions. this creates the cells and */
/* generates the pair list. */
/*/*//////////////////////////////////////////////////////////////////////////// */ */

int space_init ( struct space *s , const double *origin , const double *dim , double cutoff , unsigned int period ) {

    int i, j, k, l[3], ii, jj, kk;
    int id1, id2;
    double o[3];
    float shift[3];

    /* check inputs */
    if ( s == NULL || origin == NULL || dim == NULL )
        return space_err = space_err_null;
        
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
        return space_err = space_err_malloc;
        
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
                    return space_err = space_err_cell;
                }
            }
        }
        
    /* allocate the cell pairs array (pessimistic guess) */
    if ( (s->pairs = (struct cellpair *)malloc( sizeof(struct cellpair) * s->nr_cells * 14 )) == NULL )
        return space_err = space_err_malloc;
    
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
        return space_err = space_err_malloc;
    bzero( s->cells_taboo , sizeof(char) * s->nr_pairs );
    if ( (s->cells_owner = (char *)malloc( sizeof(char) * s->nr_pairs )) == NULL )
        return space_err = space_err_malloc;
    bzero( s->cells_owner , sizeof(char) * s->nr_pairs );
    
    /* allocate the initial partlist */
    if ( ( s->partlist = (struct part **)malloc( sizeof(struct part *) * space_partlist_incr ) ) == NULL )
        return space_err = space_err_malloc;
    if ( ( s->celllist = (struct cell **)malloc( sizeof(struct cell *) * space_partlist_incr ) ) == NULL )
        return space_err = space_err_malloc;
    s->nr_parts = 0;
    s->size_parts = space_partlist_incr;
    
    /* init the cellpair mutexes */
    if (pthread_mutex_init(&s->cellpairs_mutex,NULL) != 0 ||
        pthread_cond_init(&s->cellpairs_avail,NULL) != 0)
        return space_err = space_err_pthread;
        
    /* all is well that ends well... */
    return space_err_ok;

    }
