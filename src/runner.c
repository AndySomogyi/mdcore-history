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
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef HAVE_SETAFFINITY
    #include <sched.h>
#endif
#ifdef HAVE_MPI
    #include <mpi.h>
#endif

/* Include local headers */
#include "cycle.h"
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "fifo.h"
#include "cell.h"
#include "space.h"
#include "potential.h"
#include "engine.h"
#include "runner.h"



#ifdef CELL
    /* the SPU executeable */
    extern spe_program_handle_t runner_spu;
#endif


/* Global variables. */
/** The ID of the last error. */
int runner_err = runner_err_ok;
unsigned int runner_rcount = 0;

/* the error macro. */
#define error(id)				( runner_err = errs_register( id , runner_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *runner_err_msg[11] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "An error occured when calling a space function.",
    "A call to a pthread routine failed.",
    "An error occured when calling an engine function.",
    "An error occured when calling an SPE function.",
    "An error occured with the memory flow controler.",
    "The requested functionality is not available." ,
    "An error occured when calling an fifo function." ,
    "Error filling Verlet list: too many neighbours."
	};
    
    

/**
 * @brief Sort the particles in descending order using QuickSort.
 *
 * @param parts The particle IDs and distances in compact form
 * @param N The number of particles.
 *
 * The particle data is assumed to contain the distance in the lower
 * 16 bits and the particle ID in the upper 16 bits.
 */
 
void runner_sort_descending ( unsigned int *parts , int N ) {

    struct {
        short int lo, hi;
        } qstack[10];
    int qpos, i, j, lo, hi, pivot, imax;
    unsigned int temp;
        
    /* Sort parts in cell_i in decreasing order with quicksort */
    qstack[0].lo = 0; qstack[0].hi = N - 1; qpos = 0;
    while ( qpos >= 0 ) {
        lo = qstack[qpos].lo; hi = qstack[qpos].hi;
        qpos -= 1;
        if ( hi - lo < 15 ) {
            for ( i = lo ; i < hi ; i++ ) {
                imax = i;
                for ( j = i+1 ; j <= hi ; j++ )
                    if ( (parts[j] & 0xffff) > (parts[imax] & 0xffff) )
                        imax = j;
                if ( imax != i ) {
                    temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                    }
                }
            }
        else {
            pivot = parts[ ( lo + hi ) / 2 ] & 0xffff;
            i = lo; j = hi;
            while ( i <= j ) {
                while ( (parts[i] & 0xffff) > pivot ) i++;
                while ( (parts[j] & 0xffff) < pivot ) j--;
                if ( i <= j ) {
                    if ( i < j ) {
                        temp = parts[i]; parts[i] = parts[j]; parts[j] = temp;
                        }
                    i += 1; j -= 1;
                    }
                }
            if ( j > ( lo + hi ) / 2 ) {
                if ( lo < j ) {
                    qpos += 1;
                    qstack[qpos].lo = lo;
                    qstack[qpos].hi = j;
                    }
                if ( i < hi ) {
                    qpos += 1;
                    qstack[qpos].lo = i;
                    qstack[qpos].hi = hi;
                    }
                }
            else {
                if ( i < hi ) {
                    qpos += 1;
                    qstack[qpos].lo = i;
                    qstack[qpos].hi = hi;
                    }
                if ( lo < j ) {
                    qpos += 1;
                    qstack[qpos].lo = lo;
                    qstack[qpos].hi = j;
                    }
                }
            }
        }
                
    }
    

/**
 * @brief Sort the particles in ascending order using QuickSort.
 *
 * @param parts The particle IDs and distances in compact form
 * @param N The number of particles.
 *
 * The particle data is assumed to contain the distance in the lower
 * 16 bits and the particle ID in the upper 16 bits.
 */
 
void runner_sort_ascending ( unsigned int *parts , int N ) {

    struct {
        short int lo, hi;
        } qstack[10];
    int qpos, i, j, lo, hi, pivot, imax;
    unsigned int temp;
        
    /* Sort parts in cell_i in decreasing order with quicksort */
    qstack[0].lo = 0; qstack[0].hi = N - 1; qpos = 0;
    while ( qpos >= 0 ) {
        lo = qstack[qpos].lo; hi = qstack[qpos].hi;
        qpos -= 1;
        if ( hi - lo < 15 ) {
            for ( i = lo ; i < hi ; i++ ) {
                imax = i;
                for ( j = i+1 ; j <= hi ; j++ )
                    if ( (parts[j] & 0xffff) < (parts[imax] & 0xffff) )
                        imax = j;
                if ( imax != i ) {
                    temp = parts[imax]; parts[imax] = parts[i]; parts[i] = temp;
                    }
                }
            }
        else {
            pivot = parts[ ( lo + hi ) / 2 ] & 0xffff;
            i = lo; j = hi;
            while ( i <= j ) {
                while ( (parts[i] & 0xffff) < pivot ) i++;
                while ( (parts[j] & 0xffff) > pivot ) j--;
                if ( i <= j ) {
                    if ( i < j ) {
                        temp = parts[i]; parts[i] = parts[j]; parts[j] = temp;
                        }
                    i += 1; j -= 1;
                    }
                }
            if ( j > ( lo + hi ) / 2 ) {
                if ( lo < j ) {
                    qpos += 1;
                    qstack[qpos].lo = lo;
                    qstack[qpos].hi = j;
                    }
                if ( i < hi ) {
                    qpos += 1;
                    qstack[qpos].lo = i;
                    qstack[qpos].hi = hi;
                    }
                }
            else {
                if ( i < hi ) {
                    qpos += 1;
                    qstack[qpos].lo = i;
                    qstack[qpos].hi = hi;
                    }
                if ( lo < j ) {
                    qpos += 1;
                    qstack[qpos].lo = lo;
                    qstack[qpos].hi = j;
                    }
                }
            }
        }
                
    }
    

/**
 * @brief This is the dispatcher that passes pairs
 *      to the individual #runners.
 *
 * @param r Pointer to the #engine in which the runners reside.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 */
 
int runner_dispatcher ( struct engine *e ) {

    int k, count = 0, pid, rid, cid, cjd, oid, ojd, tid, tjd;
    int min_ind, min_count;
    
    /* Local copies of variables in engine and space. */
    struct space *s = &e->s;
    struct cellpair *pairs = s->pairs;
    struct runner *r, *runners = e->runners;
    int nr_pairs = s->nr_pairs, nr_cells = s->nr_cells;
    int nr_runners = e->nr_runners;
    char *cells_taboo = s->cells_taboo, *cells_owner = s->cells_owner;
    struct fifo *dispatch_out = &s->dispatch_out;
    
    /* List of available pairs. */
    int pairs_avail[ nr_pairs ], nr_pairs_avail = nr_pairs;
    
    /* Lists of blocked pairs. */
    int pairs_blocked[ nr_cells * 27 ], nr_pairs_blocked[ nr_cells ];
    
    
    /* Init the list of available pairs. */
    for ( pid = 0 ; pid < nr_pairs ; pid++ )
        pairs_avail[ pid ] = pid;
        
    /* Init the list of blocked pairs. */
    for ( cid = 0 ; cid < nr_cells ; cid++ )
        nr_pairs_blocked[ cid ] = 0;
        
    /* Init the taboo list. */
    bzero( cells_taboo , sizeof(char) * nr_cells );
        
        
    /* Main loop. */
    while ( count < nr_pairs ) {
    
        /* Loop over the blocked pairs. */
        for ( cid = 0 ; cid < nr_cells ; cid++ ) {
        
            /* If this cell is currently free, return all pairs to
               the available list or other blocked lists. */
            if ( cells_taboo[cid] == 0 ) {
        
                /* Loop over the pairs blocked by this cell's owner. */
                for ( k = 0 ; k < nr_pairs_blocked[cid] ; k++ ) {

                    /* Get the pair ID. */
                    pid = pairs_blocked[ cid*27 + k ];
                    tid = cells_taboo[ pairs[pid].i ];
                    tjd = cells_taboo[ pairs[pid].j ];
 
                    /* Is this pair still blocked at all? */
                    if ( ( tid == 0 ) && ( tjd == 0 ) ) {

                        /* Add it to the list of available pairs. */
                        pairs_avail[ nr_pairs_avail ] = pid;
                        nr_pairs_avail += 1;

                        }
                        
                    /* Otherwise, pop it into the correct blocked list. */
                    else {
                    
                        /* Get the correct owner ID. */
                        if ( tid > 0 )
                            cjd = pairs[pid].i;
                        else
                            cjd = pairs[pid].j;
                            
                        /* Put this pair in the blocked list for cjd. */
                        // printf( "runner_dispatcher:%i: adding pid=%i to blocked cell %i.\n" , __LINE__ , pid , cjd );
                        pairs_blocked[ cjd*27 + nr_pairs_blocked[cjd] ] = pid;
                        nr_pairs_blocked[cjd] += 1;
                    
                        }
                        
                    /* Remove it from the blocked list. */
                    nr_pairs_blocked[cid] -= 1;
                    pairs_blocked[ cid*27 + k ] = pairs_blocked[ cid*27 + nr_pairs_blocked[cid] ];
                    k -= 1;

                    }
                    
                }
                
            /* Otherwise, if there is room in the runner, grab whatever is free. */
            else {

                /* Get the owner of this cell. */
                oid = cells_owner[cid];
                r = &runners[oid];
            
                /* Loop over the pairs blocked by this cell's owner. */
                for ( k = 0 ; ( r->in.count < runner_qlen ) && ( k < nr_pairs_blocked[cid] ) ; k++ ) {

                    /* Get the pair ID. */
                    pid = pairs_blocked[ cid*27 + k ];
                    
                    /* Is this pair not blocked by anybody else? */
                    if ( ( ( cells_taboo[pairs[pid].i] == 0 ) || ( cells_owner[pairs[pid].i] == oid ) ) &&
                         ( ( cells_taboo[pairs[pid].j] == 0 ) || ( cells_owner[pairs[pid].j] == oid ) ) ) {

                        /* Update cell ownership. */
                        cells_owner[ pairs[pid].i ] = oid;
                        cells_owner[ pairs[pid].j ] = oid;
                        cells_taboo[ pairs[pid].i ] += 1;
                        cells_taboo[ pairs[pid].j ] += 1;

                        /* Add this pair to the runner. */
                        if ( fifo_push( &r->in , pid ) < 0 )
                            return error(runner_err_fifo);
                        // printf( "runner_dispatch:%i: pushed pid=%i (cid=%i,npb=%i).\n" , __LINE__ , pid , cid , nr_pairs_blocked[cid] ); fflush(stdout);

                        /* Remove it from the blocked list. */
                        nr_pairs_blocked[cid] -= 1;
                        pairs_blocked[ cid*27 + k ] = pairs_blocked[ cid*27 + nr_pairs_blocked[cid] ];
                        count += 1;
                        k -= 1;

                        }

                    } /* loop over blocked pairs for this cell. */
                    
                }
        
            } /* Loop over the blocked pairs, cell-wise. */
    
    
        /* Loop over the available pairs. */
        for ( k = 0 ; k < nr_pairs_avail ; k++ ) {
        
            /* Get the pair ID. */
            pid = pairs_avail[k];
        
            /* Get owner and taboos. */
            cid = pairs[pid].i;
            cjd = pairs[pid].j;
            oid = cells_owner[ cid ];
            ojd = cells_owner[ cjd ];
            tid = cells_taboo[ cid ];
            tjd = cells_taboo[ cjd ];
            
            /* If the pair is owned by nobody... */
            if ( ( tid == 0 ) && ( tjd == 0 ) ) {
            
                /* Find the runner with the shortest queue. */
                min_ind = 0; 
                min_count = runners[0].in.count;
                for ( rid = 1 ; rid < nr_runners ; rid++ )
                    if ( runners[rid].in.count < min_count ) {
                        min_count = runners[rid].in.count;
                        min_ind = rid;
                        }
                        
                /* Did we get a runner at all? */
                if ( min_count < runner_qlen ) {
                
                    /* Update cell ownership. */
                    cells_owner[cid] = min_ind;
                    cells_owner[cjd] = min_ind;
                    cells_taboo[cid] += 1;
                    cells_taboo[cjd] += 1;
                        
                    /* Add this pair to the runner. */
                    if ( fifo_push( &runners[min_ind].in , pid ) < 0 )
                        return error(runner_err_fifo);
                    // printf( "runner_dispatch:%i: pushed pid=%i.\n" , __LINE__ , pid ); fflush(stdout);
                        
                    /* Remove it from the avail list. */
                    nr_pairs_avail -= 1;
                    pairs_avail[k] = pairs_avail[nr_pairs_avail];
                    count += 1;
                    k -= 1;
                
                    }
                
                }
                
            /* Or it is owned by only a single runner... */
            else if ( ( tid == 0 ) || ( tjd == 0 ) || ( oid == ojd ) ) {
            
                /* Make oid the owner. */
                if ( tid == 0 )
                    oid = ojd;
                r = &runners[ oid ];
                    
                /* Does the owner have any space? */
                if ( r->in.count < runner_qlen ) {
                
                    /* Update cell ownership. */
                    cells_owner[cid] = oid;
                    cells_owner[cjd] = oid;
                    cells_taboo[cid] += 1;
                    cells_taboo[cjd] += 1;
                        
                    /* Add this pair to the runner. */
                    if ( fifo_push( &r->in , pid ) < 0 )
                        return error(runner_err_fifo);
                    // printf( "runner_dispatch:%i: pushed pid=%i.\n" , __LINE__ , pid ); fflush(stdout);
                        
                    /* Remove it from the avail list. */
                    nr_pairs_avail -= 1;
                    pairs_avail[k] = pairs_avail[nr_pairs_avail];
                    count += 1;
                    k -= 1;
                
                    }
                    
                /* Otherwise, put it in the blocked queue. */
                else {

                    /* Add pair to blocked queue of its first cell. */
                    if ( tid == 0 ) cid = cjd;
                    // printf( "runner_dispatcher:%i: adding pid=%i (%i,%i) to blocked cell %i.\n" , __LINE__ , pid , pairs[pid].i , pairs[pid].j , cid );
                    pairs_blocked[ cid*27 + nr_pairs_blocked[cid] ] = pid;
                    nr_pairs_blocked[cid] += 1;

                    /* Remove it from the available list. */
                    nr_pairs_avail -= 1;
                    pairs_avail[k] = pairs_avail[nr_pairs_avail];
                    k -= 1;

                    }
            
                }
                
            /* Nope, ownership is split, put it in the blocked queue
               of the first cell. */
            else {
            
                /* Add pair to blocked queue of its first cell. */
                // printf( "runner_dispatcher:%i: adding pid=%i to blocked cell %i.\n" , __LINE__ , pid , cid );
                pairs_blocked[ cid*27 + nr_pairs_blocked[cid] ] = pid;
                nr_pairs_blocked[cid] += 1;
                
                /* Remove it from the available list. */
                nr_pairs_avail -= 1;
                pairs_avail[k] = pairs_avail[nr_pairs_avail];
                k -= 1;
                
                }
        
            } /* loop over available pairs. */
            
        /* for ( sum = count + nr_pairs_avail , cid = 0 ; cid < nr_cells ; cid++ )
            sum += nr_pairs_blocked[cid];
        printf( "runner_dispatcher: total number of pairs is %i (done=%i,avail=%i,blocked=%i).\n" ,
            sum , count , nr_pairs_avail , sum - count - nr_pairs_avail ); */
            
            
        /* Catch any finished pairs. */
        do {

            /* Get a pair from the output fifo. */
            if ( fifo_pop( dispatch_out , &pid ) < 0 )
                return error(runner_err_fifo);

            /* Get the cell indices. */
            cid = pairs[pid].i;
            cjd = pairs[pid].j;

            /* Un-mark the taboo list. */
            cells_taboo[cid] -= 1;
            cells_taboo[cjd] -= 1;
            
            } while ( dispatch_out->count > 0 );
            
            
        } /* main loop. */
        
        
    /* Tell all the runners to stop. */
    for ( rid = 0 ; rid < nr_runners ; rid++ )
        fifo_push( &e->runners[rid].in , runner_dispatch_stop );
        
        
    /* Wait for each of the runners to finish. */
    count = e->nr_runners;
    while( count > 0 ) {
        fifo_pop( &s->dispatch_out , &pid );
        if ( pid == runner_dispatch_stop )
            count -= 1;
        }
        
        
    /* Everything is just peachy. */
    return runner_err_ok;

    }
    

/**
 * @brief The #runner's main routine.
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having passed, it picks pairs out of its "in" fifo,
 * processes them, and passes them to the "out" fifo, until a
 * #runner_stop is received.
 */

int runner_run_dispatch ( struct runner *r ) {

    int k, acc = 0;
    struct cellpair *finger;
    struct engine *e;
    struct space *s;
    struct cell *c;
    int pid, cid, cjd;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* get a pointer on the engine. */
    e = r->e;
    s = &e->s;
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running (dispatch)...\n",r->id); fflush(stdout);
    
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* Try to get the next pair. */
        if ( fifo_pop( &r->in , &pid ) < 0 )
            return error(runner_err);

        /* Quit message? */
        if ( pid == runner_dispatch_stop ) {
            
            /* Send a message back... */
            if ( fifo_push( &s->dispatch_out , runner_dispatch_stop ) < 0 )
                return error(runner_err);
            
            /* And return to the top of the loop. */
            continue;
            
            }

        // printf( "runner_run_dispatch: got pid=%i.\n" , pid ); fflush(stdout);

        /* Put a finger on the pair, get the cell ids. */
        finger = &s->pairs[pid];
        cid = finger->i;
        cjd = finger->j;

        /* for each cell, prefetch the parts involved. */
        if ( e->flags & engine_flag_prefetch ) {
            c = &( e->s.cells[cid] );
            for ( k = 0 ; k < c->count ; k++ )
                acc += c->parts[k].id;
            if ( finger->i != finger->j ) {
                c = &( e->s.cells[cjd] );
                for ( k = 0 ; k < c->count ; k++ )
                    acc += c->parts[k].id;
                }
            }

        /* Verlet list? */
        if ( e->flags & engine_flag_verlet ) {

            /* We don't do dispatched Verlet lists. */
            return error(runner_err_unavail);

            }

        /* Pairwise Verlet list? */
        else if ( e->flags & engine_flag_verlet_pairwise ) {

            /* Compute the interactions of this pair. */
            if ( e->flags & engine_flag_verlet_pairwise2 ) {
                if ( runner_dopair_verlet2( r , &(s->cells[cid]) , &(s->cells[cjd]) , finger->shift , finger ) < 0 )
                    return error(runner_err);
                }
            else {
                if ( runner_dopair_verlet( r , &(s->cells[cid]) , &(s->cells[cjd]) , finger->shift , finger ) < 0 )
                    return error(runner_err);
                }

            }

        /* Otherwise, plain old... */
        else {
            if ( runner_dopair( r , &(e->s.cells[cid]) , &(e->s.cells[cjd]) , finger->shift ) < 0 )
                return error(runner_err);
            }

        /* release this pair */
        if ( fifo_push( &s->dispatch_out , pid ) < 0 )
            return error(runner_err);

        } /* while not stopped... */

    /* end well... */
    return runner_err_ok;

    }

    
/**
 * @brief The #runner's main routine (for the Cell/BE SPU).
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having paSSEd, calls #space_getpair until there are no pairs
 * available.
 *
 * Note that this routine is only compiled if @c CELL has been defined.
 */

int runner_run_cell ( struct runner *r ) {

#ifdef CELL
    int err = 0;
    struct cellpair *p[runner_qlen];
    unsigned int buff[2];
    int i, k, count = 0;
    struct space *s;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* init some local pointers. */
    s = &(r->e->s);
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running (SPU)...\n",r->id); fflush(stdout);
    
    /* init the cellpair pointers */
    for ( k = 0 ; k < runner_qlen ; k++ )
        p[k] = NULL;
        
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(r->e) < 0)
            return error(runner_err_engine);
            
        /* write the current cell data */
        for ( i = 0 ; i < s->nr_cells ; i++ ) {
            r->celldata[i].ni = s->cells[i].count;
            r->celldata[i].ai = (unsigned long long)s->cells[i].parts;
            }

        /* emit a reload message */
        buff[0] = 0xFFFFFFFF;
        /* printf("runner_run: runner %i sending reload message...\n",r->id); */
        if ( spe_in_mbox_write( r->spe , buff , 2 , SPE_MBOX_ALL_BLOCKING ) != 2 )
            return runner_err_spe;


        /* while there are pairs... */
        while ( s->next_pair < s->nr_pairs || count > 0 ) {

            /* if we have no p[0], try to get some... */
            if ( p[0] == NULL && s->next_pair < s->nr_pairs ) {
                p[0] = space_getpair( &(r->e->s) , r->id , runner_bitesize , NULL , &err , count == 0 );
                if ( err < 0 )
                    return runner_err_space;
                }

            /* if we got a pair, send it to the SPU... */
            if ( p[0] != NULL ) {

                /* we've got an active slot! */
                count += 1;

                /* pack this pair's data */
                buff[0] = ( p[0]->i << 20 ) + ( p[0]->j << 8 ) + 1;
                if ( p[0]->shift[0] > 0 )
                    buff[0] += 1 << 6;
                else if ( p[0]->shift[0] < 0 )
                    buff[0] += 2 << 6;
                if ( p[0]->shift[1] > 0 )
                    buff[0] += 1 << 4;
                else if ( p[0]->shift[1] < 0 )
                    buff[0] += 2 << 4;
                if ( p[0]->shift[2] > 0 )
                    buff[0] += 1 << 2;
                else if ( p[0]->shift[2] < 0 )
                    buff[0] += 2 << 2;

                /* wait for the buffer to be free... */
                /* while ( !spe_in_mbox_status( r->spe ) ) */
                /*     sched_yield(); */

                /* write the data to the mailbox */
                /* printf("runner_run: sending pair 0x%llx (n=%i), 0x%llx (n=%i) with shift=[%e,%e,%e].\n",
                    (unsigned long long)s->cells[p[0]->i].parts,s->cells[p[0]->i].count,(unsigned long long)s->cells[p[0]->j].parts,s->cells[p[0]->j].count,
                    p[0]->shift[0], p[0]->shift[1], p[0]->shift[2]); fflush(stdout); */
                /* printf("runner_run: runner %i sending pair to SPU...\n",r->id); fflush(stdout); */
                if ( spe_in_mbox_write( r->spe , buff , 2 , SPE_MBOX_ALL_BLOCKING ) != 2 )
                    return runner_err_spe;
                /* printf("runner_run: runner %i sent pair to SPU.\n",r->id); fflush(stdout); */


                /* wait for the last pair to have been proceSSEd */
                if ( p[runner_qlen-1] != NULL ) {

                    /* read a word from the spe */
                    /* printf("runner_run: runner %i waiting for SPU response...\n",r->id); fflush(stdout); */
                    /* if ( spe_out_intr_mbox_read( r->spe , &buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 )
                        return runner_err_spe; */
                    /* printf("runner_run: runner %i got SPU response.\n",r->id); fflush(stdout); */

                    /* release the last pair */
                    if ( space_releasepair( s , p[runner_qlen-1]->i , p[runner_qlen-1]->j ) < 0 )
                        return runner_err_space;

                    /* we've got one less... */
                    count -= 1;

                    }

                /* move on in the chain */
                for ( k = runner_qlen-1 ; k > 0 ; k-- )
                    p[k] = p[k-1];
                if ( p[0] != NULL )
                    p[0] = p[0]->next;

                /* take a breather... */
                /* sched_yield(); */

                }

            /* is there a non-empy slot, send a flush */
            else if ( count > 0 ) {

                /* send a flush message... */
                buff[0] = 0;
                if ( spe_in_mbox_write( r->spe , buff , 2 , SPE_MBOX_ALL_BLOCKING ) != 2 )
                    return runner_err_spe;

                /* wait for the reply... */
                if ( spe_out_intr_mbox_read( r->spe , buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 )
                    return runner_err_spe;
                /* printf("runner_run: got rcount=%u.\n",buff[0]); */

                /* release the pairs still in the queue */
                for ( k = 1 ; k < runner_qlen ; k++ )
                    if ( p[k] != NULL ) {
                        if ( space_releasepair( &(r->e->s) , p[k]->i , p[k]->j ) < 0 )
                            return runner_err_space;
                        p[k] = NULL;
                        count -= 1;
                        }

                }

            }
                
        /* did things go wrong? */
        /* printf("runner_run: runner %i done pairs.\n",r->id); fflush(stdout); */
        if ( err < 0 )
            return error(runner_err_space);
    
        }

    /* end well... */
    return runner_err_ok;

#else

    /* This functionality is not available */
    return runner_err_unavail;
    
#endif

    }


/**
 * @brief The #runner's main routine (for the Cell/BE SPU, using
 *      tuples).
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having paSSEd, calls #space_gettuple until there are no
 * tuples available.
 *
 * Note that this routine is only compiled if @c CELL has been defined.
 */

int runner_run_cell_tuples ( struct runner *r ) {

#ifdef CELL
    int err = 0;
    unsigned int buff[2];
    int i, j, k, count = 0, res;
    struct space *s;
    struct celltuple *t;
    int cid[runner_qlen][2];
    FPTYPE shift[3];
    

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* init some local pointers. */
    s = &(r->e->s);
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running (SPU)...\n",r->id); fflush(stdout);
    
    /* init the cellpair pointers */
    for ( k = 0 ; k < runner_qlen ; k++ )
        cid[k][0] = -1;
        
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(r->e) < 0)
            return error(runner_err_engine);
            
        /* write the current cell data */
        for ( i = 0 ; i < s->nr_cells ; i++ ) {
            r->celldata[i].ni = s->cells[i].count;
            r->celldata[i].ai = (unsigned long long)s->cells[i].parts;
            }

        /* emit a reload message */
        buff[0] = 0xFFFFFFFF;
        /* printf("runner_run: runner %i sending reload message...\n",r->id); */
        if ( spe_in_mbox_write( r->spe , buff , 2 , SPE_MBOX_ALL_BLOCKING ) != 2 )
            return runner_err_spe;


        /* Loop over tuples... */
        while ( 1 ) {
        
            /* Try to get a tuple. */
            if ( ( res = space_gettuple( s , &t , count == 0 ) ) < 0 )
                return r->err = runner_err_space;
                
            /* Did we get a tuple back? */
            if ( res > 0 )
                
                /* Loop over the cell pairs in this tuple. */
                for ( i = 0 ; i < t->n ; i++ )
                    for ( j = i ; j < t->n ; j++ ) {

                        /* Is this pair active? */
                        if ( t->pairid[ space_pairind(i,j) ] < 0 )
                            continue;

                        /* Get the cell ids. */
                        cid[0][0] = t->cellid[i];
                        cid[0][1] = t->cellid[j];

                        /* we've got an active slot! */
                        count += 1;

                        /* Compute the shift between ci and cj. */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            shift[k] = s->cells[cid[0][1]].origin[k] - s->cells[cid[0][0]].origin[k];
                            if ( shift[k] * 2 > s->dim[k] )
                                shift[k] -= s->dim[k];
                            else if ( shift[k] * 2 < -s->dim[k] )
                                shift[k] += s->dim[k];
                            }

                        /* pack this pair's data */
                        buff[0] = ( cid[0][0] << 20 ) + ( cid[0][1] << 8 ) + 1;
                        if ( shift[0] > 0 )
                            buff[0] += 1 << 6;
                        else if ( shift[0] < 0 )
                            buff[0] += 2 << 6;
                        if ( shift[1] > 0 )
                            buff[0] += 1 << 4;
                        else if ( shift[1] < 0 )
                            buff[0] += 2 << 4;
                        if ( shift[2] > 0 )
                            buff[0] += 1 << 2;
                        else if ( shift[2] < 0 )
                            buff[0] += 2 << 2;

                        /* write the data to the mailbox */
                        /* printf("runner_run: sending pair 0x%llx (n=%i), 0x%llx (n=%i) with shift=[%e,%e,%e].\n",
                            (unsigned long long)s->cells[p[0]->i].parts,s->cells[p[0]->i].count,(unsigned long long)s->cells[p[0]->j].parts,s->cells[p[0]->j].count,
                            p[0]->shift[0], p[0]->shift[1], p[0]->shift[2]); fflush(stdout); */
                        /* printf("runner_run: runner %i sending pair to SPU...\n",r->id); fflush(stdout); */
                        if ( spe_in_mbox_write( r->spe , buff , 2 , SPE_MBOX_ALL_BLOCKING ) != 2 )
                            return runner_err_spe;
                        /* printf("runner_run: runner %i sent pair to SPU.\n",r->id); fflush(stdout); */


                        /* wait for the last pair to have been processed */
                        if ( cid[runner_qlen-1][0] >= 0 ) {

                            /* read a word from the spe */
                            /* printf("runner_run: runner %i waiting for SPU response...\n",r->id); fflush(stdout); */
                            /* if ( spe_out_intr_mbox_read( r->spe , &buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 )
                                return runner_err_spe; */
                            /* printf("runner_run: runner %i got SPU response.\n",r->id); fflush(stdout); */

                            /* release the last pair */
                            if ( space_releasepair( s , cid[runner_qlen-1][0] , cid[runner_qlen-1][1] ) < 0 )
                                return runner_err_space;

                            /* we've got one less... */
                            count -= 1;

                            }

                        /* move on in the chain */
                        for ( k = runner_qlen-1 ; k > 0 ; k-- ) {
                            cid[k][0] = cid[k-1][0];
                            cid[k][1] = cid[k-1][1];
                            }
                        cid[0][0] = -1;

                        }
            
            /* Did we get a stall? */
            else if ( s->next_tuple < s->nr_tuples ) {
            
                /* wait for the last pair to have been processed */
                if ( cid[runner_qlen-1][0] >= 0 ) {

                    /* read a word from the spe */
                    /* printf("runner_run: runner %i waiting for SPU response...\n",r->id); fflush(stdout); */
                    /* if ( spe_out_intr_mbox_read( r->spe , &buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 )
                        return runner_err_spe; */
                    /* printf("runner_run: runner %i got SPU response.\n",r->id); fflush(stdout); */

                    /* release the last pair */
                    if ( space_releasepair( s , cid[runner_qlen-1][0] , cid[runner_qlen-1][1] ) < 0 )
                        return runner_err_space;

                    /* we've got one less... */
                    count -= 1;

                    }

                /* move on in the chain */
                for ( k = runner_qlen-1 ; k > 0 ; k-- ) {
                    cid[k][0] = cid[k-1][0];
                    cid[k][1] = cid[k-1][1];
                    }
                cid[0][0] = -1;

                }
                
            /* Otherwise, we're done. */
            else
                break;
                    
            }

        /* If there is a non-empy slot, send a flush */
        if ( count > 0 ) {

            /* send a flush message... */
            buff[0] = 0;
            if ( spe_in_mbox_write( r->spe , buff , 2 , SPE_MBOX_ALL_BLOCKING ) != 2 )
                return runner_err_spe;

            /* wait for the reply... */
            if ( spe_out_intr_mbox_read( r->spe , buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 )
                return runner_err_spe;
            // printf("runner_run: got rcount=%u.\n",buff[0]);

            /* release the pairs still in the queue */
            for ( k = 1 ; k < runner_qlen ; k++ )
                if ( cid[k][0] >= 0 ) {
                    if ( space_releasepair( &(r->e->s) , cid[k][0] , cid[k][1] ) < 0 )
                        return runner_err_space;
                    cid[k][0] = -1;
                    count -= 1;
                    }

            }

        /* did things go wrong? */
        /* printf("runner_run: runner %i done pairs.\n",r->id); fflush(stdout); */
        if ( err < 0 )
            return error(runner_err_space);
    
        }

    /* end well... */
    return runner_err_ok;

#else

    /* This functionality is not available */
    return runner_err_unavail;
    
#endif

    }


/**
 * @brief The #runner's main routine (for Verlet lists).
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having passed, checks first if the Verlet list should
 * be re-built and then proceeds to traverse the Verlet list cell-wise
 * and computes its interactions.
 */

int runner_run_verlet ( struct runner *r ) {

    int res, i, ci, j, cj, k, eff_size = 0, acc = 0;
    struct engine *e;
    struct space *s;
    struct celltuple *t;
    struct cell *c;
    FPTYPE shift[3], *eff = NULL;
    int count;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* get a pointer on the engine. */
    e = r->e;
    s = &(e->s);
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running (Verlet)...\n",r->id); fflush(stdout);
    
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(e) < 0)
            return error(runner_err_engine);
            
        // runner_rcount = 0;
            
        /* Does the Verlet list need to be reconstructed? */
        if ( s->verlet_rebuild ) {
        
            /* Loop over tuples. */
            while ( 1 ) {

                /* Get a tuple. */
                if ( ( res = space_gettuple( s , &t , 1 ) ) < 0 )
                    return r->err = runner_err_space;

                /* If there were no tuples left, bail. */
                if ( res < 1 )
                    break;
                    
                /* for each cell, prefetch the parts involved. */
                if ( e->flags & engine_flag_prefetch )
                    for ( i = 0 ; i < t->n ; i++ ) {
                        c = &( s->cells[t->cellid[i]] );
                        for ( k = 0 ; k < c->count ; k++ )
                            acc += c->parts[k].id;
                        }

                /* Loop over all pairs in this tuple. */
                for ( i = 0 ; i < t->n ; i++ ) { 

                    /* Get the cell ID. */
                    ci = t->cellid[i];

                    for ( j = i ; j < t->n ; j++ ) {

                        /* Is this pair active? */
                        if ( t->pairid[ space_pairind(i,j) ] < 0 )
                            continue;

                        /* Get the cell ID. */
                        cj = t->cellid[j];

                        /* Compute the shift between ci and cj. */
                        for ( k = 0 ; k < 3 ; k++ ) {
                            shift[k] = s->cells[cj].origin[k] - s->cells[ci].origin[k];
                            if ( shift[k] * 2 > s->dim[k] )
                                shift[k] -= s->dim[k];
                            else if ( shift[k] * 2 < -s->dim[k] )
                                shift[k] += s->dim[k];
                            }

                        /* Rebuild the Verlet entries for this cell pair. */
                        if ( runner_verlet_fill( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                            return error(runner_err);
                            
                        /* release this pair */
                        if ( space_releasepair( s , ci , cj ) < 0 )
                            return error(runner_err_space);

                        }

                    }
                    
                } /* loop over tuples. */

            /* did anything go wrong? */
            if ( res < 0 )
                return error(runner_err_space);
                
            } /* reconstruct the Verlet list. */
            
        /* Otherwise, just run through the Verlet list. */
        else {
            
            /* Check if eff is large enough and re-allocate if needed. */
            if ( eff_size < s->nr_parts ) {

                /* Free old eff? */
                if ( eff != NULL )
                    free( eff );

                /* Allocate new eff. */
                eff_size = s->nr_parts * 1.1;
                if ( ( eff = (FPTYPE *)malloc( sizeof(FPTYPE) * eff_size * 4 ) ) == NULL )
                    return error(runner_err_malloc);

                }

            /* Reset the force vector. */
            bzero( eff , sizeof(FPTYPE) * s->nr_parts * 4 );

            /* Re-set the potential energy. */
            r->epot = 0.0;

            /* While there are still chunks of the Verlet list out there... */
            while ( ( count = space_getcell( s , &c ) ) > 0 ) {

                /* Dispatch the interactions to runner_verlet_eval. */
                runner_verlet_eval( r , c , eff );

                }

            /* did things go wrong? */
            if ( count < 0 )
                return error(runner_err_space);

            /* Send the forces and energy back to the space. */
            if ( space_verlet_force( s , eff , r->epot ) < 0 )
                return error(runner_err_space);
            
            }

        /* Print the rcount. */
        // printf("runner_run_verlet: runner_rcount=%i.\n", runner_rcount);
        r->err = acc;
            
        }

    /* end well... */
    return runner_err_ok;

    }

    
/**
 * @brief The #runner's main routine.
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having paSSEd, calls #space_getpair until there are no pairs
 * available.
 */

int runner_run_pairs ( struct runner *r ) {

    int k, err = 0, acc = 0;
    struct cellpair *p = NULL;
    struct cellpair *finger;
    struct engine *e;
    struct space *s;
    struct cell *ci, *cj;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* get a pointer on the engine. */
    e = r->e;
    s = &e->s;
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running (pairs)...\n",r->id); fflush(stdout);
    
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(e) < 0)
            return error(runner_err_engine);
                        
        /* while i can still get a pair... */
        /* printf("runner_run: runner %i paSSEd barrier, getting pairs...\n",r->id); */
        while ( ( p = space_getpair( s , r->id , runner_bitesize , NULL , &err , 1 ) ) != NULL ) {

            /* work this list of pair... */
            for ( finger = p ; finger != NULL ; finger = finger->next ) {

                /* Get the cells. */
                ci = &( s->cells[ finger->i ] );
                cj = &( s->cells[ finger->j ] );
                
                /* for each cell, prefetch the parts involved. */
                if ( e->flags & engine_flag_prefetch ) {
                    for ( k = 0 ; k < ci->count ; k++ )
                        acc += ci->parts[k].id;
                    if ( finger->i != finger->j )
                        for ( k = 0 ; k < cj->count ; k++ )
                            acc += cj->parts[k].id;
                    }

                /* Sorted interactions? */
                if ( e->flags & engine_flag_verlet_pairwise2 ) {
                    if ( runner_dopair_verlet2( r , ci , cj , finger->shift , finger ) < 0 )
                        return error(runner_err);
                    }
                else if ( e->flags & engine_flag_verlet_pairwise ) {
                    if ( runner_dopair_verlet( r , ci , cj , finger->shift , finger ) < 0 )
                        return error(runner_err);
                    }
                else if ( e->flags & engine_flag_unsorted ) {
                    if ( runner_dopair_unsorted( r , ci , cj , finger->shift ) < 0 )
                        return error(runner_err);
                    }
                else {
                    if ( runner_dopair( r , ci , cj , finger->shift ) < 0 )
                        return error(runner_err);
                    }

                /* release this pair */
                if ( space_releasepair( s , finger->i , finger->j ) < 0 )
                    return error(runner_err_space);

                }

            }

        /* give the reaction count */
        /* printf("runner_run: last count was %u.\n",runner_rcount); */
        r->err = acc;
            
        /* did things go wrong? */
        /* printf("runner_run: runner %i done pairs.\n",r->id); fflush(stdout); */
        if ( err < 0 )
            return error(runner_err_space);
    
        }

    /* end well... */
    return runner_err_ok;

    }

    
/**
 * @brief The #runner's main routine (#celltuple model).
 *
 * @param r Pointer to the #runner to run.
 *
 * @return #runner_err_ok or <0 on error (see #runner_err).
 *
 * This is the main routine for the #runner. When called, it enters
 * an infinite loop in which it waits at the #engine @c r->e barrier
 * and, once having passed, calls #space_gettuple until there are no
 * tuples available.
 */

int runner_run_tuples ( struct runner *r ) {

    int res, i, j, k, ci, cj, acc = 0;
    struct celltuple *t;
    FPTYPE shift[3];
    struct space *s;
    struct engine *e;
    struct cell *c;

    /* check the inputs */
    if ( r == NULL )
        return error(runner_err_null);
        
    /* Remember who the engine and the space are. */
    e = r->e;
    s = &(r->e->s);
        
    /* give a hoot */
    printf("runner_run: runner %i is up and running (tuples)...\n",r->id); fflush(stdout);
    
    /* main loop, in which the runner should stay forever... */
    while ( 1 ) {
    
        /* wait at the engine barrier */
        /* printf("runner_run: runner %i waiting at barrier...\n",r->id); */
        if ( engine_barrier(e) < 0 )
            return r->err = runner_err_engine;
            
        // runner_rcount = 0;
                        
        /* Loop over tuples. */
        while ( 1 ) {
        
            /* Get a tuple. */
            if ( ( res = space_gettuple( s , &t , 1 ) ) < 0 )
                return r->err = runner_err_space;
                
            /* If there were no tuples left, bail. */
            if ( res < 1 )
                break;
                
            /* for each cell, prefetch the parts involved. */
            if ( e->flags & engine_flag_prefetch )
                for ( i = 0 ; i < t->n ; i++ ) {
                    c = &( s->cells[t->cellid[i]] );
                    for ( k = 0 ; k < c->count ; k++ )
                        acc += c->parts[k].id;
                    }

            /* Loop over all pairs in this tuple. */
            for ( i = 0 ; i < t->n ; i++ ) { 
                        
                /* Get the cell ID. */
                ci = t->cellid[i];
                    
                for ( j = i ; j < t->n ; j++ ) {
                
                    /* Is this pair active? */
                    if ( t->pairid[ space_pairind(i,j) ] < 0 )
                        continue;
                        
                    /* Get the cell ID. */
                    cj = t->cellid[j];

                    /* Compute the shift between ci and cj. */
                    for ( k = 0 ; k < 3 ; k++ ) {
                        shift[k] = s->cells[cj].origin[k] - s->cells[ci].origin[k];
                        if ( shift[k] * 2 > s->dim[k] )
                            shift[k] -= s->dim[k];
                        else if ( shift[k] * 2 < -s->dim[k] )
                            shift[k] += s->dim[k];
                        }
                    
                    /* Sorted interactions? */
                    if ( e->flags & engine_flag_verlet_pairwise2 ) {
                        if ( runner_dopair_verlet2( r , &(s->cells[ci]) , &(s->cells[cj]) , shift , &(s->pairs[ t->pairid[ space_pairind(i,j) ] ]) ) < 0 )
                            return error(runner_err);
                        }
                    else if ( e->flags & engine_flag_verlet_pairwise ) {
                        if ( runner_dopair_verlet( r , &(s->cells[ci]) , &(s->cells[cj]) , shift , &(s->pairs[ t->pairid[ space_pairind(i,j) ] ]) ) < 0 )
                            return error(runner_err);
                        }
                    else if ( e->flags & engine_flag_unsorted ) {
                        if ( runner_dopair_unsorted( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                            return error(runner_err);
                        }
                    else {
                        if ( runner_dopair( r , &(s->cells[ci]) , &(s->cells[cj]) , shift ) < 0 )
                            return error(runner_err);
                        }

                    /* release this pair */
                    if ( space_releasepair( s , ci , cj ) < 0 )
                        return error(runner_err_space);
                        
                    }
                    
                }
                
            } /* loop over the tuples. */

        /* give the reaction count */
        // printf("runner_run_tuples: runner_rcount=%u.\n",runner_rcount);
        r->err = acc;
            
        }

    /* end well... */
    return runner_err_ok;

    }

    
/**
 * @brief Initialize the runner associated to the given engine and
 *      attach it to an SPU.
 * 
 * @param r The #runner to be initialized.
 * @param e The #engine with which it is associated.
 * @param id The ID of this #runner.
 * 
 * @return #runner_err_ok or < 0 on error (see #runner_err).
 *
 * If @c CELL is not defined, this routine will fail!
 */

int runner_init_SPU ( struct runner *r , struct engine *e , int id ) {

#ifdef CELL
    static void *data = NULL;
    static int size_data = 0;
    void *finger;
    int nr_pots = 0, size_pots = 0, *pots, i, j, k, l;
    struct potential *p;
    unsigned int buff;

    /* make sure the inputs are ok */
    if ( r == NULL || e == NULL )
        return error(runner_err_null);
        
    /* remember who i'm working for */
    r->e = e;
    r->id = id;
    
    /* if this has not been done before, init the runner data */
    if ( data == NULL ) {

        /* run through the potentials and count them and their size */
        for ( i = 0 ; i < e->max_type ; i++ )
            for ( j = i ; j < e->max_type ; j++ )
                if ( e->p[ i * e->max_type + j ] != NULL ) {
                    nr_pots += 1;
                    size_pots += e->p[ i * e->max_type + j ]->n + 1;
                    }

        /* the main data consists of a pointer to the cell data (64 bit),
           the nr of cells (int), the cutoff (float), the width of
           each cell (float[3]), the max nr of types (int)
           and an array of size max_type*max_type of offsets (int) */
        size_data = sizeof(void *) + sizeof(int) + 4 * sizeof(float) + sizeof(int) * ( 1 + e->max_type*e->max_type );

        /* stretch this data until we are aligned to 8 bytes */
        size_data = ( size_data + 7 ) & ~7;

        /* we then append nr_pots potentials consisting of three floats (alphas) */
        /* and two ints (n and flags) */
        size_data += nr_pots * ( 3 * sizeof(float) + 2 * sizeof(int) );

        /* finally, we append the data of each interval of each potential */
        /* which consists of eight floats */
        size_data += size_pots * sizeof(float) * potential_chunk;

        /* raise to multiple of 128 */
        size_data = ( size_data + 127 ) & ~127;

        /* allocate memory for the SPU data */
        if ( ( data = malloc_align( size_data , 7 ) ) == NULL )
            return error(runner_err_malloc);

        /* fill-in the engine data (without the pots) */
        finger = data;
        *((unsigned long long *)finger) = 0; finger += sizeof(unsigned long long);
        *((int *)finger) = e->s.nr_cells; finger += sizeof(int);
        *((float *)finger) = e->s.cutoff; finger += sizeof(float);
        *((float *)finger) = e->s.h[0]; finger += sizeof(float);
        *((float *)finger) = e->s.h[1]; finger += sizeof(float);
        *((float *)finger) = e->s.h[2]; finger += sizeof(float);
        *((int *)finger) = e->max_type; finger += sizeof(int);
        pots = (int *)finger; finger += e->max_type * e->max_type * sizeof(int);
        for ( i = 0 ; i < e->max_type*e->max_type ; i++ )
            pots[i] = 0;

        /* move the finger until we are at an 8-byte boundary */
        finger = (void *)( ( (unsigned long long)finger + 7 ) & ~7 );

        /* loop over the potentials */
        for ( i = 0 ; i < e->max_type ; i++ )
            for ( j = i ; j < e->max_type ; j++ )
                if ( pots[ i * e->max_type + j ] == 0 && e->p[ i * e->max_type + j ] != NULL ) {
                    p = e->p[ i * e->max_type + j ];
                    for ( k = 0 ; k < e->max_type*e->max_type ; k++ )
                        if ( e->p[k] == p )
                            pots[k] = finger - data;
                    *((int *)finger) = p->n; finger += sizeof(int);
                    *((int *)finger) = p->flags; finger += sizeof(int);
                    *((float *)finger) = p->alpha[0]; finger += sizeof(float);
                    *((float *)finger) = p->alpha[1]; finger += sizeof(float);
                    *((float *)finger) = p->alpha[2]; finger += sizeof(float);
                    /* loop explicitly in case FPTYPE is not float. */
                    for ( k = 0 ; k <= p->n ; k++ ) {
                        for ( l = 0 ; l < potential_chunk ; l++ ) {
                            *((float *)finger) = p->c[k*potential_chunk+l];
                            finger += sizeof(float);
                            }
                        }
                    }

        /* raise to multiple of 128 */
        finger = (void *)( ( (unsigned long long)finger + 127 ) & ~127 );

        /* if the effective size is smaller than the allocated size */
        /* (e.g. duplicate potentials), be clean and re-allocate the data */
        if ( finger - data < size_data ) {
            size_data = finger - data;
            if ( ( data = realloc_align( data , size_data , 7 ) ) == NULL )
                return error(runner_err_malloc);
            }

        /* say something about it all */
        /* printf("runner_init: initialized data with %i bytes.\n",size_data); */

        } /* init runner data */

    /* remember where the data is */
    r->data = data;

    /* allocate and set the cell data */
    if ( ( r->celldata = (struct celldata *)malloc_align( ceil128( sizeof(struct celldata) * r->e->s.nr_cells ) , 7 ) ) == NULL )
        return error(runner_err_malloc);
    *((unsigned long long *)data) = (unsigned long long)r->celldata;

    /* get a handle on an SPU */
    if ( ( r->spe = spe_context_create(0, NULL) ) == NULL )
        return error(runner_err_spe);

    /* load the image onto the SPU */
    if ( spe_program_load( r->spe , &runner_spu ) != 0 )
        return error(runner_err_spe);

    /* dummy function that just starts the SPU... */
    int dummy ( struct runner *r ) {
        return spe_context_run( r->spe , &(r->entry) , 0 , r->data , (void *)(unsigned long long)size_data , NULL );
        }

    /* start the SPU with a pointer to the data */
    r->entry = SPE_DEFAULT_ENTRY;
	if (pthread_create(&r->spe_thread,NULL,(void *(*)(void *))dummy,r) != 0)
		return error(runner_err_pthread);

    /* wait until the SPU is ready... */
    if ( spe_out_intr_mbox_read( r->spe , &buff , 1 , SPE_MBOX_ALL_BLOCKING ) < 1 )
        return runner_err_spe;

    /* start the runner. */
    if ( e->flags & engine_flag_tuples ) {
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run_cell_tuples,r) != 0)
		    return error(runner_err_pthread);
        }
    else {
	    if (pthread_create(&r->thread,NULL,(void *(*)(void *))runner_run_cell,r) != 0)
		    return error(runner_err_pthread);
        }

    /* all is well... */
    return runner_err_ok;
    
#else
        
    /* if not compiled for cell, then this option is not available. */
    return error(runner_err_unavail);
        
#endif
        
    }
    
    
/**
 * @brief Initialize the runner associated to the given engine.
 * 
 * @param r The #runner to be initialized.
 * @param e The #engine with which it is associated.
 * @param id The ID of this #runner.
 * 
 * @return #runner_err_ok or < 0 on error (see #runner_err).
 */

int runner_init ( struct runner *r , struct engine *e , int id ) {

    #if defined(HAVE_SETAFFINITY) && !defined(CELL)
        cpu_set_t cpuset;
    #endif

    /* make sure the inputs are ok */
    if ( r == NULL || e == NULL )
        return error(runner_err_null);
        
    /* remember who i'm working for */
    r->e = e;
    r->id = id;
    
    /* Init the input fifo for the dispatcher. */
    if ( fifo_init( &r->in , runner_qlen ) < 0 )
        return error(runner_err);
    
    /* init the thread using a dispatcher. */
    if ( e->flags & engine_flag_dispatch ) {
	    if ( pthread_create( &r->thread , NULL , (void *(*)(void *))runner_run_dispatch , r ) != 0 )
		    return error(runner_err_pthread);
        }
        
    /* init the thread using a global Verlet list. */
    else if ( e->flags & engine_flag_verlet && !(e->flags & engine_flag_verlet_pairwise) ) {
	    if ( pthread_create( &r->thread , NULL , (void *(*)(void *))runner_run_verlet , r ) != 0 )
		    return error(runner_err_pthread);
        }
        
    /* init the thread using tuples. */
    else if ( e->flags & engine_flag_tuples ) {
	    if ( pthread_create( &r->thread , NULL , (void *(*)(void *))runner_run_tuples , r ) != 0 )
		    return error(runner_err_pthread);
        }
        
    /* default: use the normal pair-list instead. */
    else {
	    if ( pthread_create( &r->thread , NULL , (void *(*)(void *))runner_run_pairs , r ) != 0 )
		    return error(runner_err_pthread);
        }
    
    /* If we can, try to restrict this runner to a single CPU. */
    #if defined(HAVE_SETAFFINITY) && !defined(CELL)
        if ( e->flags & engine_flag_affinity ) {
        
            /* Set the cpu mask to zero | r->id. */
            CPU_ZERO( &cpuset );
            CPU_SET( r->id , &cpuset );

            /* Apply this mask to the runner's pthread. */
            if ( pthread_setaffinity_np( r->thread , sizeof(cpu_set_t) , &cpuset ) != 0 )
                return error(runner_err_pthread);

            }
    #endif
    
    /* all is well... */
    return runner_err_ok;
    
    }
