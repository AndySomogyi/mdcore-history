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


/** ID of the last error. */
int engine_err = engine_err_ok;


/* the error macro. */
#define error(id)				( engine_err = errs_register( id , engine_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *engine_err_msg[19] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "An error occured when calling a space function.",
    "A call to a pthread routine failed.",
    "An error occured when calling a runner function.",
    "One or more values were outside of the allowed range.",
    "An error occured while calling a cell function.",
    "The computational domain is too small for the requested operation.",
    "mdcore was not compiled with MPI.",
    "An error occured while calling an MPI function.",
    "An error occured when calling a bond function.", 
    "An error occured when calling an angle function.",
    "An error occured when calling a reader function.",
    "An error occured while interpreting the PSF file.",
    "An error occured while interpreting the PDB file.",
    "An error occured while interpreting the CPF file.",
    "An error occured when calling a potential function.", 
    "An error occured when calling an exclusion function.", 
	};


/**
 * @brief Set all the engine timers to 0.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_timers_reset ( struct engine *e ) {

    int k;
    
    /* Check input nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Run through the timers and set them to 0. */
    for ( k = 0 ; k < engine_timer_last ; k++ )
        e->timers[k] = 0;
        
    /* What, that's it? */
    return engine_err_ok;

    }
    

/**
 * @brief Check if the Verlet-list needs to be updated.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_verlet_update ( struct engine *e ) {

    int cid, pid, k;
    double dx, w, maxdx = 0.0, skin;
    struct cell *c;
    struct part *p;
    struct space *s = &e->s;
    ticks tic;
    #ifdef HAVE_OPENMP
        int step;
        double lmaxdx;
    #endif
    
    /* Do we really need to do this? */
    if ( !(e->flags & engine_flag_verlet) )
        return engine_err_ok;
    
    /* Get the skin width. */    
    skin = fmin( s->h[0] , fmin( s->h[1] , s->h[2] ) ) - s->cutoff;
    
    /* Get the maximum particle movement. */
    #ifdef HAVE_OPENMP
        #pragma omp parallel private(c,cid,pid,p,dx,k,w,step,lmaxdx)
        {
            lmaxdx = 0.0; step = omp_get_num_threads();
            for ( cid = omp_get_thread_num() ; cid < s->nr_real ; cid += step ) {
                c = &(s->cells[s->cid_real[cid]]);
                for ( pid = 0 ; pid < c->count ; pid++ ) {
                    p = &(c->parts[pid]);
                    for ( dx = 0.0 , k = 0 ; k < 3 ; k++ ) {
                        w = p->x[k] - c->oldx[ 4*pid + k ];
                        dx += w*w;
                        }
                    lmaxdx = fmax( dx , lmaxdx );
                    }
                }
            #pragma omp critical
            maxdx = fmax( lmaxdx , maxdx );
            }
    #else
        for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
            c = &(s->cells[s->cid_real[cid]]);
            for ( pid = 0 ; pid < c->count ; pid++ ) {
                p = &(c->parts[pid]);
                for ( dx = 0.0 , k = 0 ; k < 3 ; k++ ) {
                    w = p->x[k] - c->oldx[ 4*pid + k ];
                    dx += w*w;
                    }
                maxdx = fmax( dx , maxdx );
                }
            }
    #endif

    #ifdef HAVE_MPI
    /* Collect the maximum displacement from other nodes. */
    if ( ( e->flags & engine_flag_mpi ) && ( e->nr_nodes > 1 ) ) {
        /* Do not use in-place as it is buggy when async is going on in the background. */
        if ( MPI_Allreduce( MPI_IN_PLACE , &maxdx , 1 , MPI_DOUBLE_PRECISION , MPI_MAX , e->comm ) != MPI_SUCCESS )
            return error(engine_err_mpi);
        }
    #endif

    /* Are we still in the green? */
    s->verlet_rebuild = ( 2.0*sqrt(maxdx) > skin );
        
    /* Do we have to rebuild the Verlet list? */
    if ( s->verlet_rebuild ) {
    
        /* printf("engine_verlet_update: re-building verlet lists next step...\n");
        printf("engine_verlet_update: maxdx=%e, skin=%e.\n",sqrt(maxdx),skin); */
        
        /* Wait for any unterminated exchange. */
        tic = getticks();
        if ( e->flags & engine_flag_async )
            if ( engine_exchange_wait( e ) < 0 )
                return error(engine_err);
        tic = getticks() - tic;
        e->timers[engine_timer_exchange1] += tic;
        e->timers[engine_timer_verlet] -= tic;
                
        /* Flush the ghost cells (to avoid overlapping particles) */
        #pragma omp parallel for schedule(static), private(cid)
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ )
            cell_flush( &(s->cells[s->cid_ghost[cid]]) , s->partlist , s->celllist );
        
        /* Shuffle the domain. */
        if ( space_shuffle_local( s ) < 0 )
            return error(engine_err_space);
            
        /* Get the incomming particle from other procs if needed. */
        if ( e->flags & engine_flag_mpi )
            if ( engine_exchange_incomming( e ) < 0 )
                return error(engine_err);
                        
        /* Welcome the new particles in each cell, unhook the old ones. */
        #pragma omp parallel for schedule(static), private(cid,c,k)
        for ( cid = 0 ; cid < s->nr_marked ; cid++ ) {
            c = &(s->cells[s->cid_marked[cid]]);
            if ( !(c->flags & cell_flag_ghost) )
                cell_welcome( c , s->partlist );
            else {
                for ( k = 0 ; k < c->incomming_count ; k++ )
                    e->s.partlist[ c->incomming[k].id ] = NULL;
                c->incomming_count = 0;
                }
            }

        /* Store the current positions as a reference. */
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,k)
        for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
            c = &(s->cells[s->cid_real[cid]]);
            if ( c->oldx == NULL || c->oldx_size < c->count ) {
                free(c->oldx);
                c->oldx_size = c->size + 20;
                c->oldx = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * c->oldx_size );
                }
            for ( pid = 0 ; pid < c->count ; pid++ ) {
                p = &(c->parts[pid]);
                for ( k = 0 ; k < 3 ; k++ )
                    c->oldx[ 4*pid + k ] = p->x[k];
                }
            }
            
        /* Set the nrpairs to zero. */
        if ( !( e->flags & engine_flag_verlet_pairwise ) && s->verlet_nrpairs != NULL )
            bzero( s->verlet_nrpairs , sizeof(int) * s->nr_parts );

        }
            
    /* All done! */
    return engine_err_ok;

    }
    

/**
 * @brief Read the potentials from a XPLOR parameter file.
 *
 * @param e The #engine.
 * @param xplor The open XPLOR parameter file.
 * @param kappa The PME screening width.
 * @param tol The absolute tolerance for interpolation.
 * @param rigidH Convert all bonds over a type starting with 'H'
 *      to a rigid constraint.
 *
 * If @c kappa is zero, truncated Coulomb electrostatic interactions are
 * assumed. If @c kappa is less than zero, no electrostatic interactions
 * are computed.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_read_xplor ( struct engine *e , FILE *xplor , double kappa , double tol , int rigidH ) {

    struct reader r;
    char buff[100], type1[100], type2[100], type3[100], type4[100], *endptr;
    int tid, tjd, wc[4];
    int res, j, k, n, *ind1, *ind2, nr_ind1, nr_ind2, potid;
    double K, Kr0, r0, r2, r6, *eps, *rmin, A, B, q, al, ar, am, vl, vr, vm;
    struct potential *p;
    
    /* Check inputs. */
    if ( e == NULL || xplor == NULL )
        return error(engine_err_null);
        
    /* Allocate some local memory for the index arrays. */
    if ( ( ind1 = (int *)alloca( sizeof(int) * e->nr_types ) ) == NULL ||
         ( ind2 = (int *)alloca( sizeof(int) * e->nr_types ) ) == NULL ||
         ( eps = (double *)alloca( sizeof(double) * e->nr_types ) ) == NULL ||
         ( rmin = (double *)alloca( sizeof(double) * e->nr_types ) ) == NULL )
        return error(engine_err_malloc);
    bzero( eps , sizeof(double) * e->nr_types );
    bzero( rmin , sizeof(double) * e->nr_types );
        
    /* Init the reader with the XPLOR file. */
    if ( reader_init( &r , xplor , NULL , "!{" , "\n" ) < 0 )
        return error(engine_err_reader);
        
    /* Main loop. */
    while ( !( r.flags & reader_flag_eof ) ) {
    
        /* Get the first token */
        if ( ( res = reader_gettoken( &r , buff , 100 ) ) == reader_err_eof )
            break;
        else if ( res < 0 )
            return error(engine_err_reader);

        /* Did we get a bond? */
        if ( strncasecmp( buff , "BOND" , 4 ) == 0 ) {
    
            /* Get the atom types. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type2 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Get the parameters K and r0. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);

            /* Is this a rigid bond (and do we care)? */  
            if ( rigidH && ( type1[0] == 'H' || type2[0] == 'H' ) ) {

                /* Loop over all bonds... */
                for ( k = 0 ; k < e->nr_bonds ; k++ ) {

                    /* Does this bond match the types? */
                    if ( ( strcmp( e->types[e->s.partlist[e->bonds[k].i]->type].name , type1 ) == 0 &&
                           strcmp( e->types[e->s.partlist[e->bonds[k].j]->type].name , type2 ) == 0 ) ||
                         ( strcmp( e->types[e->s.partlist[e->bonds[k].i]->type].name , type2 ) == 0 &&
                           strcmp( e->types[e->s.partlist[e->bonds[k].j]->type].name , type1 ) == 0 ) ) {

                        /* Register as a constraint. */
                        if ( engine_rigid_add( e , e->bonds[k].i , e->bonds[k].j , 0.1*r0 ) < 0 )
                            return error(engine_err);

                        /* Remove this bond. */
                        e->nr_bonds -= 1;
                        e->bonds[k] = e->bonds[e->nr_bonds];
                        k -= 1;

                        }

                    } /* Loop over all bonds. */

                }

            /* Otherwise... */
            else {

                /* Are type1 and type2 the same? */
                if ( strcmp( type1 , type2 ) == 0 ) {

                    /* Fill the ind1 array. */
                    for ( nr_ind1 = 0 , k = 0 ; k < e->nr_types ; k++ )
                        if ( strcmp( type1 , e->types[k].name ) == 0 ) {
                            ind1[nr_ind1] = k;
                            nr_ind1 += 1;
                            }

                    /* Are there any indices? */
                    if ( nr_ind1 > 0 ) {

                        /* Create the harmonic potential. */
                        if ( ( p = potential_create_harmonic( 0.05*r0 , 0.15*r0 , 418.4*K , 0.1*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);

                        /* Loop over the types and add the potential. */
                        for ( tid = 0 ; tid < nr_ind1 ; tid++ )
                            for ( tjd = tid ; tjd < nr_ind1 ; tjd++ )
                                if ( engine_bond_addpot( e , p , ind1[tid] , ind1[tjd] ) < 0 )
                                    return error(engine_err);

                        }

                    }
                /* Otherwise... */
                else {

                    /* Fill the ind1 and ind2 arrays. */
                    for ( nr_ind1 = 0 , nr_ind2 = 0 , k = 0 ; k < e->nr_types ; k++ ) {
                        if ( strcmp( type1 , e->types[k].name ) == 0 ) {
                            ind1[nr_ind1] = k;
                            nr_ind1 += 1;
                            }
                        else if ( strcmp( type2 , e->types[k].name ) == 0 ) {
                            ind2[nr_ind2] = k;
                            nr_ind2 += 1;
                            }
                        }

                    /* Are there any indices? */
                    if ( nr_ind1 > 0 && nr_ind2 > 0 ) {

                        /* Create the harmonic potential. */
                        if ( ( p = potential_create_harmonic( 0.05*r0 , 0.15*r0 , 418.4*K , 0.1*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);

                        /* Loop over the types and add the potential. */
                        for ( tid = 0 ; tid < nr_ind1 ; tid++ )
                            for ( tjd = 0 ; tjd < nr_ind2 ; tjd++ )
                                if ( engine_bond_addpot( e , p , ind1[tid] , ind2[tjd] ) < 0 )
                                    return error(engine_err);

                        }

                    }

                }
                
            } /* Is it a bond? */
        
        /* Is it an angle? */    
        else if ( strncasecmp( buff , "ANGL" , 4 ) == 0 ) {
        
            /* Get the atom types. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type2 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type3 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Check if these types even exist. */
            if ( engine_gettype( e , type1 ) < 0 && 
                 engine_gettype( e , type2 ) < 0 &&
                 engine_gettype( e , type3 ) < 0 ) {
                if ( reader_skipline( &r ) < 0 )
                    return error(engine_err_reader);
                continue;
                }

            /* Get the parameters K and r0. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);

            /* Run through the angle list and create the potential if necessary. */
            potid = -1;
            for ( k = 0 ; k < e->nr_angles ; k++ ) {

                /* Does this angle match the types? */
                if ( ( strcmp( e->types[e->s.partlist[e->angles[k].i]->type].name , type1 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->angles[k].j]->type].name , type2 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->angles[k].k]->type].name , type3 ) == 0 ) ||
                     ( strcmp( e->types[e->s.partlist[e->angles[k].i]->type].name , type3 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->angles[k].j]->type].name , type2 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->angles[k].k]->type].name , type1 ) == 0 ) ) {

                    /* Do we need to create the potential? */
                    if ( potid < 0 ) {
                        if ( ( p = potential_create_harmonic_angle( M_PI/180*(r0-45) , M_PI/180*(r0+45) , 4.184*K , M_PI/180*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);
                        if ( ( potid = engine_angle_addpot( e , p ) ) < 0 )
                            return error(engine_err);
                        /* printf( "engine_read_cpf: generated potential for angle %s %s %s with %i intervals.\n" ,
                            type1 , type2 , type3 , e->p_angle[potid]->n ); */
                        }

                    /* Add the potential to the angle. */
                    e->angles[k].pid = potid;

                    }

                }
            
            } /* Is it an angle? */
            
        /* Perhaps a propper dihedral? */
        else if ( strncasecmp( buff , "DIHE" , 4 ) == 0 ) {
        
            /* Get the atom types. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type2 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type3 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type4 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Check for wildcards. */
            wc[0] = ( strcmp( type1 , "X" ) == 0 );
            wc[1] = ( strcmp( type2 , "X" ) == 0 );
            wc[2] = ( strcmp( type3 , "X" ) == 0 );
            wc[3] = ( strcmp( type4 , "X" ) == 0 );

            /* Check if these types even exist. */
            if ( ( wc[0] || engine_gettype( e , type1 ) < 0 ) && 
                 ( wc[1] || engine_gettype( e , type2 ) < 0 ) &&
                 ( wc[2] || engine_gettype( e , type3 ) < 0 ) &&
                 ( wc[3] || engine_gettype( e , type4 ) < 0 ) ) {
                if ( reader_skipline( &r ) < 0 )
                    return error(engine_err_reader);
                continue;
                }

            /* Get the parameters K and r0. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 ) {
                printf( "engine_read_xplor: failed to parse double on line %i, col %i.\n" , r.line , r.col );
                return error(engine_err_cpf);
                }
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            n = strtol( buff , &endptr , 0 );
            if ( *endptr != 0 ) {
                printf( "engine_read_xplor: failed to parse int on line %i, col %i.\n" , r.line , r.col );
                return error(engine_err_cpf);
                }
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 ) {
                printf( "engine_read_xplor: failed to parse double on line %i, col %i.\n" , r.line , r.col );
                return error(engine_err_cpf);
                }

            /* Run through the dihedral list and create the potential if necessary. */
            potid = -1;
            for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {

                /* Does this dihedral match the types? */
                if ( ( e->dihedrals[k].pid == -1 ) &&
                     ( ( ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type1 ) == 0 ) &&
                         ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type2 ) == 0 ) &&
                         ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type3 ) == 0 ) &&
                         ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type4 ) == 0 ) ) ||
                       ( ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type4 ) == 0 ) &&
                         ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type3 ) == 0 ) &&
                         ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type2 ) == 0 ) &&
                         ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type1 ) == 0 ) ) ) ) {

                    /* Do we need to create the potential? */
                    if ( potid < 0 ) {
                        if ( ( p = potential_create_harmonic_dihedral( 4.184*K , n , M_PI/180*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);
                        if ( ( potid = engine_dihedral_addpot( e , p ) ) < 0 )
                            return error(engine_err);
                        /* printf( "engine_read_cpf: generated potential for dihedral %s %s %s %s in [%e,%e] with %i intervals.\n" ,
                            type1 , type2 , type3 , type4 , e->p_dihedral[potid]->a , e->p_dihedral[potid]->b , e->p_dihedral[potid]->n ); */
                        }

                    /* Add the potential to the dihedral. */
                    e->dihedrals[k].pid = potid;

                    }

                }
            
            } /* Dihedral? */
            
        /* Or an improper dihedral instead? */
        else if ( strncasecmp( buff , "IMPR" , 4 ) == 0 ) {
        
            /* Get the atom types. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type2 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type3 , 100 ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , type4 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Check for wildcards. */
            wc[0] = ( strcmp( type1 , "X" ) == 0 );
            wc[1] = ( strcmp( type2 , "X" ) == 0 );
            wc[2] = ( strcmp( type3 , "X" ) == 0 );
            wc[3] = ( strcmp( type4 , "X" ) == 0 );

            /* Check if these types even exist. */
            if ( ( wc[0] || engine_gettype( e , type1 ) < 0 ) && 
                 ( wc[1] || engine_gettype( e , type2 ) < 0 ) &&
                 ( wc[2] || engine_gettype( e , type3 ) < 0 ) &&
                 ( wc[3] || engine_gettype( e , type4 ) < 0 ) ) {
                if ( reader_skipline( &r ) < 0 )
                    return error(engine_err_reader);
                continue;
                }

            /* Get the parameters K and r0. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);

            /* Run through the dihedral list and create the potential if necessary. */
            potid = -1;
            for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {

                /* Does this dihedral match the types? */
                if ( ( e->dihedrals[k].pid == -2 ) &&
                     ( ( ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type1 ) == 0 ) &&
                         ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type2 ) == 0 ) &&
                         ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type3 ) == 0 ) &&
                         ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type4 ) == 0 ) ) ||
                       ( ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type4 ) == 0 ) &&
                         ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type3 ) == 0 ) &&
                         ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type2 ) == 0 ) &&
                         ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type1 ) == 0 ) ) ) ) {

                    /* Do we need to create the potential? */
                    if ( potid < 0 ) {
                        if ( ( p = potential_create_harmonic_angle( M_PI/180*(r0-45) , M_PI/180*(r0+45) , 4.184*K , M_PI/180*r0 , tol ) ) == NULL )
                            return error(engine_err_potential);
                        if ( ( potid = engine_dihedral_addpot( e , p ) ) < 0 )
                            return error(engine_err);
                        /* printf( "engine_read_cpf: generated potential for imp. dihedral %s %s %s %s with %i intervals.\n" ,
                            type1 , type2 , type3 , type4 , e->p_dihedral[potid]->n ); */
                        }

                    /* Add the potential to the dihedral. */
                    e->dihedrals[k].pid = potid;

                    }

                }
            
            } /* Improper dihedral? */
            
        /* Well then maybe a non-bonded interaction... */
        else if ( strncasecmp( buff , "NONB" , 4 ) == 0 ) {
        
            /* Get the atom type. */
            if ( reader_gettoken( &r , type1 , 100 ) < 0 )
                return error(engine_err_reader);

            /* Get the next two parameters. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            K = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            r0 = strtod( buff , &endptr );
            if ( *endptr != 0 )
                return error(engine_err_cpf);

            /* Run through the types and store the parameters for each match. */
            for ( k = 0 ; k < e->nr_types ; k++ )
                if ( strcmp( e->types[k].name , type1 ) == 0 ) {
                    eps[k] = K;
                    rmin[k] = r0;
                    }
                
            } /* non-bonded iteraction. */
            
        /* Otherwise, do, well, nothing. */
        else {
        
            }
            
        /* Skip the rest of the line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
    
        } /* Main loop. */
        
                        
    /* Loop over all the type pairs and construct the non-bonded potentials. */
    for ( j = 0 ; j < e->nr_types ; j++ )
        for ( k = j ; k < e->nr_types ; k++ ) {
            
            /* Has a potential been specified for this case? */
            if ( ( eps[j] == 0.0 || eps[k] == 0.0 ) &&
                 ( kappa < 0.0 || e->types[j].charge == 0.0 || e->types[k].charge == 0.0 ) )
                continue;
                
            /* Construct the common LJ parameters. */
            K = 4.184 * sqrt( eps[j] * eps[k] );
            r0 = 0.05 * ( rmin[j] + rmin[k] );
            r2 = r0*r0; r6 = r2*r2*r2;
            A = K*r6*r6; B = 2*K*r6;
            q = e->types[j].charge*e->types[k].charge;
                
            /* Construct the potential. */
            /* printf( "engine_read_cpf: creating %s-%s potential with A=%e B=%e q=%e.\n" ,
                e->types[j].name , e->types[k].name , 
                K*r6*r6 , K*2*r6 , e->types[j].charge*e->types[k].charge ); */
            if ( K == 0.0 ) {
                if ( q != 0.0 && kappa >= 0.0 ) {
                    if ( kappa > 0.0 ) {
                        if ( ( p = potential_create_Ewald( 0.1 , e->s.cutoff , q , kappa , tol ) ) == NULL )
                            return error(engine_err_potential);
                        }
                    else {
                        if ( ( p = potential_create_Coulomb( 0.1 , e->s.cutoff , q , tol ) ) == NULL )
                            return error(engine_err_potential);
                        }
                    }
                else
                    p = NULL;
                }
            if ( kappa < 0.0 ) {
                al = r0/2; vl = potential_LJ126( al , A , B );
                ar = r0; vr = potential_LJ126( ar , A , B );
                Kr0 = fabs( potential_LJ126( r0 , A , B ) );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B );
                    if ( fabs(vm) < 5*Kr0 ) {
                        ar = am; vr = vm;
                        }
                    else {
                        al = am; vl = vm;
                        }
                    }
                if ( ( p = potential_create_LJ126( al , e->s.cutoff , A , B , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A , B , al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
            else if ( kappa == 0.0 ) {
                al = r0/2; vl = potential_LJ126( al , A , B ) + potential_escale*q/al;
                ar = r0; vr = potential_LJ126( ar , A , B ) + potential_escale*q/ar;
                Kr0 = fabs( potential_LJ126( r0 , A , B ) + potential_escale*q/r0 );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B ) + potential_escale*q/am;
                    if ( fabs(vm) < 5*Kr0 ) {
                        ar = am; vr = vm;
                        }
                    else {
                        al = am; vl = vm;
                        }
                    }
                if ( ( p = potential_create_LJ126_Coulomb( al , e->s.cutoff , A , B , q , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e q=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A ,B , q ,
                    al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
            else  {
                al = r0/2; vl = potential_LJ126( al , A , B ) + q*potential_Ewald( al , kappa );
                ar = r0; vr = potential_LJ126( ar , A , B ) + q*potential_Ewald( ar, kappa );
                Kr0 = fabs( potential_LJ126( r0 , A , B ) + q*potential_Ewald( r0, kappa ) );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B ) + q*potential_Ewald( am , kappa );
                    if ( fabs(vm) < 5*Kr0 ) {
                        ar = am; vr = vm;
                        }
                    else {
                        al = am; vl = vm;
                        }
                    }
                if ( ( p = potential_create_LJ126_Ewald( al , e->s.cutoff , A , B , q , kappa , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e q=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A , B , q ,
                    al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
                
            /* Register it with the local authorities. */
            if ( p != NULL && engine_addpot( e , p , j , k ) < 0 )
                return error(engine_err);
                
            }
        
    /* It's been a hard day's night. */
    return engine_err_ok;
        
    }


/**
 * @brief Read the potentials from a CHARMM parameter file.
 *
 * @param e The #engine.
 * @param cpf The open CHARMM parameter file.
 * @param kappa The PME screening width.
 * @param tol The absolute tolerance for interpolation.
 * @param rigidH Convert all bonds over a type starting with 'H'
 *      to a rigid constraint.
 *
 * If @c kappa is zero, truncated Coulomb electrostatic interactions are
 * assumed. If @c kappa is less than zero, no electrostatic interactions
 * are computed.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_read_cpf ( struct engine *e , FILE *cpf , double kappa , double tol , int rigidH ) {

    struct reader r;
    char buff[100], type1[100], type2[100], type3[100], type4[100], *endptr;
    int tid, tjd, wc[4];
    int j, k, n, *ind1, *ind2, nr_ind1, nr_ind2, potid;
    double K, Kr0, r0, r2, r6, *eps, *rmin;
    double al, ar, am, vl, vr, vm, A, B, q;
    struct potential *p;
    
    /* Check inputs. */
    if ( e == NULL || cpf == NULL )
        return error(engine_err_null);
        
    /* Allocate some local memory for the index arrays. */
    if ( ( ind1 = (int *)alloca( sizeof(int) * e->nr_types ) ) == NULL ||
         ( ind2 = (int *)alloca( sizeof(int) * e->nr_types ) ) == NULL ||
         ( eps = (double *)alloca( sizeof(double) * e->nr_types ) ) == NULL ||
         ( rmin = (double *)alloca( sizeof(double) * e->nr_types ) ) == NULL )
        return error(engine_err_malloc);
    bzero( eps , sizeof(double) * e->nr_types );
    bzero( rmin , sizeof(double) * e->nr_types );
        
    /* Init the reader with the PSF file. */
    if ( reader_init( &r , cpf , NULL , "!" , "\n" ) < 0 )
        return error(engine_err_reader);
        
    /* Skip all lines starting with a "*". */
    while ( r.c == '*' )
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
    
    /* We should now have the keword starting with "BOND". */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "BOND" , 4 ) != 0 )
        return error(engine_err_cpf);
        
    /* Bond-reading loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's the start of the "ANGLe" section, break. */
        if ( strncmp( type1 , "ANGL" , 4 ) == 0 )
            break;
            
        /* Otherwise, get the next token, e.g. the second type. */
        if ( reader_gettoken( &r , type2 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Get the parameters K and r0. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
          
        /* Is this a rigid bond (and do we care)? */  
        if ( rigidH && ( type1[0] == 'H' || type2[0] == 'H' ) ) {
        
            /* Loop over all bonds... */
            for ( k = 0 ; k < e->nr_bonds ; k++ ) {
            
                /* Does this bond match the types? */
                if ( ( strcmp( e->types[e->s.partlist[e->bonds[k].i]->type].name , type1 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->bonds[k].j]->type].name , type2 ) == 0 ) ||
                     ( strcmp( e->types[e->s.partlist[e->bonds[k].i]->type].name , type2 ) == 0 &&
                       strcmp( e->types[e->s.partlist[e->bonds[k].j]->type].name , type1 ) == 0 ) ) {
                       
                    /* Register as a constraint. */
                    if ( engine_rigid_add( e , e->bonds[k].i , e->bonds[k].j , 0.1*r0 ) < 0 )
                        return error(engine_err);
                        
                    /* Remove this bond. */
                    e->nr_bonds -= 1;
                    e->bonds[k] = e->bonds[e->nr_bonds];
                    k -= 1;
                    
                    }
            
                } /* Loop over all bonds. */
        
            }
            
        /* Otherwise... */
        else {
            
            /* Are type1 and type2 the same? */
            if ( strcmp( type1 , type2 ) == 0 ) {

                /* Fill the ind1 array. */
                for ( nr_ind1 = 0 , k = 0 ; k < e->nr_types ; k++ )
                    if ( strcmp( type1 , e->types[k].name ) == 0 ) {
                        ind1[nr_ind1] = k;
                        nr_ind1 += 1;
                        }

                /* Are there any indices? */
                if ( nr_ind1 > 0 ) {

                    /* Create the harmonic potential. */
                    if ( ( p = potential_create_harmonic( 0.05*r0 , 0.15*r0 , 418.4*K , 0.1*r0 , tol ) ) == NULL )
                        return error(engine_err_potential);

                    /* Loop over the types and add the potential. */
                    for ( tid = 0 ; tid < nr_ind1 ; tid++ )
                        for ( tjd = tid ; tjd < nr_ind1 ; tjd++ )
                            if ( engine_bond_addpot( e , p , ind1[tid] , ind1[tjd] ) < 0 )
                                return error(engine_err);

                    }

                }
            /* Otherwise... */
            else {

                /* Fill the ind1 and ind2 arrays. */
                for ( nr_ind1 = 0 , nr_ind2 = 0 , k = 0 ; k < e->nr_types ; k++ ) {
                    if ( strcmp( type1 , e->types[k].name ) == 0 ) {
                        ind1[nr_ind1] = k;
                        nr_ind1 += 1;
                        }
                    else if ( strcmp( type2 , e->types[k].name ) == 0 ) {
                        ind2[nr_ind2] = k;
                        nr_ind2 += 1;
                        }
                    }

                /* Are there any indices? */
                if ( nr_ind1 > 0 && nr_ind2 > 0 ) {

                    /* Create the harmonic potential. */
                    if ( ( p = potential_create_harmonic( 0.05*r0 , 0.15*r0 , 418.4*K , 0.1*r0 , tol ) ) == NULL )
                        return error(engine_err_potential);

                    /* Loop over the types and add the potential. */
                    for ( tid = 0 ; tid < nr_ind1 ; tid++ )
                        for ( tjd = 0 ; tjd < nr_ind2 ; tjd++ )
                            if ( engine_bond_addpot( e , p , ind1[tid] , ind2[tjd] ) < 0 )
                                return error(engine_err);

                    }
                    
                }
                
            }
            
        /* Skip the rest of the line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
    
        } /* bond-reading loop. */
        
        
    /* Skip the rest of the "ANGLe" line. */
    if ( reader_skipline( &r ) < 0 )
        return error(engine_err_reader);
        
    /* Main angle loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's the start of the "DIHEdral" section, break. */
        if ( strncmp( type1 , "DIHE" , 4 ) == 0 )
            break;
            
        /* Otherwise, get the next two tokens, e.g. the second and third type. */
        if ( reader_gettoken( &r , type2 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type3 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Check if these types even exist. */
        if ( engine_gettype( e , type1 ) < 0 && 
             engine_gettype( e , type2 ) < 0 &&
             engine_gettype( e , type3 ) < 0 ) {
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
            continue;
            }
            
        /* Get the parameters K and r0. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
            
        /* Run through the angle list and create the potential if necessary. */
        potid = -1;
        for ( k = 0 ; k < e->nr_angles ; k++ ) {
        
            /* Does this angle match the types? */
            if ( ( strcmp( e->types[e->s.partlist[e->angles[k].i]->type].name , type1 ) == 0 &&
                   strcmp( e->types[e->s.partlist[e->angles[k].j]->type].name , type2 ) == 0 &&
                   strcmp( e->types[e->s.partlist[e->angles[k].k]->type].name , type3 ) == 0 ) ||
                 ( strcmp( e->types[e->s.partlist[e->angles[k].i]->type].name , type3 ) == 0 &&
                   strcmp( e->types[e->s.partlist[e->angles[k].j]->type].name , type2 ) == 0 &&
                   strcmp( e->types[e->s.partlist[e->angles[k].k]->type].name , type1 ) == 0 ) ) {
                
                /* Do we need to create the potential? */
                if ( potid < 0 ) {
                    if ( ( p = potential_create_harmonic_angle( M_PI/180*(r0-40) , M_PI/180*(r0+40) , 4.184*K , M_PI/180*r0 , tol ) ) == NULL )
                        return error(engine_err_potential);
                    if ( ( potid = engine_angle_addpot( e , p ) ) < 0 )
                        return error(engine_err);
                    /* printf( "engine_read_cpf: generated potential for angle %s %s %s with %i intervals.\n" ,
                        type1 , type2 , type3 , e->p_angle[potid]->n ); */
                    }
                
                /* Add the potential to the angle. */
                e->angles[k].pid = potid;
                
                }
        
            }
            
        /* Skip the rest of this line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
        } /* angle loop. */
        
    
    /* Skip the rest of the "DIHEdral" line. */
    if ( reader_skipline( &r ) < 0 )
        return error(engine_err_reader);
        
    /* Main dihedral loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's the start of the "IMPRoper" section, break. */
        if ( strncmp( type1 , "IMPR" , 4 ) == 0 )
            break;
            
        /* Otherwise, get the next three tokens, e.g. the second, third and fouth type. */
        if ( reader_gettoken( &r , type2 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type3 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type4 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Check for wildcards. */
        wc[0] = ( strcmp( type1 , "X" ) == 0 );
        wc[1] = ( strcmp( type2 , "X" ) == 0 );
        wc[2] = ( strcmp( type3 , "X" ) == 0 );
        wc[3] = ( strcmp( type4 , "X" ) == 0 );
            
        /* Check if these types even exist. */
        if ( ( wc[0] || engine_gettype( e , type1 ) < 0 ) && 
             ( wc[1] || engine_gettype( e , type2 ) < 0 ) &&
             ( wc[2] || engine_gettype( e , type3 ) < 0 ) &&
             ( wc[3] || engine_gettype( e , type4 ) < 0 ) ) {
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
            continue;
            }
            
        /* Get the parameters K and r0. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        n = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
            
        /* Run through the dihedral list and create the potential if necessary. */
        potid = -1;
        for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {
        
            /* Does this dihedral match the types? */
            if ( ( e->dihedrals[k].pid == -1 ) &&
                 ( ( ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type1 ) == 0 ) &&
                     ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type2 ) == 0 ) &&
                     ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type3 ) == 0 ) &&
                     ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type4 ) == 0 ) ) ||
                   ( ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type4 ) == 0 ) &&
                     ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type3 ) == 0 ) &&
                     ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type2 ) == 0 ) &&
                     ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type1 ) == 0 ) ) ) ) {
                
                /* Do we need to create the potential? */
                if ( potid < 0 ) {
                    if ( ( p = potential_create_harmonic_dihedral( 4.184*K , n , M_PI/180*r0 , tol ) ) == NULL )
                        return error(engine_err_potential);
                    if ( ( potid = engine_dihedral_addpot( e , p ) ) < 0 )
                        return error(engine_err);
                    /* printf( "engine_read_cpf: generated potential for dihedral %s %s %s %s in [%e,%e] with %i intervals.\n" ,
                        type1 , type2 , type3 , type4 , e->p_dihedral[potid]->a , e->p_dihedral[potid]->b , e->p_dihedral[potid]->n ); */
                    }
                
                /* Add the potential to the dihedral. */
                e->dihedrals[k].pid = potid;
                
                }
        
            }
            
        /* Skip the rest of this line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
        } /* dihedral loop. */
        
    
    /* Skip the rest of the "IMPRoper" line. */
    if ( reader_skipline( &r ) < 0 )
        return error(engine_err_reader);
        
    /* Main improper dihedral loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's the start of the "NONBonded" section, break. */
        if ( strncmp( type1 , "NONB" , 4 ) == 0 )
            break;
            
        /* Otherwise, get the next three tokens, e.g. the second, third and fouth type. */
        if ( reader_gettoken( &r , type2 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type3 , 100 ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , type4 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Check for wildcards. */
        wc[0] = ( strcmp( type1 , "X" ) == 0 );
        wc[1] = ( strcmp( type2 , "X" ) == 0 );
        wc[2] = ( strcmp( type3 , "X" ) == 0 );
        wc[3] = ( strcmp( type4 , "X" ) == 0 );
            
        /* Check if these types even exist. */
        if ( ( wc[0] || engine_gettype( e , type1 ) < 0 ) && 
             ( wc[1] || engine_gettype( e , type2 ) < 0 ) &&
             ( wc[2] || engine_gettype( e , type3 ) < 0 ) &&
             ( wc[3] || engine_gettype( e , type4 ) < 0 ) ) {
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
            continue;
            }
            
        /* Get the parameters K and r0. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_skiptoken( &r ) < 0 )
            return error(engine_err_reader);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
            
        /* Run through the dihedral list and create the potential if necessary. */
        potid = -1;
        for ( k = 0 ; k < e->nr_dihedrals ; k++ ) {
        
            /* Does this dihedral match the types? */
            if ( ( e->dihedrals[k].pid == -2 ) &&
                 ( ( ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type1 ) == 0 ) &&
                     ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type2 ) == 0 ) &&
                     ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type3 ) == 0 ) &&
                     ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type4 ) == 0 ) ) ||
                   ( ( wc[3] || strcmp( e->types[e->s.partlist[e->dihedrals[k].i]->type].name , type4 ) == 0 ) &&
                     ( wc[2] || strcmp( e->types[e->s.partlist[e->dihedrals[k].j]->type].name , type3 ) == 0 ) &&
                     ( wc[1] || strcmp( e->types[e->s.partlist[e->dihedrals[k].k]->type].name , type2 ) == 0 ) &&
                     ( wc[0] || strcmp( e->types[e->s.partlist[e->dihedrals[k].l]->type].name , type1 ) == 0 ) ) ) ) {
                
                /* Do we need to create the potential? */
                if ( potid < 0 ) {
                    if ( ( p = potential_create_harmonic_angle( M_PI/180*(r0-45) , M_PI/180*(r0+45) , 4.184*K , M_PI/180*r0 , tol ) ) == NULL )
                        return error(engine_err_potential);
                    if ( ( potid = engine_dihedral_addpot( e , p ) ) < 0 )
                        return error(engine_err);
                    /* printf( "engine_read_cpf: generated potential for imp. dihedral %s %s %s %s with %i intervals.\n" ,
                        type1 , type2 , type3 , type4 , e->p_dihedral[potid]->n ); */
                    }
                
                /* Add the potential to the dihedral. */
                e->dihedrals[k].pid = potid;
                
                }
        
            }
            
        /* Skip the rest of this line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
        } /* dihedral loop. */
        
    
    /* Skip the rest of the "NONBonded" line. */
    if ( reader_skipline( &r ) < 0 )
        return error(engine_err_reader);
        
    /* Main loop for non-bonded interactions. */
    while ( 1 ) {
    
        /* Get the next token. */
        if ( reader_gettoken( &r , type1 , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* Bail? */
        if ( strncmp( type1 , "HBOND" , 5 ) == 0 )
            break;
            
        /* Skip the first parameter. */
        if ( reader_skiptoken( &r ) < 0 )
            return error(engine_err_reader);
    
        /* Get the next two parameters. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        K = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        r0 = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_cpf);
            
        /* Run through the types and store the parameters for each match. */
        for ( k = 0 ; k < e->nr_types ; k++ )
            if ( strcmp( e->types[k].name , type1 ) == 0 ) {
                eps[k] = K;
                rmin[k] = r0;
                }
                
        /* Skip the rest of the line. */
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
        }
        
    /* Loop over all the type pairs and construct the non-bonded potentials. */
    for ( j = 0 ; j < e->nr_types ; j++ )
        for ( k = j ; k < e->nr_types ; k++ ) {
            
            /* Has a potential been specified for this case? */
            if ( ( eps[j] == 0.0 || eps[k] == 0.0 ) &&
                 ( kappa < 0.0 || e->types[j].charge == 0.0 || e->types[k].charge == 0.0 ) )
                continue;
                
            /* Construct the common LJ parameters. */
            K = 4.184 * sqrt( eps[j] * eps[k] );
            r0 = 0.1 * ( rmin[j] + rmin[k] );
            r2 = r0*r0; r6 = r2*r2*r2;
            A = K*r6*r6; B = 2*K*r6;
            q = e->types[j].charge*e->types[k].charge;
                
            /* Construct the potential. */
            /* printf( "engine_read_cpf: creating %s-%s potential with A=%e B=%e q=%e.\n" ,
                e->types[j].name , e->types[k].name , 
                K*r6*r6 , K*2*r6 , e->types[j].charge*e->types[k].charge ); */
            if ( K == 0.0 ) {
                if ( q != 0.0 && kappa >= 0.0 ) {
                    if ( kappa > 0.0 ) {
                        if ( ( p = potential_create_Ewald( 0.1 , e->s.cutoff , q , kappa , tol ) ) == NULL )
                            return error(engine_err_potential);
                        }
                    else {
                        if ( ( p = potential_create_Coulomb( 0.1 , e->s.cutoff , q , tol ) ) == NULL )
                            return error(engine_err_potential);
                        }
                    }
                else
                    p = NULL;
                }
            if ( kappa < 0.0 ) {
                al = r0/2; vl = potential_LJ126( al , A , B );
                ar = r0; vr = potential_LJ126( ar , A , B );
                Kr0 = fabs( potential_LJ126( r0 , A , B ) );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B );
                    if ( fabs(vm) < 5*Kr0 ) {
                        ar = am; vr = vm;
                        }
                    else {
                        al = am; vl = vm;
                        }
                    }
                if ( ( p = potential_create_LJ126( al , e->s.cutoff , A , B , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A , B , al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
            else if ( kappa == 0.0 ) {
                al = r0/2; vl = potential_LJ126( al , A , B ) + potential_escale*q/al;
                ar = r0; vr = potential_LJ126( ar , A , B ) + potential_escale*q/ar;
                Kr0 = fabs( potential_LJ126( r0 , A , B ) + potential_escale*q/r0 );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B ) + potential_escale*q/am;
                    if ( fabs(vm) < 5*Kr0 ) {
                        ar = am; vr = vm;
                        }
                    else {
                        al = am; vl = vm;
                        }
                    }
                if ( ( p = potential_create_LJ126_Coulomb( al , e->s.cutoff , A , B , q , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e q=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A ,B , q ,
                    al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
            else  {
                al = r0/2; vl = potential_LJ126( al , A , B ) + q*potential_Ewald( al , kappa );
                ar = r0; vr = potential_LJ126( ar , A , B ) + q*potential_Ewald( ar, kappa );
                Kr0 = fabs( potential_LJ126( r0 , A , B ) + q*potential_Ewald( r0, kappa ) );
                while ( ar-al > 1e-5 ) {
                    am = 0.5*(al + ar); vm = potential_LJ126( am , A , B ) + q*potential_Ewald( am , kappa );
                    if ( fabs(vm) < 5*Kr0 ) {
                        ar = am; vr = vm;
                        }
                    else {
                        al = am; vl = vm;
                        }
                    }
                if ( ( p = potential_create_LJ126_Ewald( al , e->s.cutoff , A , B , q , kappa , tol ) ) == NULL ) {
                    printf( "engine_read_xplor: failed to create %s-%s potential with A=%e B=%e q=%e on [%e,%e].\n" ,
                    e->types[j].name , e->types[k].name , 
                    A , B , q ,
                    al , e->s.cutoff );
                    return error(engine_err_potential);
                    }
                }
                
            /* Register it with the local authorities. */
            if ( p != NULL && engine_addpot( e , p , j , k ) < 0 )
                return error(engine_err);
                
            }
        
    /* It's been a hard day's night. */
    return engine_err_ok;
        
    }


/**
 * @brief Read the simulation setup from a PSF and PDB file pair.
 *
 * @param e The #engine.
 * @param psf The open PSF file.
 * @param pdb The open PDB file.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_read_psf ( struct engine *e , FILE *psf , FILE *pdb ) {

    struct reader r;
    char type[100], typename[100], buff[100], *endptr;
    int pid, pjd, pkd, pld, j, k, n, id, *resids, *typeids, typelen, bufflen;
    double q, m, x[3];
    struct part p;
    
    /* Check inputs. */
    if ( e == NULL || psf == NULL || pdb == NULL )
        return error(engine_err_null);
        
    /* Init the reader with the PSF file. */
    if ( reader_init( &r , psf , NULL , "!" , "\n" ) < 0 )
        return error(engine_err_reader);
        
    /* Read the PSF header token. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strcmp( buff , "PSF" ) != 0 )
        return error(engine_err_psf);
    
    /* Ok, now read the number of comment lines and skip them. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    for ( k = 0 ; k <= n ; k++ )
        if ( reader_skipline( &r ) < 0 )
            return error(engine_err_reader);
            
    /* Now get the number of atoms, along with the comment. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NATOM" , 5 ) != 0 )
        return error(engine_err_psf);
        
    /* Allocate memory for the type IDs. */
    if ( ( typeids = (int *)alloca( sizeof(int) * n ) ) == NULL ||
         ( resids = (int *)alloca( sizeof(int) * n ) ) == NULL )
        return error(engine_err_malloc);
        
    /* Loop over the atom list. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Skip the first two tokens (ID, segment). */
        for ( j = 0 ; j < 2 ; j++ )
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
                
        /* Get the residue id. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        resids[k] = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
        
        /* Skip the next two tokens (res name, atom name). */
        for ( j = 0 ; j < 2 ; j++ )
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
                
        /* Get the atom type. */
        if ( ( typelen = reader_gettoken( &r , type , 100 ) ) < 0 )
            return error(engine_err_reader);
    
        /* Get the atom charge. */
        if ( ( bufflen = reader_gettoken( &r , buff , 100 ) ) < 0 )
            return error(engine_err_reader);
        q = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Merge the type and charge. */
        memcpy( typename , type , typelen );
        memcpy( &typename[typelen] , buff , bufflen+1 );
        
        /* Get the atom mass. */
        if ( ( bufflen = reader_gettoken( &r , buff , 100 ) ) < 0 )
            return error(engine_err_reader);
        m = strtod( buff , &endptr );
        if ( *endptr != 0 )
            return error(engine_err_psf);

        /* Try to get a type id. */
        if ( ( id = engine_gettype2( e , typename ) ) >= 0 )
            typeids[k] = id;
        
        /* Otherwise, register a new type. */
        else if ( id == engine_err_range ) {
        
            if ( ( typeids[k] = engine_addtype( e , m , q , type , typename ) ) < 0 )
                return error(engine_err);
        
            }
            
        /* error? */
        else
            return error(engine_err);
            
        /* Read the trailing zero. */
        if ( reader_skiptoken( &r ) < 0 )
            return error(engine_err_reader);
    
        }
        
    /* Look for the number of bonds. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NBOND" , 5 ) != 0 )
        return error(engine_err_psf);
        
    /* Load the bonds. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Get the particle IDs. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pid = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pjd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Store the bond. */
        if ( engine_bond_add( e , pid-1 , pjd-1 ) < 0 )
            return error(engine_err);
    
        }
                    
    /* Look for the number of angles. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NTHETA" , 5 ) != 0 )
        return error(engine_err_psf);
        
    /* Load the angles. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Get the particle IDs. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pid = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pjd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pkd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Store the angle, we will set the potential later. */
        if ( engine_angle_add( e , pid-1 , pjd-1 , pkd-1 , -1 ) < 0 )
            return error(engine_err);
    
        }
        
    /* Look for the number of dihedrals. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NPHI" , 4 ) != 0 )
        return error(engine_err_psf);
        
    /* Load the dihedrals. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Get the particle IDs. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pid = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pjd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pkd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pld = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Store the dihedral, we will set the potential later. */
        if ( engine_dihedral_add( e , pid-1 , pjd-1 , pkd-1 , pld-1 , -1 ) < 0 )
            return error(engine_err);
    
        }
        
    /* Look for the number of improper dihedrals. */
    if ( reader_gettoken( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    n = strtol( buff , &endptr , 0 );
    if ( *endptr != 0 )
        return error(engine_err_psf);
    if ( reader_getcomment( &r , buff , 100 ) < 0 )
        return error(engine_err_reader);
    if ( strncmp( buff , "NIMP" , 4 ) != 0 )
        return error(engine_err_psf);
        
    /* Load the improper dihedrals. */
    for ( k = 0 ; k < n ; k++ ) {
    
        /* Get the particle IDs. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pid = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pjd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pkd = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
        pld = strtol( buff , &endptr , 0 );
        if ( *endptr != 0 )
            return error(engine_err_psf);
            
        /* Store the dihedral, we will set the potential later. */
        if ( engine_dihedral_add( e , pid-1 , pjd-1 , pkd-1 , pld-1 , -2 ) < 0 )
            return error(engine_err);
    
        }
        
    /* There may be more stuff in the file, but we'll ignore that for now! */
    
    /* Init the reader with the PDb file. */
    if ( reader_init( &r , pdb , NULL , NULL , NULL ) < 0 )
        return error(engine_err_reader);
        
    /* Init the part data. */
    bzero( &p , sizeof(struct part) );
        
    /* Main loop. */
    while ( 1 ) {
    
        /* Get a token. */
        if ( reader_gettoken( &r , buff , 100 ) < 0 )
            return error(engine_err_reader);
            
        /* If it's a REMARK, just skip the line. */
        if ( strncmp( buff , "REMARK" , 6 ) == 0 ) {
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
            }
            
        /* Is it an atom? */
        else if ( strncmp( buff , "ATOM" , 4 ) == 0 ) {
        
            /* Get the atom ID. */
            if ( reader_gettoken( &r , buff , 100 ) < 0 )
                return error(engine_err_reader);
            pid = strtol( buff , &endptr , 0 );
            if ( *endptr != 0 )
                return error(engine_err_pdb);
                
            /* Get the atom type. */
            if ( ( typelen = reader_gettoken( &r , type , 100 ) ) < 0 )
                return error(engine_err_reader);
                
            /* Does the type match the PSF data? */
            /* if ( strncmp( e->types[typeids[pid-1]].name , type , typelen ) != 0 )
                return error(engine_err_pdb); */
                
            /* Ignore the two following tokens. */
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
            if ( reader_skiptoken( &r ) < 0 )
                return error(engine_err_reader);
                
            /* Load the position. */
            for ( k = 0 ; k < 3 ; k++ ) {
                if ( ( bufflen = reader_gettoken( &r , buff , 100 ) ) < 0 )
                    return error(engine_err_reader);
                x[k] = fmod( e->s.dim[k] - e->s.origin[k] + 0.1 * strtod( buff , &endptr ) , e->s.dim[k] ) + e->s.origin[k];
                if ( *endptr != 0 )
                    return error(engine_err_pdb);
                }
                
            /* Add a part of the given type at the given location. */
            p.id = pid-1;
            p.vid = resids[pid-1];
            p.q = e->types[typeids[pid-1]].charge;
            p.flags = part_flag_none;
            p.type = typeids[pid-1];
            if ( space_addpart( &e->s , &p , x ) < 0 )
                return error(engine_err_space);
                
            /* Skip the rest of the line. */
            if ( reader_skipline( &r ) < 0 )
                return error(engine_err_reader);
                
            }
            
        /* Is it the end? */
        else if ( strncmp( buff , "END" , 3 ) == 0 )
            break;
            
        /* Otherwise, it's and error. */
        else
            return error(engine_err_pdb);
            
            
        } /* main PDB loop. */
        
                    
    /* We're on the road again! */
    return engine_err_ok;

    }
    

/**
 * @brief Dump the contents of the enginge to a PSF and PDB file.
 *
 * @param e The #engine.
 * @param psf A pointer to @c FILE to which to write the PSF file.
 * @param pdb A pointer to @c FILE to which to write the PDB file.
 *
 * If any of @c psf or @c pdb are @c NULL, the respective output will
 * not be generated.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_dump_PSF ( struct engine *e , FILE *psf , FILE *pdb , char *excl[] , int nr_excl ) {

    struct space *s;
    struct cell *c;
    struct part *p;
    int k, pid, bid, aid;

    /* Check inputs. */
    if ( e == NULL || ( psf == NULL && pdb == NULL ) )
        return error(engine_err_null);
        
    /* Get a hold of the space. */
    s = &e->s;
        
    /* Write the header for the psf file if needed. */
    if ( psf != NULL )
        fprintf( psf , "PSF\n0 !NTITLE\n%i !NATOM\n" , s->nr_parts );
        
    /* Loop over the cells and parts. */
    for ( pid = 0 ; pid < s->nr_parts ; pid++ ) {
        if ( ( p = s->partlist[pid] ) == NULL || ( c = s->celllist[pid] ) == NULL )
            continue;
        for ( k = 0 ; k < nr_excl ; k++ )
            if ( strcmp( e->types[p->type].name , excl[k] ) == 0 )
                break;
        if ( nr_excl > 0 && k < nr_excl )
            continue;
        if ( pdb != NULL )
            fprintf( pdb , "ATOM  %5d %4s %-4s %6d%1s %8.3f%8.3f%8.3f\n",
                p->id+1 , e->types[p->type].name , "TIP3" , p->vid+1 , "" ,
                10 * ( p->x[0] + c->origin[0] ) , 10 * ( p->x[1] + c->origin[1] ) , 10 * ( p->x[2] + c->origin[2] ) );
        if ( psf != NULL )
            fprintf( psf , "%8i %4s %4i %4s %4s %4s %15.6f %15.6f    0\n" ,
                p->id+1 , "WAT" , p->vid+1 , "TIP3" , e->types[p->type].name , e->types[p->type].name , e->types[p->type].charge , e->types[p->type].mass );
        }
        
    /* Dump bonds and angles to PSF? */
    if ( psf != NULL ) {
    
        /* Dump bonds. */
        fprintf( psf , "%i !NBOND\n" , e->nr_bonds + e->nr_angles );
        for ( bid = 0 ; bid < e->nr_bonds ; bid++ )
            if ( bid % 4 == 3 )
                fprintf( psf , " %i %i\n" , e->bonds[bid].i+1 , e->bonds[bid].j+1 );
            else
                fprintf( psf , " %i %i" , e->bonds[bid].i+1 , e->bonds[bid].j+1 );
        for ( aid = 0 ; aid < e->nr_angles ; aid++ )
            if ( aid % 4 == 3 )
                fprintf( psf , " %i %i\n" , e->angles[aid].i+1 , e->angles[aid].k+1 );
            else
                fprintf( psf , " %i %i" , e->angles[aid].i+1 , e->angles[aid].k+1 );
                
        /* Dump angles. */
        fprintf( psf , "%i !NTHETA\n" , e->nr_angles );
        for ( aid = 0 ; aid < e->nr_angles ; aid++ )
            if ( aid % 3 == 2 )
                fprintf( psf , " %i %i %i\n" , e->angles[aid].i+1 , e->angles[aid].j+1 , e->angles[aid].k+1 );
            else
                fprintf( psf , " %i %i %i" , e->angles[aid].i+1 , e->angles[aid].j+1 , e->angles[aid].k+1 );
                
        /* Dump remaining bogus headers. */
        fprintf( psf , "0 !NPHI\n" );
        fprintf( psf , "0 !NIMPHI\n" );
        fprintf( psf , "0 !NDON\n" );
        fprintf( psf , "0 !NACC\n" );
        fprintf( psf , "0 !NNB\n" );
        
        }
        
    /* We're on a road to nowhere... */
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
 * @brief Add a rigid constraint to the engine.
 *
 * @param e The #engine.
 * @param pid The ID of the first #part.
 * @param pjd The ID of the second #part.
 *
 * @return The index of the rigid constraint or < 0 on error (see #engine_err).
 *
 * Beware that currently all particles have to have been inserted before
 * the rigid constraints are added!
 */
 
int engine_rigid_add ( struct engine *e , int pid , int pjd , double d ) {

    struct rigid *dummy, *r;
    int ind, jnd, rid, rjd, k, j;

    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* If we don't have a part2rigid array, allocate and init one. */
    if ( e->part2rigid == NULL ) {
        if ( ( e->part2rigid = (int *)malloc( sizeof(int *) * e->s.nr_parts ) ) == NULL )
            return error(engine_err_malloc);
        for ( k = 0 ; k < e->s.nr_parts ; k++ )
            e->part2rigid[k] = -1;
        }
        
    /* Update the number of constraints (important for temp). */
    e->nr_constr += 1;
        
    /* Check if we already have a rigid constraint with either pid or pjd. */
    rid = e->part2rigid[pid]; rjd = e->part2rigid[pjd];
    if ( rid < 0 && rjd < 0 ) {
    
        /* Do we need to grow the rigids array? */
        if ( e->nr_rigids == e->rigids_size ) {
            e->rigids_size  *= 1.414;
            if ( ( dummy = (struct rigid *)malloc( sizeof(struct rigid) * e->rigids_size ) ) == NULL )
                return error(engine_err_malloc);
            memcpy( dummy , e->rigids , sizeof(struct rigid) * e->nr_rigids );
            free( e->rigids );
            e->rigids = dummy;
            }

        /* Store this rigid. */
        e->rigids[ e->nr_rigids ].nr_parts = 2;
        e->rigids[ e->nr_rigids ].nr_constr = 1;
        e->rigids[ e->nr_rigids ].parts[0] = pid;
        e->rigids[ e->nr_rigids ].parts[1] = pjd;
        e->rigids[ e->nr_rigids ].constr[0].i = 0;
        e->rigids[ e->nr_rigids ].constr[0].j = 1;
        e->rigids[ e->nr_rigids ].constr[0].d2 = d*d;
        e->part2rigid[pid] = e->nr_rigids;
        e->part2rigid[pjd] = e->nr_rigids;
        e->nr_rigids += 1;
    
        }
        
    /* Both particles are already in different groups. */
    else if ( rid >= 0 && rjd >= 0 && rid != rjd ) {
    
        /* Get a hold of both rigids. */
        r = &e->rigids[rid]; dummy = &e->rigids[rjd];
    
        /* Get indices for these parts in the respective rigids. */
        for ( ind = 0 ; r->parts[ind] != pid ; ind++ );
        for ( jnd = 0 ; dummy->parts[jnd] != pjd ; jnd++ );
                
        /* Merge the particles of rjd into rid. */
        for ( j = 0 ; j < dummy->nr_parts ; j++ ) {
            r->parts[ r->nr_parts + j ] = dummy->parts[j];
            e->part2rigid[ dummy->parts[j] ] = rid;
            }
            
        /* Add the constraints from dummy to rid. */
        for ( j = 0 ; j < dummy->nr_constr ; j++ ) {
            r->constr[ r->nr_constr + j ] = dummy->constr[ j ];
            r->constr[ r->nr_constr + j ].i += r->nr_parts;
            r->constr[ r->nr_constr + j ].j += r->nr_parts;
            }
            
        /* Adjust the number of parts and constr in rid. */
        r->nr_constr += dummy->nr_constr;
        r->nr_parts += dummy->nr_parts;
    
        /* Store the distance constraint. */
        r->constr[ r->nr_constr ].i = ind;
        r->constr[ r->nr_constr ].j = jnd;
        r->constr[ r->nr_constr ].d2 = d*d;
        r->nr_constr += 1;
        
        /* Remove the rigid rjd. */
        e->nr_rigids -= 1;
        if ( rjd < e->nr_rigids ) {
            e->rigids[ rjd ] = e->rigids[ e->nr_rigids ];
            for ( j = 0 ; j < e->rigids[ rjd ].nr_parts ; j++ )
                e->part2rigid[ e->rigids[rjd].parts[j] ] = rjd;
            }
        
        }
        
    /* Otherwise, one or both particles are in the same group. */
    else {
    
        /* Get a grip on the rigid. */
        if ( rid < 0 )
            rid = rjd;
        r = &e->rigids[rid];
        
        /* Try to get indices for these parts in the kth constraint. */
        ind = -1; jnd = -1;
        for ( j = 0 ; j < r->nr_parts ; j++ ) {
            if ( r->parts[j] == pid )
                ind = j;
            else if ( r->parts[j] == pjd )
                jnd = j;
            }
                
        /* Do we need to store i or j? */
        if ( ind < 0 ) {
            r->parts[ r->nr_parts ] = pid;
            ind = r->nr_parts;
            r->nr_parts += 1;
            e->part2rigid[pid] = rid;
            }
        else if ( jnd < 0 ) {
            r->parts[ r->nr_parts ] = pjd;
            jnd = r->nr_parts;
            r->nr_parts += 1;
            e->part2rigid[pjd] = rid;
            }
            
        /* Store the distance constraint. */
        r->constr[ r->nr_constr ].i = ind;
        r->constr[ r->nr_constr ].j = jnd;
        r->constr[ r->nr_constr ].d2 = d*d;
        r->nr_constr += 1;
        
        }
        
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
        #pragma omp parallel private(thread_id,nr_threads), reduction(+:epot)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) &&
             ( nr_bonds + nr_angles + nr_dihedrals ) > 0 ) {
             
            /* Get the thread ID. */
            thread_id = omp_get_thread_num();

            /* Compute the bonded interactions. */
            bond_eval_div( e->bonds , nr_bonds , nr_threads , thread_id , e , &epot );
                    
            /* Compute the angle interactions. */
            angle_eval_div( e->angles , nr_angles , nr_threads , thread_id , e , &epot );
                    
            /* Compute the dihedral interactions. */
            dihedral_eval_div( e->dihedrals , nr_dihedrals , nr_threads , thread_id , e , &epot );
                    
            /* Correct for excluded interactons. */
            exclusion_eval_div( e->exclusions , nr_exclusions , nr_threads , thread_id , e , &epot );
                    
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
        #pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff), reduction(+:epot)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) && 
             ( nr_dihedrals > engine_dihedrals_chunk ) ) {
    
            /* Allocate a buffer for the forces. */
            eff = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * s->nr_parts );
            bzero( eff , sizeof(FPTYPE) * 4 * s->nr_parts );

            /* Compute the dihedral interactions. */
            k = omp_get_thread_num();
            dihedral_evalf( &e->dihedrals[k*nr_dihedrals/nr_threads] , (k+1)*nr_dihedrals/nr_threads - k*nr_dihedrals/nr_threads , e , eff , &epot );
                    
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
        #pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff), reduction(+:epot)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) && 
             ( nr_angles > engine_angles_chunk ) ) {
    
            /* Allocate a buffer for the forces. */
            eff = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * s->nr_parts );
            bzero( eff , sizeof(FPTYPE) * 4 * s->nr_parts );

            /* Compute the angle interactions. */
            k = omp_get_thread_num();
            angle_evalf( &e->angles[k*nr_angles/nr_threads] , (k+1)*nr_angles/nr_threads - k*nr_angles/nr_threads , e , eff , &epot );
                    
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
 * @brief Split the rigids into local, semilocal and non-local.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_rigid_sort ( struct engine *e ) {

    struct cell **celllist;
    struct rigid temp;
    int nr_rigids = e->nr_rigids, nr_local, nr_ghosts, i, j, k;
    
    /* If not in parallel, then we've got nothing to do. */
    if ( e->nr_nodes == 1 ) {
        e->rigids_semilocal = e->rigids_local = e->nr_rigids;
        return engine_err_ok;
        }
    
    /* Get a handle on the celllist. */
    celllist = e->s.celllist;

    /* Split between local and completely non-local rigids. */
    i = 0; j = nr_rigids-1;
    while ( i < j ) {
        while ( i < nr_rigids ) {
            for ( nr_ghosts = 0 , k = 0 ; k < e->rigids[i].nr_parts && celllist[e->rigids[i].parts[k]] != NULL ; k++ )
                if ( celllist[e->rigids[i].parts[k]]->flags & cell_flag_ghost )
                    nr_ghosts += 1;
            if ( k < e->rigids[i].nr_parts || nr_ghosts == e->rigids[i].nr_parts )
                break;
            i += 1;
            }
        while ( j >= 0 ) {
            for ( nr_ghosts = 0 , k = 0 ; k < e->rigids[j].nr_parts && celllist[e->rigids[j].parts[k]] != NULL ; k++ )
                if ( celllist[e->rigids[j].parts[k]]->flags & cell_flag_ghost )
                    nr_ghosts += 1;
            if ( k == e->rigids[j].nr_parts && nr_ghosts < e->rigids[j].nr_parts )
                break;
            j -= 1;
            }
        if ( i < j ) {
            temp = e->rigids[i];
            e->rigids[i] = e->rigids[j];
            e->rigids[j] = temp;
            }
        }
    nr_rigids = i;

    /* Split again between strictly local and semi-local (contains ghosts). */
    i = 0; j = nr_rigids-1;
    while ( i < j ) {
        while ( i < nr_rigids ) {
            for ( k = 0 ; k < e->rigids[i].nr_parts && !(celllist[e->rigids[i].parts[k]]->flags & cell_flag_ghost) ; k++ );
            if ( k < e->rigids[i].nr_parts )
                break;
            i += 1;
            }
        while ( j >= 0 ) {
            for ( k = 0 ; k < e->rigids[j].nr_parts && !(celllist[e->rigids[j].parts[k]]->flags & cell_flag_ghost) ; k++ );
            if ( k == e->rigids[j].nr_parts )
                break;
            j -= 1;
            }
        if ( i < j ) {
            temp = e->rigids[i];
            e->rigids[i] = e->rigids[j];
            e->rigids[j] = temp;
            }
        }
    nr_local = i;


    /* Store the values in the engine. */
    e->rigids_local = nr_local;
    e->rigids_semilocal = nr_rigids;
        
    /* I'll be back... */
    return engine_err_ok;

    }

/**
 * @brief Resolve the constraints.
 * 
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Note that if in parallel, #engine_rigid_sort should be called before
 * this routine.
 */
 
int engine_rigid_eval ( struct engine *e ) {

    int nr_local = e->rigids_local, nr_rigids = e->rigids_semilocal;
    ticks tic;
    #ifdef HAVE_OPENMP
        int finger_global = 0, finger, count;
    #endif
    
    /* Do we have asynchronous communication going on, e.g. are we waiting
       for ghosts? */
    if ( e->flags & engine_flag_async ) {
    
        #ifdef HAVE_OPENMP

            /* Is it worth parallelizing? */
            #pragma omp parallel private(finger,count)
            if ( omp_get_num_threads() > 1 && nr_local > engine_rigids_chunk ) {

                /* Main loop. */
                while ( finger_global < nr_local ) {

                    /* Get a finger on the bonds list. */
                    #pragma omp critical
                    {
                        if ( finger_global < nr_local ) {
                            finger = finger_global;
                            count = engine_rigids_chunk;
                            if ( finger + count > nr_local )
                                count = nr_local - finger;
                            finger_global += count;
                            }
                        else
                            count = 0;
                        }

                    /* Compute the bonded interactions. */
                    if ( count > 0 )
                        rigid_eval_shake( &e->rigids[finger] , count , e );

                    } /* main loop. */

                }

            /* Otherwise, evaluate directly. */
            else if ( omp_get_thread_num() == 0 )
                rigid_eval_shake( e->rigids , nr_local , e );
                
                
            /* Wait for the async data to come in. */
            tic = getticks();
            if ( e->flags & engine_flag_async )
                if ( engine_exchange_wait( e ) < 0 )
                    return error(engine_err);
            tic = getticks() - tic;
            e->timers[engine_timer_exchange1] += tic;
            e->timers[engine_timer_rigid] -= tic;
                
                
            /* Is it worth parallelizing? */
            #pragma omp parallel private(finger,count)
            if ( omp_get_num_threads() > 1 && nr_rigids-nr_local > engine_rigids_chunk ) {

                /* Main loop. */
                while ( finger_global < nr_rigids ) {

                    /* Get a finger on the bonds list. */
                    #pragma omp critical
                    {
                        if ( finger_global < nr_rigids ) {
                            finger = finger_global;
                            count = engine_rigids_chunk;
                            if ( finger + count > nr_rigids )
                                count = nr_rigids - finger;
                            finger_global += count;
                            }
                        else
                            count = 0;
                        }

                    /* Compute the bonded interactions. */
                    if ( count > 0 )
                        rigid_eval_shake( &e->rigids[finger] , count , e );

                    } /* main loop. */

                }

            /* Otherwise, evaluate directly. */
            else if ( omp_get_thread_num() == 0 )
                rigid_eval_shake( &(e->rigids[nr_local]) , nr_rigids-nr_local , e );
                
        #else
        
            /* Shake local rigids. */
            if ( rigid_eval_shake( e->rigids , nr_local , e ) < 0 )
                return error(engine_err_rigid);
                
            /* Wait for exchange to come in. */
            tic = getticks();
            if ( e->flags & engine_flag_async )
                if ( engine_exchange_wait( e ) < 0 )
                    return error(engine_err);
            tic = getticks() - tic;
            e->timers[engine_timer_exchange1] += tic;
            e->timers[engine_timer_verlet] -= tic;
                
            /* Shake semi-local rigids. */
            if ( rigid_eval_shake( &(e->rigids[nr_local]) , nr_rigids-nr_local , e ) < 0 )
                return error(engine_err_rigid);
                
        #endif
    
        }
        
    /* No async, do it all at once. */
    else {
    
        #ifdef HAVE_OPENMP

            /* Is it worth parallelizing? */
            #pragma omp parallel private(finger,count)
            if ( omp_get_num_threads() > 1 && nr_rigids > engine_rigids_chunk ) {

                /* Main loop. */
                while ( finger_global < nr_rigids ) {

                    /* Get a finger on the bonds list. */
                    #pragma omp critical
                    {
                        if ( finger_global < nr_rigids ) {
                            finger = finger_global;
                            count = engine_rigids_chunk;
                            if ( finger + count > nr_rigids )
                                count = nr_rigids - finger;
                            finger_global += count;
                            }
                        else
                            count = 0;
                        }

                    /* Compute the bonded interactions. */
                    if ( count > 0 )
                        rigid_eval_shake( &e->rigids[finger] , count , e );

                    } /* main loop. */

                }

            /* Otherwise, evaluate directly. */
            else if ( omp_get_thread_num() == 0 )
                rigid_eval_shake( e->rigids , nr_rigids , e );
                
        #else
        
            if ( rigid_eval_shake( e->rigids , nr_rigids , e ) < 0 )
                return error(engine_err_rigid);
                
        #endif
    
        }
        
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
        #pragma omp parallel private(thread_id,nr_threads), reduction(+:epot)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) &&
             ( nr_exclusions > 0 ) ) {
             
            /* Get the thread ID. */
            thread_id = omp_get_thread_num();

            /* Correct for excluded interactons. */
            exclusion_eval_mod( e->exclusions , nr_exclusions , nr_threads , thread_id , e , &epot );
                    
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
        #pragma omp parallel private(k,nr_threads,c,p,cid,pid,gpid,eff), reduction(+:epot)
        if ( ( e->flags & engine_flag_parbonded ) &&
             ( ( nr_threads = omp_get_num_threads() ) > 1 ) && 
             ( nr_bonds > engine_bonds_chunk ) ) {
    
            /* Allocate a buffer for the forces. */
            eff = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * s->nr_parts );
            bzero( eff , sizeof(FPTYPE) * 4 * s->nr_parts );

            /* Compute the bonded interactions. */
            k = omp_get_thread_num();
            bond_evalf( &e->bonds[k*nr_bonds/nr_threads] , (k+1)*nr_bonds/nr_threads - k*nr_bonds/nr_threads , e , eff , &epot );
                    
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
    
    
/** 
 * @brief Wait for an asynchronous data exchange to finalize.
 *
 * @param e The #engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
#ifdef HAVE_MPI
int engine_exchange_wait ( struct engine *e ) {

    /* Try to grab the xchg_mutex, which will only be free while
       the async routine is waiting on a condition. */
    if ( pthread_mutex_lock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);
        
    /* If the async exchange was started but is not running,
       wait for a signal. */
    while ( e->xchg_started && ~e->xchg_running )
        if ( pthread_cond_wait( &e->xchg_cond , &e->xchg_mutex ) != 0 )
            return error(engine_err_pthread);
        
    /* We don't actually need this, so release it again. */
    if ( pthread_mutex_unlock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);
        
    /* The end of the tunnel. */
    return engine_err_ok;

    }
#endif


/**
 * @brief Exchange data with other nodes asynchronously.
 *
 * @param e The #engine to work with.
 * @param comm The @c MPI_Comm over which to exchange data.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Starts a new thread which handles the particle exchange. At the
 * start of the exchange, ghost cells are marked in the taboo-list
 * and only freed once their data has been received.
 *
 * The function #engine_exchange_wait can be used to wait for
 * the asynchronous communication to finish.
 */

#ifdef HAVE_MPI 
int engine_exchange_async_run ( struct engine *e ) {

    int i, k, ind, res, cid;
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c;
    struct space *s;
    FPTYPE h[3];

    /* Check the input. */
    if ( e == NULL )
        return error(engine_err_null);

    /* Get local copies of some data. */
    s = &e->s;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Start by acquiring the xchg_mutex. */
    if ( pthread_mutex_lock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);

    /* Main loop... */
    while ( 1 ) {

        /* Wait for a signal to start. */
        e->xchg_running = 0;
        if ( pthread_cond_wait( &e->xchg_cond , &e->xchg_mutex ) != 0 )
            return error(engine_err_pthread);
            
        /* Tell the world I'm alive! */
        e->xchg_started = 0; e->xchg_running = 1;
        if ( pthread_cond_signal( &e->xchg_cond ) != 0 )
            return error(engine_err_pthread);
        
        /* Start by packing and sending/receiving a counts array for each send queue. */
        for ( i = 0 ; i < e->nr_nodes ; i++ ) {

            /* Do we have anything to send? */
            if ( e->send[i].count > 0 ) {

                /* Allocate a new lengths array. */
                if ( ( counts_out[i] = (int *)malloc( sizeof(int) * e->send[i].count ) ) == NULL )
                    return error(engine_err_malloc);

                /* Pack the array with the counts. */
                totals_send[i] = 0;
                for ( k = 0 ; k < e->send[i].count ; k++ )
                    totals_send[i] += ( counts_out[i][k] = s->cells[ e->send[i].cellid[k] ].count );
                /* printf( "engine_exchange[%i]: totals_send[%i]=%i.\n" , e->nodeID , i , totals_send[i] ); */

                /* Ship it off to the correct node. */
                if ( ( res = MPI_Isend( counts_out[i] , e->send[i].count , MPI_INT , i , e->nodeID , e->comm , &reqs_send[i] ) ) != MPI_SUCCESS )
                    return error(engine_err_mpi);
                /* printf( "engine_exchange[%i]: sending %i counts to node %i.\n" , e->nodeID , e->send[i].count , i ); */

                }
            else
                reqs_send[i] = MPI_REQUEST_NULL;

            /* Are we expecting any parts? */
            if ( e->recv[i].count > 0 ) {

                /* Allocate a new lengths array for the incomming data. */
                if ( ( counts_in[i] = (int *)malloc( sizeof(int) * e->recv[i].count ) ) == NULL )
                    return error(engine_err_malloc);

                /* Dispatch a recv request. */
                if ( ( res = MPI_Irecv( counts_in[i] , e->recv[i].count , MPI_INT , i , i , e->comm , &reqs_recv[i] ) ) != MPI_SUCCESS )
                    return error(engine_err_mpi);
                /* printf( "engine_exchange[%i]: recving %i counts from node %i.\n" , e->nodeID , e->recv[i].count , i ); */

                }
            else
                reqs_recv[i] = MPI_REQUEST_NULL;

            }

        /* Wait for the counts to come in. */
        if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
            return error(engine_err_mpi);
        if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
            return error(engine_err_mpi);
        /* printf( "engine_exchange[%i]: successfully exchanged counts.\n" , e->nodeID ); */

        /* Send and receive data. */
        for ( i = 0 ; i < e->nr_nodes ; i++ ) {

            /* Do we have anything to send? */
            if ( e->send[i].count > 0 ) {

                /* Allocate a buffer for the send queue. */
                if ( ( buff_send[i] = (struct part *)malloc( sizeof(struct part) * totals_send[i] ) ) == NULL )
                    return error(engine_err_malloc);

                /* Fill the send buffer. */
                finger = buff_send[i];
                for ( k = 0 ; k < e->send[i].count ; k++ ) {
                    c = &( s->cells[e->send[i].cellid[k]] );
                    memcpy( finger , c->parts , sizeof(struct part) * c->count );
                    finger = &( finger[ c->count ] );
                    }

                /* File a send. */
                if ( ( res = MPI_Isend( buff_send[i] , totals_send[i]*sizeof(struct part) , MPI_BYTE , i , e->nodeID , e->comm , &reqs_send[i] ) ) != MPI_SUCCESS )
                    return error(engine_err_mpi);
                /* printf( "engine_exchange[%i]: sending %i parts to node %i.\n" , e->nodeID , totals_send[i] , i ); */

                }

            /* Are we expecting any parts? */
            if ( e->recv[i].count > 0 ) {

                /* Count the nr of parts to recv. */
                totals_recv[i] = 0;
                for ( k = 0 ; k < e->recv[i].count ; k++ )
                    totals_recv[i] += counts_in[i][k];

                /* Allocate a buffer for the send and recv queues. */
                if ( ( buff_recv[i] = (struct part *)malloc( sizeof(struct part) * totals_recv[i] ) ) == NULL )
                    return error(engine_err_malloc);

                /* File a recv. */
                if ( ( res = MPI_Irecv( buff_recv[i] , totals_recv[i]*sizeof(struct part) , MPI_BYTE , i , i , e->comm , &reqs_recv[i] ) ) != MPI_SUCCESS )
                    return error(engine_err_mpi);
                /* printf( "engine_exchange[%i]: recving %i parts from node %i.\n" , e->nodeID , totals_recv[i] , i ); */

                }

            }

        /* Wait for all the recvs to come in. */
        /* if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
            return error(engine_err_mpi); */

        /* Unpack the received data. */
        for ( i = 0 ; i < e->nr_nodes ; i++ ) {

            /* Wait for this recv to come in. */
            /* if ( ( res = MPI_Wait( &reqs_recv[i] , MPI_STATUS_IGNORE ) ) != MPI_SUCCESS )
                return error(engine_err_mpi); */
            res = MPI_Waitany( e->nr_nodes , reqs_recv , &ind , MPI_STATUS_IGNORE );

            /* Did we get a propper index? */
            if ( ind != MPI_UNDEFINED ) {

                /* Loop over the data and pass it to the cells. */
                finger = buff_recv[ind];
                for ( k = 0 ; k < e->recv[ind].count ; k++ ) {
                    cid = e->recv[ind].cellid[k];
                    c = &( s->cells[cid] );
                    cell_load( c , finger , counts_in[ind][k] , s->partlist , s->celllist );
                    space_releasepair( &e->s , cid , cid );
                    finger = &( finger[ counts_in[ind][k] ] );
                    }

                }

            }

        /* Wait for all the sends to come in. */
        if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
            return error(engine_err_mpi);
        /* printf( "engine_exchange[%i]: all send/recv completed.\n" , e->nodeID ); */

        /* Free the send and recv buffers. */
        for ( i = 0 ; i < e->nr_nodes ; i++ ) {
            if ( e->send[i].count > 0 ) {
                free( buff_send[i] );
                free( counts_out[i] );
                }
            if ( e->recv[i].count > 0 ) {
                free( buff_recv[i] );
                free( counts_in[i] );
                }
            }

        } /* main loop. */
        
    }
#endif
        
        
/**
 * @brief Exchange data with other nodes asynchronously.
 *
 * @param e The #engine to work with.
 * @param comm The @c MPI_Comm over which to exchange data.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Starts a new thread which handles the particle exchange. At the
 * start of the exchange, ghost cells are marked in the taboo-list
 * and only freed once their data has been received.
 *
 * The function #engine_exchange_wait can be used to wait for
 * the asynchronous communication to finish.
 */

#ifdef HAVE_MPI 
int engine_exchange_async ( struct engine *e ) {

    int k, cid;

    /* Check the input. */
    if ( e == NULL )
        return error(engine_err_null);

    /* Bail if not in parallel. */
    if ( !(e->flags & engine_flag_mpi) || e->nr_nodes <= 1 )
        return engine_err_ok;
        
    /* Mark all the ghost cells as taboo and flush them. */
    for ( k = 0 ; k < e->s.nr_ghost ; k++ ) {
        cid = e->s.cid_ghost[k];
        e->s.cells_taboo[ cid ] += 2;
        if ( cell_flush( &e->s.cells[cid] , e->s.partlist , e->s.celllist ) < 0 )
            return error(engine_err_cell);
        }
            
    /* Get a hold of the exchange mutex. */
    if ( pthread_mutex_lock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);
        
    /* Tell the async thread to get to work. */
    e->xchg_started = 1;
    if ( pthread_cond_signal( &e->xchg_cond ) != 0 )
        return error(engine_err_pthread);
        
    /* Release the exchange mutex and let the async run. */
    if ( pthread_mutex_unlock( &e->xchg_mutex ) != 0 )
        return error(engine_err_pthread);
        
    /* Done (for now). */
    return engine_err_ok;
        
    }
#endif
    
    
/**
 * @brief Exchange data with other nodes.
 *
 * @param e The #engine to work with.
 * @param comm The @c MPI_Comm over which to exchange data.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

#ifdef HAVE_MPI 
int engine_exchange ( struct engine *e ) {

    int i, k, ind, res, pid, cid, delta[3];
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c, *c_dest;
    struct part *p;
    struct space *s;
    FPTYPE h[3];
    
    /* Check the input. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Bail if not in parallel. */
    if ( !(e->flags & engine_flag_mpi) || e->nr_nodes <= 1 )
        return engine_err_ok;
        
    /* Get local copies of some data. */
    s = &e->s;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* Wait for any asynchronous calls to finish. */
    if ( e->flags & engine_flag_async )
        if ( engine_exchange_wait( e ) < 0 )
            return error(engine_err);
        
    /* Start by packing and sending/receiving a counts array for each send queue. */
    #pragma omp parallel for schedule(static), private(i,k,res)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Do we have anything to send? */
        if ( e->send[i].count > 0 ) {
        
            /* Allocate a new lengths array. */
            counts_out[i] = (int *)malloc( sizeof(int) * e->send[i].count );

            /* Pack the array with the counts. */
            totals_send[i] = 0;
            for ( k = 0 ; k < e->send[i].count ; k++ )
                totals_send[i] += ( counts_out[i][k] = s->cells[ e->send[i].cellid[k] ].count );
            /* printf( "engine_exchange[%i]: totals_send[%i]=%i.\n" , e->nodeID , i , totals_send[i] ); */

            /* Ship it off to the correct node. */
            /* printf( "engine_exchange[%i]: sending %i counts to node %i.\n" , e->nodeID , e->send[i].count , i ); */
            #pragma omp critical
            { res = MPI_Isend( counts_out[i] , e->send[i].count , MPI_INT , i , e->nodeID , e->comm , &reqs_send[i] ); }
            
            }
        else
            reqs_send[i] = MPI_REQUEST_NULL;
            
        /* Are we expecting any parts? */
        if ( e->recv[i].count > 0 ) {
    
            /* Allocate a new lengths array for the incomming data. */
            counts_in[i] = (int *)malloc( sizeof(int) * e->recv[i].count );

            /* Dispatch a recv request. */
            /* printf( "engine_exchange[%i]: recving %i counts from node %i.\n" , e->nodeID , e->recv[i].count , i ); */
            #pragma omp critical
            { res = MPI_Irecv( counts_in[i] , e->recv[i].count , MPI_INT , i , i , e->comm , &reqs_recv[i] ); }
            
            }
        else
            reqs_recv[i] = MPI_REQUEST_NULL;
    
        }
        
    /* Wait for the counts to come in. */
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    /* printf( "engine_exchange[%i]: successfully exchanged counts.\n" , e->nodeID ); */
        
    /* Send and receive data. */
    #pragma omp parallel for schedule(static), private(i,finger,k,c,res)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Do we have anything to send? */
        if ( e->send[i].count > 0 ) {
            
            /* Allocate a buffer for the send queue. */
            buff_send[i] = (struct part *)malloc( sizeof(struct part) * totals_send[i] );

            /* Fill the send buffer. */
            finger = buff_send[i];
            for ( k = 0 ; k < e->send[i].count ; k++ ) {
                c = &( s->cells[e->send[i].cellid[k]] );
                memcpy( finger , c->parts , sizeof(struct part) * c->count );
                finger = &( finger[ c->count ] );
                }

            /* File a send. */
            /* printf( "engine_exchange[%i]: sending %i parts to node %i.\n" , e->nodeID , totals_send[i] , i ); */
            #pragma omp critical
            { res = MPI_Isend( buff_send[i] , totals_send[i]*sizeof(struct part) , MPI_BYTE , i , e->nodeID , e->comm , &reqs_send[i] ); }
            
            }
            
        /* Are we expecting any parts? */
        if ( e->recv[i].count > 0 ) {
    
            /* Count the nr of parts to recv. */
            totals_recv[i] = 0;
            for ( k = 0 ; k < e->recv[i].count ; k++ )
                totals_recv[i] += counts_in[i][k];

            /* Allocate a buffer for the send and recv queues. */
            buff_recv[i] = (struct part *)malloc( sizeof(struct part) * totals_recv[i] );

            /* File a recv. */
            /* printf( "engine_exchange[%i]: recving %i parts from node %i.\n" , e->nodeID , totals_recv[i] , i ); */
            #pragma omp critical
            { res = MPI_Irecv( buff_recv[i] , totals_recv[i]*sizeof(struct part) , MPI_BYTE , i , i , e->comm , &reqs_recv[i] ); }
            
            }
            
        }

    /* Wait for all the recvs to come in. */
    /* if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi); */
        
    /* Unpack the received data. */
    #pragma omp parallel for schedule(static), private(i,ind,res,finger,k,c)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        { res = MPI_Waitany( e->nr_nodes , reqs_recv , &ind , MPI_STATUS_IGNORE ); }
        
        /* Did we get a propper index? */
        if ( ind != MPI_UNDEFINED ) {

            /* Loop over the data and pass it to the cells. */
            finger = buff_recv[ind];
            for ( k = 0 ; k < e->recv[ind].count ; k++ ) {
                c = &( s->cells[e->recv[ind].cellid[k]] );
                cell_flush( c , s->partlist , s->celllist );
                cell_load( c , finger , counts_in[ind][k] , s->partlist , s->celllist );
                finger = &( finger[ counts_in[ind][k] ] );
                }
                
            }
                
        }
        
    /* Wait for all the sends to come in. */
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    /* printf( "engine_exchange[%i]: all send/recv completed.\n" , e->nodeID ); */
        
    /* Free the send and recv buffers. */
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
        if ( e->send[i].count > 0 ) {
            free( buff_send[i] );
            free( counts_out[i] );
            }
        if ( e->recv[i].count > 0 ) {
            free( buff_recv[i] );
            free( counts_in[i] );
            }
        }
        
    /* Do we need to update cell locations? */
    if ( !( e->flags & engine_flag_verlet ) ) {
    
        /* Shuffle the particles to the correct cells. */
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,k,delta,c_dest)
        for ( cid = 0 ; cid < s->nr_marked ; cid++ ) {
            c = &(s->cells[s->cid_marked[cid]]);
            pid = 0;
            while ( pid < c->count ) {

                p = &( c->parts[pid] );
                for ( k = 0 ; k < 3 ; k++ )
                    delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );

                /* do we have to move this particle? */
                if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                    for ( k = 0 ; k < 3 ; k++ )
                        p->x[k] -= delta[k] * h[k];
                    c_dest = &( s->cells[ space_cellid( s ,
                        (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] , 
                        (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] , 
                        (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );

	                if ( c_dest->flags & cell_flag_marked ) {
                        pthread_mutex_lock(&c_dest->cell_mutex);
                        cell_add_incomming( c_dest , p );
	                    pthread_mutex_unlock(&c_dest->cell_mutex);
                        s->celllist[ p->id ] = c_dest;
                        }
                    else {
                        s->partlist[ p->id ] = NULL;
                        s->celllist[ p->id ] = NULL;
                        }

                    c->count -= 1;
                    if ( pid < c->count ) {
                        c->parts[pid] = c->parts[c->count];
                        s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
                        }
                    }
                else
                    pid += 1;
                }
            }

        /* Welcome the new particles in each cell. */
        #pragma omp parallel for schedule(static), private(c)
        for ( cid = 0 ; cid < s->nr_marked ; cid++ )
            cell_welcome( &(s->cells[s->cid_marked[cid]]) , s->partlist );
            
        }
        
    /* Call it a day. */
    return engine_err_ok;
        
    }
#endif


/**
 * @brief Exchange incomming particle data with other nodes.
 *
 * @param e The #engine to work with.
 * @param comm The @c MPI_Comm over which to exchange data.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

#ifdef HAVE_MPI 
int engine_exchange_incomming ( struct engine *e ) {

    int i, j, k, ind, res;
    int *counts_in[ e->nr_nodes ], *counts_out[ e->nr_nodes ];
    int totals_send[ e->nr_nodes ], totals_recv[ e->nr_nodes ];
    MPI_Request reqs_send[ e->nr_nodes ], reqs_recv[ e->nr_nodes ];
    struct part *buff_send[ e->nr_nodes ], *buff_recv[ e->nr_nodes ], *finger;
    struct cell *c;
    struct space *s;
    FPTYPE h[3];
    
    /* Check the input. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Bail if not in parallel. */
    if ( !(e->flags & engine_flag_mpi) || e->nr_nodes <= 1 )
        return engine_err_ok;
        
    /* Get local copies of some data. */
    s = &e->s;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* As opposed to #engine_exchange, we are going to send the incomming
       particles on ghost cells that do not belong to us. We therefore invert
       the send/recv queues, i.e. we send the incommings for the cells
       from which we usually receive data. */
        
    /* Start by packing and sending/receiving a counts array for each send queue. */
    #pragma omp parallel for schedule(static), private(i,k,res)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Do we have anything to send? */
        if ( e->recv[i].count > 0 ) {
        
            /* Allocate a new lengths array. */
            counts_out[i] = (int *)malloc( sizeof(int) * e->recv[i].count );

            /* Pack the array with the counts. */
            totals_send[i] = 0;
            for ( k = 0 ; k < e->recv[i].count ; k++ )
                totals_send[i] += ( counts_out[i][k] = s->cells[ e->recv[i].cellid[k] ].incomming_count );
            /* printf( "engine_exchange[%i]: totals_send[%i]=%i.\n" , e->nodeID , i , totals_send[i] ); */

            /* Ship it off to the correct node. */
            /* printf( "engine_exchange[%i]: sending %i counts to node %i.\n" , e->nodeID , e->send[i].count , i ); */
            #pragma omp critical
            { res = MPI_Isend( counts_out[i] , e->recv[i].count , MPI_INT , i , e->nodeID , e->comm , &reqs_send[i] ); }
            
            }
        else
            reqs_send[i] = MPI_REQUEST_NULL;
            
        /* Are we expecting any parts? */
        if ( e->send[i].count > 0 ) {
    
            /* Allocate a new lengths array for the incomming data. */
            counts_in[i] = (int *)malloc( sizeof(int) * e->send[i].count );

            /* Dispatch a recv request. */
            /* printf( "engine_exchange[%i]: recving %i counts from node %i.\n" , e->nodeID , e->recv[i].count , i ); */
            #pragma omp critical
            { res = MPI_Irecv( counts_in[i] , e->send[i].count , MPI_INT , i , i , e->comm , &reqs_recv[i] ); }
            
            }
        else
            reqs_recv[i] = MPI_REQUEST_NULL;
    
        }
        
    /* Wait for the counts to come in. */
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    /* printf( "engine_exchange[%i]: successfully exchanged counts.\n" , e->nodeID ); */
        
    /* Send and receive data. */
    #pragma omp parallel for schedule(static), private(i,finger,k,c,res)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Do we have anything to send? */
        if ( e->recv[i].count > 0 ) {
            
            /* Allocate a buffer for the send queue. */
            buff_send[i] = (struct part *)malloc( sizeof(struct part) * totals_send[i] );

            /* Fill the send buffer. */
            finger = buff_send[i];
            for ( k = 0 ; k < e->recv[i].count ; k++ ) {
                c = &( s->cells[e->recv[i].cellid[k]] );
                memcpy( finger , c->incomming , sizeof(struct part) * c->incomming_count );
                finger = &( finger[ c->incomming_count ] );
                }

            /* File a send. */
            /* printf( "engine_exchange[%i]: sending %i parts to node %i.\n" , e->nodeID , totals_send[i] , i ); */
            #pragma omp critical
            { res = MPI_Isend( buff_send[i] , totals_send[i]*sizeof(struct part) , MPI_BYTE , i , e->nodeID , e->comm , &reqs_send[i] ); }
            
            }
            
        /* Are we expecting any parts? */
        if ( e->send[i].count > 0 ) {
    
            /* Count the nr of parts to recv. */
            totals_recv[i] = 0;
            for ( k = 0 ; k < e->send[i].count ; k++ )
                totals_recv[i] += counts_in[i][k];

            /* Allocate a buffer for the send and recv queues. */
            buff_recv[i] = (struct part *)malloc( sizeof(struct part) * totals_recv[i] );

            /* File a recv. */
            /* printf( "engine_exchange[%i]: recving %i parts from node %i.\n" , e->nodeID , totals_recv[i] , i ); */
            #pragma omp critical
            { res = MPI_Irecv( buff_recv[i] , totals_recv[i]*sizeof(struct part) , MPI_BYTE , i , i , e->comm , &reqs_recv[i] ); }
            
            }
            
        }

    /* Wait for all the recvs to come in. */
    /* if ( ( res = MPI_Waitall( e->nr_nodes , reqs_recv , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi); */
        
    /* Unpack the received data. */
    #pragma omp parallel for schedule(static), private(i,j,ind,res,finger,k,c)
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
    
        /* Wait for this recv to come in. */
        #pragma omp critical
        { res = MPI_Waitany( e->nr_nodes , reqs_recv , &ind , MPI_STATUS_IGNORE ); }
        
        /* Did we get a propper index? */
        if ( ind != MPI_UNDEFINED ) {

            /* Loop over the data and pass it to the cells. */
            finger = buff_recv[ind];
            for ( k = 0 ; k < e->send[ind].count ; k++ ) {
                c = &( s->cells[e->send[ind].cellid[k]] );
                pthread_mutex_lock( &c->cell_mutex );
                cell_add_incomming_multiple( c , finger , counts_in[ind][k] );
                pthread_mutex_unlock( &c->cell_mutex );
                for ( j = 0 ; j < counts_in[ind][k] ; j++ )
                    e->s.celllist[ finger[j].id ] = c;
                finger = &( finger[ counts_in[ind][k] ] );
                }
                
            }
                
        }
        
    /* Wait for all the sends to come in. */
    if ( ( res = MPI_Waitall( e->nr_nodes , reqs_send , MPI_STATUSES_IGNORE ) ) != MPI_SUCCESS )
        return error(engine_err_mpi);
    /* printf( "engine_exchange[%i]: all send/recv completed.\n" , e->nodeID ); */
        
    /* Free the send and recv buffers. */
    for ( i = 0 ; i < e->nr_nodes ; i++ ) {
        if ( e->send[i].count > 0 ) {
            free( buff_send[i] );
            free( counts_out[i] );
            }
        if ( e->recv[i].count > 0 ) {
            free( buff_recv[i] );
            free( counts_in[i] );
            }
        }
        
    /* Call it a day. */
    return engine_err_ok;
        
    }
#endif


/**
 * @brief Set-up the engine for distributed-memory parallel operation.
 *
 * @param e The #engine to set-up.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * This function assumes that #engine_split_bisect or some similar
 * function has already been called and that #nodeID, #nr_nodes as
 * well as the #cell @c nodeIDs have been set.
 */
int engine_split ( struct engine *e ) {

    int i, k, cid, cjd;
    struct cell *ci, *cj, *ct;

    /* Check for nonsense inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Start by allocating and initializing the send/recv lists. */
    if ( ( e->send = (struct engine_comm *)malloc( sizeof(struct engine_comm) * e->nr_nodes ) ) == NULL ||
         ( e->recv = (struct engine_comm *)malloc( sizeof(struct engine_comm) * e->nr_nodes ) ) == NULL )
        return error(engine_err_malloc);
    for ( k = 0 ; k < e->nr_nodes ; k++ ) {
        if ( ( e->send[k].cellid = (int *)malloc( sizeof(int) * 100 ) ) == NULL )
            return error(engine_err_malloc);
        e->send[k].size = 100;
        e->send[k].count = 0;
        if ( ( e->recv[k].cellid = (int *)malloc( sizeof(int) * 100 ) ) == NULL )
            return error(engine_err_malloc);
        e->recv[k].size = 100;
        e->recv[k].count = 0;
        }
        
    /* Un-mark all cells. */
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ )
        e->s.cells[cid].flags &= ~cell_flag_marked;
        
    /* Loop over each cell pair... */
    for ( i = 0 ; i < e->s.nr_pairs ; i++ ) {
    
        /* Get the cells in this pair. */
        cid = e->s.pairs[i].i;
        cjd = e->s.pairs[i].j;
        ci = &( e->s.cells[ cid ] );
        cj = &( e->s.cells[ cjd ] );
        
        /* If it is a ghost-ghost pair, skip it. */
        if ( (ci->flags & cell_flag_ghost) && (cj->flags & cell_flag_ghost) )
            continue;
            
        /* Mark the cells. */
        ci->flags |= cell_flag_marked;
        cj->flags |= cell_flag_marked;
            
        /* Make cj the ghost cell and bail if both are real. */
        if ( ci->flags & cell_flag_ghost ) {
            ct = ci; ci = cj; cj = ct;
            k = cid; cid = cjd; cjd = k;
            }
        else if ( !( cj->flags & cell_flag_ghost ) )
            continue;
        
        /* Store the communication between cid and cjd. */
        /* Store the send, if not already there... */
        for ( k = 0 ; k < e->send[cj->nodeID].count && e->send[cj->nodeID].cellid[k] != cid ; k++ );
        if ( k == e->send[cj->nodeID].count ) {
            if ( e->send[cj->nodeID].count == e->send[cj->nodeID].size ) {
                e->send[cj->nodeID].size += 100;
                if ( ( e->send[cj->nodeID].cellid = (int *)realloc( e->send[cj->nodeID].cellid , sizeof(int) * e->send[cj->nodeID].size ) ) == NULL )
                    return error(engine_err_malloc);
                }
            e->send[cj->nodeID].cellid[ e->send[cj->nodeID].count++ ] = cid;
            }
        /* Store the recv, if not already there... */
        for ( k = 0 ; k < e->recv[cj->nodeID].count && e->recv[cj->nodeID].cellid[k] != cjd ; k++ );
        if ( k == e->recv[cj->nodeID].count ) {
            if ( e->recv[cj->nodeID].count == e->recv[cj->nodeID].size ) {
                e->recv[cj->nodeID].size += 100;
                if ( ( e->recv[cj->nodeID].cellid = (int *)realloc( e->recv[cj->nodeID].cellid , sizeof(int) * e->recv[cj->nodeID].size ) ) == NULL )
                    return error(engine_err_malloc);
                }
            e->recv[cj->nodeID].cellid[ e->recv[cj->nodeID].count++ ] = cjd;
            }
            
        }
        
    /* Nuke all ghost-ghost pairs. */
    i = 0;
    while ( i < e->s.nr_pairs ) {
    
        /* Get the cells in this pair. */
        ci = &( e->s.cells[ e->s.pairs[i].i ] );
        cj = &( e->s.cells[ e->s.pairs[i].j ] );
        
        /* If it is a ghost-ghost pair, skip it. */
        if ( (ci->flags & cell_flag_ghost) && (cj->flags & cell_flag_ghost) )
            e->s.pairs[i] = e->s.pairs[ --(e->s.nr_pairs) ];
        else
            i += 1;
            
        }
        
    /* Empty unmarked cells. */
    for ( k = 0 ; k < e->s.nr_cells ; k++ )
        if ( !( e->s.cells[k].flags & cell_flag_marked ) )
            cell_flush( &e->s.cells[k] , e->s.partlist , e->s.celllist );
            
    /* Set ghost markings. */
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ )
        if ( e->s.cells[cid].flags & cell_flag_ghost )
            for ( k = 0 ; k < e->s.cells[cid].count ; k++ )
                e->s.cells[cid].parts[k].flags |= part_flag_ghost;
        
    /* Fill the cid lists with marked, local and ghost cells. */
    e->s.nr_real = 0; e->s.nr_ghost = 0; e->s.nr_marked = 0;
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ )
        if ( e->s.cells[cid].flags & cell_flag_marked ) {
            e->s.cid_marked[ e->s.nr_marked++ ] = cid;
            if ( e->s.cells[cid].flags & cell_flag_ghost ) {
                e->s.cells[cid].id = -e->s.nr_cells;
                e->s.cid_ghost[ e->s.nr_ghost++ ] = cid;
                }
            else {
                e->s.cells[cid].id = e->s.nr_real;
                e->s.cid_real[ e->s.nr_real++ ] = cid;
                }
            }
        
    /* Re-build the tuples if needed. */
    if ( e->flags & engine_flag_tuples )
        if ( space_maketuples( &e->s ) < 0 )
            return error(engine_err_space);
        
    /* Done deal. */
    return engine_err_ok;

    }
        
    
/**
 * @brief Split the computational domain over a number of nodes using
 *      bisection.
 *
 * @param e The #engine to split up.
 * @param N The number of computational nodes.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_split_bisect ( struct engine *e , int N ) {

    /* Interior, recursive function that actually does the split. */
    int engine_split_bisect_rec( int N_min , int N_max , int x_min , int x_max , int y_min , int y_max , int z_min , int z_max ) {
    
        int i, j, k, m, Nm;
        int hx, hy, hz;
        unsigned int flag = 0;
        struct cell *c;
    
        /* Check inputs. */
        if ( x_max < x_min || y_max < y_min || z_max < z_min )
            return error(engine_err_domain);
            
        /* Is there nothing left to split? */
        if ( N_min == N_max ) {
        
            /* Flag as ghost or not? */
            if ( N_min != e->nodeID )
                flag = cell_flag_ghost;
                
            /* printf("engine_split_bisect: marking range [ %i..%i , %i..%i , %i..%i ] with flag %i.\n",
                x_min, x_max, y_min, y_max, z_min, z_max, flag ); */
        
            /* Run through the cells. */
            for ( i = x_min ; i < x_max ; i++ )
                for ( j = y_min ; j < y_max ; j++ )
                    for ( k = z_min ; k < z_max ; k++ ) {
                        c = &( e->s.cells[ space_cellid(&(e->s),i,j,k) ] );
                        c->flags |= flag;
                        c->nodeID = N_min;
                        }
                        
            }
            
        /* Otherwise, bisect. */
        else {
        
            hx = x_max - x_min;
            hy = y_max - y_min;
            hz = z_max - z_min;
            Nm = (N_min + N_max) / 2;
        
            /* Is the x-axis the largest? */
            if ( hx > hy && hx > hz ) {
                m = (x_min + x_max) / 2;
                if ( engine_split_bisect_rec( N_min , Nm , x_min , m , y_min , y_max , z_min , z_max ) < 0 ||
                     engine_split_bisect_rec( Nm+1 , N_max , m , x_max , y_min , y_max , z_min , z_max ) < 0 )
                    return error(engine_err);
                }
        
            /* Nope, maybe the y-axis? */
            else if ( hy > hz ) {
                m = (y_min + y_max) / 2;
                if ( engine_split_bisect_rec( N_min , Nm , x_min , x_max , y_min , m , z_min , z_max ) < 0 ||
                     engine_split_bisect_rec( Nm+1 , N_max , x_min , x_max , m , y_max , z_min , z_max ) < 0 )
                    return error(engine_err);
                }
        
            /* Then it has to be the z-axis. */
            else {
                m = (z_min + z_max) / 2;
                if ( engine_split_bisect_rec( N_min , Nm , x_min , x_max , y_min , y_max , z_min , m ) < 0 ||
                     engine_split_bisect_rec( Nm+1 , N_max , x_min , x_max , y_min , y_max , m , z_max ) < 0 )
                    return error(engine_err);
                }
        
            }
            
        /* So far, so good! */
        return engine_err_ok;
    
        }

    /* Check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Call the recursive bisection. */
    if ( engine_split_bisect_rec( 0 , N-1 , 0 , e->s.cdim[0] , 0 , e->s.cdim[1] , 0 , e->s.cdim[2] ) < 0 )
        return error(engine_err);
        
    /* Store the number of nodes. */
    e->nr_nodes = N;
        
    /* Call it a day. */
    return engine_err_ok;
    
    }
    
    
/**
 * @brief Clear all particles from this #engine.
 *
 * @param e The #engine to flush.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_flush ( struct engine *e ) {

    /* check input. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Clear the space. */
    if ( space_flush( &e->s ) < 0 )
        return error(engine_err_space);
        
    /* done for now. */
    return engine_err_ok;

    }


/**
 * @brief Clear all particles from this #engine's ghost cells.
 *
 * @param e The #engine to flush.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_flush_ghosts ( struct engine *e ) {

    /* check input. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Clear the space. */
    if ( space_flush_ghosts( &e->s ) < 0 )
        return error(engine_err_space);
        
    /* done for now. */
    return engine_err_ok;

    }


/** 
 * @brief Set the explicit electrostatic potential.
 *
 * @param e The #engine.
 * @param ep The electrostatic #potential.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * If @c ep is not @c NULL, the flag #engine_flag_explepot is set,
 * otherwise it is cleared.
 */
 
int engine_setexplepot ( struct engine *e , struct potential *ep ) {

    /* check inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* was a potential supplied? */
    if ( ep != NULL ) {
    
        /* set the flag. */
        e->flags |= engine_flag_explepot;
        
        /* set the potential. */
        e->ep = ep;
        
        }
        
    /* otherwise, just clear the flag. */
    else
        e->flags &= ~engine_flag_explepot;
        
    /* done for now. */
    return engine_err_ok;

    }
    
    
/**
 * @brief Unload a set of particle data from the #engine.
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param epot A pointer to a #double in which to store the total potential energy.
 * @param N the maximum number of particles.
 *
 * @return The number of particles unloaded or < 0 on
 *      error (see #engine_err).
 *
 * The fields @c x, @c v, @c type, @c pid, @c vid, @c q, @c epot and/or @c flags may be NULL.
 */
 
int engine_unload ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , double *epot , int N ) {

    struct part *p;
    struct cell *c;
    int j, k, cid, count = 0, *ind;
    double epot_acc = 0.0;
    
    /* check the inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Allocate and fill the indices. */
    if ( ( ind = (int *)alloca( sizeof(int) * (e->s.nr_cells + 1) ) ) == NULL )
        return error(engine_err_malloc);
    ind[0] = 0;
    for ( k = 0 ; k < e->s.nr_cells ; k++ )
        ind[k+1] = ind[k] + e->s.cells[k].count;
    if ( ind[e->s.nr_cells] > N )
        return error(engine_err_range);
        
    /* Loop over each cell. */
    #pragma omp parallel for schedule(static), private(cid,count,c,k,p,j), reduction(+:epot_acc)
    for ( cid = 0 ; cid < e->s.nr_cells ; cid++ ) {
    
        /* Get a hold of the cell. */
        c = &( e->s.cells[cid] );
        count = ind[cid];
    
        /* Collect the potential energy if requested. */
        epot_acc += c->epot;
            
        /* Loop over the parts in this cell. */
        for ( k = 0 ; k < c->count ; k++ ) {
        
            /* Get a hold of the particle. */
            p = &( c->parts[k] );
        
            /* get this particle's data, where requested. */
            if ( x != NULL )
                for ( j = 0 ; j < 3 ; j++ )
                    x[count*3+j] = c->origin[j] + p->x[j];
            if ( v != NULL)
                for ( j = 0 ; j < 3 ; j++ )
                    v[count*3+j] = p->v[j];
            if ( type != NULL )
                type[count] = p->type;
            if ( pid != NULL )
                pid[count] = p->id;
            if ( vid != NULL )
                vid[count] = p->vid;
            if ( q != NULL )
                q[count] = p->q;
            if ( flags != NULL )
                flags[count] = p->flags;
                
            /* Step-up the counter. */
            count += 1;
                
            }
            
        }
        
    /* Write back the potential energy, if requested. */
    if ( epot != NULL )
        *epot += epot_acc;

    /* to the pub! */
    return ind[e->s.nr_cells];

    }


/**
 * @brief Unload a set of particle data from the marked cells of an #engine
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param epot A pointer to a #double in which to store the total potential energy.
 * @param N the maximum number of particles.
 *
 * @return The number of particles unloaded or < 0 on
 *      error (see #engine_err).
 *
 * The fields @c x, @c v, @c type, @c pid, @c vid, @c q, @c epot and/or @c flags may be NULL.
 */
 
int engine_unload_marked ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , double *epot , int N ) {

    struct part *p;
    struct cell *c;
    int j, k, cid, count = 0, *ind;
    double epot_acc = 0.0;
    
    /* check the inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Allocate and fill the indices. */
    if ( ( ind = (int *)alloca( sizeof(int) * (e->s.nr_cells + 1) ) ) == NULL )
        return error(engine_err_malloc);
    ind[0] = 0;
    for ( k = 0 ; k < e->s.nr_cells ; k++ )
        if ( e->s.cells[k].flags & cell_flag_marked )
            ind[k+1] = ind[k] + e->s.cells[k].count;
        else
            ind[k+1] = ind[k];
    if ( ind[e->s.nr_cells] > N )
        return error(engine_err_range);
        
    /* Loop over each cell. */
    #pragma omp parallel for schedule(static), private(cid,count,c,k,p,j), reduction(+:epot_acc)
    for ( cid = 0 ; cid < e->s.nr_marked ; cid++ ) {
    
        /* Get a hold of the cell. */
        c = &( e->s.cells[e->s.cid_marked[cid]] );
        count = ind[e->s.cid_marked[cid]];
    
        /* Collect the potential energy if requested. */
        epot_acc += c->epot;
            
        /* Loop over the parts in this cell. */
        for ( k = 0 ; k < c->count ; k++ ) {
        
            /* Get a hold of the particle. */
            p = &( c->parts[k] );
        
            /* get this particle's data, where requested. */
            if ( x != NULL )
                for ( j = 0 ; j < 3 ; j++ )
                    x[count*3+j] = c->origin[j] + p->x[j];
            if ( v != NULL)
                for ( j = 0 ; j < 3 ; j++ )
                    v[count*3+j] = p->v[j];
            if ( type != NULL )
                type[count] = p->type;
            if ( pid != NULL )
                pid[count] = p->id;
            if ( vid != NULL )
                vid[count] = p->vid;
            if ( q != NULL )
                q[count] = p->q;
            if ( flags != NULL )
                flags[count] = p->flags;
                
            /* Step-up the counter. */
            count += 1;
                
            }
            
        }
        
    /* Write back the potential energy, if requested. */
    if ( epot != NULL )
        *epot += epot_acc;

    /* to the pub! */
    return ind[e->s.nr_cells];

    }


/**
 * @brief Unload real particles that may have wandered into a ghost cell.
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param epot A pointer to a #double in which to store the total potential energy.
 * @param N the maximum number of particles.
 *
 * @return The number of particles unloaded or < 0 on
 *      error (see #engine_err).
 *
 * The fields @c x, @c v, @c type, @c vid, @c pid, @c q, @c epot and/or @c flags may be NULL.
 */
 
int engine_unload_strays ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , double *epot , int N ) {

    struct part *p;
    struct cell *c;
    int j, k, cid, count = 0;
    double epot_acc = 0.0;
    
    /* check the inputs. */
    if ( e == NULL )
        return error(engine_err_null);
        
    /* Loop over each cell. */
    for ( cid = 0 ; cid < e->s.nr_real ; cid++ ) {
    
        /* Get a hold of the cell. */
        c = &( e->s.cells[e->s.cid_real[cid]] );
        
        /* Collect the potential energy if requested. */
        epot_acc += c->epot;
            
        /* Loop over the parts in this cell. */
        for ( k = c->count-1 ; k >= 0 && !(c->parts[k].flags & part_flag_ghost) ; k-- ) {
        
            /* Get a hold of the particle. */
            p = &( c->parts[k] );
            if ( p->flags & part_flag_ghost )
                continue;
        
            /* get this particle's data, where requested. */
            if ( x != NULL )
                for ( j = 0 ; j < 3 ; j++ )
                    x[count*3+j] = c->origin[j] + p->x[j];
            if ( v != NULL)
                for ( j = 0 ; j < 3 ; j++ )
                    v[count*3+j] = p->v[j];
            if ( type != NULL )
                type[count] = p->type;
            if ( pid != NULL )
                pid[count] = p->id;
            if ( vid != NULL )
                vid[count] = p->vid;
            if ( q != NULL )
                q[count] = p->q;
            if ( flags != NULL )
                flags[count] = p->flags;
                
            /* increase the counter. */
            count += 1;
            
            }
            
        }
        
    /* Write back the potential energy, if requested. */
    if ( epot != NULL )
        *epot += epot_acc;

    /* to the pub! */
    return count;

    }


/**
 * @brief Load a set of particle data.
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param N the number of particles to load.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * If the parameters @c v, @c flags, @c vid or @c q are @c NULL, then
 * these values are set to zero.
 */
 
int engine_load ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , int N ) {

    struct part p;
    struct space *s;
    int j, k;
    
    /* check the inputs. */
    if ( e == NULL || x == NULL || type == NULL )
        return error(engine_err_null);
        
    /* Get a handle on the space. */
    s = &(e->s);
        
    /* init the velocity and charge in case not specified. */
    p.v[0] = 0.0; p.v[1] = 0.0; p.v[2] = 0.0;
    p.f[0] = 0.0; p.f[1] = 0.0; p.f[2] = 0.0;
    p.q = 0.0;
    p.flags = part_flag_none;
        
    /* loop over the entries. */
    for ( j = 0 ; j < N ; j++ ) {
    
        /* set the particle data. */
        p.type = type[j];
        if ( pid != NULL )
            p.id = pid[j];
        else
            p.id = j;
        if ( vid != NULL )
            p.vid = vid[j];
        if ( flags != NULL )
            p.flags = flags[j];
        if ( v != NULL )
            for ( k = 0 ; k < 3 ; k++ )
                p.v[k] = v[j*3+k];
        if ( q != 0 )
            p.q = q[j];
            
        /* add the part to the space. */
        if ( space_addpart( s , &p , &x[3*j] ) < 0 )
            return error(engine_err_space);
    
        }
        
    /* to the pub! */
    return engine_err_ok;

    }


/**
 * @brief Load a set of particle data as ghosts
 *
 * @param e The #engine.
 * @param x An @c N times 3 array of the particle positions.
 * @param v An @c N times 3 array of the particle velocities.
 * @param type A vector of length @c N of the particle type IDs.
 * @param pid A vector of length @c N of the particle IDs.
 * @param vid A vector of length @c N of the particle virtual IDs.
 * @param q A vector of length @c N of the individual particle charges.
 * @param flags A vector of length @c N of the particle flags.
 * @param N the number of particles to load.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * If the parameters @c v, @c flags, @c vid or @c q are @c NULL, then
 * these values are set to zero.
 */
 
int engine_load_ghosts ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , int N ) {

    struct part p;
    struct space *s;
    int j, k, nr_parts;
    
    /* check the inputs. */
    if ( e == NULL || x == NULL || type == NULL )
        return error(engine_err_null);
        
    /* Get a handle on the space. */
    s = &(e->s);
    nr_parts = s->nr_parts;
        
    /* init the velocity and charge in case not specified. */
    p.v[0] = 0.0; p.v[1] = 0.0; p.v[2] = 0.0;
    p.f[0] = 0.0; p.f[1] = 0.0; p.f[2] = 0.0;
    p.q = 0.0;
    p.flags = part_flag_ghost;
        
    /* loop over the entries. */
    for ( j = 0 ; j < N ; j++ ) {
    
        /* set the particle data. */
        p.type = type[j];
        if ( pid != NULL )
            p.id = pid[j];
        else
            p.id = j;
        if ( vid != NULL )
            p.vid = vid[j];
        if ( flags != NULL )
            p.flags = flags[j] | part_flag_ghost;
        if ( v != NULL )
            for ( k = 0 ; k < 3 ; k++ )
                p.v[k] = v[j*3+k];
        if ( q != 0 )
            p.q = q[j];
            
        /* add the part to the space. */
        if ( space_addpart( s , &p , &x[3*j] ) < 0 )
            return error(engine_err_space);
    
        }
        
    /* to the pub! */
    return engine_err_ok;

    }


/**
 * @brief Look for a given type by name.
 *
 * @param e The #engine.
 * @param name The type name.
 *
 * @return The type ID or < 0 on error (see #engine_err).
 */
 
int engine_gettype ( struct engine *e , char *name ) {

    int k;
    
    /* check for nonsense. */
    if ( e == NULL || name == NULL )
        return error(engine_err_null);
        
    /* Loop over the types... */
    for ( k = 0 ; k < e->nr_types ; k++ ) {
    
        /* Compare the name. */
        if ( strcmp( e->types[k].name , name ) == 0 )
            return k;
    
        }
        
    /* Otherwise, nothing found... */
    return engine_err_range;

    }


/**
 * @brief Look for a given type by its second name.
 *
 * @param e The #engine.
 * @param name2 The type name2.
 *
 * @return The type ID or < 0 on error (see #engine_err).
 */
 
int engine_gettype2 ( struct engine *e , char *name2 ) {

    int k;
    
    /* check for nonsense. */
    if ( e == NULL || name2 == NULL )
        return error(engine_err_null);
        
    /* Loop over the types... */
    for ( k = 0 ; k < e->nr_types ; k++ ) {
    
        /* Compare the name. */
        if ( strcmp( e->types[k].name2 , name2 ) == 0 )
            return k;
    
        }
        
    /* Otherwise, nothing found... */
    return engine_err_range;

    }


/**
 * @brief Add a type definition.
 *
 * @param e The #engine.
 * @param mass The particle type mass.
 * @param charge The particle type charge.
 * @param name Particle name, can be @c NULL.
 * @param name2 Particle second name, can be @c NULL.
 *
 * @return The type ID or < 0 on error (see #engine_err).
 *
 * The particle type ID must be an integer greater or equal to 0
 * and less than the value @c max_type specified in #engine_init.
 */
 
int engine_addtype ( struct engine *e , double mass , double charge , char *name , char *name2 ) {

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( e->nr_types >= e->max_type )
        return error(engine_err_range);
    
    /* set the type. */
    e->types[e->nr_types].mass = mass;
    e->types[e->nr_types].imass = 1.0 / mass;
    e->types[e->nr_types].charge = charge;
    if ( name != NULL )
        strcpy( e->types[e->nr_types].name , name );
    else
        strcpy( e->types[e->nr_types].name , "X" );
    if ( name2 != NULL )
        strcpy( e->types[e->nr_types].name2 , name2 );
    else
        strcpy( e->types[e->nr_types].name2 , "X" );
    
    /* bring good tidings. */
    return e->nr_types++;

    }


/**
 * @brief Add an interaction potential.
 *
 * @param e The #engine.
 * @param p The #potential to add to the #engine.
 * @param i ID of particle type for this interaction.
 * @param j ID of second particle type for this interaction.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Adds the given potential for pairs of particles of type @c i and @c j,
 * where @c i and @c j may be the same type ID.
 */
 
int engine_addpot ( struct engine *e , struct potential *p , int i , int j ) {

    /* check for nonsense. */
    if ( e == NULL )
        return error(engine_err_null);
    if ( i < 0 || i >= e->max_type || j < 0 || j >= e->max_type )
        return error(engine_err_range);
        
    /* store the potential. */
    e->p[ i * e->max_type + j ] = p;
    if ( i != j )
        e->p[ j * e->max_type + i ] = p;
        
    /* end on a good note. */
    return engine_err_ok;

    }


/**
 * @brief Start the runners in the given #engine.
 *
 * @param e The #engine to start.
 * @param nr_runners The number of runners start.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Allocates and starts the specified number of #runner. Also initializes
 * the Verlet lists.
 */

int engine_start ( struct engine *e , int nr_runners ) {

    int cid, pid, k, i;
    struct cell *c;
    struct part *p;
    struct space *s = &e->s;
    struct runner *temp;
    
    /* Set up async communication? */
    if ( e->flags & engine_flag_async ) {
    
        /* Init the mutex and condition variable for the asynchronous communication. */
	    if ( pthread_mutex_init( &e->xchg_mutex , NULL ) != 0 ||
             pthread_cond_init( &e->xchg_cond , NULL ) != 0 )
            return error(engine_err_pthread);
            
        /* Set the exchange flags. */
        e->xchg_started = 0;
        e->xchg_running = 0;
            
        /* Start a thread with the async exchange. */
        if ( pthread_create( &e->thread_exchg , NULL , (void *(*)(void *))engine_exchange_async_run , e ) != 0 )
            return error(engine_err_pthread);
            
        }
        
    /* Fill-in the Verlet lists if needed. */
    if ( e->flags & engine_flag_verlet ) {
    
        /* Shuffle the domain. */
        if ( space_shuffle( s ) < 0 )
            return error(space_err);
            
        /* Store the current positions as a reference. */
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,k)
        for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
            c = &(s->cells[s->cid_real[cid]]);
            if ( c->oldx == NULL || c->oldx_size < c->count ) {
                free(c->oldx);
                c->oldx_size = c->size + 20;
                c->oldx = (FPTYPE *)malloc( sizeof(FPTYPE) * 4 * c->oldx_size );
                }
            for ( pid = 0 ; pid < c->count ; pid++ ) {
                p = &(c->parts[pid]);
                for ( k = 0 ; k < 3 ; k++ )
                    c->oldx[ 4*pid + k ] = p->x[k];
                }
            }
            
        /* Set the nrpairs to zero. */
        if ( !( e->flags & engine_flag_verlet_pairwise ) && s->verlet_nrpairs != NULL )
            bzero( s->verlet_nrpairs , sizeof(int) * s->nr_parts );
            
        /* Re-set the Verlet rebuild flag. */
        s->verlet_rebuild = 1;

        }

    /* (re)allocate the runners */
    if ( e->nr_runners == 0 ) {
        if ( ( e->runners = (struct runner *)malloc( sizeof(struct runner) * nr_runners )) == NULL )
            return error(engine_err_malloc);
        }
    else {
        if ( ( temp = (struct runner *)malloc( sizeof(struct runner) * (e->nr_runners + nr_runners) )) == NULL )
            return error(engine_err_malloc);
        memcpy( temp , e->runners , sizeof(struct runner) * e->nr_runners );
        e->runners = temp;
        }
        
    /* initialize the runners. */
    for ( i = 0 ; i < nr_runners ; i++ )
        if ( runner_init(&e->runners[e->nr_runners + i],e,e->nr_runners + i) < 0 )
            return error(engine_err_runner);
    e->nr_runners += nr_runners;
            
    /* wait for the runners to be in place */
    while (e->barrier_count != e->nr_runners)
        if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
            return error(engine_err_pthread);
        
    /* all is well... */
    return engine_err_ok;
    
    }


/**
 * @brief Start the SPU-associated runners in the given #engine.
 *
 * @param e The #engine to start.
 * @param nr_runners The number of runners start.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * Allocates and starts the specified number of #runner.
 */

int engine_start_SPU ( struct engine *e , int nr_runners ) {

    int i;
    struct runner *temp;

    /* Set up async communication? */
    if ( e->flags & engine_flag_async ) {
    
        /* Init the mutex and condition variable for the asynchronous communication. */
	    if ( pthread_mutex_init( &e->xchg_mutex , NULL ) != 0 ||
             pthread_cond_init( &e->xchg_cond , NULL ) != 0 )
            return error(engine_err_pthread);
            
        /* Set the exchange flags. */
        e->xchg_started = 0;
        e->xchg_running = 0;
            
        /* Start a thread with the async exchange. */
        if ( pthread_create( &e->thread_exchg , NULL , (void *(*)(void *))engine_exchange_async_run , e ) != 0 )
            return error(engine_err_pthread);
            
        }
        
    /* (re)allocate the runners */
    if ( e->nr_runners == 0 ) {
        if ( ( e->runners = (struct runner *)malloc( sizeof(struct runner) * nr_runners )) == NULL )
            return error(engine_err_malloc);
        }
    else {
        if ( ( temp = (struct runner *)malloc( sizeof(struct runner) * (e->nr_runners + nr_runners) )) == NULL )
            return error(engine_err_malloc);
        memcpy( temp , e->runners , sizeof(struct runner) * e->nr_runners );
        free( e->runners );
        e->runners = temp;
        }
        
    /* initialize the runners. */
    for ( i = 0 ; i < nr_runners ; i++ )
        if ( runner_init_SPU(&e->runners[e->nr_runners + i],e,e->nr_runners + i) < 0 )
            return error(engine_err_runner);
    e->nr_runners += nr_runners;
            
    /* wait for the runners to be in place */
    while (e->barrier_count != e->nr_runners)
        if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
            return error(engine_err_pthread);
        
    /* all is well... */
    return engine_err_ok;
    
    }
    
    
/**
 * @brief Compute the nonbonded interactions in the current step.
 * 
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * This routine advances the timestep counter by one, prepares the #space
 * for a timestep, releases the #runner's associated with the #engine
 * and waits for them to finnish.
 */
 
int engine_nonbond_eval ( struct engine *e ) {

    struct space *s;
    
    /* Get a grip on the space. */
    s = &(e->s);

    /* open the door for the runners */
    e->barrier_count = -e->barrier_count;
    if (pthread_cond_broadcast(&e->barrier_cond) != 0)
        return error(engine_err_pthread);

    /* wait for the runners to come home */
    while (e->barrier_count < e->nr_runners)
        if (pthread_cond_wait(&e->done_cond,&e->barrier_mutex) != 0)
            return error(engine_err_pthread);
            
    /* All in a days work. */
    return engine_err_ok;
    
    }
    
    
/**
 * @brief Update the particle velocities and positions, re-shuffle if
 *      appropriate.
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */
 
int engine_advance ( struct engine *e ) {

    int cid, pid, k, delta[3];
    struct cell *c, *c_dest;
    struct part *p;
    struct space *s;
    FPTYPE dt, w, h[3];
    double epot = 0.0;
    
    /* Get a grip on the space. */
    s = &(e->s);
    dt = e->dt;
    for ( k = 0 ; k < 3 ; k++ )
        h[k] = s->h[k];
        
    /* update the particle velocities and positions */
    if ( e->flags & engine_flag_verlet || e->flags & engine_flag_mpi ) {
    
        /* Collect potential energy from ghosts. */
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ )
            epot += s->cells[ s->cid_ghost[cid] ].epot;
        
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,w,k), reduction(+:epot)
        for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
            c = &(s->cells[ s->cid_real[cid] ]);
            epot += c->epot;
            for ( pid = 0 ; pid < c->count ; pid++ ) {
                p = &( c->parts[pid] );
                w = dt * e->types[p->type].imass;
                for ( k = 0 ; k < 3 ; k++ ) {
                    p->v[k] += p->f[k] * w;
                    p->x[k] += dt * p->v[k];
                    }
                }
            }
            
        }
    else {
    
        /* Collect potential energy from ghosts. */
        for ( cid = 0 ; cid < s->nr_ghost ; cid++ )
            epot += s->cells[ s->cid_ghost[cid] ].epot;
        
        #pragma omp parallel for schedule(static), private(cid,c,pid,p,w,k,delta,c_dest), reduction(+:epot)
        for ( cid = 0 ; cid < s->nr_real ; cid++ ) {
            c = &(s->cells[ s->cid_real[cid] ]);
            epot += c->epot;
            pid = 0;
            while ( pid < c->count ) {
            
                p = &( c->parts[pid] );
                w = dt * e->types[p->type].imass;
                for ( k = 0 ; k < 3 ; k++ ) {
                    p->v[k] += p->f[k] * w;
                    p->x[k] += dt * p->v[k];
                    delta[k] = __builtin_isgreaterequal( p->x[k] , h[k] ) - __builtin_isless( p->x[k] , 0.0 );
                    }
                    
                /* do we have to move this particle? */
                if ( ( delta[0] != 0 ) || ( delta[1] != 0 ) || ( delta[2] != 0 ) ) {
                    for ( k = 0 ; k < 3 ; k++ )
                        p->x[k] -= delta[k] * h[k];
                    c_dest = &( s->cells[ space_cellid( s ,
                        (c->loc[0] + delta[0] + s->cdim[0]) % s->cdim[0] , 
                        (c->loc[1] + delta[1] + s->cdim[1]) % s->cdim[1] , 
                        (c->loc[2] + delta[2] + s->cdim[2]) % s->cdim[2] ) ] );
                    
	                pthread_mutex_lock(&c_dest->cell_mutex);
                    cell_add_incomming( c_dest , p );
	                pthread_mutex_unlock(&c_dest->cell_mutex);
        
                    s->celllist[ p->id ] = c_dest;
                    c->count -= 1;
                    if ( pid < c->count ) {
                        c->parts[pid] = c->parts[c->count];
                        s->partlist[ c->parts[pid].id ] = &( c->parts[pid] );
                        }
                    }
                else
                    pid += 1;
                }
            }
            
        /* Welcome the new particles in each cell. */
        #pragma omp parallel for schedule(static)
        for ( cid = 0 ; cid < s->nr_marked ; cid++ )
            cell_welcome( &(s->cells[ s->cid_marked[cid] ]) , s->partlist );
            
        }
            
    /* Store the accumulated potential energy. */
    s->epot += epot;
        
    /* return quietly */
    return engine_err_ok;
    
    }


/**
 * @brief Run the engine for a single time step.
 *
 * @param e The #engine on which to run.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * This routine advances the timestep counter by one, prepares the #space
 * for a timestep, releases the #runner's associated with the #engine
 * and waits for them to finnish.
 *
 * Once all the #runner's are done, the particle velocities and positions
 * are updated and the particles are re-sorted in the #space.
 */
/* TODO: Should the velocities and positions really be updated here? */

int engine_step ( struct engine *e ) {

    ticks tic, tic_step = getticks();

    /* increase the time stepper */
    e->time += 1;
    
    /* prepare the space */
    tic = getticks();
    if ( space_prepare( &e->s ) != space_err_ok )
        return error(engine_err_space);
    e->timers[engine_timer_prepare] += getticks() - tic;

    /* Make sure the verlet lists are up to date. */
    if ( e->flags & engine_flag_verlet ) {
    
        /* Start the clock. */
        tic = getticks();
        
        /* Prepare the Verlet data. */
        if ( space_verlet_init( &(e->s) , !(e->flags & engine_flag_verlet_pairwise) ) != space_err_ok )
            return error(engine_err_space);
    
        /* Check particle movement and update cells if necessary. */
        if ( engine_verlet_update( e ) < 0 )
            return error(engine_err);
            
        /* Store the timing. */
        e->timers[engine_timer_prepare] += getticks() - tic;
            
        }
            
    /* Re-distribute the particles to the processors. */
    if ( e->flags & engine_flag_mpi ) {
        
        /* Start the clock. */
        tic = getticks();
    
        if ( e->flags & engine_flag_async ) {
            if ( engine_exchange_async( e ) != 0 )
                return error(engine_err);
            }
        else {
            if ( engine_exchange( e ) != 0 )
                return error(engine_err);
            }
            
        /* Store the timing. */
        e->timers[engine_timer_exchange1] += getticks() - tic;
            
        }
            
    /* Compute the non-bonded interactions. */
    tic = getticks();
    if ( engine_nonbond_eval( e ) < 0 )
        return error(engine_err);
    e->timers[engine_timer_nonbond] += getticks() - tic;
            
    /* Do bonded interactions. */
    tic = getticks();
    if ( engine_bonded_eval( e ) < 0 )
        return error(engine_err);
    e->timers[engine_timer_bonded] += getticks() - tic;

    /* update the particle velocities and positions. */
    tic = getticks();
    if ( engine_advance( e ) < 0 )
        return error(engine_err);
    e->timers[engine_timer_advance] += getticks() - tic;
            
    /* Shake the particle positions? */
    if ( e->nr_rigids > 0 ) {
    
        /* Sort the constraints. */
        tic = getticks();
        if ( engine_rigid_sort( e ) != 0 )
            return error(engine_err);
        e->timers[engine_timer_rigid] += getticks() - tic;
    
        if ( e->flags & engine_flag_mpi ) {
        
            /* Start the clock. */
            tic = getticks();
    
            if ( e->flags & engine_flag_async ) {
                if ( engine_exchange_async( e ) != 0 )
                    return error(engine_err);
                }
            else {
                if ( engine_exchange( e ) != 0 )
                    return error(engine_err);
                }
                
            /* Store the timing. */
            e->timers[engine_timer_exchange2] += getticks() - tic;
            
            }
            
        /* Resolve the constraints. */
        tic = getticks();
        if ( engine_rigid_eval( e ) != 0 )
            return error(engine_err);
        e->timers[engine_timer_rigid] += getticks() - tic;
    
        }
        
    /* Stop the clock. */
    e->timers[engine_timer_step] += getticks() - tic_step;
            
    /* return quietly */
    return engine_err_ok;
    
    }


/**
 * @brief Barrier routine to hold the @c runners back.
 *
 * @param e The #engine to wait on.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 *
 * After being initialized, and after every timestep, every #runner
 * calls this routine which blocks until all the runners have returned
 * and the #engine signals the next timestep.
 */

int engine_barrier ( struct engine *e ) {

    /* lock the barrier mutex */
	if (pthread_mutex_lock(&e->barrier_mutex) != 0)
		return error(engine_err_pthread);
	
    /* wait for the barrier to close */
	while (e->barrier_count < 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return error(engine_err_pthread);
	
    /* if i'm the last thread in, signal that the barrier is full */
	if (++e->barrier_count == e->nr_runners) {
		if (pthread_cond_signal(&e->done_cond) != 0)
			return error(engine_err_pthread);
		}

    /* wait for the barrier to re-open */
	while (e->barrier_count > 0)
		if (pthread_cond_wait(&e->barrier_cond,&e->barrier_mutex) != 0)
			return error(engine_err_pthread);
				
    /* if i'm the last thread out, signal to those waiting to get back in */
	if (++e->barrier_count == 0)
		if (pthread_cond_broadcast(&e->barrier_cond) != 0)
			return error(engine_err_pthread);
			
    /* free the barrier mutex */
	if (pthread_mutex_unlock(&e->barrier_mutex) != 0)
		return error(engine_err_pthread);
		
    /* all is well... */
	return engine_err_ok;
	
	}
	
	
/**
 * @brief Initialize an #engine with the given data and MPI enabled.
 *
 * @param e The #engine to initialize.
 * @param origin An array of three doubles containing the cartesian origin
 *      of the space.
 * @param dim An array of three doubles containing the size of the space.
 * @param L The minimum cell edge length, should be at least @c cutoff.
 * @param cutoff The maximum interaction cutoff to use.
 * @param period A bitmask describing the periodicity of the domain
 *      (see #space_periodic_full).
 * @param max_type The maximum number of particle types that will be used
 *      by this engine.
 * @param flags Bit-mask containing the flags for this engine.
 * @param comm The MPI comm to use.
 * @param rank The ID of this node.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

#ifdef HAVE_MPI
int engine_init_mpi ( struct engine *e , const double *origin , const double *dim , double L , double cutoff , unsigned int period , int max_type , unsigned int flags , MPI_Comm comm , int rank ) {

    /* Init the engine. */
    if ( engine_init( e , origin , dim , L , cutoff , period , max_type , flags | engine_flag_mpi ) < 0 )
        return error(engine_err);
     
    /* Store the MPI Comm and rank. */
    e->comm = comm;
    e->nodeID = rank;
    
    /* Bail. */
    return engine_err_ok;
    
    }
#endif


/**
 * @brief Initialize an #engine with the given data.
 *
 * @param e The #engine to initialize.
 * @param origin An array of three doubles containing the cartesian origin
 *      of the space.
 * @param dim An array of three doubles containing the size of the space.
 * @param L The minimum cell edge length, should be at least @c cutoff.
 * @param cutoff The maximum interaction cutoff to use.
 * @param period A bitmask describing the periodicity of the domain
 *      (see #space_periodic_full).
 * @param max_type The maximum number of particle types that will be used
 *      by this engine.
 * @param flags Bit-mask containing the flags for this engine.
 *
 * @return #engine_err_ok or < 0 on error (see #engine_err).
 */

int engine_init ( struct engine *e , const double *origin , const double *dim , double L , double cutoff , unsigned int period , int max_type , unsigned int flags ) {

    /* make sure the inputs are ok */
    if ( e == NULL || origin == NULL || dim == NULL )
        return error(engine_err_null);
        
    /* init the space with the given parameters */
    if ( space_init( &(e->s) , origin , dim , L , cutoff , period ) < 0 )
        return error(engine_err_space);
        
    /* Set some flag implications. */
    if ( flags & engine_flag_verlet_pairwise2 )
        flags |= engine_flag_verlet_pairwise;
    if ( flags & engine_flag_verlet_pairwise )
        flags |= engine_flag_verlet;
    if ( flags & engine_flag_verlet )
        flags |= engine_flag_tuples;
        
    /* Set the flags. */
    e->flags = flags;
    
    /* By default there is only one node. */
    e->nr_nodes = 1;
    
    /* Init the timers. */
    if ( engine_timers_reset( e ) < 0 )
        return error(engine_err);
    
    /* Init the runners to 0. */
    e->runners = NULL;
    e->nr_runners = 0;
    
    /* Init the bonds array. */
    e->bonds_size = 100;
    if ( ( e->bonds = (struct bond *)malloc( sizeof( struct bond ) * e->bonds_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_bonds = 0;
    
    /* Init the exclusions array. */
    e->exclusions_size = 100;
    if ( ( e->exclusions = (struct exclusion *)malloc( sizeof( struct exclusion ) * e->exclusions_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_exclusions = 0;
    
    /* Init the rigids array. */
    e->rigids_size = 100;
    if ( ( e->rigids = (struct rigid *)malloc( sizeof( struct rigid ) * e->rigids_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_rigids = 0;
    e->tol_rigid = 1e-6;
    e->nr_constr = 0;
    e->part2rigid = NULL;
    
    /* Init the angles array. */
    e->angles_size = 100;
    if ( ( e->angles = (struct angle *)malloc( sizeof( struct angle ) * e->angles_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_angles = 0;
    
    /* Init the dihedrals array. */
    e->dihedrals_size = 100;
    if ( ( e->dihedrals = (struct dihedral *)malloc( sizeof( struct dihedral ) * e->dihedrals_size ) ) == NULL )
        return error(engine_err_malloc);
    e->nr_dihedrals = 0;
    
    /* set the maximum nr of types */
    e->max_type = max_type;
    e->nr_types = 0;
    if ( ( e->types = (struct part_type *)malloc( sizeof(struct part_type) * max_type ) ) == NULL )
        return error(engine_err_malloc);
    
    /* allocate the interaction matrices */
    if ( (e->p = (struct potential **)malloc( sizeof(struct potential *) * max_type * max_type )) == NULL)
        return error(engine_err_malloc);
    bzero( e->p , sizeof(struct potential *) * max_type * max_type );
    if ( (e->p_bond = (struct potential **)malloc( sizeof(struct potential *) * max_type * max_type )) == NULL)
        return error(engine_err_malloc);
    bzero( e->p_bond , sizeof(struct potential *) * max_type * max_type );
    e->anglepots_size = 100;
    if ( (e->p_angle = (struct potential **)malloc( sizeof(struct potential *) * e->anglepots_size )) == NULL)
        return error(engine_err_malloc);
    bzero( e->p_angle , sizeof(struct potential *) * e->anglepots_size );
    e->nr_anglepots = 0;
    e->dihedralpots_size = 100;
    if ( (e->p_dihedral = (struct potential **)malloc( sizeof(struct potential *) * e->dihedralpots_size )) == NULL)
        return error(engine_err_malloc);
    bzero( e->p_dihedral , sizeof(struct potential *) * e->dihedralpots_size );
    e->nr_dihedralpots = 0;
        
    /* init the barrier variables */
    e->barrier_count = 0;
	if ( pthread_mutex_init( &e->barrier_mutex , NULL ) != 0 ||
		 pthread_cond_init( &e->barrier_cond , NULL ) != 0 ||
		 pthread_cond_init( &e->done_cond , NULL ) != 0)
		return error(engine_err_pthread);
        
    /* init the barrier */
    if (pthread_mutex_lock(&e->barrier_mutex) != 0)
        return error(engine_err_pthread);
    e->barrier_count = 0;
    
    /* Init the comm arrays. */
    e->send = NULL;
    e->recv = NULL;
        
    /* all is well... */
    return engine_err_ok;
    
    }
