/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2011 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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

/* Include some standard headers */
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "cycle.h"
#include "../config.h"

/* MPI headers. */
#include <mpi.h>

/* OpenMP headers. */
#include <omp.h>

/* Include mdcore. */
#include "mdcore.h"

/* Ticks Per Second. */
#ifndef CPU_TPS
    #define CPU_TPS 2.67e+9
#endif

/* Engine flags? */
#ifndef ENGINE_FLAGS
    #define ENGINE_FLAGS (engine_flag_tuples | engine_flag_parbonded)
#endif

/* Enumeration for the different timers */
enum {
    tid_nonbond = 0,
    tid_bonded,
    tid_advance,
    tid_shake,
    tid_exchange,
    tid_temp
    };


/* The main routine -- this is where it all happens. */

int main ( int argc , char *argv[] ) {


    /* Simulation constants. */
    double origin[3] = { -5.44306 , -5.44306 , -3.8879 };
    double dim[3] = { 10.88612 , 10.88612 , 7.7758 };
    int nr_mols = 129024, nr_parts = nr_mols*3;
    // double dim[3] = { 8.0 , 8.0 , 8.0 };
    // int nr_mols = 16128, nr_parts = nr_mols*3;
    double cutoff = 1.2;
    double Temp = 300.0;
    double pekin_max = 100.0;
    int pekin_max_time = 100;


    /* Local variables. */
    int res = 0, myrank;
    int step, i, j, k, cid;
    FPTYPE ee, eff;
    double temp, v[3];
    FILE *dump, *psf, *pdb, *xplor;
    char fname[100];
    double es[6], ekin, epot, vcom[3], vcom_x , vcom_y , vcom_z , mass_tot, w, v2;
    ticks tic, toc, tic_step, toc_step, timers[10];
    double itpms = 1000.0 / CPU_TPS;
    int nr_nodes = 1;
    int verbose = 0;
    double maxpekin, A, B, q;
    int maxpekin_id;
    
    
    /* mdcore stuff. */
    struct engine e;
    struct part *p;
    struct potential *pot;
    int typeOT, nr_runners = 1, nr_steps = 1000;
    char *excl[] = { "OT" , "HT" };
    
    
    /* Start the clock. */
    for ( k = 0 ; k < 10 ; k++ )
        timers[k] = 0;
    tic = getticks();
    
    
    /* Start by initializing MPI. */
    if ( ( res = MPI_Init( &argc , &argv ) ) != MPI_SUCCESS ) {
        printf( "main: call to MPI_Init failed with error %i.\n" , res );
        return -1;
        }
    if ( ( res = MPI_Comm_rank( MPI_COMM_WORLD , &myrank ) ) != MPI_SUCCESS ) {
        printf( "main: call to MPI_Comm_rank failed with error %i.\n" , res );
        return -1;
        }
    if ( ( res = MPI_Comm_size( MPI_COMM_WORLD , &nr_nodes ) != MPI_SUCCESS ) ) {
        printf("main[%i]: MPI_Comm_size failed with error %i.\n",myrank,res);
        errs_dump(stdout);
        return -1;
        }
    if ( myrank == 0 ) {
        printf( "main[%i]: MPI is up and running...\n" , myrank );
        fflush(stdout);
        }
    
    
    /* Initialize our own input parameters. */
    if ( argc > 3 )
        nr_runners = atoi( argv[3] );
    if ( argc > 4 )
        nr_steps = atoi( argv[4] );
        
    
    /* Initialize the engine. */
    printf( "main[%i]: initializing the engine...\n" , myrank ); fflush(stdout);
    if ( engine_init( &e , origin , dim , 1.25 , space_periodic_full , 100 , ENGINE_FLAGS | engine_flag_mpi | engine_flag_verlet_pairwise ) != 0 ) {
        printf( "main[%i]: engine_init failed with engine_err=%i.\n" , myrank , engine_err );
        errs_dump(stdout);
        return -1;
        }
    e.dt = 0.0025;
    e.time = 0;
    e.tol_rigid = 1.0e-6;
    e.s.cutoff = 1.20; e.s.cutoff2 = 1.2*1.2;
    e.nodeID = myrank;
    printf("main[%i]: engine initialized.\n",myrank);
    if ( myrank == 0 )
        printf( "main[%i]: space has %i pairs and %i tuples.\n" , myrank , e.s.nr_pairs , e.s.nr_tuples );
    if ( myrank == 0 )
        printf( "main[%i]: cell size is [ %e , %e , %e ] nm.\n" , myrank , e.s.h[0] , e.s.h[1] , e.s.h[2] );
    if ( myrank == 0 )
        printf( "main[%i]: space is [ %i , %i , %i ] cells.\n" , myrank , e.s.cdim[0] , e.s.cdim[1] , e.s.cdim[2] );
    fflush(stdout);
    
    
    /* Load the PSF/PDB files. */
    printf( "main[%i]: reading psf/pdb files....\n" , myrank ); fflush(stdout);
    if ( ( psf = fopen( argv[1] , "r" ) ) == NULL ) {
        printf("main[%i]: could not fopen the file \"%s\".\n",myrank,argv[1]);
        return -1;
        }
    if ( ( pdb = fopen( argv[2] , "r" ) ) == NULL ) {
        printf("main[%i]: could not fopen the file \"%s\".\n",myrank,argv[2]);
        return -1;
        }
    if ( engine_read_psf( &e , psf , pdb ) < 0 ) {
        printf("main[%i]: engine_read_psf failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
    fclose( psf ); fclose( pdb );
    printf( "main[%i]: read %i registered types.\n" , myrank , e.nr_types );
    printf( "main[%i]: read %i particles.\n" , myrank , e.s.nr_parts );
    printf( "main[%i]: read %i bonds.\n" , myrank , e.nr_bonds );
    printf( "main[%i]: read %i angles.\n" , myrank , e.nr_angles );
    printf( "main[%i]: read %i dihedrals.\n" , myrank , e.nr_dihedrals );
    /* for ( k = 0 ; k < e.nr_types ; k++ )
        printf( "         %2i: %s (%s), q=%f, m=%f\n" , k , e.types[k].name , e.types[k].name2 , e.types[k].charge , e.types[k].mass ); */
    
    
    /* Load the CHARMM parameter file. */
    printf( "main[%i]: reading parameter file....\n" , myrank ); fflush(stdout);
    if ( ( xplor = fopen( "apoa1.xplor" , "r" ) ) == NULL ) {
        printf("main[%i]: could not fopen the file \"apoa1.xplor\".\n",myrank);
        return -1;
        }
    if ( engine_read_xplor( &e , xplor , 0.0 , 1e-3 , 1 ) < 0 ) {
        printf("main[%i]: engine_read_xplor failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
    printf( "main[%i]: done reading parameters.\n" , myrank );
    printf( "main[%i]: generated %i constraints in %i groups.\n" , myrank , e.nr_constr , e.nr_rigids );
    fflush(stdout);
    fclose( xplor );
    
    /* Dump bond types. */
    /* for ( j = 0 ; j < e.nr_types ; j++ )
        for ( k = j ; k < e.nr_types ; k++ )
            if ( ( pot = e.p_bond[ j*e.max_type + k ] ) != NULL )
                printf( "main[%i]: got bond between types %s and %s with %i intervals.\n" ,
                    myrank , e.types[j].name2 , e.types[k].name2 , pot->n ); */
    
    /* Check for missing bonds. */
    for ( k = 0 ; k < e.nr_bonds ; k++ )
        if ( e.p_bond[ e.s.partlist[e.bonds[k].i]->type*e.max_type + e.s.partlist[e.bonds[k].j]->type ] == NULL )
            printf( "main[%i]: no potential specified for bond %i: %s %s.\n" ,
                myrank , k , e.types[e.s.partlist[e.bonds[k].i]->type].name ,
                e.types[e.s.partlist[e.bonds[k].j]->type].name );

    /* Check for missing angles. */
    for ( k = 0 ; k < e.nr_angles ; k++ )
        if ( e.angles[k].pid < 0 )
            printf( "main[%i]: no potential specified for angle %s %s %s.\n" ,
                myrank , e.types[e.s.partlist[e.angles[k].i]->type].name ,
                e.types[e.s.partlist[e.angles[k].j]->type].name ,
                e.types[e.s.partlist[e.angles[k].k]->type].name );
                
    /* Check for missing dihedrals. */
    for ( k = 0 ; k < e.nr_dihedrals ; k++ )
        if ( e.dihedrals[k].pid < 0 )
            printf( "main[%i]: no potential specified for dihedral %s %s %s %s.\n" ,
                myrank , e.types[e.s.partlist[e.dihedrals[k].i]->type].name ,
                e.types[e.s.partlist[e.dihedrals[k].j]->type].name ,
                e.types[e.s.partlist[e.dihedrals[k].k]->type].name ,
                e.types[e.s.partlist[e.dihedrals[k].l]->type].name );
                
    /* Dump potentials. */
    /* for ( j = 0 ; j < e.nr_types ; j++ )
        for ( k = j ; k < e.nr_types ; k++ )
            if ( ( pot = e.p[ j*e.max_type + k ] ) != NULL )
                printf( "main[%i]: got potential between types %s and %s with %i intervals.\n" ,
                    myrank , e.types[j].name2 , e.types[k].name2 , pot->n ); */
    
            
    /* Add exclusions. */
    for ( k = 0 ; k < e.nr_bonds ; k++ )
        if ( engine_exclusion_add( &e , e.bonds[k].i , e.bonds[k].j ) < 0 ) {
            printf("main[%i]: engine_exclusion_add failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
    for ( k = 0 ; k < e.nr_angles ; k++ )
        if ( engine_exclusion_add( &e , e.angles[k].i , e.angles[k].k ) < 0 ) {
            printf("main[%i]: engine_exclusion_add failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
    for ( k = 0 ; k < e.nr_dihedrals ; k++ )
        if ( engine_exclusion_add( &e , e.dihedrals[k].i , e.dihedrals[k].l ) < 0 ) {
            printf("main[%i]: engine_exclusion_add failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return -1;
            }
    for ( k = 0 ; k < e.nr_rigids ; k++ )
        for ( j = 0 ; j < e.rigids[k].nr_constr ; j++ )
            if ( engine_exclusion_add( &e , e.rigids[k].parts[e.rigids[k].constr[j].i] , e.rigids[k].parts[e.rigids[k].constr[j].j] ) < 0 ) {
                printf("main[%i]: engine_exclusion_add failed with engine_err=%i.\n",myrank,engine_err);
                errs_dump(stdout);
                return -1;
                }
    if ( engine_exclusion_shrink( &e ) < 0 ) {
        printf("main[%i]: engine_exclusion_shrink failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
    printf( "main[%i]: collected %i exclusions.\n" , myrank , e.nr_exclusions );
    
    /* Convert water angles to rigid constraints. */
    for ( typeOT = 0 ; typeOT < e.nr_types && strcmp( e.types[typeOT].name , "OT" ) != 0 ; typeOT++ );
    for ( k = 0 ; k < e.nr_angles ; k++ )
        if ( e.s.partlist[e.angles[k].j]->type == typeOT ) {
            if ( engine_rigid_add( &e , e.angles[k].i , e.angles[k].k , 0.15139 ) < 0 ) {
                printf("main[%i]: engine_rigid_add failed with engine_err=%i.\n",myrank,engine_err);
                errs_dump(stdout);
                return -1;
                }
            e.nr_angles -= 1;
            e.angles[k] = e.angles[e.nr_angles];
            k -= 1;
            }
            
    /* Correct the water vids. */
    for ( nr_mols = 0 , k = 0 ; k < e.s.nr_parts ; k++ )
        if ( e.s.partlist[k]->type == typeOT ) {
            nr_mols += 1;
            e.s.partlist[k]->vid = k;
            e.s.partlist[k+1]->vid = k;
            e.s.partlist[k+1]->vid = k;
            }
            
    /* Assign all particles a random initial velocity. */
    vcom[0] = 0.0; vcom[1] = 0.0; vcom[2] = 0.0; mass_tot = 0.0;
    for ( k = 0 ; k < e.s.nr_parts ; k++ ) {
        v[0] = ((double)rand()) / RAND_MAX - 0.5;
        v[1] = ((double)rand()) / RAND_MAX - 0.5;
        v[2] = ((double)rand()) / RAND_MAX - 0.5;
        temp = 2.3 * sqrt( 2.0 * e.types[e.s.partlist[k]->type].imass / ( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] ) );
        v[0] *= temp; v[1] *= temp; v[2] *= temp;
        e.s.partlist[k]->v[0] = v[0];
        e.s.partlist[k]->v[1] = v[1];
        e.s.partlist[k]->v[2] = v[2];
        mass_tot += e.types[e.s.partlist[k]->type].mass;
        vcom[0] += v[0] * e.types[e.s.partlist[k]->type].mass;
        vcom[1] += v[1] * e.types[e.s.partlist[k]->type].mass;
        vcom[2] += v[2] * e.types[e.s.partlist[k]->type].mass;
        }
    vcom[0] /= mass_tot; vcom[1] /= mass_tot; vcom[2] /= mass_tot;
    for ( k = 0 ; k < e.s.nr_parts ; k++ ) {
        e.s.partlist[k]->v[0] -= vcom[0];
        e.s.partlist[k]->v[1] -= vcom[1];
        e.s.partlist[k]->v[2] -= vcom[2];
        e.s.partlist[k]->vid = k;
        }
        
    /* Ignore angles for now. */
    // e.nr_bonds = 0;
    // e.nr_angles = 0;
    // e.nr_rigids = 0;
    // e.nr_dihedrals = 0;
    
    /* Dump a potential to make sure its ok... */
    /* pot = e.p[0];
    for ( k = 0 ; k < e.nr_types ; k++ ) {
        for ( j = k ; j < e.nr_types ; j++ )
            if ( e.p[ j*e.max_type + k ] != NULL && e.p[ j*e.max_type + k ]->n > pot->n )
                pot = e.p[ j*e.max_type + k ];
        if ( j < e.nr_types )
            break;
        }
    A = 4.184 * 0.0460 * pow(0.1*0.4000,12);
    B = 2 * 4.184 * 0.0460 * pow(0.1*0.4000,6);
    q = e.types[k].charge * e.types[j].charge;
    printf( "main: dumping potential for %s-%s (%i-%i, n=%i).\n" , e.types[k].name , e.types[j].name , k , j , pot->n );
    for ( i = 0 ; i <= 10000 ; i++ ) {
        temp = pot->a + (double)i/10000 * (pot->b - pot->a);
        potential_eval_r( pot , temp , &ee , &eff );
        printf("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n", temp , ee , eff , 
            potential_LJ126(temp,A,B) + q*potential_escale/temp, 
            potential_LJ126_p(temp,A,B) - q*potential_escale/(temp*temp) ,
            pot->alpha[0] + temp*(pot->alpha[1] + temp*(pot->alpha[2] + temp*pot->alpha[3])) );
        }
    return 0; */
    
    /* Split the engine over the processors. */
    if ( engine_split_bisect( &e , nr_nodes ) < 0 ) {
        printf("main[%i]: engine_split_bisect failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
    if ( engine_split( &e ) < 0 ) {
        printf("main[%i]: engine_split failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
    /* for ( k = 0 ; k < e.nr_nodes ; k++ ) {
        printf( "main[%i]: %i cells to send to node %i: [ " , myrank , e.send[k].count , k );
        for ( j = 0 ; j < e.send[k].count ; j++ )
            printf( "%i " , e.send[k].cellid[j] );
        printf( "]\n" );
        }
    for ( k = 0 ; k < e.nr_nodes ; k++ ) {
        printf( "main[%i]: %i cells to recv from node %i: [ " , myrank , e.recv[k].count , k );
        for ( j = 0 ; j < e.recv[k].count ; j++ )
            printf( "%i " , e.recv[k].cellid[j] );
        printf( "]\n" );
        } */
        
        
    /* Start the engine. */
    if ( engine_start( &e , nr_runners ) != 0 ) {
        printf("main[%i]: engine_start failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
        
    /* Set the number of OpenMP threads to the number of runners. */
    omp_set_num_threads( nr_runners );
        
        
    /* Give the system a quick shake before going anywhere. */
    if ( engine_rigid_eval( &e ) != 0 ) {
        printf("main: engine_rigid_eval failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return -1;
        }
    if ( engine_exchange( &e , MPI_COMM_WORLD ) != 0 ) {
        printf("main: engine_step failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return -1;
        }
        
            
    /* Timing. */    
    toc = getticks();
    if ( myrank == 0 ) {
        printf("main[%i]: setup took %.3f ms.\n",myrank,(double)(toc-tic) * itpms);
        printf("# step e_pot e_kin swaps stalls ms_tot ms_nonbond ms_bonded ms_advance ms_shake ms_xchg ms_temp\n");
        fflush(stdout);
        }
        

    /* Main time-stepping loop. */
    for ( step = 0 ; step < nr_steps ; step++ ) {
    
        /* Start the clock. */
        tic_step = getticks();
        
        /* Check for ghost parts/cells. */
        /* for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
            if ( e.s.cells[cid].flags & cell_flag_ghost ) {
                for ( k = 0 ; k < e.s.cells[cid].count ; k++ )
                    if ( !( e.s.cells[cid].parts[k].flags & part_flag_ghost ) )
                        printf( "main[%i]: non-ghost part %i in ghost cell %i.\n" , myrank , e.s.cells[cid].parts[k].id , cid );
                }
            else {
                for ( k = 0 ; k < e.s.cells[cid].count ; k++ )
                    if ( e.s.cells[cid].parts[k].flags & part_flag_ghost )
                        printf( "main[%i]: ghost part %i in non-ghost cell %i.\n" , myrank , e.s.cells[cid].parts[k].id , cid );
                } */
        

        /* Compute the non-bonded interactions. */
        tic = getticks();
        if ( engine_nonbond_eval( &e ) != 0 ) {
            printf("main: engine_nonbond_eval failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_nonbond] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: engine_step took %.3f ms.\n",myrank,(double)timers[tid_nonbond] * itpms); fflush(stdout);
            }
            
        /* Compute bonded interactions. */
        tic = getticks();
        if ( engine_bonded_eval( &e ) != 0 ) {
            printf("main: engine_bonded_eval failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_bonded] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: engine_step took %.3f ms.\n",myrank,(double)timers[tid_bonded] * itpms); fflush(stdout);
            }
             
        /* Advance the particles. */
        tic = getticks();
        if ( engine_advance( &e ) != 0 ) {
            printf("main: engine_advance failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_advance] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: engine_step took %.3f ms.\n",myrank,(double)timers[tid_advance] * itpms); fflush(stdout);
            }
            
            
        /* Dump the particles. */
        /* sprintf(fname,"parts_%03i.dump",myrank);
        dump = fopen(fname,"w");
        for ( k = 0 ; k < ppm_mpart ; k++ )
            fprintf(dump,"%e %e %e\n",xp[3*k+0],xp[3*k+1],xp[3*k+2]);
        fclose(dump); */
        
        
        /* Re-distribute the particles to the processors. */
        tic = getticks();
        if ( engine_exchange( &e , MPI_COMM_WORLD ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_exchange] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: engine_exchange took %.3f ms.\n",myrank,(double)timers[tid_exchange] * itpms); fflush(stdout);
            }
    

        /* Dump the particles. */
        /* sprintf(fname,"parts_%03i.dump",myrank);
        dump = fopen(fname,"w");
        for ( k = nr_parts ; k < ppm_mpart ; k++ )
            fprintf(dump,"%e %e %e\n",xp[3*k+0],xp[3*k+1],xp[3*k+2]);
        fclose(dump); */
        
        
        /* Update the Verlet lists. */
        tic = getticks();
        if ( engine_verlet_update( &e ) != 0 ) {
            printf("main: engine_verlet_update failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_advance] += getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: engine_verlet_update took %.3f ms.\n",myrank,(double)timers[tid_advance] * itpms); fflush(stdout);
            }
            

        /* Shake the particle positions. */
        tic = getticks();
        if ( engine_rigid_eval( &e ) != 0 ) {
            printf("main: engine_rigid_eval failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_shake] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: SHAKE took %.3f ms.\n",myrank,(double)timers[tid_shake] * itpms); fflush(stdout);
            }
        
        
        /* Re-distribute the particles to the processors. */
        tic = getticks();
        if ( engine_exchange( &e , MPI_COMM_WORLD ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_exchange] += getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: engine_exchange took %.3f ms.\n",myrank,(double)timers[tid_exchange] * itpms); fflush(stdout);
            }
    
    
        /* Dump the particles. */
        /* sprintf(fname,"parts_%03i.dump",myrank);
        dump = fopen(fname,"w");
        for ( k = 0 ; k < ppm_mpart ; k++ )
            fprintf(dump,"%e %e %e\n",xp[3*k+0],xp[3*k+1],xp[3*k+2]);
        fclose(dump); */
        
        
        /* Compute the system temperature. */
        tic = getticks();
        
        /* Get the total atomic kinetic energy, v_com and molecular kinetic energy. */
        ekin = 0.0; epot = e.s.epot;
        vcom_x = 0.0; vcom_y = 0.0; vcom_z = 0.0; maxpekin = 0.0;
        #pragma omp parallel for schedule(static), private(cid,p,j,k,v2), reduction(+:epot,ekin,vcom_x,vcom_y,vcom_z)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
            epot += e.s.cells[cid].epot;
            if ( !(e.s.cells[cid].flags & cell_flag_ghost ) )
                for ( j = 0 ; j < e.s.cells[cid].count ; j++ ) {
                    p = &( e.s.cells[cid].parts[j] );
                    v2 = p->v[0]*p->v[0] + p->v[1]*p->v[1] + p->v[2]*p->v[2];
                    /* if ( 0.5*v2*e.types[p->type].mass > maxpekin ) {
                        maxpekin = 0.5*v2*e.types[p->type].mass;
                        maxpekin_id = p->id;
                        } */
                    if ( e.time < pekin_max_time && 0.5*v2*e.types[p->type].mass > pekin_max ) {
                        /* printf( "main[%i]: particle %i (%s) was caught speeding (v2=%e).\n" ,
                            myrank , p->id , e.types[p->type].name , v2 ); */
                        p->v[0] = sqrt( 2 * pekin_max * e.types[p->type].imass ) / sqrt(v2);
                        p->v[1] = sqrt( 2 * pekin_max * e.types[p->type].imass ) / sqrt(v2);
                        p->v[2] = sqrt( 2 * pekin_max * e.types[p->type].imass ) / sqrt(v2);
                        }
                    vcom_x += p->v[0] * e.types[p->type].mass;
                    vcom_y += p->v[1] * e.types[p->type].mass;
                    vcom_z += p->v[2] * e.types[p->type].mass;
                    ekin += v2 * e.types[p->type].mass * 0.5;
                    }
            }
        vcom[0] = vcom_x; vcom[1] = vcom_y; vcom[2] = vcom_z;
        // printf( "main[%i]: max particle ekin is %e (%s:%i).\n" , myrank , maxpekin , e.types[e.s.partlist[maxpekin_id]->type].name , maxpekin_id );
            
        /* Collect vcom and ekin from all procs. */
        if ( e.nr_nodes > 1 ) {
            es[0] = epot; es[1] = ekin;
            es[2] = vcom[0]; es[3] = vcom[1]; es[4] = vcom[2];
            es[5] = mass_tot;
            if ( ( res = MPI_Allreduce( MPI_IN_PLACE , es , 6 , MPI_DOUBLE_PRECISION , MPI_SUM , MPI_COMM_WORLD ) ) != MPI_SUCCESS ) {
                printf( "main[%i]: call to MPI_Allreduce failed with error %i.\n" , myrank , res );
                return -1;
                }
            ekin = es[1]; epot = es[0];
            vcom[0] = es[2]; vcom[1] = es[3]; vcom[2] = es[4];
            mass_tot = es[5];
            }
        vcom[0] /= mass_tot; vcom[1] /= mass_tot; vcom[2] /= mass_tot;
            
        /* Compute the temperature. */
        // printf( "main[%i]: vcom is [ %e , %e , %e ].\n" , myrank , vcom[0] , vcom[1] , vcom[2] );
        temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * e.s.nr_parts );
        w = sqrt( 1.0 + 0.05 * ( Temp / temp - 1.0 ) );
        // printf( "main[%i]: ekin=%e, temp=%e, w=%e, nr_parts=%i.\n" , myrank , ekin , temp , w , e.s.nr_parts );
        // printf("main[%i]: vcom_tot is [ %e , %e , %e ].\n",myrank,vcom_tot[0],vcom_tot[1],vcom_tot[2]); fflush(stdout);
            
        if ( step < 5000 ) {
        
            /* Scale the particle velocities. */
            #pragma omp parallel for schedule(static), private(cid,j,p,k)
            for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
                if ( !(e.s.cells[cid].flags & cell_flag_ghost ) )
                    for ( j = 0 ; j < e.s.cells[cid].count ; j++ ) {
                        p = &( e.s.cells[cid].parts[j] );
                        for ( k = 0 ; k < 3 ; k++ ) {
                            p->v[k] -= vcom[k];
                            p->v[k] *= w;
                            }
                        }
                    
            }
        timers[tid_temp] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: thermostat took %.3f ms.\n",myrank,(double)timers[tid_temp] * itpms); fflush(stdout);
            }
                        
        
        /* Drop a line. */
        toc_step = getticks();
        if ( myrank == 0 ) {
            /* printf("%i %e %e %e %i %i %.3f ms\n",
                e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(double)(toc_step-tic_step) * itpms); fflush(stdout); */
            printf("%i %e %e %e %i %i %.3f %.3f %.3f %.3f %.3f %.3f %.3f ms\n",
                e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(toc_step-tic_step) * itpms,
                timers[tid_nonbond]*itpms, timers[tid_bonded]*itpms, timers[tid_advance]*itpms, timers[tid_shake]*itpms, timers[tid_exchange]*itpms, timers[tid_temp]*itpms ); fflush(stdout);
            }
        
        
        if ( myrank == 0 && e.time % 100 == 0 ) {
            sprintf( fname , "apoa1_%08i.pdb" , e.time ); pdb = fopen( fname , "w" );
            if ( engine_dump_PSF( &e , NULL , pdb , excl , 2 ) < 0 ) {
                printf("main: engine_dump_PSF failed with engine_err=%i.\n",engine_err);
                errs_dump(stdout);
                return 1;
                }
            fclose(pdb);
            }
    
        } /* main loop. */
        
    
    /* Exit gracefuly. */
    if ( ( res = MPI_Finalize() ) != MPI_SUCCESS ) {
        printf( "main[%i]: call to MPI_Finalize failed with error %i.\n" , myrank , res );
        return -1;
        }
    fflush(stdout);
    printf( "main[%i]: exiting.\n" , myrank );
    return 0;

    }