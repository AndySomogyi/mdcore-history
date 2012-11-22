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
#include "../config.h"
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <string.h>
#include <pthread.h>
#include <time.h>
#include "cycle.h"

/* MPI headers. */
#ifdef WITH_MPI
    #include <mpi.h>
#endif

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
    #define ENGINE_FLAGS (engine_flag_parbonded)
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
    double origin[3] = { 0.0 , 0.0 , 0.0 };
    double dim[3] = { 6.223 , 6.223 , 6.223 };
    int nr_mols = 129024, nr_parts = nr_mols*3;
    // double dim[3] = { 8.0 , 8.0 , 8.0 };
    // int nr_mols = 16128, nr_parts = nr_mols*3;
    double cutoff = 0.9;
    double Temp = 300.0;
    double pekin_max = 100.0;
    int pekin_max_time = 0;


    /* Local variables. */
    int res = 0, myrank = 0, prov;
    int step, incr, i, j, k, cid, w_min, w_max;
    FPTYPE ee, eff;
    double temp, v[3];
    FILE *dump, *psf, *pdb, *cpf;
    char fname[100];
    double es[6], ekin, ekin_local, epot, vcom[3], vcom_x , vcom_y , vcom_z , mass_tot, w, v2;
    ticks tic, toc, tic_step, toc_step, timers[10];
    double itpms = 1000.0 / CPU_TPS;
    int nr_nodes = 1;
    int verbose = 0;
    double A, B, q;
    int maxpekin_id;
    
    
    /* mdcore stuff. */
    struct engine e;
    struct part *p;
    struct potential *pot;
    struct cell *c;
    int typeOT, nr_runners = 1, nr_steps = 1000;
    char *excl[] = { "OT" , "HT" };
    double L[] = { cutoff , cutoff , cutoff };
    
    
    /* Start the clock. */
    for ( k = 0 ; k < 10 ; k++ )
        timers[k] = 0;
    tic = getticks();
    
    
#ifdef WITH_MPI
    /* Start by initializing MPI. */
    if ( ( res = MPI_Init_thread( &argc , &argv , MPI_THREAD_MULTIPLE , &prov ) ) != MPI_SUCCESS ) {
        printf( "main: call to MPI_Init failed with error %i.\n" , res );
        abort();
        }
    if ( prov != MPI_THREAD_MULTIPLE ) {
        printf( "main: MPI does not provide the level of threading required (MPI_THREAD_MULTIPLE).\n" );
        abort();
        }
    if ( ( res = MPI_Comm_size( MPI_COMM_WORLD , &nr_nodes ) != MPI_SUCCESS ) ) {
        printf("main[%i]: MPI_Comm_size failed with error %i.\n",myrank,res);
        errs_dump(stdout);
        abort();
        }
    if ( ( res = MPI_Comm_rank( MPI_COMM_WORLD , &myrank ) ) != MPI_SUCCESS ) {
        printf( "main: call to MPI_Comm_rank failed with error %i.\n" , res );
        abort();
        }
    if ( myrank == 0 ) {
        printf( "main[%i]: MPI is up and running...\n" , myrank );
        fflush(stdout);
        }
#endif
    
    
    /* Initialize our own input parameters. */
    if ( argc > 3 )
        nr_runners = atoi( argv[3] );
    if ( argc > 4 )
        nr_steps = atoi( argv[4] );
        
    
    /* Initialize the engine. */
    printf( "main[%i]: initializing the engine...\n" , myrank ); fflush(stdout);
#ifdef WITH_MPI
    if ( engine_init_mpi( &e , origin , dim , L , cutoff , space_periodic_full , 100 , ENGINE_FLAGS | engine_flag_async , MPI_COMM_WORLD , myrank ) != 0 ) {
#else
    if ( engine_init( &e , origin , dim , L , cutoff , space_periodic_full , 100 , ENGINE_FLAGS | engine_flag_affinity ) != 0 ) {
#endif
        printf( "main[%i]: engine_init failed with engine_err=%i.\n" , myrank , engine_err );
        errs_dump(stdout);
        abort();
        }
    e.dt = 0.0025;
    e.time = 0;
    e.tol_rigid = 1.0e-5;
    e.nodeID = myrank;
    printf("main[%i]: engine initialized.\n",myrank);
    if ( myrank == 0 )
        printf( "main[%i]: space has %i pairs and %i tuples.\n" , myrank , e.s.nr_pairs , e.s.nr_tuples );
    if ( myrank == 0 )
        printf( "main[%i]: cell size is [ %e , %e , %e ] nm.\n" , myrank , e.s.h[0] , e.s.h[1] , e.s.h[2] );
    if ( myrank == 0 )
        printf( "main[%i]: space is [ %i , %i , %i ] cells.\n" , myrank , e.s.cdim[0] , e.s.cdim[1] , e.s.cdim[2] );
    fflush(stdout);
    
    #ifdef WITH_CUDA
        if ( engine_cuda_setdevice( 0 ) != 0 ) {
            printf( "main[%i]: engine_cuda_setdevice failed with engine_err=%i.\n" , myrank , engine_err );
            errs_dump(stdout);
            abort();
            }
    #endif
    
    
    /* Load the PSF/PDB files. */
    printf( "main[%i]: reading psf/pdb files....\n" , myrank ); fflush(stdout);
    if ( ( psf = fopen( argv[1] , "r" ) ) == NULL ) {
        printf("main[%i]: could not fopen the file \"%s\".\n",myrank,argv[1]);
        abort();
        }
    if ( ( pdb = fopen( argv[2] , "r" ) ) == NULL ) {
        printf("main[%i]: could not fopen the file \"%s\".\n",myrank,argv[2]);
        abort();
        }
    if ( engine_read_psf( &e , psf , pdb ) < 0 ) {
        printf("main[%i]: engine_read_psf failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        abort();
        }
    fclose( psf ); fclose( pdb );
    
    
    /* Load the CHARMM parameter file. */
    printf( "main[%i]: reading parameter file....\n" , myrank ); fflush(stdout);
    if ( ( cpf = fopen( "par_all22_prot.inp" , "r" ) ) == NULL ) {
        printf("main[%i]: could not fopen the file \"par_all22_prot.inp\".\n",myrank);
        abort();
        }
    if ( engine_read_cpf( &e , cpf , 3.0 , 3.3e-3 , 1 ) < 0 ) {
        printf("main[%i]: engine_read_cpf failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        abort();
        }
    printf( "main[%i]: done reading parameters.\n" , myrank ); fflush(stdout);
    fclose( cpf );
    
    
    /* Correct the water vids. */
    for ( typeOT = 0 ; typeOT < e.nr_types && strcmp( e.types[typeOT].name , "OT" ) != 0 ; typeOT++ );
    for ( nr_mols = 0 , k = 0 ; k < e.s.nr_parts ; k++ )
        if ( e.s.partlist[k]->type == typeOT ) {
            nr_mols += 1;
            e.s.partlist[k]->vid = k;
            e.s.partlist[k+1]->vid = k;
            e.s.partlist[k+1]->vid = k;
            }
            
    /* Remove water angles. */
    for ( k = 0 ; k < e.nr_angles ; k++ )
        if ( e.s.partlist[e.angles[k].j]->type == typeOT ) {
            e.nr_angles -= 1;
            e.angles[k] = e.angles[e.nr_angles];
            k -= 1;
            }
            
            
    /* Print some stats. */
    if ( myrank == 0 ) {
        printf( "main[%i]: read %i registered types.\n" , myrank , e.nr_types );
        /* for ( k = 0 ; k < e.nr_types ; k++ )
            printf( "         %2i: %s (%s), q=%f, m=%f\n" , k , e.types[k].name , e.types[k].name2 , e.types[k].charge , e.types[k].mass ); */
        printf( "main[%i]: read %i particles.\n" , myrank , e.s.nr_parts );
        printf( "main[%i]: read %i bonds.\n" , myrank , e.nr_bonds );
        printf( "main[%i]: read %i angles.\n" , myrank , e.nr_angles );
        printf( "main[%i]: read %i dihedrals.\n" , myrank , e.nr_dihedrals );
        /* for ( k = 0 ; k < e.nr_types ; k++ )
            printf( "         %2i: %s (%s), q=%f, m=%f, eps=%f, rmin=%f\n" , k , e.types[k].name , e.types[k].name2 , e.types[k].charge , e.types[k].mass , e.types[k].eps , e.types[k].rmin ); */
        printf( "main[%i]: generated %i constraints in %i groups.\n" , myrank , e.nr_constr , e.nr_rigids );
        fflush(stdout);
        }
        
    
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
            abort();
            }
    for ( k = 0 ; k < e.nr_angles ; k++ )
        if ( engine_exclusion_add( &e , e.angles[k].i , e.angles[k].k ) < 0 ) {
            printf("main[%i]: engine_exclusion_add failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            abort();
            }
    for ( k = 0 ; k < e.nr_dihedrals ; k++ )
        if ( engine_exclusion_add( &e , e.dihedrals[k].i , e.dihedrals[k].l ) < 0 ) {
            printf("main[%i]: engine_exclusion_add failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            abort();
            }
    for ( k = 0 ; k < e.nr_rigids ; k++ )
        for ( j = 0 ; j < e.rigids[k].nr_constr ; j++ )
            if ( engine_exclusion_add( &e , e.rigids[k].parts[e.rigids[k].constr[j].i] , e.rigids[k].parts[e.rigids[k].constr[j].j] ) < 0 ) {
                printf("main[%i]: engine_exclusion_add failed with engine_err=%i.\n",myrank,engine_err);
                errs_dump(stdout);
                abort();
                }
    if ( engine_exclusion_shrink( &e ) < 0 ) {
        printf("main[%i]: engine_exclusion_shrink failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        abort();
        }
    printf( "main[%i]: generated %i exclusions.\n" , myrank , e.nr_exclusions ); fflush(stdout);
    
    
    /* Make the bonded sets. */
    if ( e.flags & engine_flag_sets ) {
        printf( "main[%i]: computing bonded sets...\n" , myrank ); fflush(stdout);
        if ( engine_bonded_sets( &e , 10*nr_runners ) < 0 ) {
            printf("main[%i]: engine_bonded_sets failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            abort();
            }
        w_min = w_max = e.sets[0].weight;
        for ( k = 1 ; k < e.nr_sets ; k++ )
            if ( e.sets[k].weight > w_max )
                w_max = e.sets[k].weight;
            else if ( e.sets[k].weight < w_min )
                w_min = e.sets[k].weight;
        printf( "main[%i]: have %i bonded sets, weights in [%i,%i].\n" , myrank , e.nr_sets , w_min , w_max ); fflush(stdout);
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
        }
        
    /* Ignore angles for now. */
    // e.nr_bonds = 0;
    // e.nr_angles = 0;
    // e.nr_rigids = 0;
    // e.nr_dihedrals = 0;
    // e.nr_exclusions = 0;
    
    /* Dump a potential to make sure its ok... */
    /* pot = e.p[ 26*e.max_type + 26 ];
    for ( k = 0 ; k < e.nr_types ; k++ ) {
        for ( j = k ; j < e.nr_types ; j++ )
            if ( e.p[ j*e.max_type + k ] != NULL && ( pot == NULL || pot->n < e.p[ j*e.max_type + k ]->n ) )
                pot = e.p[ j*e.max_type + k ];
        }
    k = 26; j = 26;
    A = 4.184 * 0.046000 * pow(2*0.0224500,12);
    B = 2 * 4.184 * 0.046000 * pow(2*0.0224500,6);
    q = e.types[k].charge * e.types[j].charge;
    printf( "main: dumping potential for %s-%s (%i-%i, n=%i).\n" , e.types[k].name , e.types[j].name , k , j , pot->n );
    for ( i = 0 ; i <= 10000 ; i++ ) {
        temp = pot->a + (double)i/10000 * (pot->b - pot->a);
        potential_eval_r( pot , temp , &ee , &eff );
        printf("%23.16e %23.16e %23.16e %23.16e %23.16e %23.16e\n", temp , ee , eff , 
            potential_LJ126(temp,A,B) + q*potential_Ewald(temp,3.0) , 
            potential_LJ126_p(temp,A,B) + q*potential_Ewald_p(temp,3.0) ,
            pot->alpha[0] + temp*(pot->alpha[1] + temp*(pot->alpha[2] + temp*pot->alpha[3])) );
        }
    return 0; */
    
    /* Split the engine over the processors. */
    if ( engine_split_bisect( &e , nr_nodes ) < 0 ) {
        printf("main[%i]: engine_split_bisect failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        abort();
        }
    if ( engine_split( &e ) < 0 ) {
        printf("main[%i]: engine_split failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        abort();
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
        
        
    /* Just do a single cell pair. */
    /* for ( k = 0 ; k < e.s.nr_pairs ; k++ )
        if ( e.s.pairs[k].i != e.s.pairs[k].j )
            e.s.pairs[k--] = e.s.pairs[ --e.s.nr_pairs ]; */
    // e.s.nr_pairs = 1;
    // e.s.cells[ e.s.pairs[0].i ].count = 32; e.s.cells[ e.s.pairs[0].j ].count = 32;
    /* for ( k = 0 ; k < e.s.nr_cells ; k++ )
        if ( k != e.s.pairs[0].i && k != e.s.pairs[0].j )
            cell_flush( &e.s.cells[k] , e.s.partlist , e.s.celllist ); */
    /* printf( "main[%i]: restricting myself to the pair [%i,%i].\n" , myrank ,
        e.s.pairs[0].i , e.s.pairs[0].j ); */
        
        
    /* Start the engine. */
    if ( engine_start( &e , nr_runners , nr_runners ) != 0 ) {
        printf("main[%i]: engine_start failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        abort();
        }
        
    /* Set the number of OpenMP threads to the number of runners. */
    // omp_set_num_threads( nr_runners );
        
        
    /* Give the system a quick shake before going anywhere. */
    if ( engine_rigid_unsort( &e ) != 0 ) {
        printf("main: engine_rigid_unsort failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        abort();
        }
    if ( engine_rigid_sort( &e ) != 0 ) {
        printf("main: engine_rigid_sort failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        abort();
        }
    if ( engine_rigid_eval( &e ) != 0 ) {
        printf("main: engine_rigid_eval failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        abort();
        }
#ifdef WITH_MPI
    if ( engine_exchange( &e ) != 0 ) {
        printf("main: engine_step failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        abort();
        }
#endif
        
    /* Dump the engine flags. */
    if ( myrank == 0 ) {
        printf( "main[%i]: engine flags:" , myrank );
        if ( e.flags & engine_flag_tuples ) printf( " engine_flag_tuples" );
        if ( e.flags & engine_flag_static ) printf( " engine_flag_static" );
        if ( e.flags & engine_flag_localparts ) printf( " engine_flag_localparts" );
        if ( e.flags & engine_flag_cuda ) printf( " engine_flag_cuda" );
        if ( e.flags & engine_flag_explepot ) printf( " engine_flag_explepot" );
        if ( e.flags & engine_flag_verlet ) printf( " engine_flag_verlet" );
        if ( e.flags & engine_flag_verlet_pairwise ) printf( " engine_flag_verlet_pairwise" );
        if ( e.flags & engine_flag_affinity ) printf( " engine_flag_affinity" );
        if ( e.flags & engine_flag_prefetch ) printf( " engine_flag_prefetch" );
        if ( e.flags & engine_flag_verlet_pseudo ) printf( " engine_flag_verlet_pseudo" );
        if ( e.flags & engine_flag_partlist ) printf( " engine_flag_partlist" );
        if ( e.flags & engine_flag_unsorted ) printf( " engine_flag_unsorted" );
        if ( e.flags & engine_flag_mpi ) printf( " engine_flag_mpi" );
        if ( e.flags & engine_flag_parbonded ) printf( " engine_flag_parbonded" );
        if ( e.flags & engine_flag_async ) printf( " engine_flag_async" );
        if ( e.flags & engine_flag_sets ) printf( " engine_flag_sets" );
        printf( "\n" );
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
        
        /* Compute a step. */
        if ( engine_step( &e ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            abort();
            }
            
                    
        /* Compute the system temperature. */
        tic = getticks();
        
        /* Get the total atomic kinetic energy, v_com and molecular kinetic energy. */
        ekin = 0.0; epot = e.s.epot;
        #pragma omp parallel private(c,p,j,k,v2,ekin_local,vcom_x,vcom_y,vcom_z)
        {
            vcom_x = 0.0; vcom_y = 0.0; vcom_z = 0.0; ekin_local = 0.0; 
            incr = omp_get_num_threads();
            for ( k = omp_get_thread_num() ; k < e.s.nr_real ; k += incr ) {
                c = &e.s.cells[ e.s.cid_real[k] ];
                for ( j = 0 ; j < c->count ; j++ ) {
                    p = &( c->parts[j] );
                    v2 = p->v[0]*p->v[0] + p->v[1]*p->v[1] + p->v[2]*p->v[2];
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
                    ekin_local += v2 * e.types[p->type].mass * 0.5;
                    }
                }
                
            #pragma omp critical
            {
                vcom[0] += vcom_x; vcom[1] += vcom_y; vcom[2] += vcom_z;
                ekin += ekin_local;
                }
            }
        // printf( "main[%i]: max particle ekin is %e (%s:%i).\n" , myrank , maxpekin , e.types[e.s.partlist[maxpekin_id]->type].name , maxpekin_id );
            
#ifdef WITH_MPI
        /* Collect vcom and ekin from all procs. */
        if ( e.nr_nodes > 1 ) {
            es[0] = epot; es[1] = ekin;
            es[2] = vcom[0]; es[3] = vcom[1]; es[4] = vcom[2];
            es[5] = mass_tot;
            if ( ( res = MPI_Allreduce( MPI_IN_PLACE , es , 6 , MPI_DOUBLE , MPI_SUM , MPI_COMM_WORLD ) ) != MPI_SUCCESS ) {
                printf( "main[%i]: call to MPI_Allreduce failed with error %i.\n" , myrank , res );
                abort();
                }
            ekin = es[1]; epot = es[0];
            vcom[0] = es[2]; vcom[1] = es[3]; vcom[2] = es[4];
            mass_tot = es[5];
            }
#endif
        vcom[0] /= mass_tot; vcom[1] /= mass_tot; vcom[2] /= mass_tot;
            
        /* Compute the temperature. */
        // printf( "main[%i]: vcom is [ %e , %e , %e ].\n" , myrank , vcom[0] , vcom[1] , vcom[2] );
        temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * e.s.nr_parts );
        w = sqrt( 1.0 + 0.05 * ( Temp / temp - 1.0 ) );
        // printf( "main[%i]: ekin=%e, temp=%e, w=%e, nr_parts=%i.\n" , myrank , ekin , temp , w , e.s.nr_parts );
        // printf("main[%i]: vcom_tot is [ %e , %e , %e ].\n",myrank,vcom_tot[0],vcom_tot[1],vcom_tot[2]); fflush(stdout);
            
        if ( step < 5000 ) {
        
            /* Scale the particle velocities. */
            #pragma omp parallel for schedule(static), private(cid,i,j,p,k)
            for ( i = 0 ; i < e.s.nr_real ; i++ ) {
                cid = e.s.cid_real[i];
                for ( j = 0 ; j < e.s.cells[cid].count ; j++ ) {
                    p = &( e.s.cells[cid].parts[j] );
                    for ( k = 0 ; k < 3 ; k++ ) {
                        p->v[k] -= vcom[k];
                        p->v[k] *= w;
                        }
                    }
                }
                    
            }
        timers[tid_temp] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: thermostat took %.3f ms.\n",myrank,(double)timers[tid_temp] * itpms); fflush(stdout);
            }
            
#ifdef WITH_MPI
        /* Agregate the engine timers. */
        if ( ( res = MPI_Allreduce( MPI_IN_PLACE , e.timers , engine_timer_last , MPI_UNSIGNED_LONG , MPI_SUM , MPI_COMM_WORLD ) ) != MPI_SUCCESS ) {
            printf( "main[%i]: call to MPI_Allreduce failed with error %i.\n" , myrank , res );
            abort();
            }
        for ( k = 0 ; k < engine_timer_last ; k++ )
            e.timers[k] /= e.nr_nodes;
#endif
        
        /* Drop a line. */
        toc_step = getticks();
        if ( myrank == 0 ) {
            /* printf("%i %e %e %e %i %i %.3f ms\n",
                e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(double)(toc_step-tic_step) * itpms); fflush(stdout); */
            printf("%i %e %e %e %i %i %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f ms\n",
                e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(toc_step-tic_step) * itpms,
                e.timers[engine_timer_nonbond]*itpms, e.timers[engine_timer_bonded]*itpms,
                e.timers[engine_timer_advance]*itpms, e.timers[engine_timer_rigid]*itpms,
                (e.timers[engine_timer_exchange1]+e.timers[engine_timer_exchange2])*itpms,
                e.timers[engine_timer_cuda_load]*itpms, e.timers[engine_timer_cuda_dopairs]*itpms, e.timers[engine_timer_cuda_unload]*itpms, 
                timers[tid_temp]*itpms ); fflush(stdout);
            /* printf("%i %e %e %e %i %i %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f %.3f ms\n",
                e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls, e.timers[engine_timer_step] * itpms,
                e.timers[engine_timer_nonbond]*itpms, e.timers[engine_timer_bonded]*itpms,
                e.timers[engine_timer_bonds]*itpms, e.timers[engine_timer_angles]*itpms, e.timers[engine_timer_dihedrals]*itpms, e.timers[engine_timer_exclusions]*itpms, 
                e.timers[engine_timer_advance]*itpms, e.timers[engine_timer_rigid]*itpms,
                (e.timers[engine_timer_exchange1]+e.timers[engine_timer_exchange2])*itpms, timers[tid_temp]*itpms ); fflush(stdout); */
            }
        /* printf( "main[%i]: queue lengths are [ %i " , myrank , e.queues[0].count );
        for ( i = 1 ; i < e.nr_queues ; i++ )
            printf( "%i ", e.queues[i].count );
        printf( "]\n" ); */
        
        /* Re-set the timers. */
        if ( engine_timers_reset( &e ) < 0 ) {
            printf("main: engine_timers_reset failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            abort();
            }
        
        if ( myrank == 0 && e.time % 100 == 0 ) {
            sprintf( fname , "jac_%08i.pdb" , e.time ); pdb = fopen( fname , "w" );
            if ( engine_dump_PSF( &e , NULL , pdb , excl , 2 ) < 0 ) {
                printf("main: engine_dump_PSF failed with engine_err=%i.\n",engine_err);
                errs_dump(stdout);
                abort();
                }
            fclose(pdb);
            }
    
        } /* main loop. */
        
    
    /* Exit gracefuly. */
    if ( engine_finalize( &e ) < 0 ) {
        printf("main: engine_finalize failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        abort();
        }
    #ifdef WITH_MPI
        if ( ( res = MPI_Finalize() ) != MPI_SUCCESS ) {
            printf( "main[%i]: call to MPI_Finalize failed with error %i.\n" , myrank , res );
            abort();
            }
    #endif
    fflush(stdout);
    printf( "main[%i]: exiting.\n" , myrank );
    return 0;

    }
