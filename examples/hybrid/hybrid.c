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

/* Include mdcore. */
#include "mdcore.h"

/* Ticks Per Second. */
#ifndef CPU_TPS
    #define CPU_TPS 2.67e+9
#endif

/* Engine flags? */
#ifndef ENGINE_FLAGS
    #define ENGINE_FLAGS engine_flag_tuples
#endif

/* Enumeration for the different timers */
enum {
    tid_step = 0,
    tid_shake,
    tid_exchange,
    tid_temp
    };


/* The main routine -- this is where it all happens. */

int main ( int argc , char *argv[] ) {


    /* Simulation constants. */
    double origin[3] = { 0.0 , 0.0 , 0.0 };
    // double dim[3] = { 16.0 , 16.0 , 16.0 };
    // int nr_mols = 129024, nr_parts = nr_mols*3;
    double dim[3] = { 8.0 , 8.0 , 8.0 };
    int nr_mols = 16128, nr_parts = nr_mols*3;
    double cutoff = 1.0;


    /* Local variables. */
    int res = 0, myrank;
    double *xp = NULL, *vp = NULL, x[3], v[3];
    int *pid = NULL, *vid = NULL, *ptype = NULL;
    int step, i, j, k, nx, ny, nz, id, cid;
    double hx, hy, hz, temp;
    double vtot[3] = { 0.0 , 0.0 , 0.0 };
    FILE *dump;
    char fname[100];
    double old_O[3], old_H1[3], old_H2[3], new_O[3], new_H1[3], new_H2[3];
    double v_OH1[3], v_OH2[3], v_HH[3], vp_O[3], vp_H1[3], vp_H2[3];
    double d_OH1, d_OH2, d_HH, lambda;
    double vcom_tot[6], vcom_tot_x, vcom_tot_y, vcom_tot_z, ekin, epot, vcom[3], w, v2;
    ticks tic, toc, tic_step, toc_step, timers[10];
    double itpms = 1000.0 / CPU_TPS;
    struct part *p_O, *p_H1, *p_H2, *p;
    struct cell *c_O, *c_H1, *c_H2;
    int nr_nodes = 1;
    int verbose = 0;
    
    
    /* mdcore stuff. */
    struct engine e;
    struct potential *pot_OO, *pot_OH, *pot_HH;
    int nr_runners = 1, nr_steps = 1000;
    
    
    /* Start the clock. */
    for ( k = 0 ; k < 4 ; k++ )
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
    if ( myrank == 0 ) {
        printf( "main[%i]: MPI is up and running...\n" , myrank );
        fflush(stdout);
        }
    
    
    /* Initialize our own input parameters. */
    if ( argc > 1 )
        nr_runners = atoi( argv[1] );
    if ( argc > 2 )
        nr_steps = atoi( argv[2] );
        
    
    /* Generate the particles for this simulation. */
    srand(6178);
    if ( ( xp = (double *)malloc( sizeof(double) * nr_parts * 3 ) ) == NULL ||
         ( vp = (double *)malloc( sizeof(double) * nr_parts * 3 ) ) == NULL ||
         ( pid = (int *)malloc( sizeof(int) * nr_parts ) ) == NULL ||
         ( vid = (int *)malloc( sizeof(int) * nr_parts ) ) == NULL ) {
         printf( "main[%i]: allocation of particle data failed!\n" , myrank );
         return -1;
         }
    printf("main[%i]: initializing particles... \n" , myrank); fflush(stdout);
    nx = ceil( pow( nr_mols , 1.0/3 ) ); hx = dim[0] / nx;
    ny = ceil( sqrt( ((double)nr_mols) / nx ) ); hy = dim[1] / ny;
    nz = ceil( ((double)nr_mols) / nx / ny ); hz = dim[2] / nz;
    for ( i = 0 ; i < nx ; i++ ) {
        x[0] = 0.05 + i * hx;
        for ( j = 0 ; j < ny ; j++ ) {
            x[1] = 0.05 + j * hy;
            for ( k = 0 ; k < nz && k + nz * ( j + ny * i ) < nr_mols ; k++ ) {
                id = 3 * (k + nz * ( j + ny * i ));
                x[2] = 0.05 + k * hz;
                v[0] = ((double)rand()) / RAND_MAX - 0.5;
                v[1] = ((double)rand()) / RAND_MAX - 0.5;
                v[2] = ((double)rand()) / RAND_MAX - 0.5;
                temp = 0.675 / sqrt( v[0]*v[0] + v[1]*v[1] + v[2]*v[2] );
                v[0] *= temp; v[1] *= temp; v[2] *= temp;
                vtot[0] += v[0]; vtot[1] += v[1]; vtot[2] += v[2];
                /* Add oxygen. */
                xp[ 3*id + 0 ] = x[0]; vp[ 3*id + 0 ] = v[0];
                xp[ 3*id + 1 ] = x[1]; vp[ 3*id + 1 ] = v[1];
                xp[ 3*id + 2 ] = x[2]; vp[ 3*id + 2 ] = v[2];
                pid[ id ] = id;
                vid[ id ] = k + nz * ( j + ny * i );
                x[0] += 0.1;
                /* Add first hydrogen atom. */
                id += 1;
                xp[ 3*id + 0 ] = x[0]; vp[ 3*id + 0 ] = v[0];
                xp[ 3*id + 1 ] = x[1]; vp[ 3*id + 1 ] = v[1];
                xp[ 3*id + 2 ] = x[2]; vp[ 3*id + 2 ] = v[2];
                pid[ id ] = id;
                vid[ id ] = k + nz * ( j + ny * i );
                x[0] -= 0.13333;
                x[1] += 0.09428;
                /* Add second hydrogen atom. */
                id += 1;
                xp[ 3*id + 0 ] = x[0]; vp[ 3*id + 0 ] = v[0];
                xp[ 3*id + 1 ] = x[1]; vp[ 3*id + 1 ] = v[1];
                xp[ 3*id + 2 ] = x[2]; vp[ 3*id + 2 ] = v[2];
                pid[ id ] = id;
                vid[ id ] = k + nz * ( j + ny * i );
                x[0] += 0.03333;
                x[1] -= 0.09428;
                }
            }
        }
    for ( i = 0 ; i < nr_parts ; i++ )
        for ( k = 0 ; k < 3 ; k++ )
            vp[ 3*i + k ] -= vtot[k] / nr_mols;
    printf("main[%i]: done initializing particles.\n",myrank); fflush(stdout);
    printf("main[%i]: generated %i particles.\n", myrank, nr_parts);
    /* dump = fopen("parts_000.dump","w");
    for ( i = 0 ; i < nr_parts ; i++ )
        fprintf(dump,"%e %e %e\n",xp[3*i+0],xp[3*i+1],xp[3*i+2]);
    fclose(dump);*/
    if ( ( res = MPI_Barrier( MPI_COMM_WORLD ) ) != MPI_SUCCESS ) {
        printf( "main[%i]: call to MPI_Barrier failed with error %i.\n" , myrank , res );
        return -1;
        }
    
    
    /* Distribute the particle data over all processors. */
    
    
    /* Dump the particles. */
    /* sprintf(fname,"parts_%03i.dump",myrank);
    dump = fopen(fname,"w");
    for ( i = 0 ; i < ppm_mpart ; i++ )
        fprintf(dump,"%e %e %e\n",xp[3*i+0],xp[3*i+1],xp[3*i+2]);
    fclose(dump); */
        
        
    /* Initialize the engine. */
    printf( "main[%i]: initializing the engine...\n" , myrank ); fflush(stdout);
    if ( engine_init( &e , origin , dim , cutoff , space_periodic_full , 2 , ENGINE_FLAGS | engine_flag_mpi ) != 0 ) {
        printf( "main[%i]: engine_init failed with engine_err=%i.\n" , myrank , engine_err );
        errs_dump(stdout);
        return -1;
        }
    e.dt = 0.002;
    e.time = 0;
    printf("main[%i]: engine initialized.\n",myrank);
    if ( myrank == 0 )
        printf( "main[%i]: space has %i pairs and %i tuples.\n" , myrank , e.s.nr_pairs , e.s.nr_tuples );
    fflush(stdout);
        
    
    /* Register the particle types. */
    if ( engine_addtype( &e , 15.9994 , -0.8476 , "O" , NULL ) < 0 ||
         engine_addtype( &e , 1.00794 , 0.4238 , "H" , NULL ) < 0 ) {
        printf("main[%i]: call to engine_addtype failed.\n",myrank);
        errs_dump(stdout);
        return -1;
        }
        
    /* Initialize the O-H potential. */
    if ( ( pot_OH = potential_create_Ewald( 0.1 , 1.0 , -0.35921288 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main[%i]: potential_create_Ewald failed with potential_err=%i.\n",myrank,potential_err);
        errs_dump(stdout);
        return -1;
        }
    if ( myrank == 0 ) {
        printf("main[%i]: constructed OH-potential with %i intervals.\n",myrank,pot_OH->n); fflush(stdout);
        }

    /* Initialize the H-H potential. */
    if ( ( pot_HH = potential_create_Ewald( 0.1 , 1.0 , 1.7960644e-1 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main[%i]: potential_create_Ewald failed with potential_err=%i.\n",myrank,potential_err);
        errs_dump(stdout);
        return -1;
        }
    if ( myrank == 0 ) {
        printf("main[%i]: constructed HH-potential with %i intervals.\n",myrank,pot_HH->n); fflush(stdout);
        }

    /* Initialize the O-O potential. */
    if ( ( pot_OO = potential_create_LJ126_Ewald( 0.25 , 1.0 , 2.637775819766153e-06 , 2.619222661792581e-03 , 7.1842576e-01 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main[%i]: potential_create_LJ126_Ewald failed with potential_err=%i.\n",myrank,potential_err);
        errs_dump(stdout);
        return -1;
        }
    if ( myrank == 0 ) {
        printf("main[%i]: constructed OO-potential with %i intervals.\n",myrank,pot_OO->n); fflush(stdout);
        }
    
    /* Register these potentials. */
    if ( engine_addpot( &e , pot_OO , 0 , 0 ) < 0 ||
         engine_addpot( &e , pot_HH , 1 , 1 ) < 0 ||
         engine_addpot( &e , pot_OH , 0 , 1 ) < 0 ) {
        printf("main[%i]: call to engine_addpot failed.\n",myrank);
        errs_dump(stdout);
        return -1;
        }
        
    /* Load the engine with the initial set of particles. */
    free( ptype );
    if ( ( ptype = (int *)malloc( sizeof(double) * nr_parts ) ) == NULL ) {
        printf("main[%i]: failed to re-allocate ptype.\n",myrank);
        return -1;
        }
    for ( k = 0 ; k < nr_parts ; k++ )
        ptype[k] = ( k % 3 != 0 );
    if ( ( res = engine_load( &e , xp , vp , ptype , pid , vid , NULL , NULL , nr_parts ) ) < 0 ) {
        printf("main[%i]: engine_load failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
        
        
    /* Split the engine over the processors. */
    if ( ( res = MPI_Comm_size( MPI_COMM_WORLD , &nr_nodes ) != MPI_SUCCESS ) ) {
        printf("main[%i]: MPI_Comm_size failed with error %i.\n",myrank,res);
        errs_dump(stdout);
        return -1;
        }
    if ( ( res = MPI_Comm_rank( MPI_COMM_WORLD , &e.nodeID ) != MPI_SUCCESS ) ) {
        printf("main[%i]: MPI_Comm_rank failed with error %i.\n",myrank,res);
        errs_dump(stdout);
        return -1;
        }
    if ( engine_split_bisect( &e , nr_nodes ) < 0 ) {
        printf("main[%i]: engine_split_bisect failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return -1;
        }
    if ( engine_split( &e ) < 0 ) {
        printf("main[%i]: engine_split_bisect failed with engine_err=%i.\n",myrank,engine_err);
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
        
        
    /* Timing. */    
    toc = getticks();
    if ( myrank == 0 ) {
        printf("main[%i]: setup took %.3f ms.\n",myrank,(double)(toc-tic) * itpms);
        printf("# step e_pot e_kin swaps stalls ms_tot ms_step ms_shake ms_xchg ms_temp\n");
        fflush(stdout);
        }
        

    /* Main time-stepping loop. */
    for ( step = 0 ; step < nr_steps ; step++ ) {
    
        /* Start the clock. */
        tic_step = getticks();
        

        /* Compute a step. */
        tic = getticks();
        if ( engine_step( &e ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return -1;
            }
        timers[tid_step] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: engine_step took %.3f ms.\n",myrank,(double)timers[tid_step] * itpms); fflush(stdout);
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
        
        
        /* Resolve particle global/local IDs. */
        /* tic = getticks();
        bzero( globloc , sizeof(void *) * 2 * nr_parts );
        #pragma omp parallel for schedule(static,100), private(cid,k,p)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
            if ( !(e.s.cells[cid].flags & cell_flag_ghost) )
                for ( k = 0 ; k < e.s.cells[cid].count ; k++ ) {
                    p = &e.s.cells[cid].parts[k];
                    globloc[p->vid].p = p;
                    globloc[p->vid].c = &e.s.cells[cid];
                    }
        #pragma omp parallel for schedule(static,100), private(cid,k,p)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
            if ( e.s.cells[cid].flags & cell_flag_ghost )
                for ( k = 0 ; k < e.s.cells[cid].count ; k++ ) {
                    p = &e.s.cells[cid].parts[k];
                    if ( globloc[p->vid].p == NULL ) {
                        globloc[p->vid].p = p;
                        globloc[p->vid].c = &e.s.cells[cid];
                        }
                    }
        timers[tid_resolv] = getticks() - tic;
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: resolving global/local IDs took %.3f ms.\n",myrank,(double)timers[tid_resolv] * itpms); fflush(stdout);
            } */
            
        
        /* Shake the particle positions. */
        tic = getticks();
        #pragma omp parallel for schedule(dynamic), private(cid,j,p_O,p_H1,p_H2,c_O,c_H1,c_H2,vp_O,vp_H1,vp_H2,k,new_O,new_H1,new_H2,old_O,old_H1,old_H2,v_OH1,v_OH2,v_HH,d_OH1,lambda,d_OH2,d_HH)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
            for ( j = 0 ; j < e.s.cells[cid].count ; j++ ) {
            
                /* Grab a part and check if it's an oxygen. */
                p_O = &e.s.cells[cid].parts[j];
                if ( p_O->type != 0 )
                    continue;
                
                /* Do we even own part of this molecule? */
                c_O = &e.s.cells[cid]; 
                p_H1 = e.s.partlist[p_O->id+1]; c_H1 = e.s.celllist[p_O->id+1]; 
                p_H2 = e.s.partlist[p_O->id+2]; c_H2 = e.s.celllist[p_O->id+2];
                if ( p_O == NULL || p_H1 == NULL || p_H2 == NULL ||
                    ( c_O->flags & cell_flag_ghost && c_H1->flags & cell_flag_ghost && c_H2->flags & cell_flag_ghost ) )
                    continue;

                // unwrap the data
                for ( k = 0 ; k < 3 ; k++ ) {
                    new_O[k] = p_O->x[k] + c_O->origin[k];
                    vp_O[k] = p_O->v[k];
                    new_H1[k] = p_H1->x[k] + c_H1->origin[k];
                    vp_H1[k] = p_H1->v[k];
                    new_H2[k] = p_H2->x[k] + c_H2->origin[k];
                    vp_H2[k] = p_H2->v[k];
                    }
                for ( k = 0 ; k < 3 ; k++ ) {
                    old_O[k] = new_O[k] - e.dt * vp_O[k];
                    if ( new_H1[k] - new_O[k] > dim[k] * 0.5 )
                        new_H1[k] -= dim[k];
                    else if ( new_H1[k] - new_O[k] < -dim[k] * 0.5 )
                        new_H1[k] += dim[k];
                    old_H1[k] = new_H1[k] - e.dt * vp_H1[k];
                    if ( new_H2[k] - new_O[k] > dim[k] * 0.5 )
                        new_H2[k] -= dim[k];
                    else if ( new_H2[k] - new_O[k] < -dim[k] * 0.5 )
                        new_H2[k] += dim[k];
                    old_H2[k] = new_H2[k] - e.dt * vp_H2[k];
                    v_OH1[k] = old_O[k] - old_H1[k];
                    v_OH2[k] = old_O[k] - old_H2[k];
                    v_HH[k] = old_H1[k] - old_H2[k];
                    }

                // main loop
                while ( 1 ) {

                    // correct for the OH1 constraint
                    for ( d_OH1 = 0.0 , k = 0 ; k < 3 ; k++ )
                        d_OH1 += (new_O[k] - new_H1[k]) * (new_O[k] - new_H1[k]);
                    lambda = 0.5 * ( 0.1*0.1 - d_OH1 ) /
                        ( (new_O[0] - new_H1[0]) * v_OH1[0] + (new_O[1] - new_H1[1]) * v_OH1[1] + (new_O[2] - new_H1[2]) * v_OH1[2] );
                    for ( k = 0 ; k < 3 ; k++ ) {
                        new_O[k] += lambda * 1.00794 / ( 1.00794 + 15.9994 ) * v_OH1[k];
                        new_H1[k] -= lambda * 15.9994 / ( 1.00794 + 15.9994 ) * v_OH1[k];
                        }

                    // correct for the OH2 constraint
                    for ( d_OH2 = 0.0 , k = 0 ; k < 3 ; k++ )
                        d_OH2 += (new_O[k] - new_H2[k]) * (new_O[k] - new_H2[k]);
                    lambda = 0.5 * ( 0.1*0.1 - d_OH2 ) /
                        ( (new_O[0] - new_H2[0]) * v_OH2[0] + (new_O[1] - new_H2[1]) * v_OH2[1] + (new_O[2] - new_H2[2]) * v_OH2[2] );
                    for ( k = 0 ; k < 3 ; k++ ) {
                        new_O[k] += lambda * 1.00794 / ( 1.00794 + 15.9994 ) * v_OH2[k];
                        new_H2[k] -= lambda * 15.9994 / ( 1.00794 + 15.9994 ) * v_OH2[k];
                        }

                    // correct for the HH constraint
                    for ( d_HH = 0.0 , k = 0 ; k < 3 ; k++ )
                        d_HH += (new_H1[k] - new_H2[k]) * (new_H1[k] - new_H2[k]);
                    lambda = 0.5 * ( 0.1633*0.1633 - d_HH ) /
                        ( (new_H1[0] - new_H2[0]) * v_HH[0] + (new_H1[1] - new_H2[1]) * v_HH[1] + (new_H1[2] - new_H2[2]) * v_HH[2] );
                    for ( k = 0 ; k < 3 ; k++ ) {
                        new_H1[k] += lambda * 0.5 * v_HH[k];
                        new_H2[k] -= lambda * 0.5 * v_HH[k];
                        }

                    // check the tolerances
                    if ( fabs( d_OH1 - 0.1*0.1 ) < 1e-6 &&
                        fabs( d_OH2 - 0.1*0.1 ) < 1e-6 &&  
                        fabs( d_HH - 0.1633*0.1633 ) < 1e-6 )
                        break;

                    // printf("main: mol %i: d_OH1=%e, d_OH2=%e, d_HH=%e.\n",j,sqrt(d_OH1),sqrt(d_OH2),sqrt(d_HH));
                    // getchar();

                    }

                // wrap the positions back
                for ( k = 0 ; k < 3 ; k++ ) {

                    // write O
                    p_O->x[k] = new_O[k] - c_O->origin[k];
                    p_O->v[k] = (new_O[k] - old_O[k]) / e.dt;

                    // write H1
                    p_H1->x[k] -= e.dt * p_H1->v[k];
                    p_H1->v[k] = (new_H1[k] - old_H1[k]) / e.dt;
                    p_H1->x[k] += e.dt * p_H1->v[k];

                    // write H2
                    p_H2->x[k] -= e.dt * p_H2->v[k];
                    p_H2->v[k] = (new_H2[k] - old_H2[k]) / e.dt;
                    p_H2->x[k] += e.dt * p_H2->v[k];

                    }
                    
                }
                
            } // shake molecules
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
        
        
        /* Resolve particle global/local IDs. */
        /* tic = getticks();
        for ( k = 0 ; k < nr_parts ; k++ )
            globloc[k] = -1;
        for ( k = 0 ; k < nr_parts ; k++ )
            globloc[ pid[k] ] = k;
        for ( k = nr_parts ; k < ppm_mpart ; k++ )
            if ( globloc[ pid[k] ] < 0 )
                globloc[ pid[k] ] = k;
        toc = getticks();
        if ( verbose && myrank == 0 ) {
            printf("main[%i]: resolving global/local IDs took %.3f ms.\n",myrank,(double)(toc-tic) * itpms); fflush(stdout);
            } */
            
        
        /* Compute the system temperature. */
        tic = getticks();
        
        /* Get the total atomic kinetic energy, v_com and molecular kinetic energy. */
        ekin = 0.0; epot = 0.0;
        vcom_tot_x = 0.0; vcom_tot_y = 0.0; vcom_tot_z = 0.0;
        temp = 0.0;
        #pragma omp parallel for schedule(static), private(p,p_O,p_H1,p_H2,j,k,vcom,v2), reduction(+:ekin,epot,vcom_tot_x,vcom_tot_y,vcom_tot_z,temp)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
            epot += e.s.cells[cid].epot;
            if ( !(e.s.cells[cid].flags & cell_flag_ghost ) )
                for ( j = 0 ; j < e.s.cells[cid].count ; j++ ) {
                    p = &( e.s.cells[cid].parts[j] );
                    v2 = p->v[0]*p->v[0] + p->v[1]*p->v[1] + p->v[2]*p->v[2];
                    if ( p->type == 0 )
                        ekin += v2 * 15.9994 * 0.5;
                    else
                        ekin += v2 * 1.00794 * 0.5;
                    if ( p->type != 0 )
                        continue;
                    p_O = p; p_H1 = e.s.partlist[ p_O->id + 1 ]; p_H2 = e.s.partlist[ p_O->id + 2 ];
                    for ( k = 0 ; k < 3 ; k++ )
                        vcom[k] = ( p_O->v[k] * 15.9994 +
                            p_H1->v[k] * 1.00794 +
                            p_H2->v[k] * 1.00794 ) / 1.801528e+1;
                    vcom_tot_x += vcom[0]; vcom_tot_y += vcom[1]; vcom_tot_z += vcom[2];
                    temp += 9.00764 * ( vcom[0]*vcom[0] + vcom[1]*vcom[1] + vcom[2]*vcom[2] );
                    }
            }
        vcom_tot[0] = vcom_tot_x; vcom_tot[1] = vcom_tot_y; vcom_tot[2] = vcom_tot_z;
        vcom_tot[3] = temp;
            
        /* Collect vcom and ekin from all procs and compute the temp. */
        vcom_tot[4] = epot; vcom_tot[5] = ekin;
        if ( nr_nodes > 1 )
            if ( ( res = MPI_Allreduce( MPI_IN_PLACE , vcom_tot , 6 , MPI_DOUBLE_PRECISION , MPI_SUM , MPI_COMM_WORLD ) ) != MPI_SUCCESS ) {
                printf( "main[%i]: call to MPI_Allreduce failed with error %i.\n" , myrank , res );
                return -1;
                }
        ekin = vcom_tot[5]; epot = vcom_tot[4];
        for ( k = 0 ; k < 3 ; k++ )
            vcom_tot[k] /= nr_mols * 1.801528e+1;
        temp = vcom_tot[3] / ( 1.5 * 6.022045E23 * 1.380662E-26 * nr_mols );
        w = sqrt( 1.0 + 0.1 * ( 300.0 / temp - 1.0 ) );
        // printf("main[%i]: vcom_tot is [ %e , %e , %e ].\n",myrank,vcom_tot[0],vcom_tot[1],vcom_tot[2]); fflush(stdout);
            
        /* Subtract the vcom from the molecules on this proc. */
        #pragma omp parallel for schedule(static), private(j,p_O,p_H1,p_H2,k,vcom)
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
            for ( j = 0 ; j < e.s.cells[cid].count ; j++ ) {
                p_O = &( e.s.cells[cid].parts[j] );
                if ( ( p_O->type != 0 ) ||
                     ( p_H1 = e.s.partlist[ p_O->id + 1 ] ) == NULL ||
                     ( p_H2 = e.s.partlist[ p_O->id + 2 ] ) == NULL )
                    continue;
                for ( k = 0 ; k < 3 ; k++ ) {
                    vcom[k] = ( p_O->v[k] * 15.9994 +
                        p_H1->v[k] * 1.00794 +
                        p_H2->v[k] * 1.00794 ) / 1.801528e+1;
                    vcom[k] -= vcom_tot[k];
                    vcom[k] *= ( w - 1.0 );
                    p_O->v[k] += vcom[k];
                    p_H1->v[k] += vcom[k];
                    p_H2->v[k] += vcom[k];
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
            printf("%i %e %e %e %i %i %.3f %.3f %.3f %.3f %.3f ms\n",
                e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(toc_step-tic_step) * itpms,
                timers[tid_step]*itpms, timers[tid_shake]*itpms, timers[tid_exchange]*itpms, timers[tid_temp]*itpms ); fflush(stdout);
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
