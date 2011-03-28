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
#include <pthread.h>
#include <time.h>
#include "cycle.h"

/* MPI headers. */
#include <mpi.h>

/* Include mdcore. */
#include "mdcore.h"


/* Wrappers for ppm calls. */
#include "ppm.h"

/* Ticks Per Second. */
#ifndef CPU_TPS
    #define CPU_TPS 2.67e+9
#endif


/* The main routine -- this is where it all happens. */

int main ( int argc , char *argv[] ) {


    /* Simulation constants. */
    double origin[3] = { 0.0 , 0.0 , 0.0 };
    double dim[3] = { 16.0 , 16.0 , 16.0 };
    int nr_mols = 129024, nr_parts = nr_mols*3;
    double cutoff = 1.0;


    /* Local variables. */
    int res = 0, myrank;
    double *xp = NULL, *vp = NULL, x[3], v[3];
    int *pid = NULL, *ptype = NULL;
    int i, j, k, nx, ny, nz, id;
    double hx, hy, hz, temp;
    double vtot[3] = { 0.0 , 0.0 , 0.0 };
    FILE *dump;
    char fname[100];
    int globloc[nr_parts];
    ticks tic, toc;
    
    
    /* PPM variables. */
    int ppm_debug = 0;
    int topoid = 0;
    int bc[6] = { ppm_param_bcdef_periodic , ppm_param_bcdef_periodic , ppm_param_bcdef_periodic , ppm_param_bcdef_periodic , ppm_param_bcdef_periodic , ppm_param_bcdef_periodic };
    double *cost = NULL;
    int ncost, ppm_npart = 0, ppm_mpart;
    double ppm_minphys[3], ppm_maxphys[3], ppm_dim[3];

    
    /* mdcore stuff. */
    int ENGINE_FLAGS = engine_flag_tuples;
    struct engine e;
    struct potential *pot_OO, *pot_OH, *pot_HH;
    int nr_runners = 1, nr_steps = 1000;
    
    
    /* Start the clock. */
    tick = 
    
    
    /* Start by initializing MPI. */
    if ( ( res = MPI_Init( &argc , &argv ) ) != MPI_SUCCESS ) {
        printf( "main: call to MPI_Init failed with error %i.\n" , res );
        return -1;
        }
    if ( ( res = MPI_Comm_rank( MPI_COMM_WORLD , &myrank ) ) != MPI_SUCCESS ) {
        printf( "main: call to MPI_Comm_rank failed with error %i.\n" , res );
        return -1;
        }
    printf( "main[%i]: MPI is up and running...\n" , myrank );
    fflush(stdout);
    
    
    /* Initialize our own input parameters. */
    if ( argc > 1 )
        nr_runners = atoi( argv[1] );
    if ( argc > 2 )
        nr_steps = atoi( argv[2] );
        
    
    /* Now try calling ppm_init. */
    ppm_init( 3 , ppm_kind_double , -15 , MPI_Comm_c2f(MPI_COMM_WORLD) , ppm_debug , &res , 0 , 0 , 0 );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_init failed with error %i.\n" , myrank , res );
        return -1;
        }
    printf( "main[%i]: PPM is up and running...\n" , myrank ); fflush(stdout);
        
        
    /* Generate the particles for this simulation. */
    if ( myrank == 0 ) {
        if ( ( xp = (double *)malloc( sizeof(double) * nr_parts * 3 ) ) == NULL ||
             ( vp = (double *)malloc( sizeof(double) * nr_parts * 3 ) ) == NULL ||
             ( pid = (int *)malloc( sizeof(int) * nr_parts ) ) == NULL ||
             ( ptype = (int *)malloc( sizeof(int) * nr_parts ) ) == NULL ) {
             printf( "main[%i]: allocation of particle data failed!\n" , myrank );
             return -1;
             }
        printf("main[%i]: initializing particles... " , myrank); fflush(stdout);
        nx = ceil( pow( nr_mols , 1.0/3 ) ); hx = dim[0] / nx;
        ny = ceil( sqrt( ((double)nr_mols) / nx ) ); hy = dim[1] / ny;
        nz = ceil( ((double)nr_mols) / nx / ny ); hz = dim[2] / nz;
        for ( i = 0 ; i < nx ; i++ ) {
            x[0] = 0.1 + i * hx;
            for ( j = 0 ; j < ny ; j++ ) {
                x[1] = 0.1 + j * hy;
                for ( k = 0 ; k < nz && k + nz * ( j + ny * i ) < nr_mols ; k++ ) {
                    id = 3 * (k + nz * ( j + ny * i ));
                    x[2] = 0.1 + k * hz;
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
                    pid[ id ] = id; ptype[ id ] = 0;
                    x[0] += 0.1;
                    id += 1;
                    /* Add first hydrogen atom. */
                    xp[ 3*id + 0 ] = x[0]; vp[ 3*id + 0 ] = v[0];
                    xp[ 3*id + 1 ] = x[1]; vp[ 3*id + 1 ] = v[1];
                    xp[ 3*id + 2 ] = x[2]; vp[ 3*id + 2 ] = v[2];
                    pid[ id ] = id; ptype[ id ] = 1;
                    x[0] -= 0.13333;
                    x[1] += 0.09428;
                    id += 1;
                    /* Add second hydrogen atom. */
                    xp[ 3*id + 0 ] = x[0]; vp[ 3*id + 0 ] = v[0];
                    xp[ 3*id + 1 ] = x[1]; vp[ 3*id + 1 ] = v[1];
                    xp[ 3*id + 2 ] = x[2]; vp[ 3*id + 2 ] = v[2];
                    pid[ id ] = id; ptype[ id ] = 1;
                    x[0] += 0.03333;
                    x[1] -= 0.09428;
                    }
                }
            }
        for ( i = 0 ; i < nr_parts ; i++ )
            for ( k = 0 ; k < 3 ; k++ )
                v[ 3*i + k ] -= vtot[k] / nr_mols;
        printf("done.\n"); fflush(stdout);
        printf("main[%i]: generated %i particles.\n", myrank, nr_parts);
        ppm_npart = nr_parts;
        /* dump = fopen("parts_000.dump","w");
        for ( i = 0 ; i < nr_parts ; i++ )
            fprintf(dump,"%e %e %e\n",xp[3*i+0],xp[3*i+1],xp[3*i+2]);
        fclose(dump);*/
        }
    if ( ( res = MPI_Barrier( MPI_COMM_WORLD ) ) != MPI_SUCCESS ) {
        printf( "main[%i]: call to MPI_Barrier failed with error %i.\n" , myrank , res );
        return -1;
        }
    
    
    /* Make the topology. */
    ppm_topo_mkgeom( &topoid , ppm_param_decomp_bisection , ppm_param_assign_internal , origin , dim , bc , cutoff , &cost , &ncost , &res );
    /* ppm_topo_mkpart( &topoid , xp , ppm_npart , ppm_param_decomp_cuboid , ppm_param_assign_internal , origin , dim , bc , cutoff , &cost , &ncost , &res ); */
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_mktopo failed with error %i.\n" , myrank , res );
        return -1;
        }
    printf( "main[%i]: Created topology (ncost=%i).\n" , myrank , ncost ); fflush(stdout);
        
    /* Distribute the particle data over all processors. */
    ppm_map_part_global( topoid , xp , ppm_npart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_global failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_push_2dd( vp , 3 , ppm_npart , &res , 0 );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_push_2dd failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_push_1di( pid , ppm_npart , &res , 0 );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_push_1di( ptype , ppm_npart , &res , 0 );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_send( ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_send failed with error %i.\n" , myrank , res );
        return -1;
        }
        
    /* Get the particle data back. */
    ppm_map_part_pop_1di( &ptype , ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_pop_1di( &pid , ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_pop_2dd( &vp , 3 , ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
        return -1;
        }
    ppm_map_part_pop_2dd( &xp , 3 , ppm_npart , &ppm_mpart , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
        return -1;
        }
    printf( "main[%i]: now have %i particles. \n" , myrank , ppm_mpart );
    
    /* Dump the particles. */
    /* sprintf(fname,"parts_%03i.dump",myrank);
    dump = fopen(fname,"w");
    for ( i = 0 ; i < ppm_mpart ; i++ )
        fprintf(dump,"%e %e %e\n",xp[3*i+0],xp[3*i+1],xp[3*i+2]);
    fclose(dump); */
    
    
    /* Get the extent of the domain on this Processor. */
    ppm_topo_getextent( topoid , ppm_minphys , ppm_maxphys , &res );
    if ( res != 0 ) {
        printf( "main[%i]: call to ppm_topo_getextent failed with error %i.\n" , myrank , res );
        return -1;
        }
    printf( "main[%i]: ppm_minphys is [ %e , %e , %e ].\n" , myrank , ppm_minphys[0] , ppm_minphys[1] , ppm_minphys[2] );
    printf( "main[%i]: ppm_maxphys is [ %e , %e , %e ].\n" , myrank , ppm_maxphys[0] , ppm_maxphys[1] , ppm_maxphys[2] );
    
    /* Extend the physical domain by the cutoff in all dimensions to allow for
       the ghost layers. */
    for ( k = 0 ; k < 3 ; k++ ) {
        ppm_minphys[k] -= cutoff;
        ppm_maxphys[k] += cutoff;
        ppm_dim[k] = ppm_maxphys[k] - ppm_minphys[k];
        }
    
        
    /* Initialize the engine. */
    printf( "main[%i]: initializing the engine... " , myrank ); fflush(stdout);
    if ( engine_init( &e , ppm_minphys , ppm_dim , cutoff , space_periodic_none , 2 , ENGINE_FLAGS ) != 0 ) {
        printf( "main[%i]: engine_init failed with engine_err=%i.\n" , myrank , engine_err );
        errs_dump(stdout);
        return -1;
        }
    e.dt = 0.002;
    e.time = 0;
    printf("done.\n"); fflush(stdout);
    
    
    /* Register the particle types. */
    if ( engine_addtype( &e , 0 , 15.9994 , -0.8476 ) < 0 ||
         engine_addtype( &e , 1 , 1.00794 , 0.4238 ) < 0 ) {
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
    printf("main[%i]: constructed OH-potential with %i intervals.\n",myrank,pot_OH->n); fflush(stdout);

    /* Initialize the H-H potential. */
    if ( ( pot_HH = potential_create_Ewald( 0.1 , 1.0 , 1.7960644e-1 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main[%i]: potential_create_Ewald failed with potential_err=%i.\n",myrank,potential_err);
        errs_dump(stdout);
        return -1;
        }
    printf("main[%i]: constructed HH-potential with %i intervals.\n",myrank,pot_HH->n); fflush(stdout);

    /* Initialize the O-O potential. */
    if ( ( pot_OO = potential_create_LJ126_Ewald( 0.25 , 1.0 , 2.637775819766153e-06 , 2.619222661792581e-03 , 7.1842576e-01 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main[%i]: potential_create_LJ126_Ewald failed with potential_err=%i.\n",myrank,potential_err);
        errs_dump(stdout);
        return -1;
        }
    printf("main[%i]: constructed OO-potential with %i intervals.\n",myrank,pot_OO->n); fflush(stdout);
    
    /* Register these potentials. */
    if ( engine_addpot( &e , pot_OO , 0 , 0 ) < 0 ||
         engine_addpot( &e , pot_HH , 1 , 1 ) < 0 ||
         engine_addpot( &e , pot_OH , 0 , 1 ) < 0 ) {
        printf("main[%i]: call to engine_addpot failed.\n",myrank);
        errs_dump(stdout);
        return -1;
        }
        
        
    /* Start the engine. */
    if ( engine_start( &e , nr_runners ) != 0 ) {
        printf("main[%i]: engine_start failed with engine_err=%i.\n",myrank,engine_err);
        errs_dump(stdout);
        return 1;
        }
        
        
    /* Main time-stepping loop. */
    for ( i = 0 ; i < nr_steps ; i++ ) {
    
    
        /* Get ghost data. */
        tic = getticks();
        ppm_map_part_ghost_get( topoid , xp , 3 , ppm_npart , 0 , cutoff , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_ghost_get failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_2dd( vp , 3 , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( pid , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_send( ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_send failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &pid , ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &vp , 3 , ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &xp , 3 , ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_npart = ppm_mpart;
        toc = getticks();
        printf("main: ghost setup took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS); fflush(stdout);
            
            
        /* Load the data onto the engine. */
        tic = getticks();
        free( ptype );
        if ( ( ptype = (int *)malloc( sizeof(double) * ppm_npart ) ) == NULL ) {
            printf("main[%i]: failed to re-allocate ptype.\n",myrank);
            return 1;
            }
        for ( k = 0 ; k < ppm_npart ; k++ )
            ptype[k] = ( pid[k] % 3 == 0 );
        if ( ( res = engine_load( &e , xp , vp , ptype , pid , NULL , NULL , ppm_npart ) ) < 0 ) {
            printf("main[%i]: engine_load failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return 1;
            }
        toc = getticks();
        printf("main: engine_load took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS); fflush(stdout);
            
        
        /* Compute a step. */
        tic = getticks();
        if ( engine_step( &e ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return 1;
            }
        toc = getticks();
        printf("main: engine_step took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS); fflush(stdout);
            
            
        /* Update the particle velocities & positions. */
        tic = getticks();
        if ( ( res = engine_unload( &e , xp , vp , ptype , pid , NULL , NULL , ppm_npart ) ) < 0 ) {
            printf("main[%i]: engine_unload failed with engine_err=%i.\n",myrank,engine_err);
            errs_dump(stdout);
            return 1;
            }
        toc = getticks();
        printf("main: engine_unload took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS); fflush(stdout);
        
        
        /* Exchange position data for SHAKE. */
        tic = getticks();
        ppm_map_part_ghost_get( topoid , xp , 3 , ppm_npart , 0 , 0.2 , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_ghost_get failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_push_1di( pid , ppm_npart , &res , 0 );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_push_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_send( ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_send failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_1di( &pid , ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_1di failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_map_part_pop_2dd( &xp , 3 , ppm_npart , &ppm_mpart , &res );
        if ( res != 0 ) {
            printf( "main[%i]: call to ppm_map_part_pop_2dd failed with error %i.\n" , myrank , res );
            return -1;
            }
        ppm_npart = ppm_mpart;
        toc = getticks();
        printf("main: ghost exchange (SHAKE) took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS); fflush(stdout);
        
        
        /* Resolve particle global/local IDs. */
        for ( k = 0 ; k < nr_parts ; k++ )
            globloc[k] = -1;
        for ( k = 0 ; k < ppm_npart ; k++ )
            globloc[ pid[k] ] = k;
            
        
        /* Shake the particle positions. */
        
        /* Re-distribute the particles to the processors. */
    
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
