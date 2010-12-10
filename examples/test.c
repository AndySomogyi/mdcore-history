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

// include some standard headers
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <pthread.h>
#include <time.h>
#ifdef CELL
    #include <libspe2.h>
    #include <ppu_intrinsics.h>
#else
    #include "cycle.h"
#endif

// include local headers
#include "errs.h"
#include "part.h"
#include "potential.h"
#include "cell.h"
#include "space.h"
#include "engine.h"

int main ( int argc , char *argv[] ) {

    const double origin[3] = { 0.0 , 0.0 , 0.0 };
    // const double dim[3] = { 3.166 , 3.166 , 3.166 };
    // const int nr_mols = 1000, nx = 10;
    const double dim[3] = { 6.332 , 6.332 , 6.332 };
    const int nr_mols = 8000, nx = 20;
    // const double dim[3] = { 4.0 , 4.0 , 4.0 };
    // const int nr_mols = 2016, nx = 13;
    // const double dim[3] = { 8.0 , 8.0 , 8.0 };
    // const int nr_mols = 16128, nx = 26;
    
    double x[3], vtot[3] = { 0.0 , 0.0 , 0.0 };
    double epot, ekin, v2, temp, ee, eff;
    struct engine e;
    struct part p_O, p_H;
    struct potential *pot_OO, *pot_OH, *pot_HH;
    struct cellpair cp;
    int i, j, k, cid, pid, nr_runners = 1, nr_steps = 1000;
    double old_O[3], old_H1[3], old_H2[3], new_O[3], new_H1[3], new_H2[3];
    double v_OH1[3], v_OH2[3], v_HH[3];
    double d_OH1, d_OH2, d_HH, lambda;
    double vcom[3], vcom_tot[3], w;
    #ifdef CELL
        unsigned long long tic, toc;
    #else
        ticks tic, toc;
    #endif
    
    #ifdef CELL
        tic = __mftb();
    #else
        tic = getticks();
    #endif

    // initialize the engine
    printf("main: initializing the engine... "); fflush(stdout);
    if ( engine_init( &e , origin , dim , 1.0 , space_periodic_full , 2 , engine_flag_tuples ) != 0 ) {
        printf("main: engine_init failed with engine_err=%i.\n",engine_err);
        return 1;
        }
    printf("done.\n"); fflush(stdout);
        
    // mix-up the pair list just for kicks
    // printf("main: shuffling the interaction pairs... "); fflush(stdout);
    // srand(6178);
    // for ( i = 0 ; i < e.s.nr_pairs ; i++ ) {
    //     j = rand() % e.s.nr_pairs;
    //     if ( i != j ) {
    //         cp = e.s.pairs[i];
    //         e.s.pairs[i] = e.s.pairs[j];
    //         e.s.pairs[j] = cp;
    //         }
    //     }
    // printf("done.\n"); fflush(stdout);
        

    // initialize the O-H potential
    if ( ( pot_OH = potential_create_Ewald( 0.04 , 1.0 , -0.35921288 , 3.0 , 1.0e-3 ) ) == NULL ) {
        printf("main: potential_create_Ewald failed with potential_err=%i.\n",potential_err);
        return 1;
        }
    printf("main: constructed OH-potential with %i intervals.\n",pot_OH->n); fflush(stdout);
    #ifdef EXPLICIT_POTENTIALS
        pot_OH->flags = potential_flag_Ewald;
        pot_OH->alpha[0] = 0.0;
        pot_OH->alpha[1] = 0.0;
        pot_OH->alpha[2] = -0.35921288;
    #endif

    // initialize the H-H potential
    if ( ( pot_HH = potential_create_Ewald( 0.04 , 1.0 , 1.7960644e-1 , 3.0 , 1.0e-3 ) ) == NULL ) {
        printf("main: potential_create_Ewald failed with potential_err=%i.\n",potential_err);
        return 1;
        }
    printf("main: constructed HH-potential with %i intervals.\n",pot_HH->n); fflush(stdout);
    #ifdef EXPLICIT_POTENTIALS
        pot_HH->flags = potential_flag_Ewald;
        pot_HH->alpha[0] = 0.0;
        pot_HH->alpha[1] = 0.0;
        pot_HH->alpha[2] = 1.7960644e-1;
    #endif

    // initialize the O-O potential
    if ( ( pot_OO = potential_create_LJ126_Ewald( 0.2 , 1.0 , 2.637775819766153e-06 , 2.619222661792581e-03 , 7.1842576e-01 , 3.0 , 1.0e-3 ) ) == NULL ) {
        printf("main: potential_create_LJ126_Ewald failed with potential_err=%i.\n",potential_err);
        return 1;
        }
    printf("main: constructed OO-potential with %i intervals.\n",pot_OO->n); fflush(stdout);
    #ifdef EXPLICIT_POTENTIALS
        pot_OO->flags = potential_flag_LJ126 + potential_flag_Ewald;
        pot_OO->alpha[0] = 2.637775819766153e-06;
        pot_OO->alpha[1] = 2.619222661792581e-03;
        pot_OO->alpha[2] = 7.1842576e-01;
    #endif
    // for ( i = 0 ; i < 1000 ; i++ ) {
    //     temp = 0.3 + (double)i/1000 * 0.7;
    //     potential_eval( &pot , temp*temp , &ee , &eff );
    //     printf("%e %e %e %e\n", temp , ee , eff , dfdr( temp ) );
    //     }
    // return 0;
        
    
    /* register the particle types. */
    if ( engine_addtype( &e , 0 , 15.9994 , -0.8476 ) < 0 ||
         engine_addtype( &e , 1 , 1.00794 , 0.4238 ) < 0 ) {
        printf("main: call to engine_addtype failed.\n");
        errs_dump(stdout);
        return 1;
        }
        
    // register these potentials.
    if ( engine_addpot( &e , pot_OO , 0 , 0 ) < 0 ||
         engine_addpot( &e , pot_HH , 1 , 1 ) < 0 ||
         engine_addpot( &e , pot_OH , 0 , 1 ) < 0 ) {
        printf("main: call to engine_addpot failed.\n");
        errs_dump(stdout);
        return 1;
        }
    
    // set fields for all particles
    srand(6178);
    p_O.type = 0;
    p_H.type = 1;
    p_O.flags = part_flag_none;
    p_H.flags = part_flag_none;
    for ( k = 0 ; k < 3 ; k++ ) {
        p_O.v[k] = 0.0; p_H.v[k] = 0.0;
        p_O.f[k] = 0.0; p_H.f[k] = 0.0;
        }
    #ifdef VECTORIZE
        p_O.v[3] = 0.0; p_O.f[3] = 0.0; p_O.x[3] = 0.0;
        p_H.v[3] = 0.0; p_H.f[3] = 0.0; p_H.x[3] = 0.0;
    #endif
    
    // create and add the particles
    printf("main: initializing particles... "); fflush(stdout);
    for ( i = 0 ; i < nx ; i++ ) {
        x[0] = 0.1 + i * 0.31;
        for ( j = 0 ; j < nx ; j++ ) {
            x[1] = 0.1 + j * 0.31;
            for ( k = 0 ; k < nx && k + nx * ( j + nx * i ) < nr_mols ; k++ ) {
                p_O.id = 3 * (k + nx * ( j + nx * i ));
                x[2] = 0.1 + k * 0.31;
                p_O.v[0] = ((double)rand()) / RAND_MAX - 0.5;
                p_O.v[1] = ((double)rand()) / RAND_MAX - 0.5;
                p_O.v[2] = ((double)rand()) / RAND_MAX - 0.5;
                temp = 0.7 / sqrt( p_O.v[0]*p_O.v[0] + p_O.v[1]*p_O.v[1] + p_O.v[2]*p_O.v[2] );
                p_O.v[0] *= temp; p_O.v[1] *= temp; p_O.v[2] *= temp;
                vtot[0] += p_O.v[0]; vtot[1] += p_O.v[1]; vtot[2] += p_O.v[2];
                if ( space_addpart( &(e.s) , &p_O , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    return 1;
                    }
                x[0] += 0.1;
                p_H.id = 3 * (k + nx * ( j + nx * i )) + 1;
                p_H.v[0] = p_O.v[0]; p_H.v[1] = p_O.v[1]; p_H.v[2] = p_O.v[2];
                if ( space_addpart( &(e.s) , &p_H , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    return 1;
                    }
                x[0] -= 0.13333;
                x[1] += 0.09428;
                p_H.id = 3 * (k + nx * ( j + nx * i )) + 2;
                if ( space_addpart( &(e.s) , &p_H , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    return 1;
                    }
                x[0] += 0.03333;
                x[1] -= 0.09428;
                }
            }
        }
    for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
        for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ )
            for ( v2 = 0.0 , k = 0 ; k < 3 ; k++ )
                e.s.cells[cid].parts[pid].v[k] -= vtot[k] / 8000;
    printf("done.\n"); fflush(stdout);
        
    // set the time and time-step by hand
    e.time = 0;
    e.dt = 0.002;
    
    #ifdef CELL
        toc = __mftb();
        printf("main: setup took %.3f ms.\n",(double)(toc-tic) / 25000);
    #else
        toc = getticks();
        printf("main: setup took %.3f ms.\n",elapsed(toc,tic) / 2501000);
    #endif
    
    // did the user specify a number of runners?
    if ( argc > 1 )
        nr_runners = atoi( argv[1] );
    #ifdef CELL
        else
            nr_runners = spe_cpu_info_get( SPE_COUNT_USABLE_SPES , -1 );
    #endif
        
    // start the engine
    #ifdef CELL
        if ( engine_start( &e , nr_runners ) != 0 ) {
            printf("main: engine_start failed with engine_err=%i.\n",engine_err);
            return 1;
            }
    #else
        if ( engine_start( &e , nr_runners ) != 0 ) {
            printf("main: engine_start failed with engine_err=%i.\n",engine_err);
            return 1;
            }
    #endif
    
    // did the user specify a number of steps?
    if ( argc > 2 )
        nr_steps = atoi( argv[2] );
        
    // do a few steps
    for ( i = 0 ; i < nr_steps ; i++ ) {
    
        // take a step
        #ifdef CELL
            tic = __mftb();
        #else
            tic = getticks();
        #endif
        if ( engine_step( &e ) != 0 ) {
            printf("main: engine_step failed with engine_err=%i.\n",engine_err);
            fflush(stdout);
            return 1;
            }
        #ifdef CELL
            toc = __mftb();
        #else
            toc = getticks();
        #endif
        
        // shake the water molecules
        #pragma omp for private(k,new_O,new_H1,new_H2,old_O,old_H1,old_H2,v_OH1,v_OH2,v_HH,d_OH1,lambda,d_OH2,d_HH)
        for ( j = 0 ; j < nr_mols ; j++ ) {
        
            // unwrap the data
            space_getpos( &(e.s) , j*3 , new_O );
            space_getpos( &(e.s) , j*3+1 , new_H1 );
            space_getpos( &(e.s) , j*3+2 , new_H2 );
            for ( k = 0 ; k < 3 ; k++ ) {
                old_O[k] = new_O[k] - e.dt * e.s.partlist[j*3]->v[k];
                if ( new_H1[k] - new_O[k] > e.s.dim[k] * 0.5 )
                    new_H1[k] -= e.s.dim[k];
                else if ( new_H1[k] - new_O[k] < -e.s.dim[k] * 0.5 )
                    new_H1[k] += e.s.dim[k];
                old_H1[k] = new_H1[k] - e.dt * e.s.partlist[j*3+1]->v[k];
                if ( new_H2[k] - new_O[k] > e.s.dim[k] * 0.5 )
                    new_H2[k] -= e.s.dim[k];
                else if ( new_H2[k] - new_O[k] < -e.s.dim[k] * 0.5 )
                    new_H2[k] += e.s.dim[k];
                old_H2[k] = new_H2[k] - e.dt * e.s.partlist[j*3+2]->v[k];
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
                if ( fabs( d_OH1 - 0.1*0.1 ) < 1e-10 &&
                    fabs( d_OH2 - 0.1*0.1 ) < 1e-10 &&  
                    fabs( d_HH - 0.1633*0.1633 ) < 1e-10 )
                    break;
                    
                // printf("main: mol %i: d_OH1=%e, d_OH2=%e, d_HH=%e.\n",j,sqrt(d_OH1),sqrt(d_OH2),sqrt(d_HH));
                // getchar();
                    
                }
                
            // wrap the positions back
            for ( k = 0 ; k < 3 ; k++ ) {
            
                // write O
                e.s.partlist[j*3]->x[k] = new_O[k] - e.s.celllist[j*3]->origin[k];
                e.s.partlist[j*3]->v[k] = (new_O[k] - old_O[k]) / e.dt;
                
                // write H1
                if ( new_H1[k] - e.s.celllist[j*3+1]->origin[k] > e.s.dim[k] * 0.5 )
                    e.s.partlist[j*3+1]->x[k] = new_H1[k] - e.s.celllist[j*3+1]->origin[k] - e.s.dim[k];
                else if ( new_H1[k] - e.s.celllist[j*3+1]->origin[k] < -e.s.dim[k] * 0.5 )
                    e.s.partlist[j*3+1]->x[k] = new_H1[k] - e.s.celllist[j*3+1]->origin[k] + e.s.dim[k];
                else
                    e.s.partlist[j*3+1]->x[k] = new_H1[k] - e.s.celllist[j*3+1]->origin[k];
                e.s.partlist[j*3+1]->v[k] = (new_H1[k] - old_H1[k]) / e.dt;
                
                // write H2
                if ( new_H2[k] - e.s.celllist[j*3+2]->origin[k] > e.s.dim[k] * 0.5 )
                    e.s.partlist[j*3+2]->x[k] = new_H2[k] - e.s.celllist[j*3+2]->origin[k] - e.s.dim[k];
                else if ( new_H2[k] - e.s.celllist[j*3+2]->origin[k] < -e.s.dim[k] * 0.5 )
                    e.s.partlist[j*3+2]->x[k] = new_H2[k] - e.s.celllist[j*3+2]->origin[k] + e.s.dim[k];
                else
                    e.s.partlist[j*3+2]->x[k] = new_H2[k] - e.s.celllist[j*3+2]->origin[k];
                e.s.partlist[j*3+2]->v[k] = (new_H2[k] - old_H2[k]) / e.dt;
                }
                
            } // shake molecules
            
        // re-shuffle the space just to be sure...
        if ( space_shuffle( &e.s ) < 0 ) {
            printf("main: space_shuffle failed with space_err=%i.\n",space_err);
            return 1;
            }
            
            
        // get the total COM-velocities and ekin
        vcom_tot[0] = 0.0; vcom_tot[1] = 0.0; vcom_tot[2] = 0.0;
        ekin = 0.0;
        for ( j = 0 ; j < nr_mols ; j++ ) {
            for ( k = 0 ; k < 3 ; k++ ) {
                vcom[k] = ( e.s.partlist[j*3]->v[k] * 15.9994 +
                    e.s.partlist[j*3+1]->v[k] * 1.00794 +
                    e.s.partlist[j*3+2]->v[k] * 1.00794 ) / 1.801528e+1;
                vcom_tot[k] += vcom[k];
                }
            ekin += 9.00764 * ( vcom[0]*vcom[0] + vcom[1]*vcom[1] + vcom[2]*vcom[2] );
            }
        vcom_tot[0] /= 1000 * 1.801528e+1;
        vcom_tot[1] /= 1000 * 1.801528e+1;
        vcom_tot[2] /= 1000 * 1.801528e+1;
                
        // compute the temperature and scaling
        temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * nr_mols );
        w = sqrt( 1.0 + 0.1 * ( 300.0 / temp - 1.0 ) );

        // compute the molecular heat
        if ( i < 10000 ) {
        
            // scale the COM-velocities
            for ( j = 0 ; j < nr_mols ; j++ ) {
                for ( k = 0 ; k < 3 ; k++ ) {
                    vcom[k] = ( e.s.partlist[j*3]->v[k] * 15.9994 +
                        e.s.partlist[j*3+1]->v[k] * 1.00794 +
                        e.s.partlist[j*3+2]->v[k] * 1.00794 ) / 1.801528e+1;
                    vcom[k] -= vcom_tot[k];
                    vcom[k] *= ( w - 1.0 );
                    e.s.partlist[j*3]->v[k] += vcom[k];
                    e.s.partlist[j*3+1]->v[k] += vcom[k];
                    e.s.partlist[j*3+2]->v[k] += vcom[k];
                    }
                }
                
            } // apply molecular thermostat
            
        // tabulate the total potential and kinetic energy
        epot = 0.0; ekin = 0.0;
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
            epot += e.s.cells[cid].epot;
            for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ ) {
                for ( v2 = 0.0 , k = 0 ; k < 3 ; k++ )
                    v2 += e.s.cells[cid].parts[pid].v[k] * e.s.cells[cid].parts[pid].v[k];
                ekin += 0.5 * e.types[ e.s.cells[cid].parts[pid].type ].mass * v2;
                }
            }
            
            
        printf("%i %e %e %e %i %i %.3f ms\n",
        #ifdef CELL
            // e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(double)(toc-tic) / 25000); fflush(stdout);
            e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,(double)(toc-tic) / 26664.184); fflush(stdout);
        #else
            // e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,elapsed(toc,tic) / 2300000); fflush(stdout);
            e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,elapsed(toc,tic) / 2199977); fflush(stdout);
        #endif
        
        // print some particle data
        // printf("main: part 13322 is at [ %e , %e , %e ].\n",
        //     e.s.partlist[13322]->x[0], e.s.partlist[13322]->x[1], e.s.partlist[13322]->x[2]);
            
        }
     
    // dump the particle positions, just for the heck of it
    // for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
    //     for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ ) {
    //         for ( k = 0 ; k < 3 ; k++ )
    //             x[k] = e.s.cells[cid].origin[k] + e.s.cells[cid].parts[pid].x[k];
    //         printf("%i %e %e %e\n",e.s.cells[cid].parts[pid].id,x[0],x[1],x[2]);
    //         }
        
    // clean break
    return 0;

    }
