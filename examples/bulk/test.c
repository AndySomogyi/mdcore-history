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

// include some standard headers
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <float.h>
#include <pthread.h>
#include <time.h>
#ifdef CELL
    #include <libspe2.h>
    #include <ppu_intrinsics.h>
#else
    #include "cycle.h"
#endif

/* What to do if ENGINE_FLAGS was not defined? */
#ifndef ENGINE_FLAGS
    #define ENGINE_FLAGS engine_flag_tuples
#endif
#ifndef CPU_TPS
    #define CPU_TPS 2.67e+9
#endif

// include local headers
#include "errs.h"
#include "fptype.h"
#include "part.h"
#include "potential.h"
#include "cell.h"
#include "space.h"
#include "engine.h"

int main ( int argc , char *argv[] ) {

    const double origin[3] = { 0.0 , 0.0 , 0.0 };
    // double dim[3] = { 3*3.166 , 3.166 , 3.166 };
    // int nr_mols = 3*1000;
    // const double dim[3] = { 6.332 , 6.332 , 6.332 };
    // const int nr_mols = 8000;
    // const double dim[3] = { 4.0 , 4.0 , 4.0 };
    // const int nr_mols = 2016;
    double dim[3] = { 8.0 , 8.0 , 8.0 };
    int nr_mols = 16128;
    // double dim[3] = { 16.0 , 16.0 , 16.0 };
    // int nr_mols = 129024;
    double Temp = 300.0;
    
    double x[3], vtot[3] = { 0.0 , 0.0 , 0.0 };
    double epot, ekin, temp, cutoff = 1.0, cellwidth;
    // FPTYPE ee, eff;
    struct engine e;
    struct part pO, pH, *p_O, *p_H1, *p_H2;
    struct potential *pot_OO, *pot_OH, *pot_HH;
    struct cell *c_O, *c_H1, *c_H2;
    // struct potential *pot_ee;
    int i, j, k, cid, pid, nr_runners = 1, nr_steps = 1000;
    int nx, ny, nz;
    double hx, hy, hz;
    double old_O[3], old_H1[3], old_H2[3], new_O[3], new_H1[3], new_H2[3];
    double v_OH1[3], v_OH2[3], v_HH[3];
    double d_OH1, d_OH2, d_HH, lambda;
    double vcom[3], vcom_tot[3], w;
    // struct cellpair cp;
    #ifdef CELL
        unsigned long long tic, toc, toc_step, toc_shake, toc_temp;
    #else
        ticks tic, toc, toc_step, toc_shake, toc_temp;
    #endif
    
    #ifdef CELL
        tic = __mftb();
    #else
        tic = getticks();
    #endif
    
    // did the user supply a cutoff?
    if ( argc > 4 ) {
        cellwidth = atof( argv[4] );
        nr_mols *= ( cellwidth * cellwidth * cellwidth );
        for ( k = 0 ; k < 3 ; k++ )
            dim[k] *= cellwidth * (1.0 + DBL_EPSILON);
        }
    else
        cellwidth = 1.0;
    printf("main: cell width set to %22.16e.\n", cellwidth);
    
    // initialize the engine
    printf("main: initializing the engine... "); fflush(stdout);
    if ( engine_init( &e , origin , dim , cellwidth , space_periodic_full , 2 , ENGINE_FLAGS ) != 0 ) {
        printf("main: engine_init failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
        }
    printf("done.\n"); fflush(stdout);
    
    // set the interaction cutoff
    e.s.cutoff = cutoff;
    e.s.cutoff2 = cutoff * cutoff;
    printf("main: cell dimensions = [ %i , %i , %i ].\n", e.s.cdim[0] , e.s.cdim[1] , e.s.cdim[2] );
    printf("main: cell size = [ %e , %e , %e ].\n" , e.s.h[0] , e.s.h[1] , e.s.h[2] );
    printf("main: cutoff set to %22.16e.\n", cutoff);
    printf("main: nr tuples: %i.\n",e.s.nr_tuples);
        
    /* mix-up the pair list just for kicks
    printf("main: shuffling the interaction pairs... "); fflush(stdout);
    srand(6178);
    for ( i = 0 ; i < e.s.nr_pairs ; i++ ) {
        j = rand() % e.s.nr_pairs;
        if ( i != j ) {
            cp = e.s.pairs[i];
            e.s.pairs[i] = e.s.pairs[j];
            e.s.pairs[j] = cp;
            }
        }
    printf("done.\n"); fflush(stdout); */
        

    // initialize the O-H potential
    if ( ( pot_OH = potential_create_Ewald( 0.1 , 1.0 , -0.35921288 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main: potential_create_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
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
    if ( ( pot_HH = potential_create_Ewald( 0.1 , 1.0 , 1.7960644e-1 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main: potential_create_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
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
    if ( ( pot_OO = potential_create_LJ126_Ewald( 0.25 , 1.0 , 2.637775819766153e-06 , 2.619222661792581e-03 , 7.1842576e-01 , 3.0 , 0.9e-4 ) ) == NULL ) {
        printf("main: potential_create_LJ126_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
        }
    printf("main: constructed OO-potential with %i intervals.\n",pot_OO->n); fflush(stdout);
    #ifdef EXPLICIT_POTENTIALS
        pot_OO->flags = potential_flag_LJ126 + potential_flag_Ewald;
        pot_OO->alpha[0] = 2.637775819766153e-06;
        pot_OO->alpha[1] = 2.619222661792581e-03;
        pot_OO->alpha[2] = 7.1842576e-01;
    #endif
    
    // initialize the expl. electrostatic potential
    /* if ( ( pot_ee = potential_create_Ewald( 0.1 , 1.0 , 1.0 , 3.0 , 1.0e-4 ) ) == NULL ) {
        printf("main: potential_create_LJ126_Ewald failed with potential_err=%i.\n",potential_err);
        errs_dump(stdout);
        return 1;
        }
    printf("main: constructed expl. electrostatic potential with %i intervals.\n",pot_ee->n); fflush(stdout);
    if ( engine_setexplepot( &e , pot_ee ) < 0 ) {
        printf("main: engine_setexplepot failed with engine_err=%i.\n",engine_err);
        errs_dump(stdout);
        return 1;
        } */
    
    /* dump the OO-potential to make sure its ok... 
    for ( i = 0 ; i < 1000 ; i++ ) {
        temp = 0.2 + (double)i/1000 * 0.8;
        potential_eval( pot_OO , temp*temp , &ee , &eff );
        printf("%23.16e %23.16e %23.16e %23.16e %23.16e\n", temp , ee , eff , 
             potential_LJ126(temp,2.637775819766153e-06,2.619222661792581e-03) + 7.1842576e-01*potential_Ewald(temp,3.0) ,
             potential_LJ126_p(temp,2.637775819766153e-06,2.619222661792581e-03) + 7.1842576e-01*potential_Ewald_p(temp,3.0) );
        }
    return 0; */
        
    
    /* register the particle types. */
    if ( engine_addtype( &e , 15.9994 , -0.8476 , "O" , NULL ) < 0 ||
         engine_addtype( &e , 1.00794 , 0.4238 , "H" , NULL ) < 0 ) {
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
    pO.type = 0;
    pH.type = 1;
    pO.flags = part_flag_none;
    pH.flags = part_flag_none;
    for ( k = 0 ; k < 3 ; k++ ) {
        pO.v[k] = 0.0; pH.v[k] = 0.0;
        pO.f[k] = 0.0; pH.f[k] = 0.0;
        }
    #ifdef VECTORIZE
        pO.v[3] = 0.0; pO.f[3] = 0.0; pO.x[3] = 0.0;
        pH.v[3] = 0.0; pH.f[3] = 0.0; pH.x[3] = 0.0;
    #endif
    
    // create and add the particles
    printf("main: initializing particles... "); fflush(stdout);
    nx = ceil( pow( nr_mols , 1.0/3 ) ); hx = dim[0] / nx;
    ny = ceil( sqrt( ((double)nr_mols) / nx ) ); hy = dim[1] / ny;
    nz = ceil( ((double)nr_mols) / nx / ny ); hz = dim[2] / nz;
    for ( i = 0 ; i < nx ; i++ ) {
        x[0] = 0.05 + i * hx;
        for ( j = 0 ; j < ny ; j++ ) {
            x[1] = 0.05 + j * hy;
            for ( k = 0 ; k < nz && k + nz * ( j + ny * i ) < nr_mols ; k++ ) {
                pO.vid = k + nz * ( j + ny * i );
                x[2] = 0.05 + k * hz;
                pO.v[0] = ((double)rand()) / RAND_MAX - 0.5;
                pO.v[1] = ((double)rand()) / RAND_MAX - 0.5;
                pO.v[2] = ((double)rand()) / RAND_MAX - 0.5;
                temp = 0.675 / sqrt( pO.v[0]*pO.v[0] + pO.v[1]*pO.v[1] + pO.v[2]*pO.v[2] );
                pO.v[0] *= temp; pO.v[1] *= temp; pO.v[2] *= temp;
                vtot[0] += pO.v[0]; vtot[1] += pO.v[1]; vtot[2] += pO.v[2];
                if ( space_addpart( &(e.s) , &pO , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    errs_dump(stdout);
                    return 1;
                    }
                x[0] += 0.1;
                pH.vid = pO.vid;
                pH.v[0] = pO.v[0]; pH.v[1] = pO.v[1]; pH.v[2] = pO.v[2];
                if ( space_addpart( &(e.s) , &pH , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    errs_dump(stdout);
                    return 1;
                    }
                x[0] -= 0.13333;
                x[1] += 0.09428;
                pH.vid = pO.vid;
                if ( space_addpart( &(e.s) , &pH , x ) != 0 ) {
                    printf("main: space_addpart failed with space_err=%i.\n",space_err);
                    errs_dump(stdout);
                    return 1;
                    }
                x[0] += 0.03333;
                x[1] -= 0.09428;
                }
            }
        }
    for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
        for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ )
            for ( k = 0 ; k < 3 ; k++ )
                e.s.cells[cid].parts[pid].v[k] -= vtot[k] / nr_mols;
    printf("done.\n"); fflush(stdout);
    printf("main: inserted %i particles.\n", e.s.nr_parts);
    

    // set the time and time-step by hand
    e.time = 0;
    if ( argc > 3 )
        e.dt = atof( argv[3] );
    else
        e.dt = 0.002;
    printf("main: dt set to %f fs.\n", e.dt*1000 );
    
    #ifdef CELL
        toc = __mftb();
    #else
        toc = getticks();
    #endif
    printf("main: setup took %.3f ms.\n",(double)(toc-tic) * 1000 / CPU_TPS);
    
    // did the user specify a number of runners?
    if ( argc > 1 )
        nr_runners = atoi( argv[1] );
        
    // start the engine
    #ifdef CELL
        /* if ( engine_start( &e , nr_runners ) != 0 ) {
            printf("main: engine_start failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return 1;
            } */
        if ( engine_start_SPU( &e , spe_cpu_info_get( SPE_COUNT_USABLE_SPES , -1 ) ) != 0 ) {
            printf("main: engine_start_SPU failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
            return 1;
            }
    #else
        if ( engine_start( &e , nr_runners ) != 0 ) {
            printf("main: engine_start failed with engine_err=%i.\n",engine_err);
            errs_dump(stdout);
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
            errs_dump(stdout);
            return 1;
            }
        #ifdef CELL
            toc_step = __mftb();
        #else
            toc_step = getticks();
        #endif
        
        /* Check virtual/local ids. */
        /* for ( cid = 0 ; cid < e.s.nr_cells ; cid++ )
            for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ )
                if ( e.s.cells[cid].parts[pid].id != e.s.cells[cid].parts[pid].vid )
                    printf( "main: inconsistent particle id/vid (%i/%i)!\n",
                        e.s.cells[cid].parts[pid].id, e.s.cells[cid].parts[pid].vid ); */

        /* Verify integrity of partlist. */
        /* for ( k = 0 ; k < nr_mols*3 ; k++ )
            if ( e.s.partlist[k]->id != k )
                printf( "main: inconsistent particle id/partlist (%i/%i)!\n", e.s.partlist[k]->id, k );
        fflush(stdout); */
        
        /* printf( "main: position of part %i is [ %23.16e , %23.16e , %23.16e ].\n" ,
            41140, e.s.partlist[41140]->x[0], e.s.partlist[41140]->x[1], e.s.partlist[41140]->x[2] );
        printf( "main: velocity of part %i is [ %23.16e , %23.16e , %23.16e ].\n" ,
            41140, e.s.partlist[41140]->v[0], e.s.partlist[41140]->v[1], e.s.partlist[41140]->v[2] );
        printf( "main: force on part %i is [ %23.16e , %23.16e , %23.16e ].\n" ,
            41140, e.s.partlist[41140]->f[0], e.s.partlist[41140]->f[1], e.s.partlist[41140]->f[2] ); */
            
        /* Check the max verlet_nrpairs. */
        /* if ( e.s.verlet_nrpairs != NULL ) {
            j = 0;
            for ( k = 0 ; k < e.s.nr_parts ; k++ )
                if ( e.s.verlet_nrpairs[k] > j )
                    j = e.s.verlet_nrpairs[k];
            printf( "main: max nr_pairs is %i.\n" , j );
            } */
            
        // shake the water molecules
        #pragma omp parallel for schedule(dynamic,100), private(k,p_O,p_H1,p_H2,c_O,c_H1,c_H2,new_O,new_H1,new_H2,old_O,old_H1,old_H2,v_OH1,v_OH2,v_HH,d_OH1,lambda,d_OH2,d_HH)
        for ( j = 0 ; j < nr_mols ; j++ ) {
        
            // unwrap the data
            p_O = e.s.partlist[j*3]; c_O = e.s.celllist[j*3];
            p_H1 = e.s.partlist[j*3+1]; c_H1 = e.s.celllist[j*3+1];
            p_H2 = e.s.partlist[j*3+2]; c_H2 = e.s.celllist[j*3+2];
            for ( k = 0 ; k < 3 ; k++ ) {
                new_O[k] = p_O->x[k] + c_O->origin[k];
                new_H1[k] = p_H1->x[k] + c_H1->origin[k];
                new_H2[k] = p_H2->x[k] + c_H2->origin[k];
                old_O[k] = new_O[k] - e.dt * p_O->v[k];
                if ( new_H1[k] - new_O[k] > e.s.dim[k] * 0.5 )
                    new_H1[k] -= e.s.dim[k];
                else if ( new_H1[k] - new_O[k] < -e.s.dim[k] * 0.5 )
                    new_H1[k] += e.s.dim[k];
                old_H1[k] = new_H1[k] - e.dt * p_H1->v[k];
                if ( new_H2[k] - new_O[k] > e.s.dim[k] * 0.5 )
                    new_H2[k] -= e.s.dim[k];
                else if ( new_H2[k] - new_O[k] < -e.s.dim[k] * 0.5 )
                    new_H2[k] += e.s.dim[k];
                old_H2[k] = new_H2[k] - e.dt * p_H2->v[k];
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
                if ( fabs( d_OH1 - 0.1*0.1 ) < 10*FPTYPE_EPSILON &&
                    fabs( d_OH2 - 0.1*0.1 ) < 10*FPTYPE_EPSILON &&  
                    fabs( d_HH - 0.1633*0.1633 ) < 10*FPTYPE_EPSILON )
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
                
            } // shake molecules
            
        // re-shuffle the space just to be sure (only if not verlet)...
        if ( !( e.flags & engine_flag_verlet ) )
            if ( space_shuffle( &e.s ) < 0 ) {
                printf("main: space_shuffle failed with space_err=%i.\n",space_err);
                errs_dump(stdout);
                return 1;
                }
        #ifdef CELL
            toc_shake = __mftb();
        #else
            toc_shake = getticks();
        #endif
            
            
        // get the total COM-velocities, ekin and epot
        vcom_tot[0] = 0.0; vcom_tot[1] = 0.0; vcom_tot[2] = 0.0;
        ekin = 0.0; epot = e.s.epot;
        for ( j = 0 ; j < nr_mols ; j++ ) {
            for ( k = 0 ; k < 3 ; k++ ) {
                vcom[k] = ( e.s.partlist[j*3]->v[k] * 15.9994 +
                    e.s.partlist[j*3+1]->v[k] * 1.00794 +
                    e.s.partlist[j*3+2]->v[k] * 1.00794 ) / 1.801528e+1;
                vcom_tot[k] += vcom[k];
                }
            ekin += 9.00764 * ( vcom[0]*vcom[0] + vcom[1]*vcom[1] + vcom[2]*vcom[2] );
            }
        for ( k = 0 ; k < 3 ; k++ )
            vcom_tot[k] /= nr_mols * 1.801528e+1;
        // printf("main: vcom_tot is [ %e , %e , %e ].\n",vcom_tot[0],vcom_tot[1],vcom_tot[2]); fflush(stdout);
                
        // compute the temperature and scaling
        temp = ekin / ( 1.5 * 6.022045E23 * 1.380662E-26 * nr_mols );
        w = sqrt( 1.0 + 0.1 * ( Temp / temp - 1.0 ) );

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
            
        // re-compute the potential energy.
        /* ekin = 0.0;
        for ( cid = 0 ; cid < e.s.nr_cells ; cid++ ) {
            for ( pid = 0 ; pid < e.s.cells[cid].count ; pid++ ) {
                for ( v2 = 0.0 , k = 0 ; k < 3 ; k++ )
                    v2 += e.s.cells[cid].parts[pid].v[k] * e.s.cells[cid].parts[pid].v[k];
                ekin += 0.5 * e.types[ e.s.cells[cid].parts[pid].type ].mass * v2;
                }
            } */
            
        #ifdef CELL
            toc_temp = __mftb();
        #else
            toc_temp = getticks();
        #endif
        printf("%i %e %e %e %i %i %.3f %.3f %.3f %.3f ms\n",
            e.time,epot,ekin,temp,e.s.nr_swaps,e.s.nr_stalls,
                (double)(toc_temp-tic) * 1000 / CPU_TPS,
                (double)(toc_step-tic) * 1000 / CPU_TPS,
                (double)(toc_shake-toc_step) * 1000 / CPU_TPS,
                (double)(toc_temp-toc_shake) * 1000 / CPU_TPS);
        fflush(stdout);
        
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
