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

/* Local includes. */
#include "fptype.h"

/* runner error codes */
#define runner_err_ok                    0
#define runner_err_null                  -1
#define runner_err_malloc                -2
#define runner_err_space                 -3
#define runner_err_pthread               -4
#define runner_err_engine                -5
#define runner_err_spe                   -6
#define runner_err_mfc                   -7
#define runner_err_unavail               -8


/* some constants */
/** Maximum number of cellpairs to get from space_getpair. */
#define runner_bitesize                  3

/** Number of particles to request per call to space_getverlet. */
#define runner_verlet_bitesize           200

#ifdef CELL
    /** Length of the cell pair queue between the PPU and the SPU. */
    #define runner_qlen                  6
#endif


/* the last error */
extern int runner_err;

/* the runner structure */
struct runner {

    /* the engine with which i am associated */
    struct engine *e;
    
    /* this runner's id */
    int id;
    
    /* my thread */
    pthread_t thread;
    
    #ifdef CELL
    
        /* the SPE context */
        spe_context_ptr_t spe;
        pthread_t spe_thread;
        
        /* the (re-)entry point */
        unsigned int entry;
        
        /* the initialization data */
        void *data;
        
        /* the compacted cell list */
        struct celldata *celldata;
        
    #endif
    
    /** ID of the last error on this runner. */
    int err;
    
    /** Accumulated potential energy by this runner. */
    double epot;
    
    };
    
#ifdef CELL
    struct celldata {
        int ni;
        unsigned long long ai;
        };
#endif

/* associated functions */
int runner_init ( struct runner *r , struct engine *e , int id );
int runner_init_SPU ( struct runner *r , struct engine *e , int id );
int runner_run_pairs ( struct runner *r );
int runner_run_tuples ( struct runner *r );
int runner_dopair_unsorted ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift );
int runner_dopair_unsorted_ee ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift );
int runner_dopair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift );
int runner_dopair_ee ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift );
int runner_dopair_verlet ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift , struct cellpair *cp );
int runner_dopair_verlet2 ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *pshift , struct cellpair *cp );
int runner_verlet_eval ( struct runner *r , struct cell *c , FPTYPE *f_out );
int runner_run_verlet ( struct runner *r );
