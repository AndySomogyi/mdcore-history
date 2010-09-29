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
#define runner_bitesize                  3
#ifdef CELL
    #define runner_qlen                  6
#else
    #define runner_qlen                  4
#endif
#define runner_maxparts                  400
#define runner_maxqstack                 100


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
    
    };
    
#ifdef CELL
    struct celldata {
        int ni;
        unsigned long long ai;
        };
#endif

/* associated functions */
int runner_init ( struct runner *r , struct engine *e , int id );
int runner_run ( struct runner *r );
int runner_run_tuples ( struct runner *r );
int runner_dopair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift );
int runner_sortedpair ( struct runner *r , struct cell *cell_i , struct cell *cell_j , FPTYPE *shift );
