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


/* engine error codes */
#define engine_err_ok                    0
#define engine_err_null                  -1
#define engine_err_malloc                -2
#define engine_err_space                 -3
#define engine_err_pthread               -4
#define engine_err_runner                -5
#define engine_err_range                 -6
#define engine_err_cell                  -7
#define engine_err_domain                -8
#define engine_err_nompi                 -9
#define engine_err_mpi                   -10
#define engine_err_bond                  -11
#define engine_err_angle                 -12


/* some constants */
#define engine_flag_none                 0
#define engine_flag_tuples               1
#define engine_flag_static               2
#define engine_flag_localparts           4
#define engine_flag_GPU                  8
#define engine_flag_explepot             16
#define engine_flag_verlet               32
#define engine_flag_verlet_pairwise      64
#define engine_flag_affinity             128
#define engine_flag_prefetch             256
#define engine_flag_verlet_pairwise2     512
#define engine_flag_partlist             1024
#define engine_flag_unsorted             2048
#define engine_flag_mpi                  4096

#define engine_bonds_chunk               100
#define engine_angles_chunk              100


/** ID of the last error. */
extern int engine_err;


/** 
 * The #engine structure. 
 */
struct engine {

    /** Some flags controlling how this engine works. */
    unsigned int flags;

    /** The space on which to work */
    struct space s;
    
    /** Time variables */
    int time;
    double dt;
    
    /** What is the maximum nr of types? */
    int max_type;
    
    /** The particle types. */
    struct part_type *types;
    
    /** The interaction matrix */
    struct potential **p, **p_bond, **p_angle;
    
    /** The explicit electrostatic potential. */
    struct potential *ep;
    
    /** Mutexes, conditions and counters for the barrier */
	pthread_mutex_t barrier_mutex;
	pthread_cond_t barrier_cond;
	pthread_cond_t done_cond;
    int barrier_count;
    
    /** Nr of runners */
    int nr_runners;
    
    /** The runners */
    struct runner *runners;
    
    /** The ID of the computational node we are on. */
    int nodeID;
    int nr_nodes;
    
    /** Lists of cells to exchange with other nodes. */
    struct engine_comm *send, *recv;
    
    /** List of bonds. */
    struct bond *bonds;
    
    /** Nr. of bonds. */
    int nr_bonds, bonds_size;
    
    /** List of angles. */
    struct angle *angles;
    
    /** Nr. of angles. */
    int nr_angles, angles_size, nr_anglepots, anglepots_size;
    
    };
    
    
/**
 * Structure storing which cells to send/receive to/from another node.
 */
struct engine_comm {

    /* Size and count of the cellids. */
    int count, size;
    
    int *cellid;
    
    };
    

/* associated functions */
int engine_addpot ( struct engine *e , struct potential *p , int i , int j );
int engine_addtype ( struct engine *e , int id , double mass , double charge , char *name , char *name2 );
int engine_angle_addpot ( struct engine *e , struct potential *p );
int engine_angle_add ( struct engine *e , int i , int j , int k , int pid );
int engine_angle_eval ( struct engine *e );
int engine_barrier ( struct engine *e );
int engine_bond_addpot ( struct engine *e , struct potential *p , int i , int j );
int engine_bond_add ( struct engine *e , int i , int j );
int engine_bond_eval ( struct engine *e );
int engine_dump_PSF ( struct engine *e , FILE *psf , FILE *pdb );
int engine_flush_ghosts ( struct engine *e );
int engine_flush ( struct engine *e );
int engine_init ( struct engine *e , const double *origin , const double *dim , double cutoff , unsigned int period , int max_type , unsigned int flags );
int engine_load_ghosts ( struct engine *e , double *x , double *v , int *type , int *vid , double *q , unsigned int *flags , int N );
int engine_load ( struct engine *e , double *x , double *v , int *type , int *vid , double *charge , unsigned int *flags , int N );
int engine_setexplepot ( struct engine *e , struct potential *ep );
int engine_split_bisect ( struct engine *e , int N );
int engine_split ( struct engine *e );
int engine_start_SPU ( struct engine *e , int nr_runners );
int engine_start ( struct engine *e , int nr_runners );
int engine_step ( struct engine *e );
int engine_unload_marked ( struct engine *e , double *x , double *v , int *type , int *vid , double *q , unsigned int *flags , double *epot , int N );
int engine_unload_strays ( struct engine *e , double *x , double *v , int *type , int *vid , double *q , unsigned int *flags , double *epot , int N );
int engine_unload ( struct engine *e , double *x , double *v , int *type , int *vid , double *charge , unsigned int *flags , double *epot , int N );
#ifdef HAVE_MPI
    int engine_exchange ( struct engine *e , MPI_Comm comm );
#endif
