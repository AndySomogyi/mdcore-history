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
#define engine_err_reader                -13
#define engine_err_psf                   -14
#define engine_err_pdb                   -15
#define engine_err_cpf                   -16
#define engine_err_potential             -17
#define engine_err_exclusion             -18


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
#define engine_flag_parbonded            8192
#define engine_flag_async                16384

#define engine_bonds_chunk               100
#define engine_angles_chunk              100
#define engine_rigids_chunk              100
#define engine_dihedrals_chunk           100
#define engine_exclusions_chunk          100

#define engine_bonded_maxnrthreads       16
#define engine_bonded_nrthreads          ((omp_get_num_threads()<engine_bonded_maxnrthreads)?omp_get_num_threads():engine_bonded_maxnrthreads)

#define engine_nr_timers                 10

/** Timer IDs. */
enum {
    engine_timer_step = 0,
    engine_timer_prepare,
    engine_timer_verlet,
    engine_timer_exchange1,
    engine_timer_nonbond,
    engine_timer_bonded,
    engine_timer_advance,
    engine_timer_rigid,
    engine_timer_exchange2
    };


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
    int nr_types;
    
    /** The particle types. */
    struct part_type *types;
    
    /** The interaction matrix */
    struct potential **p, **p_bond, **p_angle, **p_dihedral;
    
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
    
    /** List of exclusions. */
    struct exclusion *exclusions;
    
    /** Nr. of exclusions. */
    int nr_exclusions, exclusions_size;
    
    /** List of rigid bodies. */
    struct rigid *rigids;
    
    /** List linking parts to rigids. */
    int *part2rigid;
    
    /** Nr. of rigids. */
    int nr_rigids, rigids_size, nr_constr, rigids_local, rigids_semilocal;
    
    /** Rigid solver tolerance. */
    double tol_rigid;
    
    /** List of angles. */
    struct angle *angles;
    
    /** Nr. of angles. */
    int nr_angles, angles_size, nr_anglepots, anglepots_size;
    
    /** List of dihedrals. */
    struct dihedral *dihedrals;
    
    /** Nr. of dihedrals. */
    int nr_dihedrals, dihedrals_size, nr_dihedralpots, dihedralpots_size;
    
    /** The Comm object for mpi. */
    #ifdef HAVE_MPI
        pthread_mutex_t xchg_mutex;
        pthread_cond_t xchg_cond;
        short int xchg_started, xchg_running;
        MPI_Comm comm;
        pthread_t thread_exchg;
    #endif
    
    /** Timers. */
    ticks timers[engine_nr_timers];
    
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
int engine_addtype ( struct engine *e , double mass , double charge , char *name , char *name2 );
int engine_advance ( struct engine *e );
int engine_angle_addpot ( struct engine *e , struct potential *p );
int engine_angle_add ( struct engine *e , int i , int j , int k , int pid );
int engine_angle_eval ( struct engine *e );
int engine_barrier ( struct engine *e );
int engine_bond_addpot ( struct engine *e , struct potential *p , int i , int j );
int engine_bond_add ( struct engine *e , int i , int j );
int engine_bond_eval ( struct engine *e );
int engine_bonded_eval ( struct engine *e );
int engine_dihedral_add ( struct engine *e , int i , int j , int k , int l , int pid );
int engine_dihedral_addpot ( struct engine *e , struct potential *p );
int engine_dihedral_eval ( struct engine *e );
int engine_dump_PSF ( struct engine *e , FILE *psf , FILE *pdb , char *excl[] , int nr_excl );
int engine_exclusion_add ( struct engine *e , int i , int j );
int engine_exclusion_eval ( struct engine *e );
int engine_exclusion_shrink ( struct engine *e );
int engine_flush_ghosts ( struct engine *e );
int engine_flush ( struct engine *e );
int engine_gettype ( struct engine *e , char *name );
int engine_gettype2 ( struct engine *e , char *name2 );
int engine_init ( struct engine *e , const double *origin , const double *dim , double L , double cutoff , unsigned int period , int max_type , unsigned int flags );
int engine_load_ghosts ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , int N );
int engine_load ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *charge , unsigned int *flags , int N );
int engine_nonbond_eval ( struct engine *e );
int engine_read_cpf ( struct engine *e , FILE *cpf , double kappa , double tol , int rigidH );
int engine_read_psf ( struct engine *e , FILE *psf , FILE *pdb );
int engine_read_xplor ( struct engine *e , FILE *xplor , double kappa , double tol , int rigidH );
int engine_rigid_add ( struct engine *e , int pid , int pjd , double d );
int engine_rigid_eval ( struct engine *e );
int engine_rigid_sort ( struct engine *e );
int engine_setexplepot ( struct engine *e , struct potential *ep );
int engine_split_bisect ( struct engine *e , int N );
int engine_split ( struct engine *e );
int engine_start_SPU ( struct engine *e , int nr_runners );
int engine_start ( struct engine *e , int nr_runners );
int engine_step ( struct engine *e );
int engine_timers_reset ( struct engine *e );
int engine_unload_marked ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , double *epot , int N );
int engine_unload_strays ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *q , unsigned int *flags , double *epot , int N );
int engine_unload ( struct engine *e , double *x , double *v , int *type , int *pid , int *vid , double *charge , unsigned int *flags , double *epot , int N );
int engine_verlet_update ( struct engine *e );
#ifdef HAVE_MPI
    int engine_init_mpi ( struct engine *e , const double *origin , const double *dim , double L , double cutoff , unsigned int period , int max_type , unsigned int flags , MPI_Comm comm , int rank );
    int engine_exchange ( struct engine *e );
    int engine_exchange_async ( struct engine *e );
    int engine_exchange_async_run ( struct engine *e );
    int engine_exchange_incomming ( struct engine *e );
    int engine_exchange_wait ( struct engine *e );
#endif
