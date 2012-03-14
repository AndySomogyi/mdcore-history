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


/* space error codes */
#define space_err_ok                    0
#define space_err_null                  -1
#define space_err_malloc                -2
#define space_err_cell                  -3
#define space_err_pthread               -4
#define space_err_range                 -5
#define space_err_maxpairs              -6


/* some constants */
#define space_periodic_none             0
#define space_periodic_x                1
#define space_periodic_y                2
#define space_periodic_z                4
#define space_periodic_full             7
#define space_periodic_ghost_x          8
#define space_periodic_ghost_y          16
#define space_periodic_ghost_z          32
#define space_periodic_ghost_full       56

#define space_partlist_incr             100

/** Maximum number of cells per tuple. */
#define space_maxtuples                 4

/** Maximum number of interactions per particle in the Verlet list. */
#define space_verlet_maxpairs           750


/* some useful macros */
/** Converts the index triplet (@c i, @c j, @c k) to the cell id in the
    #space @c s. */
#define space_cellid(s,i,j,k)           (  ((i)*(s)->cdim[1] + (j)) * (s)->cdim[2] + (k) )
/** Convert tuple ids into the pairid index. */
#define space_pairind(i,j)              ( space_maxtuples*(i) - (i)*((i)+1)/2 + (j) )

/** ID of the last error */
extern int space_err;


/** The space structure */
struct space {

    /** Real dimensions. */
    double dim[3];
    
    /** Location of origin. */
    double origin[3];
    
    /** Space dimension in cells. */
    int cdim[3];
    
    /** Cell edge lengths and their inverse. */
    double h[3], ih[3];
    
    /** The cutoff and the cutoff squared. */
    double cutoff, cutoff2;
    
    /** Periodicities. */
    unsigned int period;
    
    /** Total nr of cells in this space. */
    int nr_cells;
    
    /** IDs of real, ghost and marked cells. */
    int *cid_real, *cid_ghost, *cid_marked;
    int nr_real, nr_ghost, nr_marked;
    
    /** Array of cells spanning the space. */
    struct cell *cells;
    
    /** The total number of cell pairs. */
    int nr_pairs;
    
    /** Array of cell pairs. */
    struct cellpair *pairs;
    
    /** Id of the next unprocessed pair (for #space_getpair) */
    int next_pair;
    
    /** Array of cell tuples. */
    struct celltuple *tuples;
    
    /** The number of tuples. */
    int nr_tuples;
    
    /** The ID of the next free tuple or cell. */
    int next_tuple, next_cell;
    
    /** Mutex for accessing the cell pairs. */
    pthread_mutex_t cellpairs_mutex;
    
    /** Condition to wait for free cells on. */
    pthread_cond_t cellpairs_avail;
    
    /** Taboo-list for collision avoidance */
    char *cells_taboo;
    
    /** Id of #runner owning each cell. */
    char *cells_owner;
    
    /** Counter for the number of swaps in every step. */
    int nr_swaps, nr_stalls;
    
    /** Array of pointers to the individual parts, sorted by their ID. */
    struct part **partlist;
    
    /** Array of pointers to the #cell of individual parts, sorted by their ID. */
    struct cell **celllist;
    
    /** Number of parts in this space and size of the buffers partlist and celllist. */
    int nr_parts, size_parts;
    
    /** Data for the verlet list. */
    struct verlet_entry *verlet_list;
    FPTYPE *verlet_oldx;
    int *verlet_nrpairs;
    int verlet_size, verlet_next;
    int verlet_rebuild;
    pthread_mutex_t verlet_force_mutex;
    
    /** Potential energy collected by the space itself. */
    double epot;

    /** Pointers to device data for CUDA. */
    #ifdef HAVE_CUDA
        struct part_cuda *parts_cuda, *parts_cuda_local;
        int *counts_cuda, *counts_cuda_local, *ind_cuda, *ind_cuda_local;
        struct cellpair_cuda *pairs_cuda;
        int *taboo_cuda;
    #endif
    
    };
    
    
/** Struct for each cellpair (see #space_getpair). */
struct cellpair {

    /** Indices of the cells involved. */
    int i, j;
    
    /** Relative shift between cell centres. */
    FPTYPE shift[3];
    
    /** Pairwise Verlet stuff. */
    int size, count;
    short int *pairs;
    short int *nr_pairs;
    
    /** Pointer to chain pairs together. */
    struct cellpair *next;
    
    };
    
    
#ifdef HAVE_CUDA
/** Struct for each cellpair (compact). */
struct cellpair_cuda {

    /** Indices of the cells involved. */
    int i, j;
    
    /** Relative shift between cell centres. */
    float shift[3];
    
    };
#endif
    
    
/** Struct for groups of cellpairs. */
struct celltuple {

    /** IDs of the cells in this tuple. */
    int cellid[ space_maxtuples ];
    
    /** Nr. of cells in this tuple. */
    int n;
    
    /** Ids of the underlying pairs. */
    int pairid[ space_maxtuples * (space_maxtuples + 1) / 2 ];
    
    /* Buffer value to keep the size at 64 bytes for space_maxtuples==4. */
    int buff;
    
    };
    
    
/** Struct for Verlet list entries. */
struct verlet_entry {

    /** The particle. */
    struct part *p;
    
    /** The interaction potential. */
    struct potential *pot;
    
    /** The integer shift relative to this particle. */
    signed char shift[3];
    
    };


/* associated functions */
int space_init ( struct space *s , const double *origin , const double *dim , double L , double cutoff , unsigned int period );
struct cellpair *space_getpair ( struct space *s , int owner , int count , struct cellpair *old , int *err , int wait );
int space_releasepair ( struct space *s , int ci , int cj );
int space_shuffle ( struct space *s );
int space_shuffle_local ( struct space *s );
int space_addpart ( struct space *s , struct part *p , double *x );
int space_pairs_sort ( struct space *s );
int space_prepare ( struct space *s );
int space_maketuples ( struct space *s );
int space_gettuple ( struct space *s , struct celltuple **out , int wait );
int space_getpos ( struct space *s , int id , double *x );
int space_setpos ( struct space *s , int id , double *x );
int space_flush ( struct space *s );
int space_verlet_init ( struct space *s , int list_global );
int space_verlet_get ( struct space *s , int maxcount , int *from );
int space_verlet_force ( struct space *s , FPTYPE *f , double epot );
int space_flush_ghosts ( struct space *s );
int space_getcell ( struct space *s , struct cell **out );
