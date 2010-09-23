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


/* space error codes */
#define space_err_ok                    0
#define space_err_null                  -1
#define space_err_malloc                -2
#define space_err_cell                  -3
#define space_err_pthread               -4
#define space_err_range                 -5


/* some constants */
#define space_periodic_x                1
#define space_periodic_y                2
#define space_periodic_z                4
#define space_periodic_full             7

#define space_partlist_incr             100


/* some useful macros */
#define space_cellid(s,i,j,k)           (  ((i)*(s)->cdim[1] + (j)) * (s)->cdim[2] + (k) )


/* the last error */
extern int space_err;


/* the space structure */
struct space {

    /* real dimensions */
    double dim[3];
    
    /* location of origin */
    double origin[3];
    
    /* cell dimensions */
    int cdim[3];
    
    /* cell dimensions and their inverse */
    double h[3], ih[3];
    
    /* the cutoff and the cutoff squared */
    double cutoff, cutoff2;
    
    /* periodicities */
    unsigned int period;
    
    /* total nr of cells in this space */
    int nr_cells;
    
    /* the cells spanning the space */
    struct cell *cells;
    
    /* the total number of cell pairs */
    int nr_pairs;
    
    /* the cell pairs */
    struct cellpair *pairs;
    
    /* id of the next pair */
    int next_pair;
    
    /* the mutex for accessing the cell pairs */
    pthread_mutex_t cellpairs_mutex;
    pthread_cond_t cellpairs_avail;
    
    /* the taboo-list for collision avoidance */
    char *cells_taboo;
    char *cells_owner;
    
    /* count the number of swaps in every step */
    int nr_swaps, nr_stalls;
    
    struct part **partlist;
    struct cell **celllist;
    int nr_parts, size_parts;

    };
    
struct cellpair {

    /* indices of the cells */
    int i, j;
    
    /* relative shift */
    float shift[3];
    
    /* pointer to chain pairs together */
    struct cellpair *next;
    
    };


/* associated functions */
int space_init ( struct space *s , const double *origin , const double *dim , double cutoff , unsigned int period );
struct cellpair *space_getpair ( struct space *s , int owner , int count , struct cellpair *old , int *err , int wait );
int space_releasepair ( struct space *s , struct cellpair *p );
int space_shuffle ( struct space *s );
int space_addpart ( struct space *s , struct part *p , double *x );
int space_prepare ( struct space *s );
