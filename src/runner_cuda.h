/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (gonnet@maths.ox.ac.uk)
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

/* Set the max number of parts for shared buffers. */
#define cuda_maxparts 160
#define cuda_ndiags ( ( (cuda_maxparts + 1) * cuda_maxparts ) / 2 )
#define cuda_frame 32
#define cuda_maxpots 100
#define max_fingers 3


/* Use textured or global potential data? */
#define USETEX 1
// #define EXPLPOT 1
#define PACK_PARTS 1
// #define USETEX_E 1


#ifdef PACK_PARTS
/** Reduced part struct for CUDA. */
struct part_cuda {

    /** Particle position */
    float x[3];
    
    /** Particle force */
    float f[3];
    
    /** particle type. */
    int type;
    
    /** particle charge. */
    #if defined(USETEX_E) || defined(EXPLPOT)
    float q;
    #endif
    
    };
#else
    #define part_cuda part
#endif

/* List of sorted particles. */
struct sortlist {

    /* The distance along the cell pair axis. */
    float d;
    
    /* The index of the particle. */
    int ind;
    
    };


/** Struct for each cellpair (compact). */
struct cellpair_cuda {

    /** Indices of the cells involved. */
    int i, j;
    
    /** Relative shift between cell centres. */
    float shift[3];
    
    };
    
    
/** Struct for each cellpair (compact). */
struct celltuple_cuda {

    /** Indices of the cells involved. */
    int i, j, k;
    
    /* The number of interactions in this tuple. */
    int nr_pairs;
    
    /* The bit-mask for the interactions. */
    char pairs[8];
    
    /** Relative shift between cell centres. */
    float shift_ij[3], shift_ik[3];
    
    };
    
    
    
    
    


