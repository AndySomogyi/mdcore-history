/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2012 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
#define cuda_maxparts                       512
#define cuda_maxdiags                       352
#define cuda_ndiags                         ( ( (cuda_maxdiags - 1) * cuda_maxdiags ) / 2 )
#define cuda_frame                          32
#define cuda_maxpots                        100
#define max_fingers                         1
#define cuda_fifo_size                      4
#define cuda_maxblocks                      64
#define cuda_memcpy_chunk                   6
#define cuda_sum_chunk                      3


/* Some flags that control optional behaviour */
// #define TIMERS
// #define PARTS_TEX
// #define PARTS_LOCAL
// #define FORCES_LOCAL


/** Timers for the cuda parts. */
enum {
    tid_mutex = 0,
    tid_memcpy,
    tid_update,
    tid_pack,
    tid_sort,
    tid_pair,
    tid_self,
    tid_potential,
    tid_potential4,
    tid_total,
    tid_count
    };
    

/* Timer functions. */
#ifdef TIMERS
    #define TIMER_TIC_ND if ( threadIdx.x == 0 ) tic = clock();
    #define TIMER_TOC_ND(tid) toc = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc > tic ) ? (toc - tic) : ( toc + (0xffffffff - tic) ) );
    #define TIMER_TIC clock_t tic; if ( threadIdx.x == 0 ) tic = clock();
    #define TIMER_TOC(tid) clock_t toc = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc > tic ) ? (toc - tic) : ( toc + (0xffffffff - tic) ) );
    #define TIMER_TIC2_ND if ( threadIdx.x == 0 ) tic2 = clock();
    #define TIMER_TOC2_ND(tid) toc2 = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc2 > tic2 ) ? (toc2 - tic2) : ( toc2 + (0xffffffff - tic2) ) );
    #define TIMER_TIC2 clock_t tic2; if ( threadIdx.x == 0 ) tic2 = clock();
    #define TIMER_TOC2(tid) clock_t toc2 = clock(); if ( threadIdx.x == 0 ) atomicAdd( &cuda_timers[tid] , ( toc2 > tic2 ) ? (toc2 - tic2) : ( toc2 + (0xffffffff - tic2) ) );
#else
    #define TIMER_TIC_ND
    #define TIMER_TOC_ND(tid)
    #define TIMER_TIC
    #define TIMER_TOC(tid)
    #define TIMER_TIC2
    #define TIMER_TOC2(tid)
#endif


/** Struct for each cellpair (compact). */
struct cellpair_cuda {

    /** Indices of the cells involved. */
    int i, j;
    
    /** Relative shift between cell centres. */
    float shift[3];
    
    };
    
    
/** Struct for the round-robin fifo queues. */
struct fifo_cuda {
    
    unsigned int data[ cuda_fifo_size ];
    
    int first, last, count;
    
    };
    
    


