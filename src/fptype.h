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

/* Global defines. */
#ifndef FPTYPE_DEFINED
    #ifdef FPTYPE_SINGLE
        /** The basic type was set to float. */
        typedef float FPTYPE;
        #define FPTYPE_EPSILON FLT_EPSILON
        #define FPTYPE_ONE 1.0f
        #define FPTYPE_ZERO 0.0f
        #define FPTYPE_TWO 2.0f
        #define FPTYPE_SQRT sqrtf
        #define FPTYPE_FMAX fmaxf
        #define FPTYPE_FMIN fminf
    #else
        /** The default basic type is double. */
        typedef double FPTYPE;
        #define FPTYPE_EPSILON DBL_EPSILON
        #define FPTYPE_DOUBLE
        #define FPTYPE_ONE 1.0
        #define FPTYPE_TWO 2.0
        #define FPTYPE_ZERO 0.0
        #define FPTYPE_SQRT sqrt
        #define FPTYPE_FMAX fmax
        #define FPTYPE_FMIN fmin
    #endif
    #define FPTYPE_DEFINED
#endif

/* Define some macros for single/double precision vector operations. */
#if ( defined(__AVX__) && defined(FPTYPE_SINGLE) )
    #define VEC_SINGLE
    #define VEC_SIZE 8
    #define VECTORIZE
#elif ( (defined(__SSE__) || defined(__ALTIVEC__)) && defined(FPTYPE_SINGLE) )
    #define VEC_SINGLE
    #define VEC_SIZE 4
    #define VECTORIZE
#elif ( (defined(__SSE2__) || defined(__AVX__)) && defined(FPTYPE_DOUBLE) )
    #define VEC_DOUBLE
    #define VEC_SIZE 4
    #define VECTORIZE
#endif
