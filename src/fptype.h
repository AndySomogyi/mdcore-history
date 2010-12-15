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

/* Global defines. */
#ifndef FPTYPE_DEFINED
    #ifdef FPTYPE_SINGLE
        /** The basic type was set to float. */
        typedef float FPTYPE;
        #define FPTYPE_EPSILON FLT_EPSILON
    #else
        /** The default basic type is double. */
        typedef double FPTYPE;
        #define FPTYPE_EPSILON DBL_EPSILON
        #define FPTYPE_DOUBLE
    #endif
    #define FPTYPE_DEFINED
#endif
