/*******************************************************************************
 * This file is part of mdcore.
 * Coypright (c) 2010 Pedro Gonnet (pedro.gonnet@durham.ac.uk)
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
    #ifdef FPTYPE_DOUBLE
        /** The default basic type is double. */
        typedef double FPTYPE;
        #define FPTYPE_EPSILON DBL_EPSILON
        #ifndef FPTYPE_DOUBLE
            #define FPTYPE_DOUBLE
        #endif
        #define FPTYPE_ONE 1.0
        #define FPTYPE_TWO 2.0
        #define FPTYPE_HALF 0.5
        #define FPTYPE_ZERO 0.0
        #define FPTYPE_SQRT sqrt
        #define FPTYPE_FMAX fmax
        #define FPTYPE_FMIN fmin
        #define FPTYPE_FABS fabs
        #define FPTYPE_LOG log
        #define FPTYPE_COPYSIGN copysign
    #else
        /** The basic type is set to float. */
        typedef float FPTYPE;
        #define FPTYPE_EPSILON FLT_EPSILON
        #define FPTYPE_ONE 1.0f
        #define FPTYPE_ZERO 0.0f
        #define FPTYPE_TWO 2.0f
        #define FPTYPE_HALF 0.5f
        #define FPTYPE_SQRT sqrtf
        #define FPTYPE_FMAX fmaxf
        #define FPTYPE_FMIN fminf
        #define FPTYPE_FABS fabsf
        #define FPTYPE_LOG logf
        #define FPTYPE_COPYSIGN copysignf
        #ifndef FPTYPE_SINGLE
            #define FPTYPE_SINGLE
        #endif
    #endif
    #define FPTYPE_DEFINED
#endif


/* Get the inlining right. */
#ifndef INLINE
# if __GNUC__ && !__GNUC_STDC_INLINE__
#  define INLINE extern inline
# else
#  define INLINE inline
# endif
#endif

    
/* Define some macros for single/double precision vector operations. */
#if defined(FPTYPE_SINGLE)
    #if defined(__AVX__)
        #define VEC_SINGLE
        #define VEC_SIZE 8
        #define VEC_ALIGN 32
        #define VECTORIZE
        #define VEC_TYPE __m256
        #define VEC_LOAD(a) _mm256_load_ps(a)
        #define VEC_SET1(a) _mm256_set1_ps(a)
        #define VEC_SET(a,b,c,d,e,f,g,h) _mm256_set_ps(h,g,f,e,d,c,b,a)
    #elif ( defined(__SSE__) || defined(__ALTIVEC__) )
        #define VEC_SINGLE
        #define VEC_SIZE 4
        #define VEC_ALIGN 16
        #define VEC_TYPE __m128
        #define VEC_LOAD(a) _mm_load_ps(a)
        #define VEC_SET1(a) _mm_set1_ps(a)
        #define VEC_SET(a,b,c,d) _mm_set_ps(d,c,b,a)
        #define VECTORIZE
    #endif
#else
    #if defined(__AVX__)
        #define VEC_DOUBLE
        #define VEC_SIZE 4
        #define VEC_ALIGN 32
        #define VEC_TYPE __m256d
        #define VEC_LOAD(a) _mm256_load_pd(a)
        #define VEC_SET1(a) _mm256_set1_pd(a)
        #define VEC_SET(a,b,c,d) _mm256_set_pd(d,c,b,a)
        #define VECTORIZE
    #elif defined(__SSE2__)
        #define VEC_DOUBLE
        #define VEC_SIZE 4
        #define VEC_ALIGN 16
        #define VEC_TYPE __m128d[2]
        #define VEC_LOAD(a) _mm_load_pd(a)
        #define VEC_SET1(a) _mm_set1_pd(a)
        #define VEC_SET(a,b) _mm_set_pd(b,a)
        #define VECTORIZE
    #endif
#endif


/* Get headers for intrinsic functions. */
#include <immintrin.h>


/** Macro to easily define vector types. */
#define vector(elcount, type)  __attribute__((vector_size((elcount)*sizeof(type)))) type


/* Some extra functions function for Alti-Vec instruction set. */
#ifdef __ALTIVEC__
    #include <altivec.h>
    __attribute__ ((always_inline)) INLINE vector float vec_sqrt( vector float a ) {
        vector float z = ( vector float ){ 0.0f };
        vector float estimate = vec_rsqrte( a );
        vector float estimateSquared = vec_madd( estimate, estimate, z );
        vector float halfEstimate = vec_madd( estimate, (vector float){0.5}, z );
        return vec_madd( a, vec_madd( vec_nmsub( a, estimateSquared, (vector float){1.0} ), halfEstimate, estimate ), z);
        }
    /* inline static vector float vec_load4 ( float a , float b , float c , float d ) {
        return vec_mergeh( vec_mergeh( vec_promote(a,0) , vec_promote(c,0) ) , vec_mergeh( vec_promote(b,0) , vec_promote(d,0) ) );
        } */
    #define vec_load4(a,b,c,d) vec_mergeh( vec_mergeh( vec_promote((a),0) , vec_promote((c),0) ) , vec_mergeh( vec_promote((b),0) , vec_promote((d),0) ) )
    #define vec_mul(a,b) vec_madd((a),(b),(vector float){0.0f})
#endif


/**
 * @brief Inlined function to compute the distance^2 between two vectors.
 *
 * @param x1 The first vector.
 * @param x2 The second vector.
 * @param dx An array in which @c x1 - @c x2 will be stored.
 *
 * @return The Euclidian distance squared between @c x1 and @c x2.
 *
 * Depending on the processor features, this function will use
 * SSE registers and horizontal adds.
 */
 
__attribute__ ((always_inline)) INLINE FPTYPE fptype_r2 ( FPTYPE *x1 , FPTYPE *x2 , FPTYPE *dx ) {

#if defined(VECTORIZE) && defined(FPTYPE_SINGLE) && defined(__SSE4_1__)
    union {
        vector(4,float) v;
        float f[4];
        } a, b, c, d;
        
    /* Load x1 and x2 into a and b. */
    a.v = _mm_load_ps( x1 );
    b.v = _mm_load_ps( x2 );
    
    /* Compute the difference and store in dx. */
    c.v = a.v - b.v;
    _mm_store_ps( dx , c.v );
    
    /* Use the built-in dot-product instruction. */
    d.v = _mm_dp_ps( c.v , c.v , 0x71 );
    
    /* Return the sum of squares. */
    return d.f[0];
#elif defined(VECTORIZE) && defined(FPTYPE_SINGLE) && defined(__SSE3__)
    union {
        vector(4,float) v;
        float f[4];
        } a, b, c, d;
        
    /* Load x1 and x2 into a and b. */
    a.v = _mm_load_ps( x1 );
    b.v = _mm_load_ps( x2 );
    
    /* Compute the difference and store in dx. */
    c.v = a.v - b.v;
    _mm_store_ps( dx , c.v );
    
    /* Square the entries (use a different register so that c can be stored). */
    d.v = c.v * c.v;
    
    /* Add horizontally twice to get the sum of the four entries
       in the lowest float. */
    d.v = _mm_hadd_ps( d.v , d.v );
    d.v = _mm_hadd_ps( d.v , d.v );
    
    /* Return the sum of squares. */
    return d.f[0];
#elif defined(VECTORIZE) && defined(FPTYPE_DOUBLE) && defined(__AVX__)
    union {
        __m256d v;
        double f[4];
        } a, b, c, d;
        
    /* Load x1 and x2 into a and b. */
    a.v = _mm256_load_pd( x1 );
    b.v = _mm256_load_pd( x2 );
    
    /* Compute the difference and store in dx. */
    c.v = a.v - b.v;
    _mm256_store_pd( dx , c.v );
    
    /* Square the entries (use a different register so that c can be stored). */
    d.v = c.v * c.v;
    
    /* Add horizontally twice to get the sum of the four entries
       in the lowest double. */
    d.v = _mm256_hadd_pd( d.v , d.v );
    
    /* Return the sum of squares. */
    return d.f[0] + d.f[2];
#elif defined(VECTORIZE) && defined(FPTYPE_DOUBLE) && defined(__SSE4_1__)
    union {
        vector(2,double) v;
        double f[2];
        } a1, a2, b1, b2, c1, c2, d1;
        
    /* Load x1 and x2 into a and b. */
    a1.v = _mm_load_pd( x1 );
    b1.v = _mm_load_pd( x2 );
    a2.v = _mm_load_pd( &x1[2] );
    b2.v = _mm_load_pd( &x2[2] );
    
    /* Compute the difference and store in dx. */
    c1.v = a1.v - b1.v;
    c2.v = a2.v - b2.v;
    _mm_store_pd( dx , c1.v );
    _mm_store_pd( &dx[2] , c2.v );
    
    /* Use the built-in dot-product instruction. */
    d1.v = _mm_dp_pd( c1.v , c1.v , 0x31 ) + c2.v * c2.v;
    
    /* Return the sum of squares. */
    return d1.f[0];
#elif defined(VECTORIZE) && defined(FPTYPE_DOUBLE) && defined(__SSE3__)
    union {
        vector(2,double) v;
        double f[2];
        } a1, a2, b1, b2, c1, c2, d1, d2;
        
    /* Load x1 and x2 into a and b. */
    a1.v = _mm_load_pd( x1 );
    b1.v = _mm_load_pd( x2 );
    a2.v = _mm_load_pd( &x1[2] );
    b2.v = _mm_load_pd( &x2[2] );
    
    /* Compute the difference and store in dx. */
    c1.v = a1.v - b1.v;
    c2.v = a2.v - b2.v;
    _mm_store_pd( dx , c1.v );
    _mm_store_pd( &dx[2] , c2.v );
    
    /* Square the entries (use a different register so that c can be stored). */
    d1.v = c1.v * c1.v;
    d2.v = c2.v * c2.v;
    
    /* Add horizontally twice to get the sum of the four entries
       in the lowest double. */
    d1.v = _mm_hadd_pd( d1.v , d2.v );
    d1.v = _mm_hadd_pd( d1.v , d1.v );
    
    /* Return the sum of squares. */
    return d1.f[0];
#else
    dx[0] = x1[0] - x2[0];
    dx[1] = x1[1] - x2[1];
    dx[2] = x1[2] - x2[2];
    return dx[0]*dx[0] + dx[1]*dx[1] + dx[2]*dx[2];
#endif

    }
    

/**
 * @brief Inlined function to compute the dot product of two vectors.
 *
 * @param x The firstvector.
 * @param x2 The second vector.
 *
 * @return The dot product of @c x1 and @c x2.
 *
 * Depending on the processor features, this function will use
 * SSE registers and horizontal adds.
 */
 
__attribute__ ((always_inline)) INLINE FPTYPE fptype_dprod ( FPTYPE *x1 , FPTYPE *x2 ) {

#if defined(VECTORIZE) && defined(FPTYPE_SINGLE) && defined(__SSE4_1__)
    union {
        vector(4,float) v;
        float f[4];
        } a, b, d;
        
    /* Load x1 and x2 into a and b. */
    a.v = _mm_load_ps( x1 );
    b.v = _mm_load_ps( x2 );
    
    /* Use the built-in dot-product instruction. */
    d.v = _mm_dp_ps( a.v , b.v , 0x71 );
    
    /* Return the sum of squares. */
    return d.f[0];
#elif defined(VECTORIZE) && defined(FPTYPE_SINGLE) && defined(__SSE3__)
    union {
        vector(4,float) v;
        float f[4];
        } a, b, d;
        
    /* Load x1 and x2 into a and b. */
    a.v = _mm_load_ps( x1 );
    b.v = _mm_load_ps( x2 );
    
    /* Square the entries. */
    d.v = a.v * b.v;
    
    /* Add horizontally twice to get the sum of the four entries
       in the lowest float. */
    d.v = _mm_hadd_ps( d.v , d.v );
    d.v = _mm_hadd_ps( d.v , d.v );
    
    /* Return the sum of squares. */
    return d.f[0];
#elif defined(VECTORIZE) && defined(FPTYPE_DOUBLE) && defined(__AVX__)
    union {
        __m256d v;
        double f[4];
        } a, b, d;
        
    /* Load x1 and x2 into a and b. */
    a.v = _mm256_load_pd( x1 );
    b.v = _mm256_load_pd( x2 );
    
    /* Square the entries (use a different register so that c can be stored). */
    d.v = a.v * b.v;
    
    /* Add horizontally. */
    d.v = _mm256_hadd_pd( d.v , d.v );
    
    /* Return the sum of squares. */
    return d.f[0] + d.f[2];
#elif defined(VECTORIZE) && defined(FPTYPE_DOUBLE) && defined(__SSE4_1__)
    union {
        vector(2,double) v;
        double f[2];
        } a1, a2, b1, b2, d1;
        
    /* Load x1 and x2 into a and b. */
    a1.v = _mm_load_pd( x1 );
    b1.v = _mm_load_pd( x2 );
    a2.v = _mm_load_pd( &x1[2] );
    b2.v = _mm_load_pd( &x2[2] );
    
    /* Use the built-in dot-product instruction. */
    d1.v = _mm_dp_pd( a1.v , b1.v , 0x31 ) + a2.v * b2.v;
    
    /* Return the sum of squares. */
    return d1.f[0];
#elif defined(VECTORIZE) && defined(FPTYPE_DOUBLE) && defined(__SSE3__)
    union {
        vector(2,double) v;
        double f[2];
        } a1, a2, b1, b2, c1, c2, d1, d2;
        
    /* Load x1 and x2 into a and b. */
    a1.v = _mm_load_pd( x1 );
    b1.v = _mm_load_pd( x2 );
    a2.v = _mm_load_pd( &x1[2] );
    b2.v = _mm_load_pd( &x2[2] );
    
    /* Square the entries (use a different register so that c can be stored). */
    d1.v = a1.v * b1.v;
    d2.v = a2.v * b2.v;
    
    /* Add horizontally twice to get the sum of the four entries
       in the lowest double. */
    d1.v = _mm_hadd_pd( d1.v , d2.v );
    d1.v = _mm_hadd_pd( d1.v , d1.v );
    
    /* Return the sum of squares. */
    return d1.f[0];
#else
    return x1[0]*x2[0] + x1[1]*x2[1] + x1[2]*x2[2];
#endif

    }
    

