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


/* include some standard header files */
#include <stdlib.h>
#include <stdio.h>
#include <malloc.h>
#include <math.h>
#include <float.h>
#include <string.h>
#ifdef __SSE__
    #include <xmmintrin.h>
#endif
#ifdef __SSE2__
    #include <emmintrin.h>
#endif
#ifdef __ALTIVEC__
    #include <altivec.h>
    inline vector float vec_sqrt( vector float a ) {
        vector float z = ( vector float ){ 0.0f };
        vector float estimate = vec_rsqrte( a );
        vector float estimateSquared = vec_madd( estimate, estimate, z );
        vector float halfEstimate = vec_madd( estimate, (vector float){0.5}, z );
        return vec_madd( a, vec_madd( vec_nmsub( a, estimateSquared, (vector float){1.0} ), halfEstimate, estimate ), z);
        }
    inline vector float vec_load4 ( float a , float b , float c , float d ) {
        return vec_mergeh( vec_mergeh( vec_promote(a,0) , vec_promote(c,0) ) , vec_mergeh( vec_promote(b,0) , vec_promote(d,0) ) );
        }
    #define vec_mul(a,b) vec_madd((a),(b),(vector float){0.0f})
#endif

/* include local headers */
#include "errs.h"
#include "fptype.h"
#include "potential.h"


/** The last error */
int potential_err = potential_err_ok;


/* the error macro. */
#define error(id)				( potential_err = errs_register( id , potential_err_msg[-(id)] , __LINE__ , __FUNCTION__ , __FILE__ ) )

/* list of error messages. */
char *potential_err_msg[5] = {
	"Nothing bad happened.",
    "An unexpected NULL pointer was encountered.",
    "A call to malloc failed, probably due to insufficient memory.",
    "The requested value was out of bounds.",
    "Not yet implemented.",
	};
    
    
/** 
 * @brief Evaluates the given potential at a set of points (interpolated).
 *
 * @param p Pointer to an array of pointers to the #potentials to be evaluated.
 * @param r2 Pointer to an array of the radii at which the potentials
 *      are to be evaluated, squared.
 * @param e Pointer to an array of floating-point values in which to store the
 *      interaction energies.
 * @param f Pointer to an array of floating-point values in which to store the
 *      magnitude of the interaction forces.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p.
 *
 * Computes four single-precision interactions simultaneously using vectorized
 * instructions.
 * 
 * This function is only available if mdcore was compiled with SSE or AltiVec
 * and single precision! If @c mdcore was not compiled with SSE or AltiVec,
 * this function simply calls #potential_eval on each entry.
 */

void potential_eval_vec_4single ( struct potential *p[4] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f ) {

#if defined(__SSE__) && defined(FPTYPE_SINGLE)
    int j, k;
    union {
        __v4sf v;
        __m128i m;
        float f[4];
        int i[4];
        } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r, ind;
    float *data[4];
    
    /* Get r . */
    r.v = _mm_sqrt_ps( _mm_load_ps( r2 ) );
    
    /* compute the index */
    alpha0.v = _mm_setr_ps( p[0]->alpha[0] , p[1]->alpha[0] , p[2]->alpha[0] , p[3]->alpha[0] );
    alpha1.v = _mm_setr_ps( p[0]->alpha[1] , p[1]->alpha[1] , p[2]->alpha[1] , p[3]->alpha[1] );
    alpha2.v = _mm_setr_ps( p[0]->alpha[2] , p[1]->alpha[2] , p[2]->alpha[2] , p[3]->alpha[2] );
    ind.m = _mm_cvttps_epi32( _mm_max_ps( _mm_setzero_ps() , _mm_add_ps( alpha0.v , _mm_mul_ps( r.v , _mm_add_ps( alpha1.v , _mm_mul_ps( r.v , alpha2.v ) ) ) ) ) );
    
    /* get the table offset */
    for ( k = 0 ; k < 4 ; k++ )
        data[k] = &( p[k]->c[ ind.i[k] * potential_chunk ] );
    
    /* adjust x to the interval */
    mi.v = _mm_setr_ps( data[0][0] , data[1][0] , data[2][0] , data[3][0] );
    hi.v = _mm_setr_ps( data[0][1] , data[1][1] , data[2][1] , data[3][1] );
    x.v = _mm_mul_ps( _mm_sub_ps( r.v , mi.v ) , hi.v );
    
    /* compute the potential and its derivative */
    eff.v = _mm_setr_ps( data[0][2] , data[1][2] , data[2][2] , data[3][2] );
    c.v = _mm_setr_ps( data[0][3] , data[1][3] , data[2][3] , data[3][3] );
    ee.v = _mm_add_ps( _mm_mul_ps( eff.v , x.v ) , c.v );
    for ( j = 4 ; j < potential_chunk ; j++ ) {
        c.v = _mm_setr_ps( data[0][j] , data[1][j] , data[2][j] , data[3][j] );
        eff.v = _mm_add_ps( _mm_mul_ps( eff.v , x.v ) , ee.v );
        ee.v = _mm_add_ps( _mm_mul_ps( ee.v , x.v ) , c.v );
        }

    /* store the result */
    _mm_store_ps( e , ee.v );
    _mm_store_ps( f , _mm_mul_ps( eff.v , hi.v ) );
    
#elif defined(__ALTIVEC__) && defined(FPTYPE_SINGLE)
    int j, k;
    union {
        vector float v;
        float f[4];
        } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r;
    union {
        vector unsigned int v;
        unsigned int i[4];
        } ind;
    float *data[4];
    
    /* Get r . */
    r.v = vec_sqrt( *((vector float *)r2) );
    
    /* compute the index (vec_ctu maps negative floats to 0) */
    alpha0.v = vec_load4( p[0]->alpha[0] , p[1]->alpha[0] , p[2]->alpha[0] , p[3]->alpha[0] );
    alpha1.v = vec_load4( p[0]->alpha[1] , p[1]->alpha[1] , p[2]->alpha[1] , p[3]->alpha[1] );
    alpha2.v = vec_load4( p[0]->alpha[2] , p[1]->alpha[2] , p[2]->alpha[2] , p[3]->alpha[2] );
    ind.v = vec_ctu( vec_madd( r.v , vec_madd( r.v , alpha2.v , alpha1.v ) , alpha0.v ) , 0 );
    
    /* get the table offset */
    for ( k = 0 ; k < 4 ; k++ )
        data[k] = &( p[k]->c[ ind.i[k] * potential_chunk ] );
    
    /* adjust x to the interval */
    mi.v = vec_load4( data[0][0] , data[1][0] , data[2][0] , data[3][0] );
    hi.v = vec_load4( data[0][1] , data[1][1] , data[2][1] , data[3][1] );
    x.v = vec_mul( vec_sub( r.v , mi.v ) , hi.v );
    
    /* compute the potential and its derivative */
    eff.v = vec_load4( data[0][2] , data[1][2] , data[2][2] , data[3][2] );
    c.v = vec_load4( data[0][3] , data[1][3] , data[2][3] , data[3][3] );
    ee.v = vec_madd( eff.v , x.v , c.v );
    for ( j = 4 ; j < potential_chunk ; j++ ) {
        c.v = vec_load4( data[0][j] , data[1][j] , data[2][j] , data[3][j] );
        eff.v = vec_madd( eff.v , x.v , ee.v );
        ee.v = vec_madd( ee.v , x.v , c.v );
        }

    /* store the result */
    eff.v = vec_mul( eff.v , hi.v );
    memcpy( e , &ee , sizeof(vector float) );
    memcpy( f , &eff , sizeof(vector float) );
        
#else
    int k;
    for ( k = 0 ; k < 4 ; k++ )
        potential_eval( p[k] , r2[k] , &e[k] , &f[k] );
#endif
        
    }


/** 
 * @brief Evaluates the given potential at a set of points (interpolated).
 *
 * @param p Pointer to an array of pointers to the #potentials to be evaluated.
 * @param r2 Pointer to an array of the radii at which the potentials
 *      are to be evaluated, squared.
 * @param e Pointer to an array of floating-point values in which to store the
 *      interaction energies.
 * @param f Pointer to an array of floating-point values in which to store the
 *      magnitude of the interaction forces.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p.
 *
 * Computes eight single-precision interactions simultaneously using vectorized
 * instructions.
 * 
 * This function is only available if mdcore was compiled with SSE or AltiVec
 * and single precision! If @c mdcore was not compiled with SSE or AltiVec,
 * this function simply calls #potential_eval on each entry.
 */

void potential_eval_vec_8single ( struct potential *p[8] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f ) {

#if defined(__SSE__) && defined(FPTYPE_SINGLE)
    int j;
    union {
        __v4sf v;
        __m128i m;
        float f[4];
        int i[4];
        } alpha0_1, alpha1_1, alpha2_1, mi_1, hi_1, x_1, ee_1, eff_1, c_1, r_1, ind_1,
          alpha0_2, alpha1_2, alpha2_2, mi_2, hi_2, x_2, ee_2, eff_2, c_2, r_2, ind_2;
    float *data[8];
    
    /* Get r . */
    r_1.v = _mm_sqrt_ps( _mm_load_ps( &r2[0] ) );
    r_2.v = _mm_sqrt_ps( _mm_load_ps( &r2[4] ) );
    
    /* compute the index */
    alpha0_1.v = _mm_setr_ps( p[0]->alpha[0] , p[1]->alpha[0] , p[2]->alpha[0] , p[3]->alpha[0] );
    alpha1_1.v = _mm_setr_ps( p[0]->alpha[1] , p[1]->alpha[1] , p[2]->alpha[1] , p[3]->alpha[1] );
    alpha2_1.v = _mm_setr_ps( p[0]->alpha[2] , p[1]->alpha[2] , p[2]->alpha[2] , p[3]->alpha[2] );
    alpha0_2.v = _mm_setr_ps( p[4]->alpha[0] , p[5]->alpha[0] , p[6]->alpha[0] , p[7]->alpha[0] );
    alpha1_2.v = _mm_setr_ps( p[4]->alpha[1] , p[5]->alpha[1] , p[6]->alpha[1] , p[7]->alpha[1] );
    alpha2_2.v = _mm_setr_ps( p[4]->alpha[2] , p[5]->alpha[2] , p[6]->alpha[2] , p[7]->alpha[2] );
    ind_1.m = _mm_cvttps_epi32( _mm_max_ps( _mm_setzero_ps() , _mm_add_ps( alpha0_1.v , _mm_mul_ps( r_1.v , _mm_add_ps( alpha1_1.v , _mm_mul_ps( r_1.v , alpha2_1.v ) ) ) ) ) );
    ind_2.m = _mm_cvttps_epi32( _mm_max_ps( _mm_setzero_ps() , _mm_add_ps( alpha0_2.v , _mm_mul_ps( r_2.v , _mm_add_ps( alpha1_2.v , _mm_mul_ps( r_2.v , alpha2_2.v ) ) ) ) ) );
    
    /* get the table offset */
    data[0] = &( p[0]->c[ ind_1.i[0] * potential_chunk ] );
    data[1] = &( p[1]->c[ ind_1.i[1] * potential_chunk ] );
    data[2] = &( p[2]->c[ ind_1.i[2] * potential_chunk ] );
    data[3] = &( p[3]->c[ ind_1.i[3] * potential_chunk ] );
    data[4] = &( p[4]->c[ ind_2.i[0] * potential_chunk ] );
    data[5] = &( p[5]->c[ ind_2.i[1] * potential_chunk ] );
    data[6] = &( p[6]->c[ ind_2.i[2] * potential_chunk ] );
    data[7] = &( p[7]->c[ ind_2.i[3] * potential_chunk ] );
    
    /* adjust x to the interval */
    mi_1.v = _mm_setr_ps( data[0][0] , data[1][0] , data[2][0] , data[3][0] );
    hi_1.v = _mm_setr_ps( data[0][1] , data[1][1] , data[2][1] , data[3][1] );
    mi_2.v = _mm_setr_ps( data[4][0] , data[5][0] , data[6][0] , data[7][0] );
    hi_2.v = _mm_setr_ps( data[4][1] , data[5][1] , data[6][1] , data[7][1] );
    x_1.v = _mm_mul_ps( _mm_sub_ps( r_1.v , mi_1.v ) , hi_1.v );
    x_2.v = _mm_mul_ps( _mm_sub_ps( r_2.v , mi_2.v ) , hi_2.v );
    
    /* compute the potential and its derivative */
    eff_1.v = _mm_setr_ps( data[0][2] , data[1][2] , data[2][2] , data[3][2] );
    eff_2.v = _mm_setr_ps( data[4][2] , data[5][2] , data[6][2] , data[7][2] );
    c_1.v = _mm_setr_ps( data[0][3] , data[1][3] , data[2][3] , data[3][3] );
    c_2.v = _mm_setr_ps( data[4][3] , data[5][3] , data[6][3] , data[7][3] );
    ee_1.v = _mm_add_ps( _mm_mul_ps( eff_1.v , x_1.v ) , c_1.v );
    ee_2.v = _mm_add_ps( _mm_mul_ps( eff_2.v , x_2.v ) , c_2.v );
    for ( j = 4 ; j < potential_chunk ; j++ ) {
        c_1.v = _mm_setr_ps( data[0][j] , data[1][j] , data[2][j] , data[3][j] );
        c_2.v = _mm_setr_ps( data[4][j] , data[5][j] , data[6][j] , data[7][j] );
        eff_1.v = _mm_add_ps( _mm_mul_ps( eff_1.v , x_1.v ) , ee_1.v );
        eff_2.v = _mm_add_ps( _mm_mul_ps( eff_2.v , x_2.v ) , ee_2.v );
        ee_1.v = _mm_add_ps( _mm_mul_ps( ee_1.v , x_1.v ) , c_1.v );
        ee_2.v = _mm_add_ps( _mm_mul_ps( ee_2.v , x_2.v ) , c_2.v );
        }

    /* store the result */
    _mm_store_ps( &e[0] , ee_1.v );
    _mm_store_ps( &e[4] , ee_2.v );
    _mm_store_ps( &f[0] , _mm_mul_ps( eff_1.v , hi_1.v ) );
    _mm_store_ps( &f[4] , _mm_mul_ps( eff_2.v , hi_2.v ) );
    
#elif defined(__ALTIVEC__) && defined(FPTYPE_SINGLE)
    int j;
    union {
        vector float v;
        vector unsigned int m;
        float f[4];
        unsigned int i[4];
        } alpha0_1, alpha1_1, alpha2_1, mi_1, hi_1, x_1, ee_1, eff_1, c_1, r_1, ind_1,
          alpha0_2, alpha1_2, alpha2_2, mi_2, hi_2, x_2, ee_2, eff_2, c_2, r_2, ind_2;
    float *data[8];
    
    /* Get r . */
    r_1.v = vec_sqrt( *((vector float *)&r2[0]) );
    r_2.v = vec_sqrt( *((vector float *)&r2[4]) );
    
    /* compute the index */
    alpha0_1.v = vec_load4( p[0]->alpha[0] , p[1]->alpha[0] , p[2]->alpha[0] , p[3]->alpha[0] );
    alpha1_1.v = vec_load4( p[0]->alpha[1] , p[1]->alpha[1] , p[2]->alpha[1] , p[3]->alpha[1] );
    alpha2_1.v = vec_load4( p[0]->alpha[2] , p[1]->alpha[2] , p[2]->alpha[2] , p[3]->alpha[2] );
    alpha0_2.v = vec_load4( p[4]->alpha[0] , p[5]->alpha[0] , p[6]->alpha[0] , p[7]->alpha[0] );
    alpha1_2.v = vec_load4( p[4]->alpha[1] , p[5]->alpha[1] , p[6]->alpha[1] , p[7]->alpha[1] );
    alpha2_2.v = vec_load4( p[4]->alpha[2] , p[5]->alpha[2] , p[6]->alpha[2] , p[7]->alpha[2] );
    ind_1.m = vec_ctu( vec_madd( r_1.v , vec_madd( r_1.v , alpha2_1.v , alpha1_1.v ) , alpha0_1.v ) , 0 );
    ind_2.m = vec_ctu( vec_madd( r_2.v , vec_madd( r_2.v , alpha2_2.v , alpha1_2.v ) , alpha0_2.v ) , 0 );
    
    /* get the table offset */
    data[0] = &( p[0]->c[ ind_1.i[0] * potential_chunk ] );
    data[1] = &( p[1]->c[ ind_1.i[1] * potential_chunk ] );
    data[2] = &( p[2]->c[ ind_1.i[2] * potential_chunk ] );
    data[3] = &( p[3]->c[ ind_1.i[3] * potential_chunk ] );
    data[4] = &( p[4]->c[ ind_2.i[0] * potential_chunk ] );
    data[5] = &( p[5]->c[ ind_2.i[1] * potential_chunk ] );
    data[6] = &( p[6]->c[ ind_2.i[2] * potential_chunk ] );
    data[7] = &( p[7]->c[ ind_2.i[3] * potential_chunk ] );
    
    /* adjust x to the interval */
    mi_1.v = vec_load4( data[0][0] , data[1][0] , data[2][0] , data[3][0] );
    hi_1.v = vec_load4( data[0][1] , data[1][1] , data[2][1] , data[3][1] );
    mi_2.v = vec_load4( data[4][0] , data[5][0] , data[6][0] , data[7][0] );
    hi_2.v = vec_load4( data[4][1] , data[5][1] , data[6][1] , data[7][1] );
    x_1.v = vec_mul( vec_sub( r_1.v , mi_1.v ) , hi_1.v );
    x_2.v = vec_mul( vec_sub( r_2.v , mi_2.v ) , hi_2.v );
    
    /* compute the potential and its derivative */
    eff_1.v = vec_load4( data[0][2] , data[1][2] , data[2][2] , data[3][2] );
    eff_2.v = vec_load4( data[4][2] , data[5][2] , data[6][2] , data[7][2] );
    c_1.v = vec_load4( data[0][3] , data[1][3] , data[2][3] , data[3][3] );
    c_2.v = vec_load4( data[4][3] , data[5][3] , data[6][3] , data[7][3] );
    ee_1.v = vec_madd( eff_1.v , x_1.v , c_1.v );
    ee_2.v = vec_madd( eff_2.v , x_2.v , c_2.v );
    for ( j = 4 ; j < potential_chunk ; j++ ) {
        c_1.v = vec_load4( data[0][j] , data[1][j] , data[2][j] , data[3][j] );
        c_2.v = vec_load4( data[4][j] , data[5][j] , data[6][j] , data[7][j] );
        eff_1.v = vec_madd( eff_1.v , x_1.v , ee_1.v );
        eff_2.v = vec_madd( eff_2.v , x_2.v , ee_2.v );
        ee_1.v = vec_madd( ee_1.v , x_1.v , c_1.v );
        ee_2.v = vec_madd( ee_2.v , x_2.v , c_2.v );
        }

    /* store the result */
    eff_1.v = vec_mul( eff_1.v , hi_1.v );
    eff_2.v = vec_mul( eff_2.v , hi_2.v );
    memcpy( &e[0] , &ee_1 , sizeof(vector float) );
    memcpy( &f[0] , &eff_1 , sizeof(vector float) );
    memcpy( &e[4] , &ee_2 , sizeof(vector float) );
    memcpy( &f[4] , &eff_2 , sizeof(vector float) );
    
#else
    int k;
    for ( k = 0 ; k < 8 ; k++ )
        potential_eval( p[k] , r2[k] , &e[k] , &f[k] );
#endif
        
    }


/** 
 * @brief Evaluates the given potential at a set of points (interpolated)
 *      using explicit electrostatics.
 *
 * @param p The #potential to be evaluated.
 * @param ep The electrostatics #potential.
 * @param r2 The radius at which it is to be evaluated, squared.
 * @param q The product of charges from both particles
 * @param e Pointer to a floating-point value in which to store the
 *      interaction energy.
 * @param f Pointer to a floating-point value in which to store the
 *      magnitude of the interaction force.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p.
 * 
 * This function is only available if mdcore was compiled with SSE and
 * single precision! If @c mdcore was not compiled with SSE enabled, this
 * function simply calls #potential_eval_ee on each entry.
 *
 * Note that the vectors @c r2, @c e and @c f should be aligned to 16 bytes.
 */

void potential_eval_vec_4single_ee ( struct potential *p[4] , struct potential *ep , FPTYPE *r2 , FPTYPE *q , FPTYPE *e , FPTYPE *f ) {

#if defined(__SSE__) && defined(FPTYPE_SINGLE)
    int j, k;
    union {
        __v4sf v;
        __m128i m;
        float f[4];
        int i[4];
        } alpha0, alpha1, alpha2, mi, hi, x, ee, eff, c, r, ind;
    float *data[4];
    union {
        __v4sf v;
        __m128i m;
        float f[4];
        int i[4];
        } alpha0_e, alpha1_e, alpha2_e, mi_e, hi_e, x_e, ee_e, eff_e, c_e, ind_e, qv;
    float *data_e[4];
    
    /* Get r . */
    r.v = _mm_sqrt_ps( _mm_load_ps( r2 ) );
    
    /* compute the index */
    alpha0.v = _mm_setr_ps( p[0]->alpha[0] , p[1]->alpha[0] , p[2]->alpha[0] , p[3]->alpha[0] );
    alpha1.v = _mm_setr_ps( p[0]->alpha[1] , p[1]->alpha[1] , p[2]->alpha[1] , p[3]->alpha[1] );
    alpha2.v = _mm_setr_ps( p[0]->alpha[2] , p[1]->alpha[2] , p[2]->alpha[2] , p[3]->alpha[2] );
    ind.m = _mm_cvttps_epi32( _mm_max_ps( _mm_setzero_ps() , _mm_add_ps( alpha0.v , _mm_mul_ps( r.v , _mm_add_ps( alpha1.v , _mm_mul_ps( r.v , alpha2.v ) ) ) ) ) );
    alpha0_e.v = _mm_set1_ps( ep->alpha[0] );
    alpha1_e.v = _mm_set1_ps( ep->alpha[1] );
    alpha2_e.v = _mm_set1_ps( ep->alpha[2] );
    ind_e.m = _mm_cvttps_epi32( _mm_max_ps( _mm_setzero_ps() , _mm_add_ps( alpha0_e.v , _mm_mul_ps( r.v , _mm_add_ps( alpha1_e.v , _mm_mul_ps( r.v , alpha2_e.v ) ) ) ) ) );
    
    /* get the table offset */
    for ( k = 0 ; k < 4 ; k++ )
        data[k] = &( p[k]->c[ ind.i[k] * potential_chunk ] );
    for ( k = 0 ; k < 4 ; k++ )
        data_e[k] = &( ep->c[ ind_e.i[k] * potential_chunk ] );
    
    /* adjust x to the interval */
    mi.v = _mm_setr_ps( data[0][0] , data[1][0] , data[2][0] , data[3][0] );
    hi.v = _mm_setr_ps( data[0][1] , data[1][1] , data[2][1] , data[3][1] );
    x.v = _mm_mul_ps( _mm_sub_ps( r.v , mi.v ) , hi.v );
    mi_e.v = _mm_setr_ps( data_e[0][0] , data_e[1][0] , data_e[2][0] , data_e[3][0] );
    hi_e.v = _mm_setr_ps( data_e[0][1] , data_e[1][1] , data_e[2][1] , data_e[3][1] );
    x_e.v = _mm_mul_ps( _mm_sub_ps( r.v , mi_e.v ) , hi_e.v );
    
    /* compute the potential and its derivative */
    eff.v = _mm_setr_ps( data[0][2] , data[1][2] , data[2][2] , data[3][2] );
    c.v = _mm_setr_ps( data[0][3] , data[1][3] , data[2][3] , data[3][3] );
    ee.v = _mm_add_ps( _mm_mul_ps( eff.v , x.v ) , c.v );
    eff_e.v = _mm_setr_ps( data_e[0][2] , data_e[1][2] , data_e[2][2] , data_e[3][2] );
    c_e.v = _mm_setr_ps( data_e[0][3] , data_e[1][3] , data_e[2][3] , data_e[3][3] );
    ee_e.v = _mm_add_ps( _mm_mul_ps( eff_e.v , x_e.v ) , c_e.v );
    for ( j = 4 ; j < potential_chunk ; j++ ) {
        c.v = _mm_setr_ps( data[0][j] , data[1][j] , data[2][j] , data[3][j] );
        eff.v = _mm_add_ps( _mm_mul_ps( eff.v , x.v ) , ee.v );
        ee.v = _mm_add_ps( _mm_mul_ps( ee.v , x.v ) , c.v );
        c_e.v = _mm_setr_ps( data_e[0][j] , data_e[1][j] , data_e[2][j] , data_e[3][j] );
        eff_e.v = _mm_add_ps( _mm_mul_ps( eff_e.v , x_e.v ) , ee_e.v );
        ee_e.v = _mm_add_ps( _mm_mul_ps( ee_e.v , x_e.v ) , c_e.v );
        }

    /* store the result */
    qv.v = _mm_load_ps( q );
    _mm_store_ps( e , _mm_add_ps( ee.v , _mm_mul_ps( ee_e.v , qv.v ) ) );
    _mm_store_ps( f , _mm_add_ps( _mm_mul_ps( eff.v , hi.v ) , _mm_mul_ps( _mm_mul_ps( eff_e.v , hi_e.v ) , qv.v ) ) );
        
#else
    int k;
    for ( k = 0 ; k < 4 ; k++ )
        potential_eval_ee( p[k] , ep , r2[k] , q[k] , &e[k] , &f[k] );
#endif

    }


/** 
 * @brief Evaluates the given potential at a set of points (interpolated).
 *
 * @param p Pointer to an array of pointers to the #potentials to be evaluated.
 * @param r2 Pointer to an array of the radii at which the potentials
 *      are to be evaluated, squared.
 * @param e Pointer to an array of floating-point values in which to store the
 *      interaction energies.
 * @param f Pointer to an array of floating-point values in which to store the
 *      magnitude of the interaction forces.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p.
 *
 * Computes two double-precision interactions simultaneously using vectorized
 * instructions.
 * 
 * This function is only available if mdcore was compiled with SSE2 and
 * double precision! If @c mdcore was not compiled with SSE2 enabled, this
 * function simply calls #potential_eval on each entry.
 */

void potential_eval_vec_2double ( struct potential *p[2] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f ) {

#if defined(__SSE2__) && defined(FPTYPE_DOUBLE)
    int ind[2], j;
    union {
        __v2df v;
        double f[2];
        } alpha0, alpha1, alpha2, rind, mi, hi, x, ee, eff, c, r;
    double *data[2];
    
    /* Get r . */
    r.v = _mm_sqrt_pd( _mm_load_pd( r2 ) );
    
    /* compute the index */
    alpha0.v = _mm_setr_pd( p[0]->alpha[0] , p[1]->alpha[0] );
    alpha1.v = _mm_setr_pd( p[0]->alpha[1] , p[1]->alpha[1] );
    alpha2.v = _mm_setr_pd( p[0]->alpha[2] , p[1]->alpha[2] );
    rind.v = _mm_max_pd( _mm_setzero_pd() , _mm_add_pd( alpha0.v , _mm_mul_pd( r.v , _mm_add_pd( alpha1.v , _mm_mul_pd( r.v , alpha2.v ) ) ) ) );
    ind[0] = rind.f[0];
    ind[1] = rind.f[1];
    
    /* get the table offset */
    data[0] = &( p[0]->c[ ind[0] * potential_chunk ] );
    data[1] = &( p[1]->c[ ind[1] * potential_chunk ] );
    
    /* adjust x to the interval */
    mi.v = _mm_setr_pd( data[0][0] , data[1][0] );
    hi.v = _mm_setr_pd( data[0][1] , data[1][1] );
    x.v = _mm_mul_pd( _mm_sub_pd( r.v , mi.v ) , hi.v );
    
    /* compute the potential and its derivative */
    eff.v = _mm_setr_pd( data[0][2] , data[1][2] );
    c.v = _mm_setr_pd( data[0][3] , data[1][3] );
    ee.v = _mm_add_pd( _mm_mul_pd( eff.v , x.v ) , c.v );
    for ( j = 4 ; j < potential_chunk ; j++ ) {
        c.v = _mm_setr_pd( data[0][j] , data[1][j] );
        eff.v = _mm_add_pd( _mm_mul_pd( eff.v , x.v ) , ee.v );
        ee.v = _mm_add_pd( _mm_mul_pd( ee.v , x.v ) , c.v );
        }

    /* store the result */
    _mm_store_pd( e , ee.v );
    _mm_store_pd( f , _mm_mul_pd( eff.v , hi.v ) );
        
#else
    int k;
    for ( k = 0 ; k < 2 ; k++ )
        potential_eval( p[k] , r2[k] , &e[k] , &f[k] );
#endif
        
    }


/** 
 * @brief Evaluates the given potential at a set of points (interpolated).
 *
 * @param p Pointer to an array of pointers to the #potentials to be evaluated.
 * @param r2 Pointer to an array of the radii at which the potentials
 *      are to be evaluated, squared.
 * @param e Pointer to an array of floating-point values in which to store the
 *      interaction energies.
 * @param f Pointer to an array of floating-point values in which to store the
 *      magnitude of the interaction forces.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p.
 *
 * Computes four double-precision interactions simultaneously using vectorized
 * instructions.
 * 
 * This function is only available if mdcore was compiled with SSE2 and
 * double precision! If @c mdcore was not compiled with SSE2 enabled, this
 * function simply calls #potential_eval on each entry.
 */

void potential_eval_vec_4double ( struct potential *p[4] , FPTYPE *r2 , FPTYPE *e , FPTYPE *f ) {

#if defined(__SSE2__) && defined(FPTYPE_DOUBLE)
    int ind[4], j;
    union {
        __v2df v;
        double f[2];
        } alpha0_1, alpha1_1, alpha2_1, rind_1, mi_1, hi_1, x_1, ee_1, eff_1, c_1, r_1,
        alpha0_2, alpha1_2, alpha2_2, rind_2, mi_2, hi_2, x_2, ee_2, eff_2, c_2, r_2;
    double *data[4];
    
    /* Get r . */
    r_1.v = _mm_sqrt_pd( _mm_load_pd( &r2[0] ) );
    r_2.v = _mm_sqrt_pd( _mm_load_pd( &r2[2] ) );
    
    /* compute the index */
    alpha0_1.v = _mm_setr_pd( p[0]->alpha[0] , p[1]->alpha[0] );
    alpha1_1.v = _mm_setr_pd( p[0]->alpha[1] , p[1]->alpha[1] );
    alpha2_1.v = _mm_setr_pd( p[0]->alpha[2] , p[1]->alpha[2] );
    alpha0_2.v = _mm_setr_pd( p[2]->alpha[0] , p[3]->alpha[0] );
    alpha1_2.v = _mm_setr_pd( p[2]->alpha[1] , p[3]->alpha[1] );
    alpha2_2.v = _mm_setr_pd( p[2]->alpha[2] , p[3]->alpha[2] );
    rind_1.v = _mm_max_pd( _mm_setzero_pd() , _mm_add_pd( alpha0_1.v , _mm_mul_pd( r_1.v , _mm_add_pd( alpha1_1.v , _mm_mul_pd( r_1.v , alpha2_1.v ) ) ) ) );
    rind_2.v = _mm_max_pd( _mm_setzero_pd() , _mm_add_pd( alpha0_2.v , _mm_mul_pd( r_2.v , _mm_add_pd( alpha1_2.v , _mm_mul_pd( r_2.v , alpha2_2.v ) ) ) ) );
    ind[0] = rind_1.f[0];
    ind[1] = rind_1.f[1];
    ind[2] = rind_2.f[0];
    ind[3] = rind_2.f[1];
    
    /* for ( j = 0 ; j < 4 ; j++ )
        if ( ind[j] > p[j]->n ) {
            printf("potential_eval_vec_4double: dookie.\n");
            fflush(stdout);
            } */
    
    /* get the table offset */
    data[0] = &( p[0]->c[ ind[0] * potential_chunk ] );
    data[1] = &( p[1]->c[ ind[1] * potential_chunk ] );
    data[2] = &( p[2]->c[ ind[2] * potential_chunk ] );
    data[3] = &( p[3]->c[ ind[3] * potential_chunk ] );
    
    /* adjust x to the interval */
    mi_1.v = _mm_setr_pd( data[0][0] , data[1][0] );
    hi_1.v = _mm_setr_pd( data[0][1] , data[1][1] );
    mi_2.v = _mm_setr_pd( data[2][0] , data[3][0] );
    hi_2.v = _mm_setr_pd( data[2][1] , data[3][1] );
    x_1.v = _mm_mul_pd( _mm_sub_pd( r_1.v , mi_1.v ) , hi_1.v );
    x_2.v = _mm_mul_pd( _mm_sub_pd( r_2.v , mi_2.v ) , hi_2.v );
    
    /* compute the potential and its derivative */
    eff_1.v = _mm_setr_pd( data[0][2] , data[1][2] );
    eff_2.v = _mm_setr_pd( data[2][2] , data[3][2] );
    c_1.v = _mm_setr_pd( data[0][3] , data[1][3] );
    c_2.v = _mm_setr_pd( data[2][3] , data[3][3] );
    ee_1.v = _mm_add_pd( _mm_mul_pd( eff_1.v , x_1.v ) , c_1.v );
    ee_2.v = _mm_add_pd( _mm_mul_pd( eff_2.v , x_2.v ) , c_2.v );
    for ( j = 4 ; j < potential_chunk ; j++ ) {
        c_1.v = _mm_setr_pd( data[0][j] , data[1][j] );
        c_2.v = _mm_setr_pd( data[2][j] , data[3][j] );
        eff_1.v = _mm_add_pd( _mm_mul_pd( eff_1.v , x_1.v ) , ee_1.v );
        eff_2.v = _mm_add_pd( _mm_mul_pd( eff_2.v , x_2.v ) , ee_2.v );
        ee_1.v = _mm_add_pd( _mm_mul_pd( ee_1.v , x_1.v ) , c_1.v );
        ee_2.v = _mm_add_pd( _mm_mul_pd( ee_2.v , x_2.v ) , c_2.v );
        }

    /* store the result */
    _mm_store_pd( &e[0] , ee_1.v );
    _mm_store_pd( &f[0] , _mm_mul_pd( eff_1.v , hi_1.v ) );
    _mm_store_pd( &e[2] , ee_2.v );
    _mm_store_pd( &f[2] , _mm_mul_pd( eff_2.v , hi_2.v ) );
        
#else
    int k;
    for ( k = 0 ; k < 4 ; k++ )
        potential_eval( p[k] , r2[k] , &e[k] , &f[k] );
#endif
        
    }


/** 
 * @brief Evaluates the given potential at a set of points (interpolated)
 *      with explicit electrostatics.
 *
 * @param p The #potential to be evaluated.
 * @param ep The electrostatics #potential.
 * @param r2 The radius at which it is to be evaluated, squared.
 * @param q The product of charges from both particles
 * @param e Pointer to a floating-point value in which to store the
 *      interaction energy.
 * @param f Pointer to a floating-point value in which to store the
 *      magnitude of the interaction force.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p.
 * 
 * This function is only available if mdcore was compiled with SSE2 and
 * double precision!
 */

void potential_eval_vec_2double_ee ( struct potential *p[4] , struct potential *ep , FPTYPE *r2 , FPTYPE *q , FPTYPE *e , FPTYPE *f ) {

#if defined(__SSE2__) && defined(FPTYPE_DOUBLE)
    int ind[2], j;
    union {
        __v2df v;
        double f[2];
        } alpha0, alpha1, alpha2, rind, mi, hi, x, ee, eff, c, r;
    double *data[2];
    int ind_e[2];
    union {
        __v2df v;
        double f[2];
        } alpha0_e, alpha1_e, alpha2_e, rind_e, mi_e, hi_e, x_e, ee_e, eff_e, c_e, qv;
    double *data_e[2];
    
    /* Get r . */
    r.v = _mm_sqrt_pd( _mm_load_pd( r2 ) );
    
    /* compute the index */
    alpha0.v = _mm_setr_pd( p[0]->alpha[0] , p[1]->alpha[0] );
    alpha1.v = _mm_setr_pd( p[0]->alpha[1] , p[1]->alpha[1] );
    alpha2.v = _mm_setr_pd( p[0]->alpha[2] , p[1]->alpha[2] );
    alpha0_e.v = _mm_set1_pd( ep->alpha[0] );
    alpha1_e.v = _mm_set1_pd( ep->alpha[1] );
    alpha2_e.v = _mm_set1_pd( ep->alpha[2] );
    rind.v = _mm_max_pd( _mm_setzero_pd() , _mm_add_pd( alpha0.v , _mm_mul_pd( r.v , _mm_add_pd( alpha1.v , _mm_mul_pd( r.v , alpha2.v ) ) ) ) );
    rind_e.v = _mm_max_pd( _mm_setzero_pd() , _mm_add_pd( alpha0_e.v , _mm_mul_pd( r.v , _mm_add_pd( alpha1_e.v , _mm_mul_pd( r.v , alpha2_e.v ) ) ) ) );
    ind[0] = rind.f[0]; ind[1] = rind.f[1];
    ind_e[0] = rind_e.f[0]; ind_e[1] = rind_e.f[1];
    
    /* get the table offset */
    data[0] = &( p[0]->c[ ind[0] * potential_chunk ] );
    data[1] = &( p[1]->c[ ind[1] * potential_chunk ] );
    data_e[0] = &( ep->c[ ind_e[0] * potential_chunk ] );
    data_e[1] = &( ep->c[ ind_e[1] * potential_chunk ] );
    
    /* adjust x to the interval */
    mi.v = _mm_setr_pd( data[0][0] , data[1][0] );
    hi.v = _mm_setr_pd( data[0][1] , data[1][1] );
    x.v = _mm_mul_pd( _mm_sub_pd( r.v , mi.v ) , hi.v );
    mi_e.v = _mm_setr_pd( data_e[0][0] , data_e[1][0] );
    hi_e.v = _mm_setr_pd( data_e[0][1] , data_e[1][1] );
    x_e.v = _mm_mul_pd( _mm_sub_pd( r.v , mi_e.v ) , hi_e.v );
    
    /* compute the potential and its derivative */
    eff.v = _mm_setr_pd( data[0][2] , data[1][2] );
    c.v = _mm_setr_pd( data[0][3] , data[1][3] );
    ee.v = _mm_add_pd( _mm_mul_pd( eff.v , x.v ) , c.v );
    eff_e.v = _mm_setr_pd( data_e[0][2] , data_e[1][2] );
    c_e.v = _mm_setr_pd( data_e[0][3] , data_e[1][3] );
    ee_e.v = _mm_add_pd( _mm_mul_pd( eff_e.v , x_e.v ) , c_e.v );
    for ( j = 4 ; j < potential_chunk ; j++ ) {
        c.v = _mm_setr_pd( data[0][j] , data[1][j] );
        eff.v = _mm_add_pd( _mm_mul_pd( eff.v , x.v ) , ee.v );
        ee.v = _mm_add_pd( _mm_mul_pd( ee.v , x.v ) , c.v );
        c_e.v = _mm_setr_pd( data_e[0][j] , data_e[1][j] );
        eff_e.v = _mm_add_pd( _mm_mul_pd( eff_e.v , x_e.v ) , ee_e.v );
        ee_e.v = _mm_add_pd( _mm_mul_pd( ee_e.v , x_e.v ) , c_e.v );
        }

    /* store the result */
    qv.v = _mm_load_pd( q );
    _mm_store_pd( e , _mm_add_pd( ee.v , _mm_mul_pd( ee_e.v , qv.v ) ) );
    _mm_store_pd( f , _mm_add_pd( _mm_mul_pd( eff.v , hi.v ) , _mm_mul_pd( eff_e.v , _mm_mul_pd( hi_e.v , qv.v ) ) ) );
                
#else
    int k;
    for ( k = 0 ; k < 4 ; k++ )
        potential_eval_ee( p[k] , ep , r2[k] , q[k] , &e[k] , &f[k] );
#endif

    }


/** 
 * @brief Evaluates the given potential at the given point (interpolated).
 *
 * @param p The #potential to be evaluated.
 * @param r2 The radius at which it is to be evaluated, squared.
 * @param e Pointer to a floating-point value in which to store the
 *      interaction energy.
 * @param f Pointer to a floating-point value in which to store the
 *      magnitude of the interaction force.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p.
 */

void potential_eval ( struct potential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f ) {

    int ind, k;
    FPTYPE x, ee, eff, *c, r;
    
    /* Get r for the right type. */
    #ifdef FPTYPE_SINGLE
        r = sqrtf(r2);
    #else
        r = sqrt(r2);
    #endif
    
    /* is r in the house? */
    /* if ( r < p->a || r > p->b ) */
    /*     printf("potential_eval: requested potential at r=%e, not in [%e,%e].\n",r,p->a,p->b); */
    
    /* compute the index */
    #ifdef FPTYPE_SINGLE
        ind = fmaxf( 0.0f , p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]) );
    #else
        ind = fmax( 0.0 , p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]) );
    #endif
    
    /* if ( ind > p->n ) {
        printf("potential_eval: r=%.18e.\n",r);
        fflush(stdout);
        } */
            
    /* get the table offset */
    c = &(p->c[ind * potential_chunk]);
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    
    /* compute the potential and its derivative */
    ee = c[2] * x + c[3];
    eff = c[2];
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        }

    /* store the result */
    *e = ee; *f = eff * c[1];
        
    }


/** 
 * @brief Evaluates the given potential at the given point (interpolated)
 *      with explicit electrostatics.
 *
 * @param p The #potential to be evaluated.
 * @param ep The electrostatics #potential.
 * @param r2 The radius at which it is to be evaluated, squared.
 * @param q The product of charges from both particles
 * @param e Pointer to a floating-point value in which to store the
 *      interaction energy.
 * @param f Pointer to a floating-point value in which to store the
 *      magnitude of the interaction force.
 *
 * Note that for efficiency reasons, this function does not check if any
 * of the parameters are @c NULL or if @c sqrt(r2) is within the interval
 * of the #potential @c p or @c ep.
 */

void potential_eval_ee ( struct potential *p , struct potential *ep , FPTYPE r2 , FPTYPE q , FPTYPE *e , FPTYPE *f ) {

    int ind, k;
    FPTYPE x, ee, eff, *c, r;
    int ind_e;
    FPTYPE x_e, ee_e, eff_e, *c_e;
    
    /* Get r for the right type. */
    #ifdef FPTYPE_SINGLE
        r = sqrtf(r2);
    #else
        r = sqrt(r2);
    #endif
    
    /* is r in the house? */
    /* if ( r < p->a || r > p->b ) */
    /*     printf("potential_eval: requested potential at r=%e, not in [%e,%e].\n",r,p->a,p->b); */
    
    /* compute the index */
    #ifdef FPTYPE_SINGLE
        ind = fmaxf( 0.0f , p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]) );
        ind_e = fmaxf( 0.0f , ep->alpha[0] + r * (ep->alpha[1] + r * ep->alpha[2]) );
    #else
        ind = fmax( 0.0 , p->alpha[0] + r * (p->alpha[1] + r * p->alpha[2]) );
        ind_e = fmax( 0.0 , ep->alpha[0] + r * (ep->alpha[1] + r * ep->alpha[2]) );
    #endif
    
    /* get the table offset */
    c = &(p->c[ind * potential_chunk]);
    c_e = &(p->c[ind_e * potential_chunk]);
    
    /* adjust x to the interval */
    x = (r - c[0]) * c[1];
    x_e = (r - c_e[0]) * c_e[1];
    
    /* compute the potential and its derivative */
    ee = c[2] * x + c[3];
    eff = c[2];
    ee_e = c_e[2] * x_e + c_e[3];
    eff_e = c_e[2];
    for ( k = 4 ; k < potential_chunk ; k++ ) {
        eff = eff * x + ee;
        ee = ee * x + c[k];
        eff_e = eff_e * x_e + ee_e;
        ee_e = ee_e * x_e + c_e[k];
        }

    /* store the result */
    *e = ee + q*ee_e; *f = eff * c[1] + q*eff_e*c_e[1];
        
    }


/**
 * @brief Evaluates the given potential at the given radius explicitly.
 * 
 * @param p The #potential to be evaluated.
 * @param r2 The radius squared.
 * @param e A pointer to a floating point value in which to store the
 *      interaction energy.
 * @param f A pointer to a floating point value in which to store the
 *      magnitude of the interaction force
 *
 * Assumes that the parameters for the potential forms given in the value
 * @c flags of the #potential @c p are stored in the array @c alpha of
 * @c p.
 *
 * This way of evaluating a potential is not extremely efficient and is
 * intended for comparison and debugging purposes.
 *
 * Note that for performance reasons, this function does not check its input
 * arguments for @c NULL.
 */

void potential_eval_expl ( struct potential *p , FPTYPE r2 , FPTYPE *e , FPTYPE *f ) {

    const FPTYPE isqrtpi = 0.56418958354775628695;
    const FPTYPE kappa = 3.0;
    FPTYPE r = sqrt(r2), ir = 1.0 / r, ir2 = ir * ir, ir4, ir6, ir12, t1, t2;
    FPTYPE ee = 0.0, eff = 0.0;

    /* Do we have a Lennard-Jones interaction? */
    if ( p->flags & potential_flag_LJ126 ) {
    
        /* init some variables */
        ir4 = ir2 * ir2; ir6 = ir4 * ir2; ir12 = ir6 * ir6;
        
        /* compute the energy and the force */
        ee = ( p->alpha[0] * ir12 - p->alpha[1] * ir6 );
        eff = 6.0 * ir * ( -2.0 * p->alpha[0] * ir12 + p->alpha[1] * ir6 );
    
        }
        
    /* Do we have an Ewald short-range part? */
    if ( p->flags & potential_flag_Ewald ) {
    
        /* get some values we will re-use */
        t2 = r * kappa;
        t1 = erfc( t2 );
    
        /* compute the energy and the force */
        ee += p->alpha[2] * t1 * ir;
        eff += p->alpha[2] * ( -2.0 * isqrtpi * exp( -t2 * t2 ) * kappa * ir - t1 * ir2 );
    
        }
    
    /* Do we have a Coulomb interaction? */
    if ( p->flags & potential_flag_Coulomb ) {
    
        /* get some values we will re-use */
        t2 = r * kappa;
        t1 = erfc( t2 );
    
        /* compute the energy and the force */
        ee += p->alpha[2] * ir;
        eff += -p->alpha[2] * ir2;
    
        }
        
    /* store the potential and force. */
    *e = ee;
    *f = eff;
    
    }


/**
 * @brief A basic 12-6 Lennard-Jones potential.
 *
 * @param r The interaction radius.
 * @param A First parameter of the potential.
 * @param B Second parameter of the potential.
 *
 * @return The potential @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$
 *      evaluated at @c r.
 */

inline double potential_LJ126 ( double r , double A , double B ) {

    double ir = 1.0/r, ir2 = ir * ir, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

    return ( A * ir12 - B * ir6 );

    }
    
/**
 * @brief A basic 12-6 Lennard-Jones potential (first derivative).
 *
 * @param r The interaction radius.
 * @param A First parameter of the potential.
 * @param B Second parameter of the potential.
 *
 * @return The first derivative of the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$
 *      evaluated at @c r.
 */

inline double potential_LJ126_p ( double r , double A , double B ) {

    double ir = 1.0/r, ir2 = ir * ir, ir4 = ir2*ir2, ir12 = ir4 * ir4 * ir4;

    return 6 * ir * ( -2 * A * ir12 + B * ir4 * ir2 );

    }
    
/**
 * @brief A basic 12-6 Lennard-Jones potential (sixth derivative).
 *
 * @param r The interaction radius.
 * @param A First parameter of the potential.
 * @param B Second parameter of the potential.
 *
 * @return The sixth derivative of the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$
 *      evaluated at @c r.
 */

inline double potential_LJ126_6p ( double r , double A , double B ) {

    double r2 = r * r, ir2 = 1.0 / r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;

    return 10080 * ir12 * ( 884 * A * ir6 - 33 * B );
        
    }
    
/**
 * @brief The short-range part of an Ewald summation.
 *
 * @param r The interaction radius.
 * @param kappa The screening length of the Ewald summation.
 *
 * @return The potential @f$ \frac{\mbox{erfc}( \kappa r )}{r} @f$
 *      evaluated at @c r.
 */

inline double potential_Ewald ( double r , double kappa ) {

    return erfc( kappa * r ) / r;

    }
    
/**
 * @brief The short-range part of an Ewald summation (first derivative).
 *
 * @param r The interaction radius.
 * @param kappa The screening length of the Ewald summation.
 *
 * @return The first derivative of the potential @f$ \frac{\mbox{erfc}( \kappa r )}{r} @f$
 *      evaluated at @c r.
 */

inline double potential_Ewald_p ( double r , double kappa ) {

    double r2 = r * r, ir = 1.0 / r, ir2 = ir * ir;
    const double isqrtpi = 0.56418958354775628695;

    return -2 * exp( -kappa*kappa * r2 ) * kappa * ir * isqrtpi -
        erfc( kappa * r ) * ir2;

    }
    
/**
 * @brief The short-range part of an Ewald summation (sixth derivative).
 *
 * @param r The interaction radius.
 * @param kappa The screening length of the Ewald summation.
 *
 * @return The sixth derivative of the potential @f$ \frac{\mbox{erfc}( \kappa r )}{r} @f$
 *      evaluated at @c r.
 */

inline double potential_Ewald_6p ( double r , double kappa ) {

    double r2 = r*r, ir2 = 1.0 / r2, r4 = r2*r2, ir4 = ir2*ir2, ir6 = ir2*ir4;
    double kappa2 = kappa*kappa;
    double t6, t23;
    const double isqrtpi = 0.56418958354775628695;

    t6 = erfc(kappa*r);
    t23 = exp(-kappa2*r2);
    return 720.0*t6/r*ir6+(1440.0*ir6+(960.0*ir4+(384.0*ir2+(144.0+(-128.0*r2+64.0*kappa2*r4)*kappa2)*kappa2)*kappa2)*kappa2)*kappa*isqrtpi*t23;
    
    }
        

/**
 * @brief Creates a #potential representing the real-space part of an Ewald 
 *      potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param q The charge scaling of the potential.
 * @param kappa The screening distance of the Ewald potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ q\frac{\mbox{erfc}(\kappa r}{r} @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_Ewald ( double a , double b , double q , double kappa , double tol ) {

    struct potential *p;
    
    /* the potential functions */
    double f ( double r ) {
        return q * potential_Ewald( r , kappa );
        }
        
    double dfdr ( double r ) {
        return q * potential_Ewald_p( r , kappa );
        }
        
    double d6fdr6 ( double r ) {
        return q * potential_Ewald_6p( r , kappa );
        }
        
    /* allocate the potential */
    if ( ( p = (struct potential *)malloc( sizeof( struct potential ) ) ) == NULL ) {
        error(potential_err_malloc);
        return NULL;
        }
        
    /* fill this potential */
    if ( potential_init( p , &f , &dfdr , &d6fdr6 , a , b , tol ) < 0 ) {
        free(p);
        return NULL;
        }
    
    /* return it */
    return p;

    }
    

/**
 * @brief Creates a #potential representing the sum of a
 *      12-6 Lennard-Jones potential and the real-space part of an Ewald 
 *      potential.
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param q The charge scaling of the potential.
 * @param kappa The screening distance of the Ewald potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 */

struct potential *potential_create_LJ126_Ewald ( double a , double b , double A , double B , double q , double kappa , double tol ) {

    struct potential *p;
    
    /* the potential functions */
    double f ( double r ) {
        return potential_LJ126 ( r , A , B ) +
            q * potential_Ewald( r , kappa );
        }
        
    double dfdr ( double r ) {
        return potential_LJ126_p ( r , A , B ) +
            q * potential_Ewald_p( r , kappa );
        }
        
    double d6fdr6 ( double r ) {
        return potential_LJ126_6p ( r , A , B ) +
            q * potential_Ewald_6p( r , kappa );
        }
        
    /* allocate the potential */
    if ( ( p = (struct potential *)malloc( sizeof( struct potential ) ) ) == NULL ) {
        error(potential_err_malloc);
        return NULL;
        }
        
    /* fill this potential */
    if ( potential_init( p , &f , &dfdr , &d6fdr6 , a , b , tol ) < 0 ) {
        free(p);
        return NULL;
        }
    
    /* return it */
    return p;

    }
    

/**
 * @brief Creates a #potential representing a 12-6 Lennard-Jones potential
 *
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param A The first parameter of the Lennard-Jones potential.
 * @param B The second parameter of the Lennard-Jones potential.
 * @param tol The tolerance to which the interpolation should match the exact
 *      potential.
 *
 * @return A newly-allocated #potential representing the potential
 *      @f$ \left( \frac{A}{r^{12}} - \frac{B}{r^6} \right) @f$ in @f$[a,b]@f$
 *      or @c NULL on error (see #potential_err).
 *
 */

struct potential *potential_create_LJ126 ( double a , double b , double A , double B , double tol ) {

    struct potential *p;
    
    double f ( double r2 ) {
    
        double ir2 = 1.0/r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;
    
        return ( A * ir12 - B * ir6 );
        
        }

    double fp ( double r2 ) {
    
        double ir2 = 1.0/r2, ir4 = ir2*ir2, ir12 = ir4 * ir4 * ir4;
    
        return 3 * ( -2 * A * ir12 * ir2 + B * ir4 * ir4 );
        
        }

    double f6p ( double r2 ) {
    
        double ir2 = 1.0 / r2, ir6 = ir2*ir2*ir2, ir12 = ir6 * ir6;
    
        return 10080 * ( 33 * A * ir12 * ir12 - 2 * B * ir12 * ir6 );
        
        }
        
    /* allocate the potential */
    if ( ( p = (struct potential *)malloc( sizeof( struct potential ) ) ) == NULL ) {
        error(potential_err_malloc);
        return NULL;
        }
        
    /* fill this potential */
    if ( potential_init( p , &f , &fp , &f6p , a , b , tol ) < 0 ) {
        free(p);
        return NULL;
        }
    
    /* return it */
    return p;

    }
    

/**
 * @brief Construct a #potential from the given function.
 *
 * @param p A pointer to an empty #potential.
 * @param f A pointer to the potential function to be interpolated.
 * @param fp A pointer to the first derivative of @c f.
 * @param f6p A pointer to the sixth derivative of @c f.
 * @param a The smallest radius for which the potential will be constructed.
 * @param b The largest radius for which the potential will be constructed.
 * @param tol The absolute tolerance to which the interpolation should match
 *      the exact potential.
 *
 * @return #potential_err_ok or <0 on error (see #potential_err).
 *
 * Computes an interpolated potential function from @c f in @c [a,b] to the
 * absolute tolerance @c tol.
 *
 * The sixth derivative @c f6p is used to compute the optimal node
 * distribution. If @c f6p is @c NULL, the derivative is approximated
 * numerically.
 *
 * The zeroth interval contains a linear extension of @c f for values < a.
 */

int potential_init ( struct potential *p , double (*f)( double ) , double (*fp)( double ) , double (*f6p)( double ) , FPTYPE a , FPTYPE b , FPTYPE tol ) {

    FPTYPE alpha, w;
    int l = potential_ivalsmin, r = potential_ivalsmax, m;
    FPTYPE err_l, err_r, err_m;
    FPTYPE *xi_l, *xi_r, *xi_m;
    FPTYPE *c_l, *c_r, *c_m;
    int i, k;
    FPTYPE e;

    /* check inputs */
    if ( p == NULL || f == NULL || fp == NULL )
        return error(potential_err_null);
        
    /* check if we have a user-specified 6th derivative or not. */
    if ( f6p == NULL )
        return error(potential_err_nyi);
        
    /* Stretch the domain ever so slightly to accommodate for rounding
       error when computing the index. */
    b *= (1.0 + sqrt(FPTYPE_EPSILON));
        
    /* set the boundaries */
    p->a = a; p->b = b;
    
    /* compute the optimal alpha for this potential */
    alpha = potential_getalpha(f6p,a,b);
    /* printf("potential_init: alpha is %e\n",alpha); fflush(stdout); */
    
    /* compute the interval transform */
    w = 1.0 / (a - b); w *= w;
    p->alpha[0] = -( a * ( alpha * b - a ) ) * w;
    p->alpha[1] = ( alpha * ( b + a ) - 2 * a ) * w;
    p->alpha[2] = -(alpha - 1) * w;
    
    /* compute the smallest interpolation... */
    /* printf("potential_init: trying l=%i...\n",l); fflush(stdout); */
    xi_l = (FPTYPE *)malloc( sizeof(FPTYPE) * (l + 1) );
    c_l = (FPTYPE *)malloc( sizeof(FPTYPE) * (l+1) * potential_chunk );
    if ( posix_memalign( (void **)&c_l , potential_align , sizeof(FPTYPE) * (l+1) * potential_chunk ) < 0 )
        return error(potential_err_malloc);
    for ( i = 0 ; i <= l ; i++ ) {
        xi_l[i] = a + (b - a) * i / l;
        while ( fabs( (e = i - l * (p->alpha[0] + xi_l[i]*(p->alpha[1] + xi_l[i]*p->alpha[2]))) ) > 3 * l * FPTYPE_EPSILON )
            xi_l[i] += e / (l * (p->alpha[1] + 2*xi_l[i]*p->alpha[2]));
        }
    if ( potential_getcoeffs(f,fp,xi_l,l,&c_l[potential_chunk],&err_l) < 0 )
        return error(potential_err);
    /* fflush(stderr); printf("potential_init: err_l=%e.\n",err_l); */
        
    /* if this interpolation is good enough, stop here! */
    if ( err_l < tol ) {
        p->n = l;
        p->c = c_l;
        p->alpha[0] *= p->n; p->alpha[1] *= p->n; p->alpha[2] *= p->n;
        p->alpha[0] += 1;
        p->c[0] = a; p->c[1] = 1.0 / a;
        p->c[potential_degree+2] = f(a);
        p->c[potential_degree+1] = fp(a) * a;
        p->c[potential_degree] = 0.0;
        for ( k = 2 ; k <= potential_degree ; k++ )
            p->c[potential_degree] += k * (k-1) * p->c[2*potential_chunk-k-1] * ( 1 - 2*(k%2) );
        p->c[potential_degree] *= a * a * p->c[potential_degree+4] * p->c[potential_degree+4];
        for ( k = 2 ; k < potential_degree ; k++ )
            p->c[k] = 0.0;
        free(xi_l);
        return potential_err_ok;
        }
        
    /* loop until we have an upper bound on the right... */
    while ( 1 ) {
    
        /* compute the larger interpolation... */
        /* printf("potential_init: trying r=%i...\n",r); fflush(stdout); */
        xi_r = (FPTYPE *)malloc( sizeof(FPTYPE) * (r + 1) );
        if ( posix_memalign( (void **)&c_r , potential_align , sizeof(FPTYPE) * (r+1) * potential_chunk ) != 0 )
            return error(potential_err_malloc);
        for ( i = 0 ; i <= r ; i++ ) {
            xi_r[i] = a + (b - a) * i / r;
            while ( fabs( (e = i - r*(p->alpha[0] + xi_r[i]*(p->alpha[1] + xi_r[i]*p->alpha[2]))) ) > 3 * r * FPTYPE_EPSILON )
                xi_r[i] += e / (r * (p->alpha[1] + 2*xi_r[i]*p->alpha[2]));
            }
        if ( potential_getcoeffs(f,fp,xi_r,r,&c_r[potential_chunk],&err_r) < 0 )
            return error(potential_err);
        /* printf("potential_init: err_r=%e.\n",err_r); fflush(stdout); */
            
        /* if this is better than tolerance, break... */
        if ( err_r < tol )
            break;
            
        /* otherwise, l=r and r = 2*r */
        else {
            l = r; err_l = err_r;
            free(xi_l); xi_l = xi_r;
            free(c_l); c_l = c_r;
            r *= 2;
            }

        } /* loop until we have a good right estimate */
        
    /* we now have a left and right estimate -- binary search! */
    while ( r - l > 1 ) {
    
        /* find the middle */
        m = 0.5 * ( r + l );
        
        /* construct that interpolation */
        // printf("potential_init: trying m=%i...\n",m); fflush(stdout);
        xi_m = (FPTYPE *)malloc( sizeof(FPTYPE) * (m + 1) );
        if ( posix_memalign( (void **)&c_m , potential_align , sizeof(FPTYPE) * (m+1) * potential_chunk ) != 0 )
            return error(potential_err_malloc);
        for ( i = 0 ; i <= m ; i++ ) {
            xi_m[i] = a + (b - a) * i / m;
            while ( fabs( (e = i - m*(p->alpha[0] + xi_m[i]*(p->alpha[1] + xi_m[i]*p->alpha[2]))) ) > 3 * m * FPTYPE_EPSILON )
                xi_m[i] += e / (m * (p->alpha[1] + 2*xi_m[i]*p->alpha[2]));
            }
        if ( potential_getcoeffs(f,fp,xi_m,m,&c_m[potential_chunk],&err_m) != 0 )
            return error(potential_err);
        // printf("potential_init: err_m=%e.\n",err_m); fflush(stdout);
            
        /* go left? */
        if ( err_m > tol ) {
            l = m; err_l = err_m;
            free(xi_l); xi_l = xi_m;
            free(c_l); c_l = c_m;
            }
            
        /* otherwise, go right... */
        else {
            r = m; err_r = err_m;
            free(xi_r); xi_r = xi_m;
            free(c_r); c_r = c_m;
            }
                
        } /* binary search */
        
    /* as of here, the right estimate is the smallest interpolation below */
    /* the requested tolerance */
    p->n = r;
    p->c = c_r;
    p->alpha[0] *= p->n; p->alpha[1] *= p->n; p->alpha[2] *= p->n;
    p->alpha[0] += 1.0;
    p->c[0] = a; p->c[1] = 1.0 / a;
    p->c[potential_degree+2] = f(a);
    p->c[potential_degree+1] = fp(a) * a;
    p->c[potential_degree] = 0.0;
    for ( k = 2 ; k <= potential_degree ; k++ )
        p->c[potential_degree] += k * (k-1) * p->c[2*potential_chunk-k-1] * ( 1 - 2*(k%2) );
    p->c[potential_degree] *= a * a * p->c[potential_chunk+1] * p->c[potential_chunk+1];
    for ( k = 2 ; k < potential_degree ; k++ )
        p->c[k] = 0.0;
    free(xi_r);
    free(xi_l); free(c_l);
        
    /* all is well that ends well... */
    return potential_err_ok;
    
    }
    
    
/**
 * @brief Compute the interpolation coefficients over a given set of nodes.
 * 
 * @param f Pointer to the function to be interpolated.
 * @param fp Pointer to the first derivative of @c f.
 * @param xi Pointer to an array of nodes between whicht the function @c f
 *      will be interpolated.
 * @param n Number of nodes in @c xi.
 * @param c Pointer to an array in which to store the interpolation
 *      coefficients.
 * @param err Pointer to a floating-point value in which an approximation of
 *      the interpolation error is stored.
 *
 * @return #potential_err_ok or < 0 on error (see #potential_err).
 *
 * Compute the coefficients of the function @c f with derivative @c fp
 * over the @c n intervals between the @c xi and store an estimate of the
 * maximum absolute interpolation error in @c err.
 *
 * The array to which @c c points must be large enough to hold at least
 * #potential_degree x @c n values of type #FPTYPE.
 */

int potential_getcoeffs ( double (*f)( double ) , double (*fp)( double ) , FPTYPE *xi , int n , FPTYPE *c , FPTYPE *err ) {

    int i, j, k, ind;
    FPTYPE phi[7], cee[6], fa, fb, dfa, dfb;
    FPTYPE h, m, w, e, x;
    FPTYPE fx[potential_N];

    /* check input sanity */
    if ( f == NULL || fp == NULL || xi == NULL || err == NULL )
        return error(potential_err_null);
        
    /* init the maximum interpolation error */
    *err = 0.0;
    
    /* loop over all intervals... */
    for ( i = 0 ; i < n ; i++ ) {
    
        /* set the initial index */
        ind = i * (potential_degree + 3);
        
        /* get the interval centre and width */
        m = (xi[i] + xi[i+1]) / 2;
        h = (xi[i+1] - xi[i]) / 2;
        
        /* evaluate f and fp at the edges */
        fa = f(xi[i]); fb = f(xi[i+1]);
        dfa = fp(xi[i]) * h; dfb = fp(xi[i+1]) * h;
        // printf("potential_getcoeffs: xi[i]=%22.16e\n",xi[i]);
        
        /* compute the coefficients phi of f */
        for ( k = 0 ; k < potential_N ; k++ )
            fx[k] = f( m + h * cos( k * M_PI / potential_N ) );
        for ( j = 0 ; j < 7 ; j++ ) {
            phi[j] = (fa + (1-2*(j%2))*fb) / 2;
            for ( k = 1 ; k < potential_N ; k++ )
                phi[j] += fx[k] * cos( j * k * M_PI / potential_N );
            phi[j] *= 2.0 / potential_N;
            }
        
        /* compute the first four coefficients */
        cee[0] = (4*(fa + fb) + dfa - dfb) / 4;
        cee[1] = -(9*(fa - fb) + dfa + dfb) / 16;
        cee[2] = (dfb - dfa) / 8;
        cee[3] = (fa - fb + dfa + dfb) / 16;
        cee[4] = 0.0;
        cee[5] = 0.0;
        
        /* add the 4th correction... */
        w = ( 6 * ( cee[0] - phi[0]) - 4 * ( cee[2] - phi[2] ) - phi[4] ) / ( 36 + 16 + 1 );
        cee[0] += -6 * w;
        cee[2] += 4 * w;
        cee[4] = -w;

        /* add the 5th correction... */
        w = ( 2 * ( cee[1] - phi[1]) - 3 * ( cee[3] - phi[3] ) - phi[5] ) / ( 4 + 9 + 1 );
        cee[1] += -2 * w;
        cee[3] += 3 * w;
        cee[5] = -w;
        
        /* convert to monomials on the interval [-1,1] */
        c[ind+7] = cee[0]/2 - cee[2] + cee[4];
        c[ind+6] = cee[1] - 3*cee[3] + 5*cee[5];
        c[ind+5] = 2*cee[2] - 8*cee[4];
        c[ind+4] = 4*cee[3] - 20*cee[5];
        c[ind+3] = 8*cee[4];
        c[ind+2] = 16*cee[5];
        c[ind+1] = 1.0 / h;
        c[ind] = m;

        /* compute a local error estimate (klutzy) */
        for ( k = 0 ; k < potential_N ; k++ ) {
            x = cos( k * M_PI / potential_N );
            e = fabs( fx[k] - c[ind+7]
                -x * ( c[ind+6] + 
                x * ( c[ind+5] + 
                x * ( c[ind+4] + 
                x * ( c[ind+3] + 
                x * c[ind+2] )))) );
            if ( e > *err )
                *err = e;
            }
        
        }
        
    /* all is well that ends well... */
    return potential_err_ok;

    }
    
    
/**
 * @brief Compute the parameter @f$\alpha@f$ for the optimal node distribution.
 *
 * @param f6p Pointer to a function representing the 6th derivative of the
 *      interpoland.
 * @param a Left limit of the interpolation.
 * @param b Right limit of the interpolation.
 *
 * @return The computed value for @f$\alpha@f$.
 *
 * The value @f$\alpha@f$ is computed using Brent's algortihm to 4 decimal
 * digits.
 */
 
double potential_getalpha ( double (*f6p)( double ) , double a , double b ) {

    double xi[potential_N], fx[potential_N];
    int i, j;
    double temp;
    double alpha[4], fa[4];
    const double golden = 2.0 / (1 + sqrt(5));
    
    /* start by evaluating f6p at the N nodes between 'a' and 'b' */
    for ( i = 0 ; i < potential_N ; i++ ) {
        xi[i] = a + (b-a) * i / (potential_N - 1);
        fx[i] = f6p( xi[i] );
        }
        
    /* set the initial values for alpha */
    alpha[0] = 0; alpha[3] = 2;
    alpha[1] = alpha[3] - 2 * golden; alpha[2] = alpha[0] + 2 * golden;
    for ( i = 0 ; i < 4 ; i++ ) {
        fa[i] = 0.0;
        for ( j = 0 ; j < potential_N ; j++ ) {
            temp = fabs( pow( alpha[i] + 2 * (1 - alpha[i]) * xi[j] , -6 ) * fx[j] );
            if ( temp > fa[i] )
                fa[i] = temp;
            }
        }
        
    /* main loop (brent's algorithm) */
    while ( alpha[3] - alpha[0] > 1.0e-4 ) {
    
        /* go west? */
        if ( fa[1] < fa[2] ) {
            alpha[3] = alpha[2]; fa[3] = fa[2];
            alpha[2] = alpha[1]; fa[2] = fa[1];
            alpha[1] = alpha[3] - (alpha[3] - alpha[0]) * golden;
            i = 1;
            }
            
        /* nope, go east... */
        else {
            alpha[0] = alpha[1]; fa[0] = fa[1];
            alpha[1] = alpha[2]; fa[1] = fa[2];
            alpha[2] = alpha[0] + (alpha[3] - alpha[0]) * golden;
            i = 2;
            }
            
        /* compute the new value */
        fa[i] = 0.0;
        for ( j = 0 ; j < potential_N ; j++ ) {
            temp = fabs( pow( alpha[i] + 2 * (1 - alpha[i]) * xi[j] , -6 ) * fx[j] );
            if ( temp > fa[i] )
                fa[i] = temp;
            }
    
        } /* main loop */
        
    /* return the average */
    return (alpha[0] + alpha[3]) / 2;

    }

