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

/* include some standard headers. */
#include <stdio.h>
#include <stdlib.h>

/* include local headers. */
#include "errs.h"
#include "btree.h"

/* define some constants. */
#define N 20


int main ( int argc , char *argv[] ) {

    int j, k, temp, data[N], *res, found;
    struct btree b;
    
    /* btree mapping function. */
    int map_print ( int *data , void *dummy ) {
        printf("map_print: node %3i.\n",*data);
        return 0;
        }
    
    /* fill and mix the data. */
    for ( k = 0 ; k < N ; k++ )
        data[k] = k;
    for ( k = 0 ; k < N ; k++ ) {
        j = rand() % N;
        temp = data[j];
        data[j] = data[k];
        data[k] = temp;
        }
        
    /* init the btree. */
    if ( btree_init( &b ) < 0 ) {
        printf("btree_test: call to btree_init failed.\n");
        errs_dump(stdout);
        return 0;
        }
        
    /* fill the btree with the data. */
    for ( k = 0 ; k < N ; k++ ) {
    
        if ( btree_insert( &b , data[k] , (void *)&data[k] ) < 0 ) {
            printf("btree_test: btree_insert failed for %ith data.\n",k);
            errs_dump(stdout);
            return 0;
            }
            
        }
        
    /* map the function 'map_print' to the data in the btree.
    if ( btree_map( &b , (btree_maptype)&map_print , NULL ) < 0 ) {
        printf("btree_test: call to btree_map failed.\n");
        errs_dump(stdout);
        return 0;
        } */
        
    /* look for the 13th entry.
    if ( ( found = btree_find( &b , 13 , (void **)&res ) ) < 0 ) {
        printf("btree_test: call to btree_find failed.\n");
        errs_dump(stdout);
        return 0;
        }
    printf("btree_test: data for node 13 is %3i (found=%i).\n",*res,found);
    */
    
    /* loop... */
    while ( 1 ) {
    
        /* dump the btree. */
        btree_dump( &b , stdout );
        
        /* get the key to delete. */
        printf("btree_test: enter key to delete: ");
        if ( scanf("%i",&k) < 1 )
            break;
        
        /* delete this key. */
        if ( ( found = btree_delete( &b , k , (void **)&res ) ) < 0 ) {
            printf("btree_test: call to btree_delete failed.\n");
            errs_dump(stdout);
            return 0;
            }
            
        /* display the returned key. */
        if ( found )
            printf("btree_test: deleted key %i had data %i.\n",k,*res);
        
        }
        
    /* to the pub! */
    return 0;
    
    }

