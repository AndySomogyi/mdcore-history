
# This file is part of mdcore.
# Coypright (c) 2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Automake stuff
AUTOMAKE_OPTIONS=gnu

# Add the source directory and debug to CFLAGS
AM_CFLAGS = -g -I../src -Wall $(OPENMP_CFLAGS)

# Set the linker flags if parmetis is present
if HAVEPARMETIS
AM_LDFLAGS = $(PARMETIS_LDFLAGS)
endif

# Set-up the library
bin_PROGRAMS = test test_single test_verlet test_single_verlet test_pwverlet test_single_pwverlet btree_test

# Plain old tests
# Sources for test
test_SOURCES = test.c
test_LDADD = ../src/.libs/libmdcore.a

# Sources for test_single
test_single_SOURCES = test.c
test_single_CFLAGS = -DENGINE_FLAGS="engine_flag_tuples" -DFPTYPE_SINGLE $(AM_CFLAGS)
test_single_LDADD = ../src/.libs/libmdcore_single.a


# Using Verlet lists
# Sources for test_verlet
test_verlet_SOURCES = test.c
test_verlet_CFLAGS = -DENGINE_FLAGS="engine_flag_verlet" $(AM_CFLAGS)
test_verlet_LDADD = ../src/.libs/libmdcore.a

# Sources for test_single_verlet
test_single_verlet_SOURCES = test.c
test_single_verlet_CFLAGS = -DFPTYPE_SINGLE -DENGINE_FLAGS="engine_flag_verlet" $(AM_CFLAGS)
test_single_verlet_LDADD = ../src/.libs/libmdcore_single.a


# Using pairwise Verlet lists
# Sources for test_pwverlet
test_pwverlet_SOURCES = test.c
test_pwverlet_CFLAGS = -DENGINE_FLAGS="engine_flag_verlet_pairwise" $(AM_CFLAGS)
test_pwverlet_LDADD = ../src/.libs/libmdcore.a

# Sources for test_single_verlet
test_single_pwverlet_SOURCES = test.c
test_single_pwverlet_CFLAGS = -DFPTYPE_SINGLE -DENGINE_FLAGS="engine_flag_verlet_pairwise" $(AM_CFLAGS)
test_single_pwverlet_LDADD = ../src/.libs/libmdcore_single.a

# Sources for btree_test
btree_test_SOURCES = btree_test.c
btree_test_LDADD = ../src/.libs/libmdcore.a

# Dummy target to make the Cell/BE version
if HAVECELL
test_cell: test.c ../src/.libs/libmdcore_cell.a
	make -f Makefile_cell
clean_cell:
	make -f Makefile_cell clean
	rm -f test_cell
clean-recursive: clean_cell
all: test_cell
endif

