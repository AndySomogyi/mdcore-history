
# This file is part of mdcore.
# Coypright (c) 2010 Pedro Gonnet (gonnet@maths.ox.ac.uk)
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Automake stuff
AUTOMAKE_OPTIONS=gnu

# Add the source directory and debug to CFLAGS
AM_CFLAGS = -g -I../../src -Wall $(OPENMP_CFLAGS)

# Set the linker flags if metis is present
AM_LDFLAGS = $(METIS_LDFLAGS)

# Set-up the library
bin_PROGRAMS = bulk bulk_single bulk_verlet bulk_single_verlet bulk_pwverlet bulk_single_pwverlet bulk_pwverlet2 bulk_single_pwverlet2

# Plain old bulks
# Sources for bulk
bulk_SOURCES = bulk.c
bulk_LDADD = ../../src/.libs/libmdcore.a

# Sources for bulk_single
bulk_single_SOURCES = bulk.c
bulk_single_CFLAGS = -DENGINE_FLAGS="engine_flag_tuples" -DFPTYPE_SINGLE $(AM_CFLAGS)
bulk_single_LDADD = ../../src/.libs/libmdcore_single.a


# Using Verlet lists
# Sources for bulk_verlet
bulk_verlet_SOURCES = bulk.c
bulk_verlet_CFLAGS = -DENGINE_FLAGS="engine_flag_verlet" $(AM_CFLAGS)
bulk_verlet_LDADD = ../../src/.libs/libmdcore.a

# Sources for bulk_single_verlet
bulk_single_verlet_SOURCES = bulk.c
bulk_single_verlet_CFLAGS = -DFPTYPE_SINGLE -DENGINE_FLAGS="engine_flag_verlet" $(AM_CFLAGS)
bulk_single_verlet_LDADD = ../../src/.libs/libmdcore_single.a


# Using pairwise Verlet lists
# Sources for bulk_pwverlet
bulk_pwverlet_SOURCES = bulk.c
bulk_pwverlet_CFLAGS = -DENGINE_FLAGS="engine_flag_verlet_pairwise" $(AM_CFLAGS)
bulk_pwverlet_LDADD = ../../src/.libs/libmdcore.a

# Sources for bulk_single_verlet
bulk_single_pwverlet_SOURCES = bulk.c
bulk_single_pwverlet_CFLAGS = -DFPTYPE_SINGLE -DENGINE_FLAGS="engine_flag_verlet_pairwise" $(AM_CFLAGS)
bulk_single_pwverlet_LDADD = ../../src/.libs/libmdcore_single.a


# Using pairwise Verlet lists of the second kind
# Sources for bulk_pwverlet2
bulk_pwverlet2_SOURCES = bulk.c
bulk_pwverlet2_CFLAGS = -DENGINE_FLAGS="engine_flag_verlet_pairwise2" $(AM_CFLAGS)
bulk_pwverlet2_LDADD = ../../src/.libs/libmdcore.a

# Sources for bulk_single_verlet2
bulk_single_pwverlet2_SOURCES = bulk.c
bulk_single_pwverlet2_CFLAGS = -DFPTYPE_SINGLE -DENGINE_FLAGS="engine_flag_verlet_pairwise2" $(AM_CFLAGS)
bulk_single_pwverlet2_LDADD = ../../src/.libs/libmdcore_single.a

# Dummy target to make the Cell/BE version
if HAVECELL
bulk_cell: bulk.c ../../src/.libs/libmdcore_cell.a
	make -f Makefile_cell
clean_cell:
	make -f Makefile_cell clean
	rm -f bulk_cell
clean-recursive: clean_cell
all: bulk_cell
endif

