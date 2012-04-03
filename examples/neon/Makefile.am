
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
bin_PROGRAMS = neon neon_verlet neon_pwverlet neon_pwverlet2

# Plain old tests
# Sources for test
neon_SOURCES = neon.c
neon_CFLAGS = -DENGINE_FLAGS="engine_flag_none" -DFPTYPE_SINGLE $(AM_CFLAGS)
neon_LDADD = ../../src/.libs/libmdcore_single.a


# Using Verlet lists
# Sources for neon_verlet
neon_verlet_SOURCES = neon.c
neon_verlet_CFLAGS = -DFPTYPE_SINGLE -DENGINE_FLAGS="engine_flag_verlet" $(AM_CFLAGS)
neon_verlet_LDADD = ../../src/.libs/libmdcore_single.a


# Using pairwise Verlet lists
# Sources for neon_pwverlet
neon_pwverlet_SOURCES = neon.c
neon_pwverlet_CFLAGS = -DFPTYPE_SINGLE -DENGINE_FLAGS="engine_flag_verlet_pairwise" $(AM_CFLAGS)
neon_pwverlet_LDADD = ../../src/.libs/libmdcore_single.a


# Using pairwise Verlet lists of the second kind
# Sources for neon_pwverlet2
neon_pwverlet2_SOURCES = neon.c
neon_pwverlet2_CFLAGS = -DFPTYPE_SINGLE -DENGINE_FLAGS="engine_flag_verlet_pairwise2" $(AM_CFLAGS)
neon_pwverlet2_LDADD = ../../src/.libs/libmdcore_single.a
