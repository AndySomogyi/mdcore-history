
# This file is part of libchebfun.
# Coypright (c) 2010 The Chebfun Team.
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
AM_CFLAGS = -g -I../src -O2 $(CFLAGS) $(OPENMP_CFLAGS)

# Set-up the library
bin_PROGRAMS = test test_single btree_test

# Sources for test
test_SOURCES = test.c
test_LDADD = ../src/.libs/libmdcore.a

# Sources for test_single
test_single_SOURCES = test.c
test_single_CFLAGS = -DFPTYPE_SINGLE $(AM_CFLAGS)
test_single_LDADD = ../src/.libs/libmdcore_single.a

# Sources for fun_test
btree_test_SOURCES = btree_test.c
btree_test_LDADD = ../src/.libs/libmdcore.a

