
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

# Set-up CFLAGS for all targets
AM_CFLAGS = -g -Wall $(SIMD_FLAGS)
AM_LDFLAGS = -version-info 0:0:0 $(METIS_LDFLAGS)

# List required headers
include_HEADERS = cell.h part.h runner.h engine.h potential.h space.h fptype.h errs.h

# Build the libchebfun libraries
lib_LTLIBRARIES = libmdcore.la libmdcore_single.la


# Set-up for general (double-precision) case
libmdcore_la_SOURCES = cell.c part.c runner.c engine.c potential.c space.c errs.c btree.c


# Set-up for single-precision case
libmdcore_single_la_SOURCES = cell.c part.c runner.c engine.c potential.c space.c errs.c btree.c
libmdcore_single_la_SHORTNAME = sp
libmdcore_single_la_CFLAGS = -DFPTYPE_SINGLE $(AM_CFLAGS)


# Dummy target to make the Cell/BE version
if HAVECELL
libmdcore_cell: *.[ch] spu/runner_spu-embed64.o
	make -f Makefile_cell
	mv libmdcore_cell.a .libs/
clean_cell:
	make -f Makefile_cell clean
	rm -f ppu64_*.[ch]
	rm -f .libs/libmdcore_cell.a
clean-recursive: clean_cell
all: libmdcore_cell
endif

# Show the way...
SUBDIRS = spu
