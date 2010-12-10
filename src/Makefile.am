
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

# Add the debug flag to the whole thing
AM_CFLAGS = $(SIMD_FLAGS) $(CFLAGS) -DFPTYPE_SINGLE

# Build the libchebfun library
lib_LTLIBRARIES = libmdcore.la
libmdcore_la_SOURCES = cell.c part.c runner.c engine.c potential.c space.c errs.c btree.c

# List required headers
include_HEADERS = cell.h part.h runner.h engine.h potential.h space.h fptype.h

# Assign a "safe" version number
libmdcore_la_LDFLAGS = -version-info 0:0:0

