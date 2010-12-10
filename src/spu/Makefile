
# Targets
PROGRAM_spu := runner_spu
LIBRARY_embed64 := librunner_spu.a

# Objects
OBJS = runner_spu.o

# Local Defines
IMPORTS = -lm -lmisc -lsimdmath

# Compiler flags and defines
CC_OPT_LEVEL := -O3
CFLAGS := -g
CFLAGS := $(CFLAGS) -DVECTORIZE
CFLAGS := $(CFLAGS) -DSORTED_INTERACTIONS
# CFLAGS := $(CFLAGS) -DEXPLICIT_POTENTIALS
# CFLAGS := $(CFLAGS) -DUSE_DOUBLES

# Footer
ifdef CELL_TOP
	include $(CELL_TOP)/buildutils/make.footer
else
	include /opt/cell/sdk/buildutils/make.footer
endif
