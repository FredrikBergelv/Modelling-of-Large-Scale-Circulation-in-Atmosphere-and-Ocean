#ON LOCAL MACHINES
# using brew (both linux and macos) is the easiest to set up
# install with `brew install netcdf` and `brew install netcdf-fortran`
# then look up the path for `gfortran` and `netcdf-fortran` with
# `brew info netcdf-fortran` respectively
# one needs to also set the gfortran path, since gfortran might also
# be also available in the system outside of brew
# for example
# GFORTRAN = /usr/local/Cellar/gcc/13.1.0/bin
# NCDF_ROOT = /usr/local/Cellar/netcdf-fortran/4.6.1
#
# ON MISU COMPUTERS
# GFORTRAN should be left empty here, and `NCDF_ROOT = /usr/local/netcdf`

# user defined paths (user input usually needed)

NCDF_ROOT = /home/linuxbrew/.linuxbrew/Cellar/netcdf-fortran/4.6.2

# compilation (no user input needed)
FC = gfortran
FF = -g -O3  -fbounds-check -fbacktrace

LIB = -L$(NCDF_ROOT)/lib -lnetcdf -lnetcdff
OBJ = mod_write_nc.o

NAME = st_venant.f90
MODNAME = src/mod_write_nc.f90

All: st_venant.x

st_venant.x: src/$(NAME)
	$(FC) $(FF) -o st_venant.x ${MODNAME} src/$(NAME) -I $(NCDF_ROOT)/include $(LIB)

hej:
	@echo hej!

clean:
	rm -f *.x *.bin src/*.o src/*.mod *.mod *~ *\#

