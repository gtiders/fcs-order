#!/bin/bash
#PBS -S /bin/bash
#PBS -l walltime=100:00:00
#PBS -q six_hours
#PBS -l nodes=1:ppn=40
#PBS -N fc3rd_3RD.POSCAR.002
#PBS -V

cd "${PBS_O_WORKDIR}"

source /opt/intel/compilers_and_libraries_2018/linux/bin/compilervars.sh intel64
source /opt/intel/mkl/bin/mklvars.sh intel64
source /opt/intel/impi/2018.1.163/bin64/mpivars.sh

mpirun -np 40 /opt/software/vasp/vasp.5.4.4/vasp_std > log.dat
