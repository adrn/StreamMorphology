#!/bin/sh

# Directives
#PBS -N scf_spherical_2.5e6
#PBS -W group_list=yetiastro
#PBS -l nodes=1:ppn=1,walltime=8:00:00,mem=8gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

# print date and time to file
date

#Command to execute Python program
cd /vega/astro/users/amp2217/projects/morphology/simulations/runs/spherical_2.5e6
make
./scf
/vega/astro/users/amp2217/projects/gary/bin/moviesnap --path=/vega/astro/users/amp2217/projects/morphology/simulations/runs/spherical_2.5e6

date

#End of script
