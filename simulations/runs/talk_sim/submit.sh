#!/bin/sh

# Directives
#PBS -N scf_talk_sim
#PBS -W group_list=yetiastro
#PBS -l nodes=1:ppn=1,walltime=1:00:00,mem=4gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

# print date and time to file
date

#Command to execute Python program
cd /vega/astro/users/amp2217/projects/morphology/simulations/runs/talk_sim
make
./scf
/vega/astro/users/amp2217/tmp/moviesnap --path=/vega/astro/users/amp2217/projects/morphology/simulations/runs/talk_sim

date

#End of script
