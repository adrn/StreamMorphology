#!/bin/sh

# Directives
#PBS -N scf_thin
#PBS -W group_list=yetiastro
#PBS -l nodes=1:ppn=1,walltime=8:00:00,mem=42gb
#PBS -M amp2217@columbia.edu
#PBS -m abe
#PBS -V

# Set output and error directories
#PBS -o localhost:/vega/astro/users/amp2217/pbs_output
#PBS -e localhost:/vega/astro/users/amp2217/pbs_output

# print date and time to file
date

#Command to execute Python program
/vega/astro/users/amp2217/anaconda/bin/python /vega/astro/users/amp2217/projects/morphology/scripts/debris_actions.py -f /vega/astro/users/amp2217/projects/morphology/simulations/runs/thin/SNAP015 --output=/vega/astro/users/amp2217/projects/morphology/simulations/runs/thin/actions -v --nsteps=500000 --dt=0.25 

date

#End of script
