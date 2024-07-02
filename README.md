# deepnwp_oper

$ idev to the GPU - More info at https://docs.tacc.utexas.edu/hpc/lonestar6/

$ module load tacc-apptainer

$ apptainer build Apptainer_sfno.sif Apptainer_sfno.def

$ apptainer exec --bind /scratch/08105/ms86336:/opt/notebooks --nv Apptainer_sfno.sif  python sfno_embedding.py

$ apptainer exec --bind /scratch/08105/ms86336:/opt/notebooks --nv Apptainer_sfno.sif  python train_sfno.py



# For inference

1. Run graphcast to bring the graphcast output in a folder

2. Run the command below to bring the data in the form that can be ingested by graphcast

$ apptainer exec --bind /scratch/08105/ms86336:/opt/notebooks --nv Apptainer_sfno.sif python prepare_input_output_v1.py

$ apptainer exec --bind /scratch/08105/ms86336:/opt/notebooks --nv Apptainer_sfno.sif python inference_sfno.py graphcast_2021_01_02.nc
