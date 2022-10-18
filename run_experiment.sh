# Running on server 2:
#for nblocks in 1 2 3
#do
#python run_experiment.py --n-blocks $nblocks
#done

# Running the sresnet independent twice # Modified to have random flip
python run_experiment.py --diff-aug
python run_experiment.py # same aug




