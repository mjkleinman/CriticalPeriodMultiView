python plot_experiment.py -s -f pdf  -t "Blur Pathway" e99579 -n 'plots/sresnet.pdf'
#python plot_experiment.py -s -f pdf  -t "Strabismus" da5174 -n 'plots/indep_views_sresnet.pdf'
python plot_experiment.py -s -f pdf  -t "Strabismus" f0e5f1 -n 'plots/indep_views_sresnet2.pdf'
#python plot_experiment.py -s -f pdf  -t "Strabismus" 90f69a -n 'plots/indep_views_sresnet3.pdf'

# No deficit first epoch
python plot_experiment.py -s -f pdf  -t "Strabismus" b9ba66 -n 'plots/indep_views_sresnet4.pdf'

python plot_experiment.py -s -f pdf  -t "Strabismus" 5b4d77 -n 'plots/indep_views_sresnet_240p.pdf'

python plot_experiment.py -s -f pdf  -t "Strabismus" c28c82 -n 'plots/indep_views_sresnet_240p_lr0p09.pdf'

# Blur with information tracking
python plot_experiment.py -s -f pdf  -t "Strabismus" 094de6 -n 'plots/sresnet_blur_information_tracking.pdf'
python scripts/plot_information_all_experiment.py 094de6
python scripts/plot_dominance_experiment.py 094de6 --dominance_save_name 'blur_info_tracking.pdf'

# Plotting dominance
python scripts/plot_dominance_experiment.py b9ba66 --dominance_save_name 'blur_normalized_sum.pdf'

# Independent multihead
python plot_experiment.py -s -f pdf  -t "Strabismus" abe7ef -n 'plots/sresnet_multihead.pdf'
python scripts/plot_dominance_experiment.py abe7ef --dominance_save_name 'indep_multihead.pdf' --arch 'sresnetmulti'

# View size 16 Aug 1
python plot_experiment.py -s -f pdf  -t "Strabismus" 0634f1 -n 'plots/sresnet_16view_multihead.pdf'
python scripts/plot_dominance_experiment.py 0634f1 --dominance_save_name 'indep_16view_multihead.pdf' --arch 'sresnetmulti' --view_size 16

python plot_experiment.py -s -f pdf  -t "Strabismus" df7678 -n 'plots/sresnet_16view_multihead_augmented.pdf'
python plot_experiment.py -s -f pdf  -t "Strabismus" 980907 -n 'plots/sresnet_16view_multihead_augmented.pdf'
python plot_experiment.py -s -f pdf  -t "Strabismus" 1b606c -n 'plots/sresnet_16view_multihead_augmented.pdf'
python plot_experiment.py -s -f pdf  -t "Strabismus" c89467 -n 'plots/sresnet_16view_multihead_augmented.pdf'
python scripts/plot_dominance_experiment.py c89467 --dominance_save_name 'indep_16view_multihead.pdf' --arch 'sresnetmulti' --view_size 14

# Aug 4
python plot_experiment.py -s -f pdf  -t "Strabismus" 9b4eb8 -n 'plots/sresnet_16view_multihead_augmented.pdf'
python scripts/plot_dominance_experiment.py 9b4eb8 --dominance_save_name 'indep_16view_multihead.pdf' --arch 'sresnetmulti' --view_size 16

python plot_experiment.py -s -f pdf  -t "Strabismus" c49172 -n 'plots/sresnet_14view_multihead_augmented.pdf'
python scripts/plot_dominance_experiment.py c49172 --dominance_save_name 'indep_14view_multihead.pdf' --arch 'sresnetmulti' --view_size 14

