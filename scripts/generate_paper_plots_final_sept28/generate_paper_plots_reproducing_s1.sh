# Server 1
python scripts/plot_dominance_experiment.py bc6454 --arch sresnet --view_size 14 --dominance_save_name view14_lr0.075_sresnet_blur_diffaug_bc6454 --plot_dominances
python scripts/plot_dominance_experiment.py 32c2bb --arch sresnet --view_size 16 --dominance_save_name view16_lr0.075_sresnet_blur_diffaug_32c2bb --plot_dominances
python scripts/plot_dominance_experiment.py 8b17cd --arch sresnet --view_size 18 --dominance_save_name view18_lr0.075_sresnet_blur_diffaug_8b17cd --plot_dominances
python scripts/plot_dominance_experiment.py bd5fe9 --arch sresnet --view_size 16 --dominance_save_name view16_lr0.075_sresnet_blur_diffaug_randzero_bd5fe9 --plot_dominances
python scripts/plot_information_all_experiment.py bd5fe9 --save_dir view16_lr0.075_sresnet_blur_diffaug_randzero_bd5fe9

