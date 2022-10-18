# Server 1
python scripts/plot_dominance_experiment.py bc6454 --arch sresnet --view_size 14 --dominance_save_name view14_lr0.075_sresnet_blur_diffaug_bc6454 # --plot_dominances
#python scripts/plot_dominance_experiment.py 7a08d0 --arch sresnet --view_size 16 --dominance_save_name view16_lr0.075_sresnet_blur_diffaug_7a08d0 # --plot_dominances
#python scripts/plot_dominance_experiment.py eb2000 --arch sresnet --view_size 16 --dominance_save_name view16_lr0.1_sresnet_blur_diffaug_eb2000 # --plot_dominances
python scripts/plot_dominance_experiment.py 32c2bb --arch sresnet --view_size 16 --dominance_save_name view16_lr0.075_sresnet_blur_diffaug_32c2bb # --plot_dominances
python scripts/plot_dominance_experiment.py 8b17cd --arch sresnet --view_size 18 --dominance_save_name view18_lr0.075_sresnet_blur_diffaug_8b17cd # --plot_dominances
python scripts/plot_dominance_experiment.py bd5fe9 --arch sresnet --view_size 16 --dominance_save_name view16_lr0.075_sresnet_blur_diffaug_randzero_bd5fe9 #--plot_dominances
#python scripts/plot_dominance_experiment_indep_original.py 8d9c4f --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.075_sresnetmulti_ind_diffaug_8d9c4f #--plot_dominances
#python scripts/plot_dominance_experiment_indep_original.py 516ec8 --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.05_sresnetmulti_ind_diffaug_516ec8 #--plot_dominances
#python scripts/plot_dominance_experiment_indep_original.py 98265c --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.05_wd0.00025_sresnetmulti_ind_diffaug_98265c #--plot_dominances
python scripts/plot_information_all_experiment.py bd5fe9 --save_dir view16_lr0.075_sresnet_blur_diffaug_randzero_bd5fe9

# Server 2
# python plot_multiple.py 5c9388 73522a 73522a --name deepness.pdf
#python scripts/plot_dominance_experiment_indep.py 52abd6 --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.075_sresnetmulti_ind_RHF_diffaug_52abd6 --plot_dominances
#python scripts/plot_dominance_experiment_indep.py 2cbc35 --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.075_sresnetmulti_ind_RHF_sameaug_2cbc35 --plot_dominances
#python scripts/plot_dominance_experiment_indep.py 7b58a9 --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.05_sresnetmulti_ind_RHF_diffaug_7b58a9 --plot_dominances
# python scripts/plot_dominance_experiment_indep.py e5eb36 --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.05_wd0.00025_sresnetmulti_ind_RHF_diffaug_e5eb36 --plot_dominances
# python scripts/plot_dominance_experiment_indep.py 1ad396 --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.05_wd0_sresnetmulti_ind_RHF_diffaug_1ad396 --plot_dominances
#python scripts/plot_dominance_experiment_indep.py 4ff6c0 --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.05_sresnetmulti_ind_RHF_sameaug_4ff6c0 --plot_dominances
# python scripts/plot_dominance_experiment_indep.py b2830e --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.05_sresnetmulti_ind_RHF_noaug_b2830e --plot_dominances
#python scripts/plot_dominance_experiment_indep.py dcd6cd --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.05_wd0_sresnetmulti_ind_RHF_noaug_dcd6cd --plot_dominances

# Server 3: only using one image (main_independent1.py)
#python scripts/plot_dominance_experiment.py 02dee1 --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.075_sresnetmulti_ind_oneImage_diffaug_02dee1 --plot_dominances
#python scripts/plot_dominance_experiment.py ead6b4 --arch sresnetmulti --view_size 16 --dominance_save_name view16_lr0.05_sresnetmulti_ind_oneImage_diffaug_ead6b4 --plot_dominances
