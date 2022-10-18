Run `python singular_analytical.py` in PyCharm to reproduce Fig. 3 of Saxe et al. 2019, and investigate how perturbations to the input-output correlation matrix affect the dynamics for learning features.

Set the flag `use_perturbed=True` to use a modified input-output correlation matrix.

After running `python singular_analytical.py` with the `use_perturbed` flag set to `True` and `False`, run `plot_difference.py` to plot the norm of the rows.

`plot_synthetic_fsv.py` computes the fsv distribution for a synthetic generative model.
