"""DPMC: mechanism selection for entrywise differentially private matrix completion.

The package contains everything needed to reproduce the experiments of
"Mechanism Selection for Private Matrix Completion":

* ``channels``      entrywise eps-DP mechanisms on a bounded rating grid, their public
                    channel statistics (attenuation, conditional noise, effective noise)
                    and samplers, plus the data-free selection rule.
* ``staircase``     the staircase mechanism retuned for squared error.
* ``closed_forms``  closed-form effective noise of the canonical designs, the two
                    crossover budgets, the reference-law tilt sweep and law-free dominance.
* ``optimal``       the linear program that lower-bounds effective noise over the whole
                    affine-mean class, randomized response on bins, and the exact channel
                    tables used in the end-to-end runs.
* ``mf``            the bias-aware matrix factorization fitted by alternating least squares.
* ``nuclear``       the nuclear-norm reconstruction used as an ablation.
* ``dpsgd``         the gradient-perturbation baselines and their privacy accountant.
* ``data``          dataset loaders, cached blocks, splits and the error metric.
* ``selection``     cross-validation on privatized values and deterministic seeding.
* ``stats``         paired t-test p-values and rank correlation.
"""
__version__ = "1.0.0"
