## Code for "[Robust Bayesian Satisficing](https://arxiv.org/abs/2308.08291)", (NeurIPS 2023)

### Setup
```setup
conda env create --name rbs --file environment.yml
```

## Reproduction

<!-- For each simulation in our paper, we present a Jupyter Notebook. `synthetic_exp.ipynb` is for the proof of concept experiment. `insulin_dosage_exp.ipynb` includes the insulin dosage experiment with both setup. `sensitivity_tau_vs_eps_exp.ipynb` includes the sensitivity analysis of $\tau$ and $r$ for RoBOS and DRBO, respectively.

--- -->

For each of the simulations described in our paper, we have provided a corresponding Jupyter Notebook. Below is a list of the notebooks and a brief description of each:

- **[`synthetic_exp.ipynb`](synthetic_exp.ipynb)**: This notebook demonstrates the proof of concept experiment.

- **[`insulin_dosage_exp.ipynb`](insulin_dosage_exp.ipynb)**: This notebook demonstrates the insulin dosage experiment with both setup scenarios.

- **[`sensitivity_tau_vs_eps_exp.ipynb`](sensitivity_tau_vs_eps_exp.ipynb)**: This notebook provides the sensitivity analysis of $\tau$ for RoBOS and $r$ for DRBO.

**Note**: Please make sure to follow the comments within each notebook to ensure smooth execution.


## You can cite RBS as below:
```
@inproceedings{
  saday2023rbs,
  title={Robust Bayesian Satisficing},
  author={Saday, Artun and Yıldırım, Y. Cahit and Tekin, Cem},
  booktitle={Advances in Neural Information Processing Systems 37},
  year={2023},
  url={https://arxiv.org/abs/2308.08291}
}
```
