## Grokking, Neural Collapse and Relative Flatness

The code is implemented based on the implementation of paper [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) by Alethea Power, Yuri Burda, Harri Edwards, Igor Babuschkin, and Vedant Misra

The original github link: https://github.com/openai/grok

## Installation

```bash
pip install -e .
```

## Dependency Correction 
We suggest using the following library versions to ensure the experiments run correctly:

```bash
pip install torch==1.13.0 pytorch-lightning==1.2.10
```

After training, you can use `read_loss_acc.ipynb` to process the loss and accuracy logs for the run in the **default** directory. After that, you can use `plot_grok_cluster.ipynb` to plot all results together. Please copy and paste both notebooks into the corresponding results directories before running them.

