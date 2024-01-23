# FL_Hunter
## Table of Contents
=================

  
  * [Usage](#usage)
  * [Datasets](#datasets)
  * [Introduce](#introduce)
  



### Usage
| Algorithm             | Usage                                                                                                                                                                                      |
|-----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FL_Hunter             | Run for dataset A1: `python exp/run_GAT_node_classification.py -H=4 -L=8 -fe=GRU -bal=True --data_dir=data/A1`                                                                             |
| DejaVu                | Run for dataset A1: `python exp/run_GAT_node_classification.py -H=4 -L=8 -fe=GRU -bal=True --data_dir=data/A1`                                                                             |
| Brandon               | Run for dataset A1: `python exp/FL_Hunter/run_Brandon.py --data_dir=data/A1`                                                                                                               |
| iSQUAD                | Run for dataset A1: `python exp/FL_Hunter/run_iSQ.py --data_dir=data/A1`                                                                                                                   |
| Decision Tree         | Run for dataset A1: `python exp/run_DT_node_classification.py --data_dir=data/A1`                                                                                                          |
| RandomWalk@Metric     | Run for dataset A1: `python exp/FL_Hunter/run_random_walk_single_metric.py --data_dir=data/A1 --window_size 60 10 --score_aggregation_method=min`                                          |
| RandomWalk@FI         | Run for dataset A1: `python exp/FL_Hunter/run_random_walk_failure_instance.py --data_dir=data/A1 --window_size 60 10 --anomaly_score_aggregation_method=min --corr_aggregation_method=max` |
| Global interpretation | Run `notebooks/explain.py` as a jupyter notebook with `jupytext`                                                                                                                           |
| Local interpretation  | `FL_Hunter/explanability/similar_faults.py`                                                                                                                                                |

The commands would print a `one-line summary` in the end, including the following fields: `A@1`, `A@2`, `A@3`, `A@5`, `MAR`, `Time`, `Epoch`, `Valid Epoch`, `output_dir`, `val_loss`, `val_MAR`, `val_A@1`, `command`, `git_commit_url`, which are the desrired results.

Totally, the main experiment commands of FL_Hunter should output as follows:
- FDG message, including the data paths, edge types, the number of nodes (failure units), the number of metrics, the metrics of each failure class.
- Traning setup message: the faults used for training, validation and testing.
- Model architecture: model parameters in each part, total params
- Training process: the training/validation/testing loss and accuracy
- Time Report.
- command output one-line summary.



## Datasets

The datasets A, B, C, D are public at :
- https://www.dropbox.com/sh/ist4ojr03e2oeuw/AAD5NkpAFg1nOI2Ttug3h2qja?dl=0
- https://doi.org/10.5281/zenodo.6955909 (including the raw data of the Train-Ticket dataset)
In each dataset, `graph.yml` or `graphs/*.yml` are FDGs, `metrics.csv` is metrics, and `faults.csv` is failures (including ground truths).
`FDG.pkl` is a pickle of the FDG object, which contains all the above data.
Note that the pickle files are not compatible in different Python and Pandas versions. So if you cannot load the pickles, just ignore and delete them. They are only used to speed up data load.



## Introduce


The complex dependencies between the characteristics of online service system failures pose significant challenges for fault localization. 
To address the challenges of long-distance dependencies among fault characteristics and imbalanced distribution of fault knowledge, we present a fault localization model, named FL-Hunter. 
FL-Hunter consists of three core components. The feature encoding component relies on graph attention networks and gated recurrent units to capture the complex temporal and spatial dependency fault features. 
The fault location component captures the dependency relationships among fault features in a three-stage manner based on multi-attention. The fault knowledge balancing component addresses the issue of imbalanced fault knowledge distribution through the use of a weighted Kullback-Leibler divergence loss function. 
Extensive experiments were conducted on datasets from four real-world projects: a microservice system from a major ISP, a service-oriented system from a commercial bank and an Oracle database from another commercial bank and an open-source microservice benchmark system. 
Experimental results show that FL-Hunter outperforms state-of-the-art approaches in fault localization



