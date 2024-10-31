# An unified approach to link prediction in collaboration networks

This repository contains the code and associated files for the paper **"Comparison of Traditional and Machine Learning Models for Link Prediction in Graphs"**. It explores and compares various approaches, from traditional statistical models to machine learning techniques, assessing their accuracy and efficiency in predicting links in complex networks.

## Repository Structure

- **Code:** Implementation of the models and techniques used in the study, including statistical models such as ERGM and machine learning techniques like GCN and Word2Vec.
- **Data:** Network datasets used in the analysis, including Astro-Ph, Cond-Mat, Gr-Qc, Hep-Ph, and Hep-Th.
- **Models:** Files with the fitted models.
## Requirements

To reproduce the experiments, ensure you have:

- Python 3.8 or higher
- Required libraries: `networkx`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `torch` y `tensorflow` 

## Execution

1. Clone this repository:

   ```bash
   git clone https://github.com/damartinezsi/An-unified-approach-to-link-prediction-in-collaboration-networks.git
   cd An-unified-approach-to-link-prediction-in-collaboration-networks

2. Run the notebooks in the `code` directory to replicate experiments and view results. To skip model fitting, models can be loaded directly from the "models" folder.

## Contact

For questions or comments about the code or the paper, feel free to reach out at damartinezsi@unal.edu.co
