# Population Morphology Implies a Common Developmental Blueprint for *Drosophila* Motion Detectors

Nikolas Drummond, Arthur Zhao, Alexander Borst

[BiorXiv](https://www.biorxiv.org/content/10.1101/2025.11.15.688637v1)

This is the repository for the paper: "Population Morphology Implies a Common Developmental Blueprint for *Drosophila* Motion Detectors". it includes datasets of extracted dendrite morphology metrics, and code for reproducing analysis and figures.

## Abstract

> Detailed characterization of neuronal morphology provides vital clues for understanding the wiring logic and development of neural circuits. As neuron arbours can both span large distances, and be densely interwoven, one major challenge is acquiring and quantifying fine structures for independent arbours over large spatial extents. Recent whole-brain Electron Microscopy (EM) connectomes of *Drosophila melanogaster* provide an ideal opportunity to study fly neuronal morphology at scale. Utilising this rich resource, we developed novel computational methods and morphological metrics to perform the most comprehensive morphological analysis on the dendrites of the T4 and T5 neurons in the fly brain.

> T4 and T5 neurons are the first direction-selective neurons in the visual pathway. They are the most numerous cell types in the fly brain (~6000 within each optic lobe) and as a population, their compact dendritic arbours span the entire visual field. They are classified into four subtypes (a,b,c, and d). Each subtype encodes one of four orthogonal motion directions (up, down, forwards, backwards). The dendrites of these neurons form in two distinct neuropils, the Medulla (T4) and the Lobula (T5), and respond to ON (light increments, T4) and OFF (light decrements, T5) motions. T4 and T5 neurons' dendrites are oriented against their preferred direction of motion. However, the differences beyond their characteristic orientation, both between T4 and T5, as well as within subtypes, has remained poorly understood.

> Our analysis reveals a high degree of structural similarity between T4 and T5, and within their subtypes. Particularly, the geometry of branching, section orientation, and tree-graph structure of these dendrites shows only minor variability, with no consistent separation between T4 and T5, or their subtypes.
 
> These results indicate that, despite forming in different neuropils, and serving distinct motion directions, T4 and T5 dendrites follow closely aligned morphological patterns. This suggests a shared developmental mechanism.

## Repository Structure 

```
.
├── Data
├── Notebooks
└── src
```

`Data` contains pickled `pandas.DataFrame` files which contain all the data needed to run analysis and reconstruct all figures within the paper. Additionally a `.csv` file is included which contains the neuron ID of every neuron used within this paper in [flywire](flywire.ai).

`Notebooks` contains a set of jupyter notebooks for running statistical analysis using the provided data, as well as reproduce all figures within the manuscript.

`src` contains `.py` files with functions needed to run the various analysis and plotting notebooks.

### Data

#### `Neuron_ids.csv`



|Metric|File|dtype|source|
|------|----|-----|------|


### Notebooks


### src

## Datasets



## Usage

