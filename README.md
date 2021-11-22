# Preparation Course Python Notebooks

This repository contains the PCP Notebooks, which introduce some basic material on Python programming as required for more advanced lab courses offered at FAU study programmes such as <a href="https://www.cme.studium.fau.de/">Communications and Multimedia Engineering (CME)</a> or <a href="https://www.asc.studium.fau.de/">Advanced Signal Processing and Communications Engineering (ASC)</a>. Furthermore, the PCP notebooks may be used as a gentle introduction to programming as needed in the more advanced <a href="https://www.audiolabs-erlangen.de/FMP">FMP Notebooks on Fundamentals of Music Processing</a>. While the first half of the PCP notebooks covers general Python concepts, the second half introduces and requires fundamental concepts in signal processing. The PCP notebooks are not intended to give a comprehensive overview of Python programming, nor are the notebooks self-contained. For a systematic introduction to Python programming, we refer to online sources such as  <a href="https://docs.python.org/3/tutorial/index.html">The Python Tutorial</a> or the <a href="https://scipy-lectures.org/">Scipy Lecture Notes</a>. The PCP notebooks have been inspired and borrow material from the <a href="https://www.audiolabs-erlangen.de/FMP">FMP Notebooks on Fundamentals of Music Processing</a>. The PCP Notebooks are freely accessible under the MIT License.

If a static view of the PCP notebooks is enough for you, the [exported HTML versions](https://www.audiolabs-erlangen.de/PCP) can be used right away without any installation. All material including the explanations, the figures, and the audio examples can be accessed by just following the **HTML links**. If you want to **execute** the Python code cells, you have to clone/download the notebooks (along with the data), create an environment, and start a Jupyter server. You then need to follow the **IPYNB links** within the Jupyter session. The necessary steps are explained in detail in the [PCP notebook on how to get started](https://www.audiolabs-erlangen.de/resources/MIR/PCP/PCP_getstarted.html).

## Installing Local Environment for Executing PCP Notebooks
This is the preferred and tested variant for using the PCP notebooks.

```
conda env create -f environment.yml
conda activate PCP
jupyter notebook
```

## Using Web-Based Services for Executing PCP Notebooks

### Google colab
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/meinardmueller/PCP/blob/master/PCP.ipynb)

The PCP notebooks may be executed using [Google colab](https://colab.research.google.com/). However, this needs some preparation. First, you need to be logged in with a Google account. The starting notebook can be accessed via:

https://colab.research.google.com/github/meinardmueller/PCP/blob/master/PCP.ipynb

For the other notebooks, clone the PCP repository to get access to data and the functions in `libpcp`. To this end, for each colab session, include and execute a code cell at the beginning of the notebook containing the following lines:

```
%%bash
git clone https://github.com/meinardmueller/PCP.git PCP_temp
mv PCP_temp/* .
rm -rd PCP_temp
```

### Binder
[![Open In Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/meinardmueller/PCP/master)

One can also use [Binder](https://mybinder.org/) to execute the PCP notebooks. This clones the repository and automatically creates a conda environment. This may take several (maybe even up to ten) minutes when starting binder.

https://mybinder.org/v2/gh/meinardmueller/PCP/master

## Acknowledgements

We want to thank the various people who have contributed to the design, implementation, and code examples of the notebooks. We mention the main contributors in alphabetical order: Michael Krause, Heinrich Löllmann, Meinard Müller, Sebastian Rosenzweig, Frank Zalkow. The [International Audio Laboratories Erlangen](https://www.audiolabs-erlangen.de/) are a joint institution of the [Friedrich-Alexander-Universität Erlangen-Nürnberg (FAU)](https://www.fau.eu/) and [Fraunhofer Institute for Integrated Circuits IIS](https://www.iis.fraunhofer.de/en.html).
