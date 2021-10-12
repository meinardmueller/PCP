# Utility Script for Exporting the PCP Notebooks to HTML

The following commands should be run from the top level of this repository:
`python tools/notebook_batch.py --mode clean .` (to remove output of executed cells)
`python tools/notebook_batch.py --mode execute .` (to execute all notebooks)
`python tools/notebook_batch.py --mode html .` (to export all executed notebooks to HTML)
