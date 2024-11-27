#!/bin/bash

for notebook in Notebooks/DatasetGenerator/*.ipynb; do
    echo "Running $notebook"
    jupyter-nbconvert --to notebook --execute --inplace "$notebook"
done