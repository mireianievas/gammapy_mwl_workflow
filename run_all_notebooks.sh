#!/bin/bash

for notebook in Notebooks/*.ipynb; do
    echo "Running $notebook"
    jupyter-nbconvert --to notebook --execute --inplace "$notebook"
done