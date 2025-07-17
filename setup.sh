#!/bin/bash

pip install spacy
python -m spacy download en_core_web_sm

# Manually link the model
python -m spacy link en_core_web_sm en_core_web_sm

