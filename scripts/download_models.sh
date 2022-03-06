#!/bin/bash
# Create directory to host the models
mkdir $DLDLM/resources/models/
# Download the corpus
# TODO add correct download address
wget ... -P $DLDLM/resources/ --no-check-certificate
# Unzip downloaded models
unzip -j $DLDLM/resources/models.zip -d $DLDLM/resources/
# Delete archive
rm $DLDLM/resources/models.zip
rm -rf __*