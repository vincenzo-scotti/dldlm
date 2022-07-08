#!/bin/bash
# NOTE this script must be run from the repository root
# NOTE to run in background detached:
# $ nohup ./scripts/download_data.sh > download.out &

# Print help if required
help()
{
  # Print usage instructions
  echo "DLDLM data download script."
  echo
  echo "Syntax: download_data.sh [-h]"
  echo "Options:"
  echo "-h     Displays help."
  echo
}
# Check options
while getopts :h: option;
do
  case "${option}" in
    h) help
       exit;;
  esac
done

# Create directory to host the corpora
mkdir -p ./resources/data/
# Create directory to host the raw corpora
mkdir -p ./resources/data/raw/
# Create directory to host the cache
mkdir -p ./resources/data/cache/

# Download and unpack all corpora
# Create directory to host the corpus
mkdir -p $DLDLM/resources/data/raw/dailydialog/
# Download the corpus
wget https://parl.ai/downloads/dailydialog/dailydialog.tar.gz -P $DLDLM/resources/data/raw/ --no-check-certificate
# Unpack downloaded archive
tar -xzf $DLDLM/resources/data/raw/dailydialog.tar.gz -C $DLDLM/resources/data/raw/dailydialog/
# Delete archive
rm $DLDLM/resources/data/raw/dailydialog.tar.gz
# Create directory to host the corpus
mkdir -p $DLDLM/resources/data/raw/empatheticdialogues/
# Download the corpus
wget https://parl.ai/downloads/empatheticdialogues/empatheticdialogues.tar.gz -P $DLDLM/resources/data/raw/ --no-check-certificate
# Unpack downloaded archive
tar -xzf $DLDLM/resources/data/raw/empatheticdialogues.tar.gz -C $DLDLM/resources/data/raw/empatheticdialogues/ --strip=1
# Delete archive
rm $DLDLM/resources/data/raw/empatheticdialogues.tar.gz
# Create directory to host the corpus
mkdir -p $DLDLM/resources/data/raw/personachat/
# Download the corpus
wget https://parl.ai/downloads/personachat/personachat.tgz -P $DLDLM/resources/data/raw/ --no-check-certificate
# Unpack downloaded archive
tar -xzf $DLDLM/resources/data/raw/personachat.tgz -C $DLDLM/resources/data/raw/personachat/ --strip=1
# Delete archive
rm $DLDLM/resources/data/raw/personachat.tgz
# Create directory to host the corpus
mkdir -p $DLDLM/resources/data/raw/wizard_of_wikipedia
# Download the corpus
wget https://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz -P $DLDLM/resources/data/raw/ --no-check-certificate
# Unpack downloaded archive
tar -xzf $DLDLM/resources/data/raw/wizard_of_wikipedia.tgz -C $DLDLM/resources/data/raw/wizard_of_wikipedia/
# Delete archive
rm $DLDLM/resources/data/raw/wizard_of_wikipedia.tgz