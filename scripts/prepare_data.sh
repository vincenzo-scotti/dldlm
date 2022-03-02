# NOTE this script must be run from the repository root
# Print help if required
help()
{
  # Print usage instructions
  echo "DLDLM data download and preparation script."
  echo
  echo "Syntax: prepare_data.sh [-h|d|s|p]"
  echo "Options:"
  echo "-h     Displays help."
  echo "-d     Download the raw versions of the data sets."
  echo "-s     Calls the scripts to standardise the data sets."
  echo "-p     Calls the scripts to prepare the corpus."
  echo
}
# Check options
download=false
standardise=false
prepare=false
while getopts :hpa: option;
do
  case "${option}" in
    h) help
       exit;;
    d) download=true;;
    s) standardise=true;;
    p) prepare=true;;
  esac
done

# Create directory to host the corpora
mkdir $DLDLM/resources/data/

# Create directory to host the raw corpora
mkdir $DLDLM/resources/data/raw/
if $download
then
  # Create directory to host the corpus
  mkdir $DLDLM/resources/data/raw/dailydialog/
  # Download the corpus
  wget https://parl.ai/downloads/dailydialog/dailydialog.tar.gz -P $DLDLM/resources/data/raw/ --no-check-certificate
  # Unpack downloaded archive
  tar -xzf $DLDLM/resources/data/raw/dailydialog.tar.gz -C $DLDLM/resources/data/raw/dailydialog/
  # Delete archive
  rm $DLDLM/resources/data/raw/dailydialog.tar.gz
  # Create directory to host the corpus
  mkdir $DLDLMresources/data/raw/empatheticdialogues/
  # Download the corpus
  wget https://parl.ai/downloads/empatheticdialogues/empatheticdialogues.tar.gz -P $DLDLMresources/data/raw/ --no-check-certificate
  # Unpack downloaded archive
  tar -xzf $DLDLMresources/data/raw/empatheticdialogues.tar.gz -C $DLDLMresources/data/raw/empatheticdialogues/ --strip=1
  # Delete archive
  rm $DLDLMresources/data/raw/empatheticdialogues.tar.gz
  # Create directory to host the corpus
  mkdir $DLDLM/resources/data/raw/personachat/
  # Download the corpus
  wget https://parl.ai/downloads/personachat/personachat.tgz -P $DLDLM/resources/data/raw/ --no-check-certificate
  # Unpack downloaded archive
  tar -xzf $DLDLM/resources/data/raw/personachat.tgz -C $DLDLM/resources/data/raw/personachat/ --strip=1
  # Delete archive
  rm $DLDLM/resources/data/raw/personachat.tgz
  # Create directory to host the corpus
  mkdir $DLDLM/resources/data/raw/wizard_of_wikipedia
  # Download the corpus
  wget https://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz -P $DLDLM/resources/data/raw/ --no-check-certificate
  # Unpack downloaded archive
  tar -xzf $DLDLM/resources/data/raw/wizard_of_wikipedia.tgz -C $DLDLM/resources/data/raw/wizard_of_wikipedia/
  # Delete archive
  rm $DLDLM/resources/data/raw/wizard_of_wikipedia.tgz
fi

# Create directory to host the standardised corpora
mkdir $DLDLM/resources/data/preprocessed
if $standardise
then
  # Run preprocessing scripts
  python $DLDLDM/standardize_daily_dialog.py
  python $DLDLDM/standardize_empathetic_dialogues.py
  python $DLDLDM/standardize_persona_chat_dataset.py
  python $DLDLDM/standardize_wizard_of_wikipedia.py
fi

# Create directory to host the prepared corpus
mkdir $DLDLDM/resources/data/dialogue_corpus
if $prepare
then
  # Run preprocessing scripts
  python $DLDLDM/prepare_dialogue_corpus.py
fi