# Data

This directory is used to host the data set(s).
Data set(s) are available at the following links:

- [Counseling and Psychotherapy Transcripts: Volume II](https://search.alexanderstreet.com/ctrn/browse/title?showall=true): hosted by [Alexander Street](https://search.alexanderstreet.com) 
- [DailyDialog](https://www.aclweb.org/anthology/I17-1099/): hosted by [ParlAI](https://parl.ai) ([download link](https://parl.ai/downloads/dailydialog/dailydialog.tar.gz))
- [EmpatheticDialogues](https://www.aclweb.org/anthology/P19-1534/): hosted by [ParlAI](https://parl.ai) ([download link](https://parl.ai/downloads/empatheticdialogues/empatheticdialogues.tar.gz))
- [HOPE Dataset](https://dl.acm.org/doi/10.1145/3488560.3498509): hosted by authors with restricted ([request form link](https://docs.google.com/forms/d/e/1FAIpQLSfX_7yzABPtdo5FuhEPw8mosHJmHt|-|-3W6s4nTkL1ot7OCCiA/viewform))
- [Persona-Chat](https://aclanthology.org/P18-1205/): hosted by [ParlAI](https://parl.ai) ([download link](https://parl.ai/downloads/personachat/personachat.tgz))
- [Wizard of Wikipedia](https://arxiv.org/abs/1811.01241): hosted by [ParlAI](https://parl.ai) ([download link](https://parl.ai/downloads/wizard_of_wikipedia/wizard_of_wikipedia.tgz))

Directory structure:
```
 |- data/
    |- cache/
      |- ...
    |- raw/
      |- Counseling_and_Psychotherapy_Transcripts_Volume_II/
        |- 00000.txt
        |- 00001.txt
        |- ...
      |- dailydialog/
        |- test.json
        |- train.json
        |- valid.json
      |- empatheticdialogues/
        |- test.csv
        |- train.csv
        |- valid.csv
      |- HOPE_WSDM_2022/
        |- Test/
          |- Copy of 2.csv
          |- Copy of 4.csv
          |- ...
        |- Train/
          |- Copy of 1.csv
          |- Copy of 3.csv
          |- ...
        |- Validation/
          |- Copy of 16.csv
          |- Copy of 19.csv
          |- ...
      |- personachat/
        |- test_both_original.txt
        |- test_both_revised.txt
        |- ...
      |- wizard_of_wikipedia/
        |- data.json
        |- test_random_split.json
        |- ...
```