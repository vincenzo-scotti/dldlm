from enum import Enum
from .dialogue_corpus import DialogueCorpus, EpisodicDialogueCorpus


IGNORE_INDEX = -100


class DataSetSplit(Enum):
    DEVELOPMENT: str = 'dev'
    TRAIN: str = 'train'
    VALIDATION: str = 'validation'
    TEST: str = 'test'
