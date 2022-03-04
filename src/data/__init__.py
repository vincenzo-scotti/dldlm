from enum import Enum
from .dialogue_corpus import DialogueCorpus, EpisodicDialogueCorpus


IGNORE_INDEX = -100


class DataSetType(Enum):
    DEVELOPMENT: str = 'dev'
    TRAIN: str = 'train'
    VALIDATION: str = 'validation'
    TEST: str = 'test'
