import os

from sklearn.model_selection import train_test_split
from .utils import DataSetSplit, DialogueCorpus

from joblib import Parallel
from joblib import delayed
from joblib import parallel_backend

import re

import json
import pandas as pd

from typing import List, Dict, Pattern, Iterable, Union, Optional


# TODO move context management to dialogue corpus and threat it here as a list of strings
class DailyDialog(DialogueCorpus):
    IDENTIFIER = 'dailydialog'

    def _preprocess_dialogue(self, original_dialogue, dialogue_idx: int) -> List[Dict]:
        # Correct misspelled topic label
        if original_dialogue['topic'] == 'culture_and_educastion':
            original_dialogue['topic'] = 'culture_and_education'
        # Dialogue samples
        dialogue_turns = [
            (self._preprocess_text(turn['text']), turn['emotion'], turn['act'], original_dialogue['topic'])
            for turn in original_dialogue['dialogue']
        ]
        # Dialogue contexts
        dialogue_contexts = self._get_dialogue_contexts([utterance for utterance, *_ in dialogue_turns])
        # Pre-processed dialogue
        dialogue: List[Dict] = [
            {
                'split': self.data_set_split.value,
                'corpus': 'DailyDialog',
                'conversation_idx': dialogue_idx,
                'turn_idx': turn_idx,
                'context': context,
                'response': response,
                'emotion': emotion,
                'dialogue_act': act,
                'topic': topic,
            }
            for turn_idx, (context, (response, emotion, act, topic)) in enumerate(zip(dialogue_contexts, dialogue_turns))
        ]
        return dialogue

    def _load_samples(self) -> List[Dict]:
        # Get split file name
        if self.data_set_split == DataSetSplit.TRAIN:
            file_name = 'train.json'
        elif self.data_set_split == DataSetSplit.VALIDATION:
            file_name = 'valid.json'
        elif self.data_set_split == DataSetSplit.TEST:
            file_name = 'test.json'
        else:
            raise ValueError()

        # Load split of the corpus
        with open(os.path.join(self.corpus_dir_path, file_name)) as f:
            dialogues = [json.loads(line) for line in f.readlines()]

        # Standardise corpus
        with parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            return sum(Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue, idx) for idx, dialogue in enumerate(dialogues)
            ), [])


class EmpatheticDialogues(DialogueCorpus):
    IDENTIFIER = 'empatheticdialogues'

    COLUMNS = [
        'conv_id', 'utterance_idx', 'context', 'prompt', 'speaker_idx', 'utterance', 'selfeval', 'tags', 'distractors'
    ]
    DTYPES = [str, int, str, str, int, str, str, str, str]

    def _preprocess_text(self, text: str) -> str:
        text = text.replace('_comma_', ',')
        return super(EmpatheticDialogues, self)._preprocess_text(text)

    def _preprocess_dialogue(self, original_dialogue, dialogue_idx: int) -> List[Dict]:
        # TODO return also emotions, distractors, and other info
        # Dialogue samples generator
        dialogue_turns = [
            self._preprocess_text(turn['utterance'])
            for _, turn in original_dialogue.sort_values('utterance_idx').iterrows()
        ]
        # Dialogue contexts
        dialogue_contexts = self._get_dialogue_contexts(list(dialogue_turns))
        # Pre-processed dialogue
        dialogue: List[Dict] = [
            {
                'split': self.data_set_split.value,
                'corpus': 'EmpatheticDialogues',
                'conversation_idx': dialogue_idx,
                'turn_idx': turn_idx,
                'context': context,
                'response': response,
            }
            for turn_idx, (context, response) in
            enumerate(zip(dialogue_contexts, dialogue_turns))
        ]
        return dialogue

    def _load_samples(self) -> List[Dict]:
        # Get split file name
        if self.data_set_split == DataSetSplit.TRAIN:
            file_name = 'train.csv'
        elif self.data_set_split == DataSetSplit.VALIDATION:
            file_name = 'valid.csv'
        elif self.data_set_split == DataSetSplit.TEST:
            file_name = 'test.csv'
        else:
            raise ValueError()

        # Load split of the corpus
        with open(os.path.join(self.corpus_dir_path, file_name)) as f:
            dialogues = [line.strip().split(',') for line in f]
        dialogues.pop(0)
        dialogues = [sample if len(sample) == len(self.COLUMNS) else sample + [None] for sample in dialogues]
        dialogues = [[t(x) if x is not None else x for x, t in zip(sample, self.DTYPES)] for sample in dialogues]
        dialogues = pd.DataFrame(data=dialogues, columns=self.COLUMNS)

        # Standardise corpus
        with parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            return sum(Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue, idx)
                for idx, (_, dialogue) in enumerate(dialogues.groupby('conv_id'))
            ), [])


class PersonaChat(DialogueCorpus):
    IDENTIFIER = 'personachat'

    DISTRACTORS_SPLIT_SYM = '\t\t'
    DISTRACTORS_SEPARATOR_SYM = '|'
    RESPONSE_SPLIT_SYM = '\t'
    Y_PERSONA_SYM = 'your persona:'
    P_PERSONA_SYM = 'partner\'s persona:'

    def _preprocess_dialogue(self, original_dialogue, dialogue_idx: int) -> List[Dict]:
        # TODO return also persona, alterantive persona and distractors
        # Dialogue pairs generator
        dialogue_turn_pairs = [
            (
                turn_pair.split(self.RESPONSE_SPLIT_SYM),
                [self._preprocess_text(distractor) for distractor in distractors.split(self.DISTRACTORS_SEPARATOR_SYM)]
            )
            for turn_pair, distractors in (
                    sample.split(self.DISTRACTORS_SPLIT_SYM) for sample in original_dialogue
                    if not (self.Y_PERSONA_SYM in sample or self.P_PERSONA_SYM in sample)
            )
        ]
        # Dialogue samples generator
        dialogue_turns = [
            (self._preprocess_text(turn), distractors)
            for turn_pair, distractors in dialogue_turn_pairs for turn in turn_pair
        ]
        # Dialogue contexts
        dialogue_contexts = self._get_dialogue_contexts([utterance for utterance, *_ in dialogue_turns])
        # Pre-processed dialogue
        dialogue: List[Dict] = [
            {
                'split': self.data_set_split.value,
                'corpus': 'Persona-Chat',
                'conversation_idx': dialogue_idx,
                'turn_idx': turn_idx,
                'context': context,
                'response': response,
                'distractors': distractors,
            }
            for turn_idx, (context, (response, distractors)) in enumerate(zip(dialogue_contexts, dialogue_turns))
        ]
        return dialogue

    def _load_samples(self) -> List[Dict]:
        # Get split file name
        if self.data_set_split == DataSetSplit.TRAIN:
            file_name = 'train_both_original.txt'
        elif self.data_set_split == DataSetSplit.VALIDATION:
            file_name = 'valid_both_original.txt'
        elif self.data_set_split == DataSetSplit.TEST:
            file_name = 'test_both_original.txt'
        else:
            raise ValueError()

        # Load split of the corpus
        with open(os.path.join(self.corpus_dir_path, file_name)) as f:
            dialogues = [line.split(' ', 1) for line in f.readlines()]
        start_idxs = [l_idx for l_idx, (i, _) in enumerate(dialogues) if int(i) == 1]
        end_idxs = start_idxs[1:] + [len(dialogues)]
        dialogues = [[elem for _, elem in dialogues[s_idx:e_idx]] for s_idx, e_idx in zip(start_idxs, end_idxs)]

        # Standardise corpus
        with parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            return sum(Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue, idx) for idx, dialogue in enumerate(dialogues)
            ), [])


class WizardOfWikipedia(DialogueCorpus):
    IDENTIFIER = 'wizard_of_wikipedia'

    def _preprocess_dialogue(self, original_dialogue, dialogue_idx: int) -> List[Dict]:
        # TODO return additional data
        # Dialogue samples generator
        dialogue_turns = [self._preprocess_text(turn['text']) for turn in original_dialogue['dialog']]
        # Dialogue contexts
        dialogue_contexts = self._get_dialogue_contexts(list(dialogue_turns))
        # Pre-processed dialogue
        dialogue: List[Dict] = [
            {
                'split': self.data_set_split.value,
                'corpus': 'Wizard of Wikipedia',
                'conversation_idx': dialogue_idx,
                'turn_idx': turn_idx,
                'context': context,
                'response': response,
            }
            for turn_idx, (context, response) in enumerate(zip(dialogue_contexts, dialogue_turns))
        ]
        return dialogue

    def _load_samples(self) -> List[Dict]:
        # Get split file name
        if self.data_set_split == DataSetSplit.TRAIN:
            file_name = 'train.json'
        elif self.data_set_split == DataSetSplit.VALIDATION:
            file_name = 'valid_random_split.json'
        elif self.data_set_split == DataSetSplit.TEST:
            file_name = 'test_random_split.json'
        else:
            raise ValueError()

        # Load split of the corpus
        with open(os.path.join(self.corpus_dir_path, file_name)) as f:
            dialogues = json.loads(f.read())

        # Standardise corpus
        with parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            return sum(Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue, idx) for idx, dialogue in enumerate(dialogues)
            ), [])


class CounsellingAndPsychotherapyCorpus(DialogueCorpus):
    IDENTIFIER = 'Counseling_and_Psychotherapy_Transcripts_Volume_II'
    VALID_LINE_REGEX: Pattern[str] = re.compile(r'^(CLIENT|THERAPIST): .+$')
    ROLES_DECODER: Dict = {'THERAPIST': 'Therapist', 'CLIENT': 'Patient'}  # TODO ask mark how to approach this

    def __init__(
            self,
            corpus_dir_path: str,
            data_set_split: str,
            *args,
            validation_size: Union[int, float] = 50,
            test_size: Union[int, float] = 50,
            random_seed: Optional[int] = 0,
            **kwargs
    ):
        # Get file list
        self.file_list: List[str] = os.listdir(corpus_dir_path)
        # Get indices list
        idxs = range(len(self.file_list))
        # Do train/validation/test split on the indices
        train_idxs, test_idxs = train_test_split(idxs, test_size=test_size, random_state=random_seed)
        train_idxs, validation_idxs = train_test_split(train_idxs, test_size=validation_size, random_state=random_seed)
        # Load the desired split
        if DataSetSplit(data_set_split) == DataSetSplit.TRAIN:
            self.file_list = [self.file_list[idx] for idx in train_idxs]
        elif DataSetSplit(data_set_split) == DataSetSplit.VALIDATION:
            self.file_list = [self.file_list[idx] for idx in validation_idxs]
        elif DataSetSplit(data_set_split) == DataSetSplit.TEST:
            self.file_list = [self.file_list[idx] for idx in test_idxs]
        else:
            raise ValueError(f"Unsupported data split: {data_set_split}")
        # Finally call super-class constructor
        super(CounsellingAndPsychotherapyCorpus, self).__init__(corpus_dir_path, data_set_split, *args, **kwargs)

    def _preprocess_dialogue(self, original_dialogue, dialogue_idx: int) -> List[Dict]:
        lines: List[str] = original_dialogue.strip().split('\n')

        if not all(self.VALID_LINE_REGEX.match(line) for line in lines):
            return []
        # Replace missing/corrupted parts symbol with a generic indicator
        lines = [line.replace('<|unknown|>', '###').strip() for line in lines]
        # Dialogue samples
        dialogue_turns = [
            (utterance.strip(), self.ROLES_DECODER[speaker.strip()])
            for speaker, utterance in (line.split(': ', 1) for line in lines)
        ]
        # Dialogue contexts
        dialogue_contexts = self._get_dialogue_contexts([utterance for utterance, *_ in dialogue_turns])
        # Pre-processed dialogue
        dialogue: List[Dict] = [
            {
                'split': self.data_set_split.value,
                'corpus': 'Counselling and Psychotherapy Transcripts Volume II',
                'conversation_idx': dialogue_idx,
                'turn_idx': turn_idx,
                'context': context,
                'response': response,
                'speaker': speaker,
            }
            for turn_idx, (context, (response, speaker)) in enumerate(zip(dialogue_contexts, dialogue_turns))
        ]
        return dialogue

    def _load_samples(self) -> List[Dict]:
        # Define helper function to load the samples
        def load_txt(file_path):
            with open(file_path) as f:
                raw_text = f.read()
            return raw_text

        # Load split of the corpus
        dialogues = [
            load_txt(os.path.join(self.corpus_dir_path, file_name)) for file_name in self.file_list
        ]

        # Standardise corpus
        with parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            return sum(Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue, idx) for idx, dialogue in enumerate(dialogues)
            ), [])


class Hope(DialogueCorpus):
    IDENTIFIER = 'HOPE_WSDM_2022'

    DATA_DIR_DECODER: Dict = {
        DataSetSplit.TRAIN: 'Train',
        DataSetSplit.VALIDATION: 'Validation',
        DataSetSplit.TEST: 'Test'
    }
    FILE_NAME_REGEX: Pattern[str] = re.compile(r'^Copy of \d+\.csv$')

    ACTIONS_DECODER: Dict = {
        'acak': 'Acknowledgement',
        'ack': 'Acknowledgement',
        # 'ap': '',  # 2 # Answer positive?
        # 'ay': '',  # 2 # Answer yes?
        'cd': 'Clarification Delivery',
        'cdd': 'Clarification Delivery',
        'ci': 'General Chat',  # 50 # Maybe??
        # 'com': '',  # 9
        # 'comp': '',  # 2
        'cq': 'Clarification Delivery',  # Typo?
        'cr': 'General Chat',  # 110 # Maybe??
        'crq': 'Clarification Request',
        'cv': 'General Chat',  # 61 # Maybe??
        'gc': 'General Chat',
        'gt': 'Greeting',
        # 'hp': '',  # 1
        'id': 'Information Delivery',
        'in': 'Information Delivery',  # Typo?
        'irq': 'Information Request',
        'irrq': 'Information Request',
        'o': 'Opinion Delivery',  # Opinion?
        'od': 'Opinion Delivery',
        'on': 'Negative Answer',  # Opinion Negative?
        'op': 'Positive Answer',  # Opinion Positive?
        'orq': 'Opinion Request',
        'urq': 'Opinion Request',  # Typo?
        # 'vc': '',  # 1
        # 'yk': '',  # 1
        'yq': 'Yes/No question'
    }
    ACTION_REGEX: Pattern[str] = re.compile(r'(\w+)([^\w\n]+\w*)*')
    ROLES_DECODER: Dict = {'T': 'Therapist', 'P': 'Patient'}  # TODO ask mark how to approach this

    def _preprocess_dialogue(self, original_dialogue: pd.DataFrame, dialogue_idx: int) -> List[Dict]:
        # TODO return additional
        # Get identifier for actions
        if 'Dialogue Act' in original_dialogue.columns:
            dialogue_act_col = 'Dialogue Act'
        elif 'Dialogue_Act' in original_dialogue.columns:
            dialogue_act_col = 'Dialogue_Act'
        elif 'Dialog_Act' in original_dialogue.columns:
            dialogue_act_col = 'Dialog_Act'
        elif 'Dilog_Act' in original_dialogue.columns:
            dialogue_act_col = 'Dilog_Act'
        elif 'Unnamed: 3' in original_dialogue.columns:
            dialogue_act_col = 'Unnamed: 3'
        else:
            raise ValueError()

        # Dialogue samples
        dialogue_turns = [
            (
                self._preprocess_text(turn['Utterance']),
                self.ROLES_DECODER.get(turn['Type']),
                self.ACTIONS_DECODER.get(
                    self.ACTION_REGEX.search(str(turn[dialogue_act_col])).group(1).strip()
                ),
            )
            for _, turn in original_dialogue.iterrows()
        ]
        # Dialogue contexts
        dialogue_contexts = self._get_dialogue_contexts([utterance for utterance, *_ in dialogue_turns])
        # Pre-processed dialogue
        dialogue: List[Dict] = [
            {
                'split': self.data_set_split.value,
                'corpus': 'HOPE',
                'conversation_idx': dialogue_idx,
                'turn_idx': turn_idx,
                'context': context,
                'response': response,
                'speaker': speaker,
                'dialogue_act': act
            }
            for turn_idx, (context, (response, speaker, act)) in enumerate(zip(dialogue_contexts, dialogue_turns))
        ]
        return dialogue

    def _load_samples(self) -> List[Dict]:
        # Get split sub-dir name
        sub_dir_name = self.DATA_DIR_DECODER[self.data_set_split]
        # Get iterator over file to load
        file_name_iterator: Iterable[str] = (
            file_name
            for file_name in os.listdir(os.path.join(self.corpus_dir_path, sub_dir_name))
            if self.FILE_NAME_REGEX.match(file_name)
        )
        # Load split of the corpus
        dialogues = [
            pd.read_csv(os.path.join(self.corpus_dir_path, sub_dir_name, file_name)) for file_name in file_name_iterator
        ]

        # Standardise corpus
        with parallel_backend(self.parallel_backend, n_jobs=self.n_jobs):
            return sum(Parallel(verbose=self.verbosity_level)(
                delayed(self._preprocess_dialogue)(dialogue, idx) for idx, dialogue in enumerate(dialogues)
            ), [])
