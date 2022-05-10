import random
from enum import Enum

import torch
from torch.utils.data import Dataset
import pandas as pd

from model import DLDLMTokenizer

from typing import List, Tuple, Union, Optional


class DataSetSplit(Enum):
    DEVELOPMENT: str = 'dev'
    TRAIN: str = 'train'
    VALIDATION: str = 'validation'
    TEST: str = 'test'


class DialogueCorpus(Dataset):
    def __init__(
            self,
            corpus_csv_file_path: str,
            tokenizer: DLDLMTokenizer,
            data_set_split: str,
            distractor: bool = True
    ):
        super(DialogueCorpus, self).__init__()
        # Tokeniser to prepare inputs
        self.tokenizer: DLDLMTokenizer = tokenizer
        # Data split identifier
        self.data_set_split: DataSetSplit = DataSetSplit(data_set_split)
        # Use distractor sample
        self.distractor: bool = distractor
        # Path to corpus data frame
        self.corpus_csv_file_path: str = corpus_csv_file_path

        # Load data
        self.data_df: pd.DataFrame = pd.read_csv(self.corpus_csv_file_path)
        self.data_df = self.data_df[self.data_df.split == self.data_set_split.value]
        self.data_df.context.fillna("", inplace=True)
        self.reward_columns_id: List[str] = [
            column_id for column_id in self.data_df.columns if column_id.endswith('_reward')
        ]

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data_df)

    def __getitem__(self, index: int) -> Union[
        Tuple[Optional[List[str]], str, List[float]], Tuple[Optional[List[str]], str, List[float], str]
    ]:
        context: Optional[List[str]] = (
            self.data_df.iloc[index].context.split('\n') if self.data_df.iloc[index].context != "" else None
        )
        turn: str = self.data_df.iloc[index].turn
        rewards: List[float] = self.data_df.iloc[index][self.reward_columns_id].values.tolist()
        if self.distractor:
            distractor: str = random.choice(self.data_df.iloc[index].distractors.split('\n'))
            return context, turn, rewards, distractor
        else:
            return context, turn, rewards

    def collate(self, mini_batch) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor
    ]:
        # Unpack mini-mini_batch
        try:
            contexts, responses, rewards, distractors = zip(*mini_batch)
        except ValueError:
            contexts, responses, rewards = zip(*mini_batch)
            distractors = None
        # Prepare contexts
        try:  # In case all contexts are None
            contexts = [
                str().join(
                    self.tokenizer.bos_token + turn + self.tokenizer.eos_token for turn in context
                ) if context is not None else ""
                for context in contexts
            ]
            context_ids, context_attentions = self.tokenizer(contexts, padding=True, return_tensors='pt').values()
        except IndexError:
            context_ids, context_attentions = None, None
        # Prepare response
        responses = [self.tokenizer.bos_token + response + self.tokenizer.eos_token for response in responses]
        response_ids, response_attentions = self.tokenizer(responses, padding=True, return_tensors='pt').values()
        labels = response_ids.clone()
        labels[~response_attentions.bool()] = -100  # Index to ignore in loss computation
        # Reward
        rewards = torch.tensor(rewards)
        # Prepare negative sample
        if distractors is not None:
            distractors = [
                self.tokenizer.bos_token + distractor + self.tokenizer.eos_token for distractor in distractors
            ]
            distractor_ids, distractor_attentions = self.tokenizer(distractors, padding=True, return_tensors='pt').values()
        else:
            distractor_ids = distractor_attentions = None

        return (
            context_ids,  # Shape (batch_size, max_context_length)
            context_attentions,  # Shape (batch_size, max_context_length)
            response_ids,  # Shape (batch_size, max_response_length)
            response_attentions,  # Shape (batch_size, max_response_length)
            distractor_ids,  # Shape (batch_size, max_distractor_response_length)
            distractor_attentions,  # Shape (batch_size, max_distractor_response_length)
            labels,  # Shape (batch_size, max_response_length)
            rewards  # Shape (batch_size, num_rewards)
        )


class EpisodicDialogueCorpus(Dataset):
    epsilon = 1e-9

    def __init__(
            self,
            corpus_csv_file_path: str,
            tokenizer: DLDLMTokenizer,
            data_set_split: Union[str, List[str]] = 'train',
            data_sub_set: Optional[Union[str, List[str]]] = None,
            distractor: bool = True,
            discount_factor: float = 0.0,
            reward_weights: List[float] = 0.0,
    ):
        super(EpisodicDialogueCorpus, self).__init__()
        # Tokeniser to prepare inputs
        self.tokenizer: DLDLMTokenizer = tokenizer
        # Data split identifier
        self.data_set_split: List[DataSetSplit] = [
            DataSetSplit(split) for split in ([data_set_split] if isinstance(data_set_split, str) else data_set_split)
        ]
        # Data sub-set identifier
        self.data_sub_set: Optional[List[str]] = [data_sub_set] if isinstance(data_sub_set, str) else data_sub_set
        # Use distractor sample
        self.distractor: bool = distractor
        # Path to corpus data frame
        self.corpus_csv_file_path: str = corpus_csv_file_path
        # Discount factor (gamma)
        self.discount_factor = discount_factor
        # Weights to sum the rewards
        self.reward_weights: List[float] = reward_weights

        # Load data
        self.data_group_df: pd.DataFrame = pd.read_csv(self.corpus_csv_file_path)
        if self.data_sub_set is not None:
            self.data_group_df = self.data_group_df[
                self.data_group_df.split.isin([split_id.value for split_id in self.data_set_split]) &
                self.data_group_df.data_set.isin(self.data_sub_set)
                ]
        else:
            self.data_group_df = self.data_group_df[self.data_group_df.split.isin([split_id.value for split_id in self.data_set_split])]

        self.data_group_df.context.fillna("", inplace=True)
        self.reward_columns_id: List[str] = [
            column_id for column_id in self.data_group_df.columns if column_id.endswith('_reward')
        ]
        self.reward_trace_column_ids: List[str] = [
            column_id for column_id in self.data_group_df.columns if column_id.endswith('_reward_trace')
        ]
        self.data_group_df = self.data_group_df.groupby('conversation_id', sort=False)
        self.key_list: List[int] = list(self.data_group_df.groups)

    def __len__(self) -> int:
        # Number of sequences within the data set
        return len(self.data_group_df)

    def __getitem__(self, index: int) -> List[Union[
        Tuple[Optional[List[str]], str, List[float], List[Tuple[float]]],
        Tuple[Optional[List[str]], str, List[float], str, List[Tuple[float]]]
    ]]:
        # Load episode at specified index
        conversation_df: pd.DataFrame = self.data_group_df.get_group(self.key_list[index])
        # Prepare episode
        if self.distractor:
            episode: List[Tuple[Optional[List[str]], str, List[float], str, List[Tuple[float]]]] = [
                (
                    sample.context.split('\n') if sample.context != "" else None,
                    sample.turn,
                    sample[self.reward_columns_id].values.tolist(),
                    random.choice(sample.distractors.split('\n')),
                    [*zip(*(
                        [float(rew_str) for rew_str in rew_trace_str.split('\n')]
                        for rew_trace_str in sample[self.reward_trace_column_ids]
                    ))]
                )
                for idx, sample in conversation_df.iterrows()
            ]
        else:
            episode: List[Tuple[Optional[List[str]], str, List[float], List[Tuple[float]]]] = [
                (
                    sample.context.split('\n') if sample.context != "" else None,
                    sample.turn,
                    sample[self.reward_columns_id].values.tolist(),
                    [*zip(*(
                        [float(rew_str) for rew_str in rew_trace_str.split('\n')]
                        for rew_trace_str in sample[self.reward_trace_column_ids]
                    ))]
                )
                for idx, sample in conversation_df.iterrows()
            ]

        return episode

    def collate(self, mini_batch) -> Tuple[
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        Optional[torch.Tensor],
        Optional[torch.Tensor],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor
    ]:
        # Unpack mini-mini_batch
        try:
            contexts, responses, rewards, distractors, reward_traces = [
                sum(elem, tuple()) for elem in zip(*[list(zip(*episode)) for episode in mini_batch])
            ]
        except ValueError:
            contexts, responses, rewards, reward_traces = [
                sum(elem, tuple()) for elem in zip(*[list(zip(*episode)) for episode in mini_batch])
            ]
            distractors = None
        # Prepare contexts
        try:  # In case all contexts are None
            contexts = [
                str().join(
                    self.tokenizer.bos_token + turn + self.tokenizer.eos_token for turn in context
                ) if context is not None else ""
                for context in contexts
            ]
            context_ids, context_attentions = self.tokenizer(contexts, padding=True, return_tensors='pt').values()
        except IndexError:
            context_ids, context_attentions = None, None
        # Prepare response
        responses = [self.tokenizer.bos_token + response + self.tokenizer.eos_token for response in responses]
        response_ids, response_attentions = self.tokenizer(responses, padding=True, return_tensors='pt').values()
        labels = response_ids.clone()
        labels[~response_attentions.bool()] = -100  # Index to ignore in loss computation
        # Reward
        rewards = torch.tensor(rewards)
        # Prepare negative sample
        if distractors is not None:
            distractors = [
                self.tokenizer.bos_token + distractor + self.tokenizer.eos_token for distractor in distractors
            ]
            distractor_ids, distractor_attentions = self.tokenizer(distractors, padding=True, return_tensors='pt').values()
        else:
            distractor_ids = distractor_attentions = None
        # Normalised discounted reward for REINFORCE step
        discounted_rewards = torch.tensor([
            sum(
                w * r * (self.discount_factor ** t)
                for t, rewards in enumerate(reward_trace)
                for w, r in zip(self.reward_weights, rewards)
            )
            for reward_trace in reward_traces
        ])
        discounted_rewards_std, discounted_rewards_mean = torch.std_mean(discounted_rewards)
        normalised_discounted_rewards = (
                (discounted_rewards - discounted_rewards_mean) /
                torch.max(torch.tensor(self.epsilon), discounted_rewards_std)
        )

        return (
            context_ids,  # Shape (batch_size, max_context_length)
            context_attentions,  # Shape (batch_size, max_context_length)
            response_ids,  # Shape (batch_size, max_response_length)
            response_attentions,  # Shape (batch_size, max_response_length)
            distractor_ids,  # Shape (batch_size, max_distractor_response_length)
            distractor_attentions,  # Shape (batch_size, max_distractor_response_length)
            labels,  # Shape (batch_size, max_response_length)
            rewards,  # Shape (batch_size, num_rewards)
            normalised_discounted_rewards  # Shape (batch_size,)
        )
