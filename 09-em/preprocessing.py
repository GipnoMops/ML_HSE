from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import xml.etree.ElementTree as ET


@dataclass(frozen=True)
class SentencePair:
    """
    Contains lists of tokens (strings) for source and target sentence
    """
    source: List[str]
    target: List[str]


@dataclass(frozen=True)
class TokenizedSentencePair:
    """
    Contains arrays of token vocabulary indices (preferably np.int32) for source and target sentence
    """
    source_tokens: np.ndarray
    target_tokens: np.ndarray


@dataclass(frozen=True)
class LabeledAlignment:
    """
    Contains arrays of alignments (lists of tuples (source_pos, target_pos)) for a given sentence.
    Positions are numbered from 1.
    """
    sure: List[Tuple[int, int]]
    possible: List[Tuple[int, int]]


def extract_sentences(filename: str) -> Tuple[List[SentencePair], List[LabeledAlignment]]:
    """
    Given a file with tokenized parallel sentences and alignments in XML format, return a list of sentence pairs
    and alignments for each sentence.

    Args:
        filename: Name of the file containing XML markup for labeled alignments

    Returns:
        sentence_pairs: list of `SentencePair`s for each sentence in the file
        alignments: list of `LabeledAlignment`s corresponding to these sentences
    """

    sentence_pairs = []
    aligments = []

    raw = open(filename, 'r').read().replace('&', '&amp;')
    root = ET.fromstring(raw)

    for pair in root:
        english, czech, sure, possible = [item.text.split() if item.text is not None else '' for item in pair[:4]]

        sentence_pairs.append(SentencePair(english, czech))

        sure_list = [(int(ints.split('-')[0]), int(ints.split('-')[1])) for ints in sure]
        pos_list  = [(int(ints.split('-')[0]), int(ints.split('-')[1])) for ints in possible]

        aligments.append(LabeledAlignment(sure_list, pos_list))

    return sentence_pairs, aligments


def get_token_to_index(sentence_pairs: List[SentencePair], freq_cutoff=None) -> Tuple[Dict[str, int], Dict[str, int]]:
    """
    Given a parallel corpus, create two dictionaries token->index for source and target language.

    Args:
        sentence_pairs: list of `SentencePair`s for token frequency estimation
        freq_cutoff: if not None, keep only freq_cutoff most frequent tokens in each language

    Returns:
        source_dict: mapping of token to a unique number (from 0 to vocabulary size) for source language
        target_dict: mapping of token to a unique number (from 0 to vocabulary size) target language

    """
    source_dict = dict()
    target_dict = dict()
    for cur_pair in sentence_pairs:
        for word in cur_pair.source:
            if source_dict.get(word) is None:
                source_dict[word] = 1
            else:
                source_dict[word] += 1

        for word in cur_pair.target:
            if target_dict.get(word) is None:
                target_dict[word] = 1
            else:
                target_dict[word] += 1

    if freq_cutoff is not None:
        source_dict = dict(sorted(source_dict.items(), key=lambda x: x[1], reverse=True)[:freq_cutoff])
        target_dict = dict(sorted(target_dict.items(), key=lambda x: x[1], reverse=True)[:freq_cutoff])

    new_source_dict = {word: idx for idx, word in enumerate(source_dict)}
    new_target_dict = {word: idx for idx, word in enumerate(target_dict)}

    return new_source_dict, new_target_dict


def tokenize_sents(sentence_pairs: List[SentencePair], source_dict, target_dict) -> List[TokenizedSentencePair]:
    """
    Given a parallel corpus and token_to_index for each language, transform each pair of sentences from lists
    of strings to arrays of integers. If either source or target sentence has no tokens that occur in corresponding
    token_to_index, do not include this pair in the result.
    
    Args:
        sentence_pairs: list of `SentencePair`s for transformation
        source_dict: mapping of token to a unique number for source language
        target_dict: mapping of token to a unique number for target language

    Returns:
        tokenized_sentence_pairs: sentences from sentence_pairs, tokenized using source_dict and target_dict
    """

    tokenized_sentence_pairs = []

    for sentence in sentence_pairs:
        source = sentence.source
        target = sentence.target
        cur_sour = []
        cur_targ = []
        for word in source:
            if word in source_dict:
                cur_sour.append(source_dict[word])

        for word in target:
            if word in target_dict:
                cur_targ.append(target_dict[word])

        if len(cur_sour) > 0 and len(cur_targ) > 0:
            tokenized_sentence_pairs.append(TokenizedSentencePair(np.array(cur_sour), np.array(cur_targ)))
    return tokenized_sentence_pairs



