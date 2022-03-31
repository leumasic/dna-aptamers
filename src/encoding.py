from typing import List
import tensorflow as tf
import numpy as np

baseToClass = {
    'A': 0,
    'C': 1,
    'G': 2,
    'T': 3
}

def clampSequences(sequences: List[str], maxLength: int) -> List[str]:
    return [sequence[:maxLength] for sequence in sequences]

def numberEncode(sequence: str) -> List[int]:
    numEncoding = [baseToClass[char] for char in sequence]
    return numEncoding

def numberEncodeMany(sequences: List[str]) -> List[List[int]]:
    encodedSequences: List[List[int]] = []

    for sequence in sequences:
        numEncoding = numberEncode(sequence)
        encodedSequences.append(numEncoding)

    return encodedSequences

def oneHotEncode(sequence: str) -> np.ndarray:
    numEncoding = numberEncode(sequence)

    return tf.one_hot(numEncoding, 4, dtype=tf.uint8).numpy()

def frequencyEncode(sequence: str) -> np.ndarray:
    """Encodes a sequence based on the count of distinct nucleotides

    Args:
        sequence (str): Sequence to encode

    Returns:
        [count of A, count of C, count of G, count of T]
    """
    charToCount = { 'A': 0, 'C': 0, 'G': 0, 'T': 0 }

    for char in sequence:
        charToCount[char] += 1

    return np.array([charToCount['A'], charToCount['C'], charToCount['G'],
        charToCount['T']])

def frequencyEncodeMany(sequences: List[str]) -> np.ndarray:
    """Wrapper around frequencyEncode that encodes a list of sequences

    Args:
        sequences (List[str]): List of sequences to encode

    Returns:
        Numpy array of frequency encoded sequences
    """
    numSequences = len(sequences)
    encoded = np.zeros((numSequences, 4))

    for i, seq in enumerate(sequences):
        encoded[i] = frequencyEncode(seq)

    return encoded

def oneHotEncodeMany(sequences: List[str], seqLength = 60) -> np.ndarray:
    """

    Args:
        seqLength (int): Sequences of smaller length than this param get padded
        and those longer get truncated
        sequences (List[str]): Sequences to hot encode

    Returns:
        A numpy ndarray of shape (numSequences, seqLength * 4)
    """
    numSequences = len(sequences)
    clampedSequences = clampSequences(sequences, seqLength)
    numEncoding = numberEncodeMany(clampedSequences)

    encoded = np.zeros((numSequences, seqLength * 4))

    for i, numSequence in enumerate(numEncoding):
        sequenceLength = len(numSequence)
        oneHot = tf.one_hot(numSequence, 4, dtype=tf.uint8)
        padded = np.pad(oneHot, ((0, seqLength - sequenceLength), (0, 0)), mode='constant', constant_values=0) # type: ignore
        encoded[i] = padded.flatten()

    return encoded


# Add other encodings below as necessary

# Only some tests in the if block
if __name__ == '__main__':
    # Clampity clamp tests
    sequences = clampSequences(["Sami", "Samuel"], 4)
    assert(sequences == ["Sami", "Samu"])

    numberEncoding = numberEncode("ACGT")
    assert(numberEncoding == [0, 1, 2, 3])

    oneHotMany = oneHotEncodeMany(["ACC", "AGTT"], 5)
    assert(oneHotMany.shape == (2, 5, 4))
