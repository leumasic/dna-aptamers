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

def oneHotEncode(sequence: str) -> np.ndarray:
    numEncoding = numberEncode(sequence)

    return tf.one_hot(numEncoding, 4, dtype=tf.uint8).numpy()

def frequencyEncode(sequence: str) -> np.ndarray:
    charToCount = { 'A': 0, 'C': 0, 'G': 0, 'T': 0 }

    for char in sequence:
        charToCount[char] += 1

    return np.array([charToCount['A'], charToCount['C'], charToCount['G'],
        charToCount['T']])

# Add other encodings below as necessary

# Only some tests in the if block
if __name__ == '__main__':
    # Clampity clamp tests
    sequences = clampSequences(["Sami", "Samuel"], 4)
    assert(sequences == ["Sami", "Samu"])

    numberEncoding = numberEncode("ACGT")
    assert(numberEncoding == [0, 1, 2, 3])
