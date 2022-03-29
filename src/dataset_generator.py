from typing import List
import random
import numpy as np
import pandas as pd
from nupack import Strand, Complex, ComplexSet, Model, SetSpec, complex_analysis
import torch
from torch.utils.data import TensorDataset
from encoding import oneHotEncodeMany



def generateSequence(length: int):
    return "".join(random.choice("CGTA") for _ in range(length))


def generateSequences(n = 10000, minLength = 20, maxLength = 60):
    sequences = []

    for _ in range(n):
        sequences.append(generateSequence(random.randint(minLength, maxLength)))

    return sequences


def getFreeEnergy(sequences: List[str]):
    temperature = 310.0  # Kelvin
    ionic_strength = 1.0  # molar

    strands = [Strand(seq, name=f"strand{idx}") for idx, seq in enumerate(sequences)]

    complexes = [
        Complex([strand], name=f"comp{idx}") for idx, strand in enumerate(strands)
    ]
    complex_set = ComplexSet(
        strands=strands, complexes=SetSpec(max_size=1, include=tuple(complexes))
    )
    model = Model(material="dna", celsius=temperature - 273, sodium=ionic_strength)
    results = complex_analysis(complex_set, model=model, compute=["mfe"])

    energies = [results[comp].mfe[0].energy for comp in complexes]

    return energies

def loadDataset(fileName="variable_length_dataset.csv", encoding = 'onehot'):
    data = pd.read_csv(fileName, header=None, delimiter=',', dtype={0: str, 1: float}) # type: ignore

    numpified = data.to_numpy() # type: ignore
    sequences = numpified[:, 0].astype(str)
    energies = numpified[:, 1].astype(float)

    x = None

    if encoding == 'onehot':
        x = oneHotEncodeMany(sequences)

    xTensor= torch.tensor(x, dtype=torch.float32)
    yTensor = torch.tensor(energies, dtype=torch.float32)

    return TensorDataset(xTensor, yTensor)


if __name__ == "__main__":
    variableLengthSequences = generateSequences()

    variableLengthEnergy = getFreeEnergy(variableLengthSequences)

    sequencesAugEnergy = np.column_stack(
        (variableLengthSequences, variableLengthEnergy)
    )

    np.savetxt(
        "variable_length_dataset.csv", sequencesAugEnergy, delimiter=",", fmt="%s"
    )
