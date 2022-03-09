import random
import numpy as np
from nupack import Strand, Complex, ComplexSet, Model, SetSpec, complex_analysis


def sequence(length):
    return "".join(random.choice("CGTA") for _ in range(length))


def aptamer_generator(n=10000, length_lower=20, length_upper=60):
    seq_list = []
    for i in range(n):
        seq_list.append(sequence(random.randint(length_lower, length_upper)))
    return seq_list


def free_energy(sequences):
    temperature = 310.0  # Kelvin
    ionic_strength = 1.0  # molar
    strands = [Strand(seq, name=f"strand{idx}") for idx, seq in enumerate(sequences)]
    complexes = [
        Complex([strand], name=f"comp{idx}") for idx, strand in enumerate(strands)
    ]
    complex_set = ComplexSet(
        strands=strands, complexes=SetSpec(max_size=1, include=complexes)
    )
    model = Model(material="dna", celsius=temperature - 273, sodium=ionic_strength)
    results = complex_analysis(complex_set, model=model, compute=["mfe"])

    energies = [results[comp].mfe[0].energy for comp in complexes]
    return energies


variable_length_sequences = aptamer_generator()

variable_length_energy = free_energy(variable_length_sequences)

variable_length_dataset = np.column_stack(
    (variable_length_sequences, variable_length_energy)
)

# np.savetxt(
#     "variable_length_dataset.csv", variable_length_dataset, delimiter=",", fmt="%s"
# )
