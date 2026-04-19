import numpy as np

# hydrophobicity scale
hydrophobicity = {
    'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5,
    'C': 2.5, 'Q': -3.5, 'E': -3.5, 'G': -0.4,
    'H': -3.2, 'I': 4.5, 'L': 3.8, 'K': -3.9,
    'M': 1.9, 'F': 2.8, 'P': -1.6, 'S': -0.8,
    'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
}

# charge scale
charge = {
    'R': 1, 'K': 1, 'H': 0.5,
    'D': -1, 'E': -1
}


def extract_physicochemical_features(seq):
    """
    Convert protein sequence into physicochemical feature vector.
    Returns:
        - amino acid composition (20 features)
        - length
        - hydrophobicity
        - net charge
    """
    seq = str(seq)
    length = len(seq)

    if length == 0:
        return [0] * 23

    aa_counts = [seq.count(aa) / length for aa in hydrophobicity.keys()]

    hydro = np.mean([hydrophobicity.get(aa, 0) for aa in seq])
    net_charge = sum([charge.get(aa, 0) for aa in seq]) / length

    return aa_counts + [length, hydro, net_charge]