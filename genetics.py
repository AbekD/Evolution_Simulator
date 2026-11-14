import numpy as np

alleles = ["A","a"]

def validate_allele_probs(allele_probs):
    # Must be a dict with the two alleles, {"A": 0.7, "a": 0.3}
    if not isinstance(allele_probs, dict) or set(allele_probs.keys()) != set(alleles):
        raise ValueError("allele_probs must be a dict with keys 'A' and 'a'")

    for allele in alleles:
        if not 0 <= allele_probs[allele] <= 1:
            raise ValueError("Each allele probability must be between 0 and 1")

    # np.isclose avoids floating point (0.1 + 0.2 != 0.3)
    if not np.isclose(sum(allele_probs.values()), 1.0):
        raise ValueError("Allele probabilities must sum to 1")

def get_phenotype(genotype):
  if 'A' in genotype:
    return False
  
  else:
    return True


def create_human(genotype=None, allele_probs=None):
    if genotype is None:
        if allele_probs is None:
            genotype = [np.random.choice(alleles), np.random.choice(alleles)]
        else:
            validate_allele_probs(allele_probs)
            probs = [allele_probs["A"], allele_probs["a"]]
            genotype = [np.random.choice(alleles, p=probs),
                        np.random.choice(alleles, p=probs)]

    genotype = sorted(genotype)

    return {
        'genotype': genotype,
        'has_large_hand': get_phenotype(genotype)
    }


def reproduce(parent1, parent2):
    allele1 = np.random.choice(parent1['genotype'])
    allele2 = np.random.choice(parent2['genotype'])

    return create_human([allele1, allele2])