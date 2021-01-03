def get_feature_datatype(similarity_measure, molecular_descriptor):
    """ Get the datatype required for feature based on rules
    conditional on similarity measure and molecular_descriptor

    Parameters
    ----------
    similarity_measure: str
        Label for the similarity measure used.
    molecular_descriptor: str
        Label for the molecular descriptor used.

    Returns
    -------
    str
        Label for datatype to be used
    """
    if similarity_measure == 'tanimoto':
        if molecular_descriptor in ['topological_fingerprint',
                                    'morgan_fingerprint',
                                    ]:
            return 'rdkit'
    elif similarity_measure in ['-l0', '-l1', '-l2']:
        if molecular_descriptor in ['topological_fingerprint',
                                    'morgan_fingerprint',
                                    ]:
            return 'numpy'
    else:
        raise NotImplementedError(f'{similarity_measure} similarity'
                                  'does not work with '
                                  f'{molecular_descriptor} featurization')