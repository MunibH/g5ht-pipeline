def map_nir(data_dict):
    '''
    makes a mapping of each confocal stack to its corresponding nir frame.
    this mapping is incomplete as it does not account for any issues in saving
    and may be longer than actual amount of nir frames saved
    '''
    input_h5 = data_dict['input_h5']
    n_stack = data_dict['n_stack']
    new_dict = {}

    #FOR CONFOCAL (SIMPLE): confocal is on when daqmx_ai is high and pick when each stack starts
    f = h5py.File(input_h5, 'r')
    confocal_ons = (f['daqmx_ai'][0] > 0.5).astype(int)
    confocal_starts = np.where((np.diff(confocal_ons) == 1))[0]
    confocal_stack_starts = confocal_starts[range(0, len(confocal_starts), n_stack)]

    #FOR NIR (COMPLICATED): nir is on when daqmx_di is high
    nir_ons = (f['daqmx_di'][1] > 0.5).astype(int)
    nir_starts = np.where((np.diff(nir_ons) == 1))[0]
    
    #FOR NIR (COMPLICATED): i believe the camera only captures at certain nir ids
    ids = np.array(f['img_metadata']['img_id'])
    ids -= ids[0]

    #FOR NIR (COMPLICATED): i believe the camera only adddionally saves every other frame
    q = np.array(f['img_metadata']['q_iter_save']) > 0.5
    filtered_nir_starts = nir_starts[ids[q]]

    #combine the mappings
    nir_to_conf_mapping = []
    for i in confocal_stack_starts:
        index = np.argmax(filtered_nir_starts > i)
        if index > 0:
            nir_to_conf_mapping.append(index)
        else:
            break
    new_dict['nir_to_conf_mapping'] = np.array(nir_to_conf_mapping)
    data_dict.update(new_dict)
    return new_dict