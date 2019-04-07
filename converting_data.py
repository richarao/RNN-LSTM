def save_as_pandas_df(filename):
    import h5py as h5
    import pandas as pd
    import numpy as np

    cols_tmp = [('index', '<i8'), ('j_ptfrac', '<f4'), ('j_pt', '<f4'), ('j_eta', '<f4'), ('j_mass', '<f4'), ('j_tau1_b1', '<f4'), ('j_tau2_b1', '<f4'), ('j_tau3_b1', '<f4'), ('j_tau1_b2', '<f4'), ('j_tau2_b2', '<f4'), ('j_tau3_b2', '<f4'), ('j_tau32_b1', '<f4'), ('j_tau32_b2', '<f4'), ('j_zlogz', '<f4'), ('j_c1_b0', '<f4'), ('j_c1_b1', '<f4'), ('j_c1_b2', '<f4'), ('j_c2_b1', '<f4'), ('j_c2_b2', '<f4'), ('j_d2_b1', '<f4'), ('j_d2_b2', '<f4'), ('j_d2_a1_b1', '<f4'), ('j_d2_a1_b2', '<f4'), ('j_m2_b1', '<f4'), ('j_m2_b2', '<f4'), ('j_n2_b1', '<f4'), ('j_n2_b2', '<f4'), ('j_tau1_b1_mmdt', '<f4'), ('j_tau2_b1_mmdt', '<f4'), ('j_tau3_b1_mmdt', '<f4'), ('j_tau1_b2_mmdt', '<f4'), ('j_tau2_b2_mmdt', '<f4'), ('j_tau3_b2_mmdt', '<f4'), ('j_tau32_b1_mmdt', '<f4'), ('j_tau32_b2_mmdt', '<f4'), ('j_c1_b0_mmdt', '<f4'), ('j_c1_b1_mmdt', '<f4'), ('j_c1_b2_mmdt', '<f4'), ('j_c2_b1_mmdt', '<f4'), ('j_c2_b2_mmdt', '<f4'), ('j_d2_b1_mmdt', '<f4'), ('j_d2_b2_mmdt', '<f4'), ('j_d2_a1_b1_mmdt', '<f4'), ('j_d2_a1_b2_mmdt', '<f4'), ('j_m2_b1_mmdt', '<f4'), ('j_m2_b2_mmdt', '<f4'), ('j_n2_b1_mmdt', '<f4'), ('j_n2_b2_mmdt', '<f4'), ('j_mass_trim', '<f4'), ('j_mass_mmdt', '<f4'), ('j_mass_prun', '<f4'), ('j_mass_sdb2', '<f4'), ('j_mass_sdm1', '<f4'), ('j_multiplicity', '<f4'), ('j1_px', '<f4'), ('j1_py', '<f4'), ('j1_pz', '<f4'), ('j1_e', '<f4'), ('j1_pdgid', '<f4'), ('j1_erel', '<f4'), ('j1_pt', '<f4'), ('j1_ptrel', '<f4'), ('j1_eta', '<f4'), ('j1_etarel', '<f4'), ('j1_etarot', '<f4'), ('j1_phi', '<f4'), ('j1_phirel', '<f4'), ('j1_phirot', '<f4'), ('j1_deltaR', '<f4'), ('j1_costheta', '<f4'), ('j1_costhetarel', '<f4'), ('j1_e1mcosthetarel', '<f4'), ('j_g', '<i4'), ('j_q', '<i4'), ('j_w', '<i4'), ('j_z', '<i4'), ('j_t', '<i4'), ('j_undef', '<i4'), ('j_index', '<i4')]
    cols = []
    for c in cols_tmp:
        cols.append(c[0])

    cols = cols[:-1]
    data = np.array([])

    with h5.File(filename) as hf:
        arr = hf['t_allpar_new'][()]
        for a in arr:
            if data.size == 0:
                data = cast_to_np(a)
            else:
                data = np.vstack((data, cast_to_np(a)))

    data = data[:, 1:]
    y = data[:, -6:]
    data = data[:, :-6]

    write_name = filename.split('.')[0]
    write_name = 'after_processed_' + write_name + '.h5'
    pd_data = pd.DataFrame(data=data, columns=cols[1:-6])
    pd_data.to_hdf(write_name, key='data', mode='w')

    pd_y = pd.DataFrame(data=y, columns=cols[-6:])
    pd_y.to_hdf(write_name, key='labels', mode='a')


def cast_to_np(arr):
    import numpy as np
    result = np.array([], dtype=float)
    for a in arr:
        result = np.concatenate((result, [float(a)]))
    return result


def usage_example():
    import pandas as pd
    import matplotlib.pyplot as plt
    import os

    colors = ['orange', 'mediumseagreen', 'gold', 'purple', 'k', 'b']

    filename = 'C:\\Users\\richa\\research\\histogram\\particles\\after_processed_processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_withPars_truth_0.h5'

    data = pd.read_hdf(filename, key='data')  # type: pd.DataFrame
    y = pd.read_hdf(filename, key='labels')  # type: pd.DataFrame

    print(data)
    print(y)

    image_dir = 'img'
    if not os.path.exists(image_dir):
        os.makedirs(image_dir)

    for c_data in data.columns:
        for ydx, c_y in enumerate(y.columns):
            plt.hist(data.loc[y[c_y] == 1., c_data], bins=100, histtype='step', label=c_y, color=colors[ydx])
        plt.title(c_data)
        plt.legend()
        plt.savefig(os.path.join(image_dir, '{} hist'.format(c_data)))
        plt.show()
        plt.cla()
        plt.clf()


if __name__ == '__main__':
    filename = 'C:\\Users\\richa\\research\\histogram\\particles\\processed-pythia82-lhc13-all-pt1-50k-r1_h022_e0175_t220_nonu_withPars_truth_0.z'
    save_as_pandas_df(filename)
	
    usage_example()
