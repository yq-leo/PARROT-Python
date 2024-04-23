import scipy.io as sio
import numpy as np

if __name__ == "__main__":
    data = sio.loadmat('mid/rwrs.mat')
    py_data = sio.loadmat('mid/py_rwrs.mat')

    rwr1, rwr2 = data['rwr1'], data['rwr2']
    py_rwr1, py_rwr2 = py_data['rwr1'], py_data['rwr2']

    rwrcost_data = sio.loadmat('mid/rwrcost.mat')
    py_rwrcost_data = sio.loadmat('mid/py_rwrcost.mat')

    rwrcost = rwrcost_data['rwrCost']
    py_rwrcost = py_rwrcost_data['rwrCost']

    intraC_data = sio.loadmat('mid/intraC.mat')
    py_intraC_data = sio.loadmat('mid/py_intraC.mat')

    intraC1, intraC2 = intraC_data['intraC1'], intraC_data['intraC2']
    py_intraC1, py_intraC2 = py_intraC_data['intraC1'], py_intraC_data['intraC2']

    crossC_data = sio.loadmat('mid/crossC.mat')
    py_crossC_data = sio.loadmat('mid/py_crossC.mat')

    crossC = crossC_data['crossC']
    py_crossC = py_crossC_data['crossC']

    pass
