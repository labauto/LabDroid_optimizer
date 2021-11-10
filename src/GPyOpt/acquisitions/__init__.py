from .base import AcquisitionBase
from .EI import AcquisitionEI
from GPyOpt.acquisitions.EI_mcmc import AcquisitionEI_MCMC
from .MPI import AcquisitionMPI
from .MPI_mcmc import AcquisitionMPI_MCMC
from .LCB import AcquisitionLCB
from .LCB_EST import AcquisitionLCB_EST
from .LCB_mcmc import AcquisitionLCB_MCMC
from .LP import AcquisitionLP
from .ES import AcquisitionEntropySearch
from .EST import AcquisitionEST
from .MV import AcquisitionMV

def select_acquisition(name):
    '''
    Acquisition selector
    '''
    if name == 'EI':
        return AcquisitionEI
    elif name == 'EI_MCMC':
        return AcquisitionEI_MCMC
    elif name == 'LCB':
        return AcquisitionLCB
    elif name == 'LCB_EST':
        return AcquisitionLCB_EST
    elif name == 'LCB_MCMC':
        return AcquisitionLCB_MCMC
    elif name == 'MPI':
        return AcquisitionMPI
    elif name == 'MPI_MCMC':
        return AcquisitionMPI_MCMC
    elif name == 'LP':
        return AcquisitionLP
    elif name == 'ES':
        return AcquisitionEntropySearch
    elif name == 'EST':
        return AcquisitionEST
    elif name == 'MV':
        return AcquisitionMV

    else:
        raise Exception('Invalid acquisition selected.')