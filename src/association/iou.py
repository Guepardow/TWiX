import numpy as np


def get_cost_matrix_IoU(obsColl1, obsColl2, measure, **kwargs):
    """
    Compute a cost matrix based on the position of observations

    :param obsColl1: ObsCollection with the observations from the first frame
    :param obsColl2: ObsCollection with the observations from the second frame
    :param measure: measure using IoU (IoU, mIoU, BIoU, GIoU, DIoU, sIoU)
    """

    cost_matrix = np.zeros((len(obsColl1), len(obsColl2)))
    for i, obsi in enumerate(obsColl1):
        for j, obsj in enumerate(obsColl2):
            if measure == 'IoU':
                cost_matrix[i, j] = 1-obsi.locator.IoU(obsj.locator) if obsi.locator is not None else 0.0
            elif measure == 'mIoU':
                cost_matrix[i, j] = 1-obsi.locator.mIoU(obsj.locator) if obsi.locator is not None else 0.0
            elif measure == 'BIoU':
                cost_matrix[i, j] = 1-obsi.locator.BIoU(obsj.locator, b=kwargs['b']) if obsi.locator is not None else 0.0
            elif measure == 'GIoU':
                cost_matrix[i, j] = 1-obsi.locator.GIoU(obsj.locator) if obsi.locator is not None else 0.0            
            elif measure == 'DIoU':
                cost_matrix[i, j] = 1-obsi.locator.DIoU(obsj.locator) if obsi.locator is not None else 0.0  
            elif measure == 'sIoU':
                cost_matrix[i, j] = 1-obsi.locator.sIoU(obsj.locator) if obsi.locator is not None else 0.0                    
            else:
                raise NotImplementedError(f'The measure {measure} is not implemented yet!')

    return cost_matrix
