#import simplified
import contour


def tune(Gc, Gp, method='simplified'):
    if method == 'simplified':
        raise NotImplementedError()
        #return simplified.tune(Gc, Gp)
    elif method == 'contour':
        return contour.tune(Gc, Gp)
    else:
        raise ValueError('Unrecognized tuning method')
