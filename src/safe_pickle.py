"""Restricted unpickler for the MiliPoint .pkl files: only numpy array
reconstruction is allowed, so a tampered file cannot execute code."""
import pickle
ALLOWED = {('numpy.core.multiarray', '_reconstruct'), ('numpy._core.multiarray', '_reconstruct'),
           ('numpy', 'ndarray'), ('numpy', 'dtype'), ('numpy.core.multiarray', 'scalar'),
           ('numpy._core.multiarray', 'scalar')}
class _U(pickle.Unpickler):
    def find_class(self, module, name):
        if (module, name) in ALLOWED:
            return super().find_class(module, name)
        raise pickle.UnpicklingError(f'blocked global {module}.{name}')
def load(path):
    with open(path, 'rb') as f:
        return _U(f).load()
