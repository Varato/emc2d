from .transformations import expand, compose
from .drift_setup import DriftSetup


class Corrector(object):
    def __init__(self, drift_setup: DriftSetup):
        self._drift_setup = drift_setup
        self._current_model = None
        self._pmat = None




