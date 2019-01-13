import os
import json

from bmtk.simulator.core.config import ConfigDict


class Config(ConfigDict):
    @property
    def jitter(self):
        conds = self.conditions
        has_lj = 'jitter_lower' in conds
        has_uj = 'jitter_upper' in conds

        if has_lj and has_uj:
            return (conds['jitter_lower'], conds['jitter_upper'])
        elif has_lj ^ has_uj:
            raise Exception('Please define both jitter_upper and jitter_lower parameters')
        else:
            return None