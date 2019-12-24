'''
Specifies examples of the full set of parameters required to solve various
consumption-saving models.  These models can be found in ConsIndShockModel,
ConsAggShockModel, ConsPrefShockModel, and ConsMarkovModel.
'''

import copy
import os
import yaml

config_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ConsumerParameters.yaml")

config = yaml.load(open(config_path).read())

def inherit(params):
    if 'EXTENDS' in params:
        original = params
        extensions = inherit(config[params['EXTENDS'][0]])
        # TODO: No list

        new = copy.copy(extensions)
        new.update(original)
        del new['EXTENDS']

        return new
    else:
        return params

all_params = {k : inherit(config[k]) for k in config.keys()}
locals().update(all_params)
