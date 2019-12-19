import copy
import yaml

config_path = "ConsumptionSaving/ConsumerParameters.yaml"

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

