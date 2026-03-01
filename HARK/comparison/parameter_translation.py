"""
Parameter translation utilities for model comparison.

This module handles the mapping between unified primitive parameters
and method-specific parameter names and structures.
"""

from copy import deepcopy


class ParameterTranslator:
    """
    Handles parameter translation between different solution methods.

    Different solution methods often use different parameter names and
    structures for the same economic concepts. This class provides
    mappings to translate between them.
    """

    def __init__(self):
        """Initialize parameter translator with standard mappings."""
        # Define parameter name mappings for different methods
        self.param_mappings = {
            "krusell_smith/HARK": {
                "DiscFac": "DiscFac",
                "CRRA": "CRRA",
                "Rfree": "Rfree",
                "PermGroFac": "PermGroFac",
                "CapShare": "CapShare",
                "DeprFac": "DeprFac",
                "LbrInd": "LbrInd",
                "ProdB": "ProdB",
                "ProdG": "ProdG",
                "UrateB": "UrateB",
                "UrateG": "UrateG",
                "DurMeanB": "DurMeanB",
                "DurMeanG": "DurMeanG",
                "SpellMeanB": "SpellMeanB",
                "SpellMeanG": "SpellMeanG",
            },
            "aiyagari/HARK": {
                "DiscFac": "DiscFac",
                "CRRA": "CRRA",
                "Rfree": "Rfree",
                "LivPrb": "LivPrb",
                "PermGroFac": "PermGroFac",
                "TranShkStd": "TranShkStd",
                "PermShkStd": "PermShkStd",
                "UnempPrb": "UnempPrb",
                "IncUnemp": "IncUnemp",
                "BoroCnstArt": "BoroCnstArt",
            },
            "ssj": {
                "DiscFac": "beta",
                "CRRA": "eis",  # Note: EIS = 1/CRRA
                "Rfree": "r_ss",
                "CapShare": "alpha",
                "DeprFac": "delta",
                "LbrInd": "L_ss",
            },
            "maliar_winant_euler": {
                "DiscFac": "beta",
                "CRRA": "gamma",
                "Rfree": "R",
                "CapShare": "alpha",
                "DeprFac": "delta",
            },
            "maliar_winant_bellman": {
                "DiscFac": "beta",
                "CRRA": "gamma",
                "Rfree": "R",
                "CapShare": "alpha",
                "DeprFac": "delta",
            },
            "maliar_winant_reward": {
                "DiscFac": "beta",
                "CRRA": "gamma",
                "Rfree": "R",
                "CapShare": "alpha",
                "DeprFac": "delta",
            },
        }

        # Special transformations needed for some methods
        self.transformations = {
            "ssj": {
                "eis": lambda params: 1.0 / params.get("CRRA", 2.0),
                "r_ss": lambda params: params.get("Rfree", 1.03) - 1.0,
            }
        }

    def translate(self, primitives: dict, method: str, method_config: dict) -> dict:
        """
        Translate primitive parameters to method-specific format.

        Parameters
        ----------
        primitives : dict
            Unified primitive parameters
        method : str
            Target solution method
        method_config : dict
            Method-specific configuration (grids, etc.)

        Returns
        -------
        translated : dict
            Parameters in method-specific format
        """
        # Start with a copy of primitives
        translated = deepcopy(primitives)

        # Add method-specific configuration
        translated.update(deepcopy(method_config))

        # Apply parameter name mappings if available
        if method in self.param_mappings:
            mapping = self.param_mappings[method]
            remapped = {}

            for prim_name, method_name in mapping.items():
                if prim_name in primitives:
                    remapped[method_name] = primitives[prim_name]

            # Update with remapped parameters
            translated.update(remapped)

        # Apply any special transformations
        if method in self.transformations:
            transforms = self.transformations[method]
            for param_name, transform_func in transforms.items():
                translated[param_name] = transform_func(primitives)

        # Handle method-specific requirements
        translated = self._add_method_specific_params(translated, method, primitives)

        return translated

    def _add_method_specific_params(
        self, params: dict, method: str, primitives: dict
    ) -> dict:
        """Add any method-specific parameters that aren't direct translations."""

        if method == "krusell_smith/HARK":
            # KS model needs some additional parameters
            if "AgentCount" not in params:
                params["AgentCount"] = 10000
            if "act_T" not in params:
                params["act_T"] = 11000
            if "T_discard" not in params:
                params["T_discard"] = 1000
            if "DampingFac" not in params:
                params["DampingFac"] = 0.5
            if "tolerance" not in params:
                params["tolerance"] = 0.0001

            # Markov transition parameters
            if "RelProbBG" not in params:
                params["RelProbBG"] = 0.75
            if "RelProbGB" not in params:
                params["RelProbGB"] = 1.25

        elif method == "aiyagari/HARK":
            # Aiyagari model specifics
            if "AgentCount" not in params:
                params["AgentCount"] = 10000
            if "T_cycle" not in params:
                params["T_cycle"] = 1
            if "cycles" not in params:
                params["cycles"] = 0  # Infinite horizon

            # Income process discretization
            if "TranShkCount" not in params:
                params["TranShkCount"] = 7
            if "PermShkCount" not in params:
                params["PermShkCount"] = 7

            # Asset grid
            if "aXtraMin" not in params:
                params["aXtraMin"] = 0.001
            if "aXtraMax" not in params:
                params["aXtraMax"] = 50.0
            if "aXtraCount" not in params:
                params["aXtraCount"] = 48
            if "aXtraNestFac" not in params:
                params["aXtraNestFac"] = 3

        elif method == "ssj":
            # SSJ needs some specific calibration
            if "Y_ss" not in params:
                params["Y_ss"] = 1.0
            if "w_ss" not in params:
                # Compute steady state wage from production function
                alpha = params.get("alpha", params.get("CapShare", 0.36))
                params["w_ss"] = (1 - alpha) * params["Y_ss"]

        elif method.startswith("maliar_winant"):
            # Deep learning methods need neural network configuration
            if "n_agents" not in params:
                params["n_agents"] = 10000
            if "n_periods" not in params:
                params["n_periods"] = 1000

            # Neural network architecture (if not provided)
            if "nn_layers" not in params:
                params["nn_layers"] = [64, 64, 32]
            if "activation" not in params:
                params["activation"] = "relu"
            if "learning_rate" not in params:
                params["learning_rate"] = 0.001
            if "batch_size" not in params:
                params["batch_size"] = 256
            if "epochs" not in params:
                params["epochs"] = 100

        return params

    def reverse_translate(self, method_params: dict, method: str) -> dict:
        """
        Translate method-specific parameters back to primitive format.

        Parameters
        ----------
        method_params : dict
            Method-specific parameters
        method : str
            Source solution method

        Returns
        -------
        primitives : dict
            Unified primitive parameters
        """
        primitives = {}

        if method in self.param_mappings:
            # Reverse the mapping
            reverse_mapping = {v: k for k, v in self.param_mappings[method].items()}

            for method_name, prim_name in reverse_mapping.items():
                if method_name in method_params:
                    primitives[prim_name] = method_params[method_name]

        # Handle special reverse transformations
        if method == "ssj":
            if "eis" in method_params:
                primitives["CRRA"] = 1.0 / method_params["eis"]
            if "r_ss" in method_params:
                primitives["Rfree"] = 1.0 + method_params["r_ss"]

        return primitives
