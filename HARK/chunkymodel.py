"""
A new experimental module for demonstrating MNW's "model chunks" framework.
"""

from copy import deepcopy
import inspect
import numpy as np
from HARK.core import get_arg_names


class MissingObject:
    """
    Dummy class to represent a missing object or empty space.
    """

    def __init__(self):
        pass

    def __repr__(self):
        return "missing object"


class Chunk:
    """
    A class for representing "model chunks", which can be connected together to
    make a larger model. Subclasses of Chunk can provide default values for many
    of the core attributes. All arguments passed to the initializer are combined
    or appended to the default values.

    Parameters
    ----------
    requirements : [str], optional
        A list of strings naming objects that must exists for this Chunk to be
        *run* or *executed*. It is appended to the class attribute _requirements.
    parameters : dict, optional
        A dictionary of raw parameters assigned to this chunk. In typical usage,
        the user should only *directly* interact with the parameters dictionary,
        not other dictionaries. It is combined with the class attribute _parameters.
    constructors : dict, optional
        A dictionary of functions that construct more complicated objects; this
        can include objects that we would usually think of as being part of the
        "solution". It is combined with the class attribute _constructors.
    exposes : [str], optional
        A list of strings naming objects of this chunk that are *exposed* for
        other chunks-- "open ports", so to speak. When the expose() method is
        used, the attribute called exposed is updated. This input is combined with
        the class attribute _exposes.
    links : dict, optional
        A dictionary of references to Chunks, assigning objects to be pulled from
        each of those chunks. When the pull_links() method is used, the dictionary
        called linked is updated with values pulled from those chunks.
    dynamics : str, optional
        A string describing what *actually happens* when this chunk is *run* or
        *executed*. The format is not specified here because this is what Seb is
        working on.

    Returns
    -------
    None.
    """

    _requirements = []
    _parameters = {}
    _constructors = {}
    _exposes = []
    _links = {}
    _dynamics = None

    def __init__(
        self,
        requirements=[],
        parameters={},
        constructors={},
        exposes=[],
        links={},
        dynamics=None,
    ):
        """
        Make a new model chunk.
        """
        # Make requirements an attribute of self
        if type(requirements) is not list:
            raise TypeError("requirements must be a list of strings!")
        my_reqs = deepcopy(self._requirements)
        for req in requirements:
            if req not in my_reqs:
                my_reqs.append(req)
        self.requirements = my_reqs

        # Make parameters an attribute of self
        if type(parameters) is not dict:
            raise TypeError("parameters must be a dictionary!")
        my_params = deepcopy(self._parameters)
        my_params.update(**parameters)
        self.parameters = my_params

        # Make constructors an attribute of self
        if type(constructors) is not dict:
            raise TypeError("constructors must be a dictionary!")
        my_constr = deepcopy(self._constructors)
        my_constr.update(**constructors)
        self.constructors = my_constr

        # Make requirements an attribute of self
        if type(exposes) is not list:
            raise TypeError("exposes must be a list of strings!")
        my_exp = deepcopy(self._exposes)
        for exp in exposes:
            if exp not in my_exp:
                my_exp.append(exp)
        self.exposes = my_exp

        # Make links an attribute of self
        if type(links) is not dict:
            raise TypeError("links must be a dictionary!")
        my_links = deepcopy(self._links)
        my_links.update(**links)
        self.links = my_links

        # Make dynamics an attribute of self
        if dynamics is not None:
            if type(dynamics) is not str:
                raise TypeError("dynamics must be a string!")
            self.dynamics = self._dynamics + dynamics
        else:
            self.dynamics = self._dynamics

        # Make empty dictionary attributes (some will be added later)
        self.linked = {}
        self.built = {}
        self.exposed = {}
        self.data = {}
        self.tests = {}

    def assign_parameters(self, **kwds):
        """
        Assign an arbitrary number of attributes to this agent.

        Parameters
        ----------
        **kwds : keyword arguments
            Any number of keyword arguments of the form key=value.  Each value
            will be assigned to the attribute named in self.

        Returns
        -------
        None
        """
        self.parameters.update(kwds)

    def __getitem__(self, key):
        if type(key) is not str:
            raise TypeError("Requested key must be a string!")
        if key in self.linked.keys():
            out = self.linked[key]
        elif key in self.built.keys():
            out = self.built[key]
        elif key in self.parameters:
            out = self.parameters[key]
        else:
            raise ValueError("Can't find object named " + key)
        return out

    def construct(self, *args, force=False):
        """
        Top-level method for building constructed objects. If called without any
        inputs, construct builds each of the objects named in the keys of the
        constructors dictionary; it draws inputs for the constructors from the
        parameters dictionary and adds its results to the same. If passed one or
        more strings as arguments, the method builds only the named keys. The
        method will do multiple "passes" over the requested keys, as some cons-
        tructors require inputs built by other constructors. If any requested
        constructors failed to build due to missing data, those keys (and the
        missing data) will be named in self._missing_key_data. Other errors are
        recorded in the dictionary attribute _constructor_errors.

        Parameters
        ----------
        *args : str, optional
            Keys of self.constructors that are requested to be constructed. If
            no arguments are passed, *all* elements of the dictionary are implied.
        force : bool, optional
            When True, the method will force its way past any errors, including
            missing constructors, missing arguments for constructors, and errors
            raised during execution of constructors. Information about all such
            errors is stored in the dictionary attributes described above. When
            False (default), any errors or exception will be raised.

        Returns
        -------
        None
        """
        # Set up the requested work
        if len(args) > 0:
            keys = args
        else:
            keys = list(self.constructors.keys())
        N_keys = len(keys)
        keys_complete = np.zeros(N_keys, dtype=bool)

        # Get the dictionary of constructor errors
        if not hasattr(self, "_constructor_errors"):
            self._constructor_errors = {}
        errors = self._constructor_errors

        # As long as the work isn't complete and we made some progress on the last
        # pass, repeatedly perform passes of trying to construct objects
        any_keys_incomplete = np.any(np.logical_not(keys_complete))
        go = any_keys_incomplete
        while go:
            anything_accomplished_this_pass = False  # Nothing done yet!
            missing_key_data = []  # Keep this up-to-date on each pass

            # Loop over keys to be constructed
            for i in range(N_keys):
                if keys_complete[i]:
                    continue  # This key has already been built

                # Get this key and its constructor function
                key = keys[i]
                try:
                    constructor = self.constructors[key]
                except Exception as not_found:
                    errors[key] = "No constructor found for " + str(not_found)
                    self.built[key] = MissingObject()
                    if force:
                        continue
                    else:
                        raise ValueError("No constructor found for " + key) from None

                # Get the names of arguments for this constructor and try to gather them
                args_needed = get_arg_names(constructor)
                has_no_default = {
                    k: v.default is inspect.Parameter.empty
                    for k, v in inspect.signature(constructor).parameters.items()
                }
                temp_dict = {}
                any_missing = False
                missing_args = []
                for j in range(len(args_needed)):
                    this_arg = args_needed[j]
                    try:
                        temp_dict[this_arg] = self[this_arg]
                    except:
                        if has_no_default[this_arg]:
                            # Record missing key-data pair
                            any_missing = True
                            missing_key_data.append((key, this_arg))
                            missing_args.append(this_arg)

                # If all of the required data was found, run the constructor and
                # store the result in parameters (and on self)
                if not any_missing:
                    try:
                        temp = constructor(**temp_dict)
                    except Exception as problem:
                        errors[key] = str(type(problem)) + ": " + str(problem)
                        self.built[key] = MissingObject()
                        if force:
                            continue
                        else:
                            raise
                    setattr(self, key, temp)
                    self.parameters[key] = temp
                    if key in errors:
                        del errors[key]
                    keys_complete[i] = True
                    anything_accomplished_this_pass = True  # We did something!
                else:
                    msg = "Missing required arguments:"
                    for arg in missing_args:
                        msg += " " + arg + ","
                    msg = msg[:-1]
                    errors[key] = msg
                    self.built[key] = MissingObject()
                    # Never raise exceptions here, as the arguments might be filled in later

            # Check whether another pass should be performed
            any_keys_incomplete = np.any(np.logical_not(keys_complete))
            go = any_keys_incomplete and anything_accomplished_this_pass

    def expose(self):
        """
        Updates the dictionary attribute called exposed with each of the attributes
        named in the list called exposes. If cannot be found, a MissingObject is used.

        Returns
        -------
        None.
        """
        exposed = {}
        for name in self.exposes:
            try:
                exposed[name] = self[name]
            except:
                exposed[name] = MissingObject()
        self.exposed = exposed

    def pull_links(self, copy=False):
        """
        Updates the dictionary attribute called linked by pulling information from
        the Chunks referenced in the dictionary links. If something cannot be found,
        a MissingObject is used.

        Parameters
        ----------
        copy : bool, optional
            Indicator for whether the linked information should be copied (True)
            or just referenced (False).

        Returns
        -------
        None
        """
        linked = {}
        for key in self.links.keys():
            other = self.links[key]
            if not isinstance(other, Chunk):
                raise TypeError("References in links dictionary must be to chunks!")
            try:
                fetched = other.exposed[key]
            except:
                fetched = MissingObject()

            if copy:
                linked[key] = deepcopy(fetched)
            else:
                linked[key] = fetched
        self.linked = linked

    def build(self):
        """
        Top level method to build or update the chunk by pulling in linked objects,
        constructing as much as possible, and then exposing designated objects for
        other chunks to view and use.

        Returns
        -------
        None
        """
        self.pull_links()
        self.construct()
        self.expose()

    def link_to(self, other, keys=[]):
        """
        Link one or more named keys to one particular chunk by adding references
        to the links dictionary attribute.

        Parameters
        ----------
        other : Chunk
            A reference to another model Chunk.
        keys : str or [str]
            One or more strings naming objects that should be linked from the other Chunk.

        Returns
        -------
        None
        """
        if not isinstance(other, Chunk):
            raise TypeError("The reference must be to another Chunk!")

        if type(keys) is str:
            self.links[keys] = other
            return
        elif type(keys) is list:
            for key in keys:
                if type(key) is not str:
                    raise TypeError("Each key in keys must be a string!")
                self.links[key] = other
            return
        else:
            raise TypeError("The keys must be a string or list of strings!")

    def check_reqs(self, verbose=False):
        """
        Evaluate whether this Chunk's requirements exist so that it can be *run*
        or *executed*, returning a boolean. Can optionally print status to screen.

        Parameters
        ----------
        verbose : bool, optional
            Indicator for whether to print status of requirements to screen. The
            default is False.

        Returns
        -------
        ready : bool
            Indicator for whether all objects named in requirements are present,
            meaning that this Chunk can be *run* or *executed*.
        """
        N = len(self.requirements)
        checklist = np.zeros(N, dtype=bool)

        for i in range(N):
            key = self.requirements[i]
            try:
                fetch = self[key]
                if not isinstance(fetch, MissingObject):
                    checklist[i] = True
            except:
                pass

        if verbose:
            print("Verbose functionality of check_reqs is missing!")

        ready = np.all(checklist)
        return ready

    def run(self):
        """
        Run or execute this model chunk. Does absolutely nothing right now.
        """
        pass
