from warnings import warn

import numpy as np


def distance_lists(list_a, list_b):
    """
    If both inputs are lists, then the distance between them is the maximum
    distance between corresponding elements in the lists.  If they differ in
    length, the distance is the difference in lengths.
    """
    len_a = len(list_a)
    len_b = len(list_b)
    if len_a == len_b:
        return np.max([distance_metric(list_a[n], list_b[n]) for n in range(len_a)])
    return np.abs(len_a - len_b)


def distance_dicts(dict_a, dict_b):
    """
    If both inputs are dictionaries, call distance on the list of its elements.
    If both dictionaries have matching distance_criteria entries, compare only those keys.
    If they do not have the same keys, return 1000 and raise a warning. Nothing
    in HARK should ever hit that warning.
    """
    # Check whether the dictionaries have matching distance_criteria
    if ("distance_criteria" in dict_a.keys()) and (
        "distance_criteria" in dict_b.keys()
    ):
        crit_a = dict_a["distance_criteria"]
        crit_b = dict_b["distance_criteria"]
        if len(crit_a) == len(crit_b):
            check = [crit_a[j] == crit_b[j] for j in range(len(crit_a))]
            if np.all(check):
                # Compare only their distance_criteria
                return np.max(
                    [distance_metric(dict_a[key], dict_b[key]) for key in crit_a]
                )

    # Otherwise, compare all their keys
    if set(dict_a.keys()) != set(dict_b.keys()):
        warn("Dictionaries with keys that do not match are being compared.")
        return 1000.0
    return np.max([distance_metric(dict_a[key], dict_b[key]) for key in dict_a.keys()])


def distance_arrays(arr_a, arr_b):
    """
    If both inputs are array-like, return the maximum absolute difference b/w
    corresponding elements (if same shape). If they don't even have the same number
    of dimensions, return 10000 times the difference in dimensions. If they have
    the same number of dimensions but different shapes, return the sum of differences
    in size for each dimension.
    """
    shape_A = arr_a.shape
    shape_B = arr_b.shape
    if shape_A == shape_B:
        return np.max(np.abs(arr_a - arr_b))

    if len(shape_A) != len(shape_B):
        return 10000 * np.abs(len(shape_A) - len(shape_B))

    dim_diffs = np.abs(np.array(shape_A) - np.array(shape_B))
    return np.sum(dim_diffs)


def distance_class(cls_a, cls_b):
    """
    If none of the above cases, but the objects are of the same class, call the
    distance method of one on the other.
    """
    if isinstance(cls_a, type(lambda: None)):
        warn("Cannot compare lambda functions. Returning large distance.")
        return 1000.0
    return cls_a.distance(cls_b)


def distance_metric(thing_a, thing_b):
    """
    A "universal distance" metric that can be used as a default in many settings.

    Parameters
    ----------
    thing_a : object
        A generic object.
    thing_b : object
        Another generic object.

    Returns:
    ------------
    distance : float
        The "distance" between thing_a and thing_b.
    """

    # If both inputs are numbers, return their difference
    if isinstance(thing_a, (int, float)) and isinstance(thing_b, (int, float)):
        return np.abs(thing_a - thing_b)

    if isinstance(thing_a, list) and isinstance(thing_b, list):
        return distance_lists(thing_a, thing_b)

    if isinstance(thing_a, np.ndarray) and isinstance(thing_b, np.ndarray):
        return distance_arrays(thing_a, thing_b)

    if isinstance(thing_a, dict) and isinstance(thing_b, dict):
        return distance_dicts(thing_a, thing_b)

    if isinstance(thing_a, type(thing_b)):
        return distance_class(thing_a, thing_b)

    # Failsafe: the inputs are very far apart
    return 1000.0


def describe_metric(thing, n=0, label=None, D=100000):
    """
    Generate a description of an object's distance metric.

    Parameters
    ----------
    thing : object
        A generic object.
    n : int
        Recursive depth of this call.
    label : str or None
        Name/label of the thing, which might be a dictionary key, attribute name,
        list index, etc.
    D : int
        Maximum recursive depth; if n > D, empty output is returned.

    Returns
    -------
    desc : str
        Description of this object's distance metric, indented 2n spaces.
    """
    pad = 2
    if n > D:
        return ""

    if label is None:
        desc = ""
    else:
        desc = pad * n * " " + "- " + label + " "

    # If both inputs are numbers, distance is their difference
    if isinstance(thing, (int, float)):
        desc += "(scalar): absolute difference of values\n"

    elif isinstance(thing, list):
        J = len(thing)
        desc += "(list) largest distance among:\n"
        if n == D:
            desc += pad * (n + 1) * " " + "SUPPRESSED OUTPUT\n"
        else:
            for j in range(J):
                desc += describe_metric(thing[j], n + 1, label="[" + str(j) + "]", D=D)

    elif isinstance(thing, np.ndarray):
        desc += (
            "(array"
            + str(thing.shape)
            + "): greatest absolute difference among elements\n"
        )

    elif isinstance(thing, dict):
        if "distance_criteria" in thing.keys():
            my_keys = thing["distance_criteria"]
        else:
            my_keys = thing.keys()
        desc += "(dict): largest distance among these keys:\n"
        if n == D:
            desc += pad * (n + 1) * " " + "SUPPRESSED OUTPUT\n"
        else:
            for key in my_keys:
                try:
                    desc += describe_metric(thing[key], n + 1, label=key, D=D)
                except:
                    desc += key + " (missing): CAN'T COMPARE\n"

    elif isinstance(thing, MetricObject):
        my_keys = thing.distance_criteria
        desc += (
            "(" + type(thing).__name__ + "): largest distance among these attributes:\n"
        )
        if len(my_keys) == 0:
            desc += pad * (n + 1) * " " + "NO distance_criteria SPECIFIED\n"
        if n == D:
            desc += pad * (n + 1) * " " + "SUPPRESSED OUTPUT\n"
        else:
            for key in my_keys:
                if hasattr(thing, key):
                    desc += describe_metric(getattr(thing, key), n + 1, label=key, D=D)
                else:
                    desc += key + " (missing): CAN'T COMPARE\n"

    else:
        # Something has gone wrong
        desc += "WARNING: INCOMPARABLE\n"

    return desc


class MetricObject:
    """
    A superclass for object classes in HARK.  Comes with two useful methods:
    a generic/universal distance method and an attribute assignment method.
    """

    distance_criteria = []  # This should be overwritten by subclasses.

    def distance(self, other):
        """
        A generic distance method, which requires the existence of an attribute
        called distance_criteria, giving a list of strings naming the attributes
        to be considered by the distance metric.

        Parameters
        ----------
        other : object
            Another object to compare this instance to.

        Returns
        -------
        (unnamed) : float
            The distance between this object and another, using the "universal
            distance" metric.
        """
        try:
            return np.max(
                [
                    distance_metric(getattr(self, attr_name), getattr(other, attr_name))
                    for attr_name in self.distance_criteria
                ]
            )
        except (AttributeError, ValueError):
            return 1000.0

    def describe_distance(self, display=True, max_depth=None):
        """
        Generate a description for how this object's distance metric is computed.
        By default, the description is printed to screen, but it can be returned.

        Like the distance metric itself, the description is built recursively.

        Parameters
        ----------
        display : bool, optional
            Whether the description should be printed to screen (default True).
            Otherwise, it is returned as a string.
        max_depth : int or None
            If specified, the maximum recursive depth of the description.

        Returns
        -------
        out : str
            Description of how this object's distance metric is computed, if
            display=False.
        """
        max_depth = max_depth if max_depth is not None else np.inf

        keys = self.distance_criteria
        if len(keys) == 0:
            out = "No distance criteria are specified; please name them in distance_criteria.\n"
        else:
            out = describe_metric(self, D=max_depth)
        out = out[:-1]
        if display:
            print(out)
            return
        return out
