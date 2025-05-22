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
    If they do not have the same keys, return 1000 and raise a warning. Nothing
    in HARK should ever hit that warning.
    """
    if set(dict_a.keys()) != set(dict_b.keys()):
        warn("Dictionaries with keys that do not match are being compared.")
        return 1000.0
    return np.max([distance_metric(dict_a[key], dict_b[key]) for key in dict_a.keys()])


def distance_arrays(arr_a, arr_b):
    """
    If both inputs are array-like, return the maximum absolute difference b/w
    corresponding elements (if same shape); return difference in size if shapes
    do not align.
    """
    if arr_a.shape == arr_b.shape:
        return np.max(np.abs(arr_a - arr_b))
    return np.abs(arr_a.size - arr_b.size)


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
