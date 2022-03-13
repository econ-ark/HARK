from collections import OrderedDict
import copy
from sre_constants import SRE_FLAG_ASCII
from HARK import AgentType, Model
from HARK.distribution import Distribution, TimeVaryingDiscreteDistribution
import itertools

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np


class Frame():
    """
    An object representing a single 'frame' of an optimization problem.
    A frame defines some variables of a model, including what other variables
    (if any) they depend on for their values.

    Parameters
    ----------
    target : tuple
        A tuple of variable names
    scope : tuple
        A tuple of variable names. The variables this frame depends on for transitions.
    default : Distribution
        Default values for these target variables for simulation initialization.
    transition : function
        A function from scope variables to target variables.
    objective : function
        A function for use in the solver. [??]
    aggregate : bool, default False
        True if the frame is an aggregate state variable.
    control : bool, default False
        True if the frame targets are control variables.
    reward : bool, default False
        True if the frame targets are reward variables.
    context : dict, Optional
        A dictionary of additional values used by the transition function.
    Attributes
    -----------

    parents : dict
        A dictionary of frames on which these frames depend.
        May include backward references.

    children : dict
        A dictionary of frames that depend on this frame.
        May include forward references.
    """

    def __init__(
            self,
            target : tuple,
            scope : tuple,
            default = None,
            transition = None,
            objective = None,
            aggregate = False,
            control = False,
            reward = False,
            context = None,
    ):
        """
        """

        self.target = target if isinstance(target, tuple) else (target,) # tuple of variables
        self.scope = scope # tuple of variables
        self.default = default # default value used in simBirth; a dict

        ## Careful! Transition functions need to return a tuple, even if there is only one state value
        self.transition = transition # for use in simulation
        self.objective = objective # for use in solver
        self.aggregate = aggregate
        self.control = control
        self.reward = reward

        # Context specific to this node
        self.context = {}
        if context is not None:
            self.context.update(context)

        # to be filled with references to other frames
        self.children = {}
        self.parents = {}

    def __repr__(self):
        return f"<{self.__class__}, target:{self.target}, scope:{self.scope}>"

    def name(self):
        target = self.target
        return str(target[0]) if len(target) == 1 else str(self.target)

    def clear_relations(self):
        """
        Empties the references to parents and children.

        TODO: Better handling of this aspect of frame state
              e.g. setters for the relations
        """
        self.children = {}
        self.parents = {}

    def add_suffix(self, suffix :str):
        """
        Change the names of all variables in this frame's target and scope
        (except for backward references) to include an additional suffix.

        This is used when copying or repreating frames.
        """
        self.target = tuple((var + suffix for var in self.target))

        self.scope = tuple((var 
                            if any(var in pa and isinstance(self.parents[pa], BackwardFrameReference)
                                   for pa in self.parents)
                            else var + suffix
                            for var in self.scope))

    def add_backwards_suffix(self, suffix : str):
        """
        Change the names of any scope variables that are backward references to
        include an additional suffix.
        """
        self.scope = tuple((var + suffix 
                        if any(var in pa and isinstance(self.parents[pa], BackwardFrameReference)
                        for pa in self.parents)
                        else var
                        for var in self.scope))

class ForwardFrameReference():
    """
    A 'reference' to a frame that is in the next period.

    The graphical children of frames that are at the "end" of a period will have these
    references pointing to frames that are at the begining of the next
    period.

    Parameters
    ----------

    frame : Frame
        The frame to which this reference refers.
    """

    def __init__(self, frame):
        self.frame = frame
        self.target = frame.target

        self.reward = frame.reward
        self.control = frame.control
        self.aggregate = frame.aggregate

    def name(self):
        return self.frame.name() + "'"

    def __repr__(self):
        return f"<FFR:{self.frame.target}>"

class BackwardFrameReference():
    """
    A 'reference' to a frame that is in the previous period.

    The graphical parents of frames that are at the "beginning"
    of a period will be these references to frames in the previous
    period.

    Parameters
    ----------

    frame : Frame
        The frame to which this reference refers.
    """

    def __init__(self, frame):
        self.frame = frame
        self.target = frame.target

        self.reward = frame.reward
        self.control = frame.control
        self.aggregate = frame.aggregate

    def name(self):
        return self.frame.name() + "-"

    def __repr__(self):
        return f"<BFR:{self.frame.target}>"


class FrameSet(OrderedDict):
    """
    A data structure for a collection of frames.

    Wraps an ordered dictionary, where keys are tuples of variable names,
    and values are Frames.

    Preserves order. Is sliceable and has index() functions like a list.
    Supports lookup of frame by variable name.
    """
    def __getitem__(self, k):
        if not isinstance(k, slice):
            return OrderedDict.__getitem__(self, k)
        return FrameSet(itertools.islice(self.items(), k.start, k.stop))

    def k_index(self, key):
        return list(self.keys()).index(key)

    def v_index(self, value):
        return list(self.keys()).index(value)

    def var(self, var_name):
        """
        Returns the frame in this frame set that includes the
        named variable as a target.

        Parameters
        ----------

        var_name : str
            The name of a variable
        """
        ## Can be sped up with a proper index.
        for k in self:
            if var_name in k:
                return self[k]

        return None

    def iloc(self, k):
        """
        Returns the frame in this frame set that corresponds
        to the given numerical index.

        Parameters
        ----------

        k : int
            The numerical index of the frame in the FrameSet
        """
        return list(self.values())[k]


class FrameModel(Model):
    """
    A class that represents a model, defined in terms of Frames.

    Frames can be transitional/functional, or they can be control frames
    (subject to an agent's policy), or a reward frame.

    FrameModels can be composed with other FrameModels into new models.

    Parameters
    ------------

    frames : [Frame]
        List of frames to include in the FrameSet.

    parameters : dict

    infinite: bool
        True if the model is an infinite model, such that state variables are assumed to be
        available as scope for the next period's transitions.

    Attributes
    ----------

    frames : FrameSet[Frame]
        #Keys are tuples of strings corresponding to model variables.
        #Values are methods.
        #Each frame method should update the the variables
        #named in the key.
        #Frame order is significant here.
    """

    def __init__(self, frames, parameters, infinite = True):
        super().__init__()

        self.frames = FrameSet([(fr.target, fr) for fr in frames])
        self.infinite = infinite

        self.assign_parameters(**parameters)

        for frame in self.frames.values():
            # relations for the frame -- internal links to other frames -- are reset in model initiation
            frame.clear_relations()

        for frame_target in self.frames:

            frame = self.frames[frame_target]

            if frame.scope is not None:
                for var in frame.scope:

                    ## Should replace this with a new data structure that allows for multiple keys into the same frame
                    scope_frames = [self.frames[frame_target] for frame_target in self.frames if var in frame_target]

                    ## There should only be one frame in this list.
                    for scope_frame in scope_frames:
                        if self.frames.k_index(frame_target) > self.frames.k_index(scope_frame.target):
                            if frame not in scope_frame.children:
                                ## should probably use frame data structure here
                                scope_frame.children[frame_target] = frame

                            if scope_frame not in frame.parents:
                                frame.parents[scope_frame.target] = scope_frame
                        else:
                            
                            ## Do I need to keep backward references even in a finite model, because these
                            ## are initial conditions?
                            bfr = BackwardFrameReference(frame)
                            frame.parents[scope_frame.target] = bfr

                            # ignoring equivalence checks for now
                            if infinite:
                                ffr = ForwardFrameReference(frame)
                                scope_frame.children[frame_target] = ffr

    def prepend(self, model, suffix='_0'):
        """
        Combine this FrameModel with another FrameModel.

        TODO: Checks to make sure the endpoints match.

        Parameters
        ------------

        model: FrameModel

        suffix: str
            A suffix to add to any variables in the prepended model that have
            a name conflict with the old model.


        Returns
        --------

        FrameModel
        """

        pre_frames = list(copy.deepcopy(model.frames).values())

        suffix = "_"

        for frame in pre_frames:
            frame.add_suffix(suffix)
    
        frames = list(copy.deepcopy(self.frames).values())

        for frame in frames:
    
            frame.add_backwards_suffix(suffix)

        return FrameModel(pre_frames + frames, self.parameters, infinite = self.infinite)

    def make_terminal(self):
        """
        Remove the forward references from the end of the model,
        making the model "finite".

        Returns
        --------

        FrameModel
        """

        # Is this copying the old frames right?
        new_frames = copy.deepcopy(list(self.frames.values()))

        for frame in new_frames:
            forward_references = [child for child in frame.children if isinstance(child, ForwardFrameReference)]

            for fref in forward_references:
                frame.children.remove(fref)
        
        return FrameModel(new_frames, self.parameters, infinite = False)
            

    def repeat(self, tv_parameters):
        """
        Returns a new FrameModel consisting of this model repeated N times.

        Parameters
        -----------

        tv_parameters : dict
            A dictionary of 'time-varying' parameters.
            Keys are (original) variable names. Values are dictionaries with:
               Keys are parameter names. Values as iterable contain time-varying
               parameter values. All time-varying values assumes to be of same length, N.

        """
        # getting length of first iterable thing passed to it.
        repeat_n = len(list(list(tv_parameters.values())[0].values())[0])

        catalog = {}

        new_frames = [copy.deepcopy(self.frames) for t in range(repeat_n)]

        for frame in self.frames:
            # catalog is a convenient alternative index of the new frames
            catalog[frame] = [new_frames[t][frame] for t in range(repeat_n)]

            # distribute any time-varying parameters.
            for t, t_frame in enumerate(catalog[frame]):
                t_frame.add_suffix(f"_{t}")

                if t > 0:
                    t_frame.add_backwards_suffix(f"_{t-1}")
        
        for var_name in tv_parameters:
            for param in tv_parameters[var_name]:
                for t, pv in enumerate(tv_parameters[var_name][param]):
                    new_frames[t].var(var_name).context[param] = pv

        return FrameModel(
            itertools.chain.from_iterable([
                frame_set.values()
                for frame_set in 
                new_frames
                ]),
            self.parameters,
            infinite = self.infinite
            )


class FrameAgentType(AgentType):
    """
    A variation of AgentType that uses Frames to organize
    its simulation steps.

    The FrameAgentType is initalizaed with a FrameModel,
    which contains all the information needed to execute
    generic simulation methods.

    Parameters
    -----------

    model : FrameModel

    Attributes
    -----------

    decision_rules : dict
        A dictionary of decision rules used to determine the
        transitions of control variables.


    """

    cycles = 0 # for now, only infinite horizon models.

    def __init__(self, model, **kwds):
        self.model = model

        ### kludge?
        self.frames = self.model.frames

        # decision rules are added here which are then used in simulation.
        self.decision_rules = {}

    def initialize_sim(self):

        for frame in self.frames.values():
            for var in frame.target:

                if frame.aggregate:
                    val = np.empty(1)
                    if frame.default is not None and var in frame.default:
                        val[:] = frame.default[var]
                else:
                    val = np.empty(self.AgentCount)

                if frame.control:
                    self.controls[var] = val
                elif  isinstance(frame.transition, Distribution):
                    self.shocks[var] = val
                else:
                    self.state_now[var] = val
    
        super().initialize_sim()

    def sim_one_period(self):
        """
        Simulates one period for this type.
        Calls each frame in order.
        These should be defined for
        AgentType subclasses, except getMortality (define
        its components simDeath and simBirth instead)
        and readShocks.

        Parameters
        ----------
        None

        Returns
        -------
        None
        """
        if not hasattr(self, "solution"):
            raise Exception(
                "Model instance does not have a solution stored. To simulate, it is necessary"
                " to run the `solve()` method of the class first."
            )

        # Mortality adjusts the agent population
        self.get_mortality()  # Replace some agents with "newborns"

        # state_{t-1}
        for frame in self.frames.values():
            for var in frame.target:
                if var in self.state_now:
                    self.state_prev[var] = self.state_now[var]
                
                    if not frame.aggregate:
                        self.state_now[var] = np.empty(self.AgentCount)
                    else:
                        self.state_now[var] = np.empty(1)

        # transition the variables in the frame
        for frame in self.frames.values():
            self.transition_frame(frame)

        # Advance time for all agents
        self.t_age = self.t_age + 1  # Age all consumers by one period
        self.t_cycle = self.t_cycle + 1  # Age all consumers within their cycle
        self.t_cycle[
            self.t_cycle == self.T_cycle
        ] = 0  # Resetting to zero for those who have reached the end

    def sim_birth(self, which_agents):
        """
        Makes new agents for the simulation.
        Takes a boolean array as an input, indicating which
        agent indices are to be "born".

        Populates model variable values with value from `init`
        property

        Parameters
        ----------
        which_agents : np.array(Bool)
            Boolean array of size self.AgentCount indicating which agents should be "born".

        Returns
        -------
        None
        """
        which_agents = which_agents.astype(bool)

        for frame in self.frames.values():
            if not frame.aggregate:
                for var in frame.target:

                    N = np.sum(which_agents)

                    if frame.default is not None and var in frame.default:
                        if callable(frame.default[var]):
                            value = frame.default[var](self, N)
                        else:
                            value = frame.default[var]

                        if var in self.state_now:
                            ## need to check in case of aggregate variables.. PlvlAgg
                            if hasattr(self.state_now[var],'__getitem__'):
                                self.state_now[var][which_agents] = value
                        elif var in self.controls:
                            self.controls[var][which_agents] = value
                        elif var in self.shocks:
                            ## assuming no aggregate shocks... 
                            self.shocks[var][which_agents] = value

        # from ConsIndShockModel. Needed???
        self.t_age[which_agents] = 0  # How many periods since each agent was born
        self.t_cycle[
            which_agents
        ] = 0  # Which period of the cycle each agent is currently in

        ## simplest version of this.
    def transition_frame(self, frame):
        """
        Updates the model variables in `target`
        using the `transition` function.
        The transition function will use current model
        variable state as arguments.
        """
        # build a context object based on model state variables
        # and 'self' reference for 'global' variables
        context = {} # 'self' : self}
        context.update(self.shocks)
        context.update(self.controls)
        context.update(self.state_prev)

        # use the "now" version of variables that have already been targetted.
        for pre_frame in self.frames[:self.frames.k_index(frame.target)].values():
            for var in pre_frame.target:
                if var in self.state_now:
                    context.update({var : self.state_now[var]})

        ## Get these parameters from the FrameModel.parameters
        ## ... Unless there are also _simulation_ parameters attached to the AgentType
        context.update(self.parameters)

        # The "most recently" computed value of the variable is used.
        # This could be the value from the 'previous' time step.

        # limit context to scope of frame
        local_context = {
            var : context[var]
            for var
            in frame.scope
        } if frame.scope is not None else context.copy()

        ## TODO
        ## - A repeated model may have transition equations that do not reference the right "suffixes",
        ##   so contextual lookup will require matching on the shared prefix
        ##
        ## - Local context can be loaded onto a node in the FrameModel.repeat() step with
        ##   age-varying parameters
        ##
        ## - Consider relationship between AgentType simulation mechanics (here) and the FrameModel definition.

        if frame.control:
            new_values = self.control_transition_age_varying(frame.target, **local_context)

        elif frame.transition is not None:
            if isinstance(frame.transition, Distribution):
                # assume this is an IndexDistribution keyed to age (t_cycle)
                # for now
                # later, t_cycle should be included in local context, etc.
                if frame.aggregate:
                    new_values = (frame.transition.draw(1),)
                else:    
                    new_values = (frame.transition.draw(self.t_cycle),)

            else: # transition is function of state variables not an exogenous shock
                new_values = frame.transition(
                    # self,
                    **local_context
                )

        else:
            raise Exception(f"Frame has None for transition: {frame}")

        # because we want to alter the 'now' not 'prev' table
        context.update(self.state_now)

        # because the context was a shallow update,
        # the model values can be modified directly(?)
        for i,t in enumerate(frame.target):
            if t in context:
                context[t][:] = new_values[i]
            else:
                raise Exception(f"From frame {frame.target}, target {t} is not in the context object.")

    def control_transition_age_varying(self, target, **context):
        """
        Generic transition method for a control frame for when the
        variable has an age-varying decision rule.

        """
        frame = self.model.frames[target]
        scope = frame.scope

        target_values = tuple((np.zeros(self.AgentCount) + np.nan for var in scope))

        # Loop over each period of the cycle, getting controls separately depending on "age"
        for t in range(self.T_cycle):
            these = t == self.t_cycle

            ## maybe scope instead of context here
            ntv = self.decision_rules[target][t](**context)

            # this is ugly because of the way ages are looped through. See #890
            for i, tv in enumerate(target_values):
                tv[these] = ntv[i]

        return target_values


def draw_frame_model(frame_model: FrameModel, figsize = (8,8), dot = False):
    """
    Draws a FrameModel as an influence diagram.

    Round nodes : chance variables
    Square nodes: control variables
    Rhombus nodes: reward variables
    Hexagon nodes: aggregate variables
    """
    
    g = nx.DiGraph()

    g.add_nodes_from([
        (frame.name(), 
         {'control' : frame.control, 'reward' : frame.reward, 'aggregate' : frame.aggregate})
        for frame in frame_model.frames.values()
    ])

    for frame in frame_model.frames.values():
        for child_target in frame.children:
            child = frame.children[child_target]
            g.add_nodes_from([
                (child.name(), 
                 {
                     'control' : child.control,
                     'reward' : child.reward,
                     'aggregate' : child.aggregate
                 })])
            g.add_edge(frame.name(), child.name())

    if dot:
        pos = nx.drawing.nx_pydot.graphviz_layout(g, prog='dot')
    else:
        pos = nx.drawing.layout.kamada_kawai_layout(g)

    node_options = {
        "node_size": 2500,
        "node_color": "white",
        "edgecolors": "black",
        "linewidths": 1,
        "pos" : pos
    }

    edge_options = {
        "node_size": 2500,
        "width": 2,
        "pos" : pos
    }

    label_options = {
        "font_size": 12,
         #"labels" : {node : str(node[0]) if len(node) == 1 else str(node) for node in g.nodes},
        "pos" : pos
    }

    reward_nodes = [k for k,v in g.nodes(data = True) if v['reward']]
    control_nodes = [k for k,v in g.nodes(data = True) if v['control']]
    aggregate_nodes = [k for k,v in g.nodes(data = True) if v['aggregate']]

    chance_nodes = [node for node in g.nodes() 
                    if node not in reward_nodes 
                    and node not in control_nodes
                    and node not in aggregate_nodes
                   ]

    plt.figure(figsize=figsize)

    nx.draw_networkx_nodes(g, nodelist = chance_nodes, node_shape = 'o', **node_options)
    nx.draw_networkx_nodes(g, nodelist = reward_nodes, node_shape = 'd', **node_options)
    nx.draw_networkx_nodes(g, nodelist = control_nodes, node_shape = 's', **node_options)
    nx.draw_networkx_nodes(g, nodelist = aggregate_nodes, node_shape = 'h', **node_options)
    nx.draw_networkx_edges(g, **edge_options)

    nx.draw_networkx_labels(g, **label_options)
