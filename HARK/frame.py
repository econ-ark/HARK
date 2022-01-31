from HARK import AgentType
from HARK.distribution import Distribution, TimeVaryingDiscreteDistribution
import numpy as np

class Frame():
    """
    An object representing a single 'frame' of an optimization problem.
    A frame defines some variables of a model, including what other variables
    (if any) they depend on for their values.
    """

    def __init__(
            self,
            target,
            scope,
            default = None,
            transition = None,
            objective = None,
            aggregate = False,
            control = False,
            reward = False
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

    def __repr__(self):
        return f"<{self.__class__}, target:{self.target}, scope:{self.scope}>"

    def name(self):
        target = self.target
        return str(target[0]) if len(target) == 1 else str(self.target)


class FrameAgentType(AgentType):
    """
    A variation of AgentType that uses Frames to organize
    its simulation steps.

    Frames allow for state, control, and shock resolutions
    in a specified order, rather than assuming that they
    are resolved as shocks -> states -> controls -> poststates.

    Attributes
    ----------

    frames : [Frame]
        #Keys are tuples of strings corresponding to model variables.
        #Values are methods.
        #Each frame method should update the the variables
        #named in the key.
        #Frame order is significant here.
    """

    cycles = 0 # for now, only infinite horizon models.

    # frames property
    frames = [
        Frame(
            ('y'),('x'),
            transition = lambda x: x^2
        )
    ]

    def __init__(self, **kwds):

        ## set up relationships between frames
        for frame in self.frames:
            frame.children = []
            frame.parents = []

        for frame in self.frames:
            if frame.scope is not None:
                for var in frame.scope:
                    scope_frames = [frm for frm in self.frames if var in frm.target]

                    for scope_frame in scope_frames:
                        if self.frames.index(frame) > self.frames.index(scope_frame):
                            if frame not in scope_frame.children:
                                scope_frame.children.append(frame)

                            if scope_frame not in frame.parents:
                                frame.parents.append(scope_frame)
                        else:
                            ffr = ForwardFrameReference(frame)
                            bfr = BackwardFrameReference(frame)

                            # ignoring equivalence checks for now
                            scope_frame.children.append(ffr)
                            frame.parents.append(bfr)

        # Initialize a basic AgentType
        #AgentType.__init__(
        #    self,
        #    **kwds
        #)

    def initialize_sim(self):

        for frame in self.frames:
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
        for frame in self.frames:
            for var in frame.target:
                if var in self.state_now:
                    self.state_prev[var] = self.state_now[var]
                
                    if not frame.aggregate:
                        self.state_now[var] = np.empty(self.AgentCount)
                    else:
                        self.state_now[var] = np.empty(1)

        # transition the variables in the frame
        for frame in self.frames:
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

        for frame in self.frames:
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
        for pre_frame in self.frames[:self.frames.index(frame)]:
            for var in pre_frame.target:
                if var in self.state_now:
                    context.update({var : self.state_now[var]})

        context.update(self.parameters)

        # The "most recently" computed value of the variable is used.
        # This could be the value from the 'previous' time step.

        # limit context to scope of frame
        local_context = {
            var : context[var]
            for var
            in frame.scope
        } if frame.scope is not None else context.copy()

        if frame.transition is not None:
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
                    self,
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

class ForwardFrameReference():
    """
    A 'reference' to a frame that is in the next period
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