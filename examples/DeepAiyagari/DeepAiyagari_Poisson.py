# Import TensorFlow and NumPy
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# Aiyagari problem parameters
gamma = 2 # CRRA Utility Parameter
r = 0.03 # Stationary Interest Rate 
rho = 0.05 # Discount Rate


z1 = .1 
z2 = .2
z = np.array([z1, z2]) # Income State
la1 = 0.02
la2 = 0.03
la = np.array([la1, la2]) # Poisson Intensity of Income State

eps = tf.constant(1e-10, dtype=tf.float32) # Small Constant

X_low = np.array([-0.02])       # wealth lower bound
X_high = np.array([2])          # wealth upper bound



def u(c):
    return c**(1-gamma)/(1-gamma)


def u_deriv(c):
    return c**(-gamma)


def u_deriv_inv(c):
    return c**(-1/gamma)

# Define model architecture
class DCGMNet(tf.keras.Model):
    """ Set basic architecture of the model."""

    def __init__(self, X_low, X_high,
                 input_dim, output_dim,
                 n_layers_FFNN, layer_width,
                 activation_FFNN,
                 kernel_initializer='glorot_normal',
                 **kwargs):
        super().__init__(**kwargs)
        
        self.X_low = X_low
        self.X_high = X_high
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        self.n_layers_FFNN = n_layers_FFNN
        self.layer_width = layer_width
        
        self.activation_FFNN = activation_FFNN
        # print(activation_FFNN)
        
        # Define NN architecture
        self.initial_scale = tf.keras.layers.Lambda(
            lambda x: 2.0*(x - X_low)/(X_high - X_low) - 1.0)
        self.hidden = [tf.keras.layers.Dense(layer_width,
                                             activation=tf.keras.activations.get(
                                                 activation_FFNN),
                                             kernel_initializer=kernel_initializer)
                       for _ in range(self.n_layers_FFNN)]
        self.out = tf.keras.layers.Dense(output_dim)

    def call(self, X):
        """Forward-pass through neural network."""
        Z = self.initial_scale(X)
        for i in range(self.n_layers_FFNN):
            Z = self.hidden[i](Z) +Z
        return self.out(Z)


# neural network parameters
num_layers_FFNN = 4    # Depth of Neural Network
nodes_per_layer = 50   # Width of Neural Network
starting_learning_rate = 0.001  # Learning Rate of Optimizer
activation_FFNN = 'tanh' # Activation Function in Neural Network
# Training parameters
sampling_stages  = 6000   # number of times to resample new time-space domain points
steps_per_sample = 10    # number of SGD steps to take before re-sampling, avoiding trapp in local minimum 

# Sampling parameters
nSim_interior = 128  # Number of Points sampled in Interior of State Space
nSim_boundary = 1    # Number of Points sampled in Boundary of State Space

dim_input = 1        # Dimensionality of Input, wealth as a single state variable
dim_output = 2       # Dimensionality of output, since we have two Poisson state, each represented by one value function.
 
model = DCGMNet(X_low, X_high,  
                 dim_input, dim_output, 
                 num_layers_FFNN, nodes_per_layer,
                 activation_FFNN)


def sampler(nSim_interior, nSim_boundary):
    ''' Sample space points from the function's domain
    
    Args:
        nSim_interior: number of space points in the interior of the function's domain to sample 
        nSim_terminal: number of space points at boundary to sample 
    ''' 
    
    # Sampler #1: domain interior    

    X_interior = tf.random.uniform(
        shape=[nSim_interior, 1], minval=X_low, maxval=X_high, dtype=tf.float32)


    a_alower = X_low[0] * tf.ones(shape = (nSim_boundary, 1), dtype=tf.float32)
    X_alower = a_alower

    return X_interior, X_alower


def loss_differentialoperator(model, X):
    a = X[:,0:1]

    V = model(tf.stack([a[:,0]], axis=1))
    V_a = tf.concat([tf.gradients(V[:,0], a)[0],tf.gradients(V[:,1], a)[0]],axis=1)
    V_a = tf.math.maximum(eps*tf.ones_like(V), V_a)


    c = u_deriv_inv(V_a)
    u_c = u(c) 
    
    diff_V_z1 = -rho * V[:, 0] + u_c[:, 0] + V_a[:, 0] * (z[0]+r*a[:, 0]-c[:, 0]) + la[0] * (V[:, 1] - V[:, 0])
    diff_V_z2 = -rho * V[:, 1] + u_c[:, 1] + V_a[:,1] * (z[1]+r*a[:,0]-c[:,1]) + la[1]* (V[:,0] - V[:,1])

    diff_V = tf.concat([diff_V_z1,diff_V_z2], axis=0)
        
    L = tf.reduce_mean(tf.square(diff_V))
    return diff_V, L

def loss_differentialoperator_alower(model, X):
    a = X[:,0:1]

    V = model(tf.stack([a[:,0]], axis=1))
    V_a = tf.concat([tf.gradients(V[:,0], a)[0],tf.gradients(V[:,1], a)[0]],axis=1)
    V_a = tf.math.maximum( tf.zeros_like(V), V_a)
    V_a_new = tf.math.maximum(u_deriv(z+ r* a), V_a)

    c_new = u_deriv_inv(V_a_new)
    u_c_new = u(c_new) 
    

    diff_V_z1 = -rho * V[:,0] + u_c_new[:,0] + V_a[:,0]*(z[0]+r*a[:,0]-c_new[:,0]) + la[0]* (V[:,1] - V[:,0])
    diff_V_z2 = -rho * V[:,1] + u_c_new[:,1] + V_a[:,1]*(z[1]+r*a[:,0]-c_new[:,1]) + la[1]* (V[:,0] - V[:,1])

    diff_V = tf.concat([diff_V_z1,diff_V_z2], axis=0)
        

    L = tf.reduce_mean(tf.square(diff_V))
    return diff_V, L

def loss_concave(model, X):
    a = X[:,0:1]

    V = model(tf.stack([a[:,0]], axis=1))
    V_a = tf.concat([tf.gradients(V[:,0], a)[0],tf.gradients(V[:,1], a)[0]],axis=1)
    V_aa = tf.concat([tf.gradients(V_a[:,0], a)[0],tf.gradients(V_a[:,1], a)[0]],axis=1)
    concave_V = tf.maximum(V_aa, tf.zeros_like(V))

    L = tf.reduce_mean( tf.square(concave_V ) )
    return L


def compute_loss(model, X_interior, X_alower):
    ''' Compute total loss for training.
    
    Args:
        model:      DGM model object
        X_interior, X_alower:      Points

    ''' 

    Loss_V_interior, L1 = loss_differentialoperator(model, X_interior)

    Loss_V_alower, L2 = loss_differentialoperator_alower(model, X_alower)
    
    L3 =  loss_concave(model, X_interior)

    L = L1 + L2 + L3 
    
    return L
    
    
def get_grad(model, X_interior, X_alower):
    
    with tf.GradientTape(persistent=True) as tape:

        tape.watch(model.trainable_variables)
        loss = compute_loss(model, X_interior, X_alower)

    grad = tape.gradient(loss, model.trainable_variables)
    del tape
    
    return loss, grad

optimizer = tf.keras.optimizers.Adam(learning_rate=starting_learning_rate)
# optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=starting_learning_rate)




@tf.function
def train_step(X_interior, X_alower):
    # Compute current loss and gradient w.r.t. parameters
    loss, grad_theta = get_grad(model, X_interior, X_alower)

    # Perform gradient descent step
    optimizer.apply_gradients(zip(grad_theta, model.trainable_variables))

    return loss


hist = []

for i in range(sampling_stages):

    # sample uniformly from the required regions

    X_interior, X_alower = sampler(nSim_interior, nSim_boundary)

    for _ in range(steps_per_sample):
        loss = train_step(X_interior, X_alower)
    
    hist.append(loss.numpy())
    
    if i%100==0:
        tf.print("Progress: {}/{}, Loss: {}".format(i,sampling_stages,loss))
    

aspace = np.linspace(X_low, X_high, 500)
# A = np.meshgrid(aspace)
A = aspace
X_interior = np.vstack([A.flatten()]).T

X_alower = np.vstack([A[A==X_low].flatten()]).T


with tf.GradientTape(persistent=True) as tape:
    a = tf.cast(X_interior, dtype=tf.float32)[:,0:1]
    tape.watch(a)
    V = model(tf.stack([a[:, 0]], axis=1))
    Va_1 = tape.gradient(V[:,0], a)
    Va_2 = tape.gradient(V[:, 1], a)
    Va = tf.concat([Va_1,Va_2],axis=1)

fitted_V = V.numpy().reshape(500, 2)
fitted_Va = Va.numpy().reshape(500,2)
fitted_saving = z+r*X_interior - u_deriv_inv(fitted_Va)

fitted_saving[0,:] = np.maximum(fitted_saving[0,:], np.zeros_like(fitted_saving[0,:]))

fig = plt.figure(figsize=(16, 9))
plt.plot(X_interior[:, 0], fitted_V[:, 0], label="High Income")
plt.plot(X_interior[:, 0], fitted_V[:, 1], label="Low Income")
plt.xlabel('$a$')
plt.legend()
plt.grid(linestyle=":")
plt.xlim(X_low, X_high)
plt.title("Value Function")
plt.savefig("./ValueFunction.png")

fig = plt.figure(figsize=(16, 9))
plt.plot(X_interior[:, 0], fitted_Va[:, 0], label="High Income")
plt.plot(X_interior[:, 0], fitted_Va[:, 1], label="Low Income")
plt.xlabel('$a$')
plt.legend()
plt.grid(linestyle=":")
plt.xlim(X_low, X_high)
plt.title("Value Function Derivative")
plt.savefig("./ValueFunctionDerivative.png")

fig = plt.figure(figsize=(16, 9))
plt.plot(X_interior[:, 0], fitted_saving[:, 0], label="High Income")
plt.plot(X_interior[:, 0], fitted_saving[:, 1], label="Low Income")
plt.xlabel('$a$')
plt.legend()
plt.grid(linestyle=":")
plt.xlim(X_low,X_high)
plt.title("Saving")
plt.savefig("./Saving.png")
