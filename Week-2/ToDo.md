# To do list: 24/03/25 - 30/03/25

---
Notes from the 1v1 session: 
Reproduce the flip flop task in python. 
Theorem to look into: Between two attractive attractors, you will always have a repulsive one in between. 
Reproduce it using first SGD then EG

How unstable are the green crosses (saddle points) in each method?
How many dimensions are at play in both methods?
What are the second derivatives of the saddle points?
Do the different methods present different statistics on the speed of shuttling? Maybe
---

Using the following [github](https://github.com/mattgolub/fixed-point-finder) to make the Flip Flop example. 
The requirements.txt file is part of that github. The conda evironment I created to run the code associated with this project is: gatsby-eg. To run the experiment, run: python run_FlipFlop.py in the examples/torch folder. 


I need to understand all the code in the github and make a shorter version (j'aimerais m'approprier le truc un peu déjà), une fois que
ça c'est fait je vais pouvoir regarder le colab de Johnathan pour entrainer le RNN avec EG. Une fois que j'ai ça, je fais une analyse des résultats. 

---
Understanding what the code does: (high level overview)
- Train an RNN on a simple flip-flop (N-bit memory) task. In this task, the RNN’s job is to remember which input bit was most recently “flipped” to 1.
- Generate predictions (hidden states, outputs) on a set of validation trials.
- Find the fixed points of the trained RNN by selecting initial states (with random noise) and using an optimization routine to discover states that do not change under the RNN’s update rules.
- Visualize those fixed points in a 3D principal components (PCA) space, with colored stable vs. unstable points and the network’s hidden state trajectories overlaid.
- Enter a debug mode at the end so that you can interact with the figure and objects in a Python debugger.

What some key functions of the github code do: 
**train_FlipFlop()** : Trains a simple recurrent neural network (RNN) to perform the N-bit memory (flip-flop) task. Returns a trained FlipFlop model and the model’s predictions on validation data. The network sees a 3-bit input (3 separate “flip” channels). n_train is the number of training trials, n_valid is the number of validation trials. 16 hidden units in the RNN, using a simple tanh RNN cell. (There is also a commented note that gru can be used but may not work as expected in this PyTorch example for fixed point finding.) Instantiates a data generator and creates random input-output pairs suitable for the flip-flop task. FlipFlop is a PyTorch module that builds either a tanh or gru-based RNN for the flip-flop task. The training routine updates the RNN’s parameters via gradient descent. (The training loop is implemented inside FlipFlop.train(...).) After training completes, the model is tested on the validation set to get hidden states, outputs, etc. Returns the trained model and the valid_predictions dictionary (containing validation outputs, hidden states, etc.)

**find_fixed_points(model, valid_predictions)** : Finds fixed points of the trained RNN. Plots the discovered fixed points in 3D PCA space alongside the network’s hidden state trajectories. NOISE_SCALE sets the standard deviation of Gaussian noise added to sampled hidden states to create diverse initial states for the optimizer. N_INITS is how many random initial points the optimizer will attempt. fpf_hps is a dict of parameters for the fixed point finding routine, e.g. max iterations, initial learning rate, etc. This creates an object that knows how to do gradient-based search for hidden states that map to themselves under the RNN update. The hidden states from the validation trials are taken and perturbed with noise, providing diverse seeds for the fixed point finder. This means we are seeking fixed points with no new “flip” signals coming in. unique_fps is a list of unique fixed points (pruned of duplicates). all_fps are all discovered fixed points (including duplicates). Uses PCA on the hidden states, then plots the RNN trajectories, stable fixed points (black markers), and unstable fixed points (red markers or lines).

**main()** : Orchestrates the two steps above. Train the RNN via train_FlipFlop(). Find and plot fixed points via find_fixed_points(...). Enter debug mode so you can interact with variables. main() is invoked. Inside main(), train_FlipFlop() is called. Returns (model, valid_predictions) after training. Still inside main(), find_fixed_points(model, valid_predictions) is called. Finds the fixed points and plots them. The code then prints some instructions and calls pdb.set_trace(), letting you inspect variables and the figure in debug mode.
