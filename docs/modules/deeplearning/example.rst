Getting started
===============


Linear perceptron
-----------------

You can replicate a linear regression (`OLS <../econometrics/ols/ols.html>`_) with a MLP with no hidden layer and a linear output activation.

.. code-block:: python

    from statinf.data import generate_dataset
    from statinf.ml import MLP, Layer

    # Generate the synthetic dataset
    data = generate_dataset(coeffs=[1.2556, -6.465, 1.665414, 1.5444], n=1000, std_dev=1.6)
    new_df = generate_dataset(coeffs=[1.2556, -6.465, 1.665414, 1.5444], n=500, std_dev=1.6)

    Y = ['Y']
    X = [c for c in data.columns if c not in Y]

    # Initialize the network and its architecture
    nn = MLP(loss='mse')
    nn.add(Layer(4, 1, activation='linear'))

    # Train the neural network
    nn.train(data=data, X=X, Y=Y, epochs=1, learning_rate=0.001)

    # Predict new output
    pred = nn.predict(new_data=new_df)

    # Get the network's parameters
    print(nn.get_weights())


The output will be:

.. code-block:: python

    {'weights 0': array([[ 1.32005564],
       [-6.38121934],
       [ 1.64515704],
       [ 1.48571785]]), 'bias 0': array([0.81190412])}




Non-linear model
----------------

You can fit complex data structures by stacking several layers.
Keep in mind that the input dimension of on layer corresponds to the output size of the previous
(except for the first layer, having the number of features as input dimension).

The below example shows how to fit a complex binary classification problem.

.. code-block:: python

    from statinf.data import generate_dataset
    from statinf.ml import MLP, Layer

    # Generate the synthetic dataset
    data = generate_dataset(coeffs=[1.2556, -6.465, 1.665414, 1.5444], n=1000, std_dev=1.6)
    new_df = generate_dataset(coeffs=[1.2556, -6.465, 1.665414, 1.5444], n=500, std_dev=1.6)

    data['Y'] = data['Y'].map(lambda x: 1. if x > 0. else 0.)

    Y = ['Y']
    X = [c for c in data.columns if c not in Y]

    # Initialize the network and its architecture
    nn = MLP(loss='binary_cross_entropy')
    nn.add(Layer(4, 8, activation='relu'))
    nn.add(Layer(8, 2, activation='relu'))
    nn.add(Layer(2, 1, activation='sigmoid'))

    # Train the neural network
    nn.train(data=data, X=X, Y=Y, epochs=100, learning_rate=0.001, optimizer='adam')

    # Predict new output
    pred = nn.predict(new_data=new_df)



Run STATINF with GPU
--------------------

See `theano documentation <http://deeplearning.net/software/theano/tutorial/using_gpu.html>`_ for more details and examples. 