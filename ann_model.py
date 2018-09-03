import tensorflow as tf
import numpy as np
import input_data

NUM_FEATURE = 3

with np.load('features_positive_10') as fp:
    HAA = fp['HAA']
    HJC = fp['HJC']
    HRA = fp['HRA']

# create input function
def input_evaluation_set():
    features = {}
    labels = np.array([0, 1])
    return features, labels

def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)

def test_input_fn():
    pass

# Define the feature columnn
HAA = tf.feature_column.numeric_column('HAA')
HJC = tf.feature_column.numeric_column('HJC')
HRA = tf.feature_column.numeric_column('HRA')

# Tensorflow placeholder
X = tf.placeholder(dtype=tf.float32, shape=[None, NUM_FEATURE], name="features")
y = tf.placeholder(dtype=tf.uint8, shape=[None,], name="labels")

# Instantiate an estimator, passing the feature columns.
estimator = tf.estimator.DNNClassifier(
    feature_columns=[HAA, HJC, HRA],
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=2
)

# Train the Model.
features, labels = input_evaluation_set()

estimator.train(
    input_fn=lambda: train_input_fn(features, labels, 2000),
    steps=2000)






