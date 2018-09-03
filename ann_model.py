import tensorflow as tf
import numpy as np

# tf.logging.set_verbosity(tf.logging.INFO)


# create input function
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features=dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# load the training data
with np.load('features_positive.npz') as fp:
    HAA_train_positive = fp['HAA']
    HJC_train_positive = fp['HJC']
    HRA_train_positive = fp['HRA']

train_features = {'HAA': np.array(HAA_train_positive[1000:2000]),
                  'HJC': np.array(HJC_train_positive[1000:2000]),
                  'HRA': np.array(HRA_train_positive[1000:2000])}

train_labels = np.array([1 for x in range(len(HAA_train_positive[1000:2000]))])

# load the test data
test_features = {'HAA': np.array(HAA_train_positive[0:999]),
                 'HJC': np.array(HJC_train_positive[0:999]),
                 'HRA': np.array(HRA_train_positive[0:999])}
test_labels = np.array([1 for x in range(len(HAA_train_positive[0:999]))])

# Define the feature column (describe how to use the features)
HAA = tf.feature_column.numeric_column('HAA')
HJC = tf.feature_column.numeric_column('HJC')
HRA = tf.feature_column.numeric_column('HRA')

# Instantiate an estimator(2 hidden layer DNN with 10, 10 units respectively), passing the feature columns.
estimator = tf.estimator.DNNClassifier(
    feature_columns=[HAA, HJC, HRA],
    # Two hidden layers of 10 nodes each.
    hidden_units=[10, 10],
    # The model must choose between 3 classes.
    n_classes=2
)

# Train the Model.
estimator.train(
    input_fn=lambda: train_input_fn(train_features, train_labels, 100),
    steps=1000)

# eval_result = estimator.evaluate(
#     input_fn=lambda: eval_input_fn(test_features, test_labels, 100))
#
# print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Making Predictions
predict_x = {
    'HAA': np.array(HAA[0:3]),
    'HJC': np.array(HJC[0:3]),
    'HRA': np.array(HRA[0:3])
}


# predictions = estimator.predict(
#     input_fn=tf.estimator.inputs.pandas_input_fn(x=predict_x,
#     num_epochs=1,
#     shuffle=False))
predictions = estimator.predict(input_fn=lambda:eval_input_fn(test_features, None, 100))
for p in predictions:
    class_id = p['class_ids'][0]
    probability = p['probabilities'][class_id]
    print(class_id, probability)

