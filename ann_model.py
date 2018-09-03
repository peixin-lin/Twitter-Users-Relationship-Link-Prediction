import tensorflow as tf
import numpy as np


# create input function
def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    return dataset.shuffle(1000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
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


def main():
    # load the training data
    with np.load('features_positive.npz') as fp:
        HAA = fp['HAA']
        HJC = fp['HJC']
        HRA = fp['HRA']

    train_features = {'HAA': np.array(HAA[1000:2000]),
                      'HJC': np.array(HJC[1000:2000]),
                      'HRA': np.array(HRA[1000:2000])}

    train_labels = np.array([1 for x in range(len(HAA[1000:2000]))])

    # load the test data
    test_features = {'HAA': np.array(HAA[0:999]),
                     'HJC': np.array(HJC[0:999]),
                     'HRA': np.array(HRA[0:999])}
    test_labels = np.array([1 for x in range(len(HAA[0:999]))])

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

    eval_result = estimator.evaluate(
        input_fn=lambda: eval_input_fn(test_features, test_labels, 100))

    print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Making Predictions
    expected = [0, 1]
    predict_x = {
        'HAA': np.array(HAA[0:999]),
        'HJC': np.array(HJC[0:999]),
        'HRA': np.array(HRA[0:999])
    }

    predictions = estimator.predict(input_fn=lambda: eval_input_fn(predict_x, labels=sNone, batch_size=100))
    template = ('\nPrediction is "{}" ({:.1f}%), expected "{}"')

    for pred_dict, expec in zip(predictions, expected):
        class_id = pred_dict['class_ids'][0]
        probability = pred_dict['probabilities'][class_id]

        print(template.format(expected[class_id],
                              100 * probability, expec))

if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.app.run(main)
    main()
