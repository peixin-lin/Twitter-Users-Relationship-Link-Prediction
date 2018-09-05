import tensorflow as tf
import numpy as np
import pandas as pd


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

    # Convert the inputs to a dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset

def merge_data(data_1, data_2):
    data = []
    for i in range(min(len(data_1), len(data_2))):
        data.append(data_1[i])
        data.append(data_2[i])

    if len(data_1)>len(data_2):
        data.append(data_1[i+1:])
    else:
        data+=data_2[i+1:]
    return data


# load the training data
with np.load('new_positive_500k.npz') as fp:
    HAA_train_positive = fp['HAA']
    HJC_train_positive = fp['HJC']
    HRA_train_positive = fp['HRA']
    SD_train_positive = fp['SD']

with np.load('new_negative_500k.npz') as fn:
    HAA_train_negative = fn['HAA']
    HJC_train_negative = fn['HJC']
    HRA_train_negative = fn['HRA']
    SD_train_negative = fn['SD']

print('the size of positive set: ', len(HAA_train_positive))
print('the size of negative set: ', len(HAA_train_negative))

feature_HAA = merge_data(list(HAA_train_positive), list(HAA_train_negative))
feature_HJC = merge_data(list(HJC_train_positive), list(HJC_train_negative))
feature_HRA = merge_data(list(HRA_train_positive), list(HJC_train_negative))
feature_SD = merge_data(list(SD_train_positive), list(SD_train_negative))

train_features = {'HAA': np.array(feature_HAA[2000:10000000]),
                  'HJC': np.array(feature_HJC[2000:10000000]),
                  'HRA': np.array(feature_HRA[2000:10000000]),
                  'SD': np.array(feature_SD[2000:10000000])}

print("the size of training set",len(feature_HAA))

train_labels = merge_data([1 for x in range(len(HAA_train_positive))],
                          [0 for x in range(len(HAA_train_negative))])[2000:10000000]
print('the size of the label set', len(train_labels))


# load the test data
with np.load('new_test.npz') as tft:
    HAA_test = tft['HAA']
    HJC_test = tft['HJC']
    HRA_test = tft['HRA']
    SD_test = tft['SD']

test_features = {'HAA': np.array(HAA_test),
                 'HJC': np.array(HJC_test),
                 'HRA': np.array(HRA_test),
                 'SD': np.array(SD_test)}
print('the size of test set', len(HAA_test))


# load the eval data
eval_features = {'HAA': np.array(feature_HAA[:2000]),
                 'HJC': np.array(feature_HJC[:2000]),
                 'HRA': np.array(feature_HRA[:2000]),
                 'SD': np.array(feature_SD[:2000])}
eval_labels = merge_data([1 for x in range(len(HAA_train_positive))],
                         [0 for x in range(len(HAA_train_negative))])[:2000]

print('@@@@@@')
print(np.var(feature_HAA))
print(np.var(feature_HJC))
print(np.var(feature_HRA))
print(np.var(feature_SD))


# Define the feature column (describe how to use the features)
HAA = tf.feature_column.numeric_column('HAA')
HJC = tf.feature_column.numeric_column('HJC')
HRA = tf.feature_column.numeric_column('HRA')
SD = tf.feature_column.numeric_column('SD')

# Instantiate an estimator(2 hidden layer DNN with 10, 10 units respectively), passing the feature columns.
estimator = tf.estimator.DNNClassifier(
    feature_columns=[HAA, HJC, HRA, SD],
    # Two hidden layers of 10 nodes each.
    hidden_units=[16,32,16],
    # The model must choose between 3 classes.
    n_classes=2
)

# Train the Model.
estimator.train(
    input_fn=lambda: train_input_fn(train_features, train_labels, 128),
    steps=500)

eval_result = estimator.evaluate(
    input_fn=lambda: eval_input_fn(eval_features, eval_labels, 128))

print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

# Making Predictions
predictions = estimator.predict(input_fn=lambda:eval_input_fn(test_features, None, 128))
i = 0
id_list = []
prediction_list = []
print('id ', 'probability')
for p in predictions:
    i += 1
    class_id = p['class_ids'][0]
    probability = p['probabilities'][1]
    id_list.append(i)
    prediction_list.append(probability)
    print(i, " " , probability)

dataframe = pd.DataFrame({'Id':id_list,'Prediction':prediction_list})
dataframe.to_csv("prediction.csv",index=True,sep=',')


