# Sentiment Analysis task study on IMDB Reviews dataset

<div align="left">
  <img src="https://img.shields.io/badge/Python-FFD43B?style=for-the-badge&logo=python&logoColor=blue" />
  <img src="https://img.shields.io/badge/Jupyter-F37626.svg?&style=for-the-badge&logo=Jupyter&logoColor=white" />
<img src="https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=Keras&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit_learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=TensorFlow&logoColor=white" />
<div/>

The IMDB dataset is a comprehensive collection of data related to movies, TV shows, and other entertainment media, curated by the Internet Movie Database (IMDb). It includes a wide range of information about these media, such as titles (names of movies, TV shows, and episodes), ratings, summaries, etc.

The IMDb dataset is often used in data analysis and machine learning projects due to its rich and detailed information. Researchers and developers use it for tasks, like **Sentiment Analysis**, that is **analyzing user reviews to determine the overall sentiment**. This is exactly what we are going to do in this project using neural approaches, exploiting the **LSTM neural network**.

[**An LSTM (Long Short-Term Memory)** neural network](https://en.wikipedia.org/wiki/Long_short-term_memory) is a type of [**recurrent neural network (RNN)**](https://en.wikipedia.org/wiki/Recurrent_neural_network) designed to better capture and utilize information over long sequences of data, which can span hundreds or thousands of time steps. LSTMs are particularly effective for tasks where the context of the data over time is crucial, such as language modeling, time series forecasting, and sequence classification.

<div align="center">
<img src="https://github.com/iamluirio/lstm-sentiment-analysis/assets/118205581/e917b27e-744b-4dba-88f4-a312caf7ab0f" />
</div>
<div style="margin-bottom: 20px;">‎ </div>

Aim of the project is:
- **Dataset**
  - Exploration and preparation of the dataset to better understand its structure and label distributions.
  - Split of the dataset into training sets and test sets to evaluate the model performance.
  - Data preprocessing: cleaning reviews by removing punctuation, stop words, and other unwanted characters. Tokenizing reviews by breaking them down into  individual words or subsequences of words (e.g., n-grams).

- **LSTM Model Creation**
  - Definition of the LSTM model architecture.
  - Configuration of the model input to accept variable-length sequences (tokenized reviews).
  - Set of the model output as a single unit with an activation function appropriate for binary classification (for example, the sigmoid activation function).

- **Model Training**
  - Model inputs and outputs preparation using the training set.
  - Defininition of a loss function suitable for binary classification.
  - Optimizer usage (Adam or RMSprop) to update the model weights during training.
  - Model training for a defined number of epochs, monitoring evaluation metrics such as accuracy.
 
- **Model Evaluation**
  - Evaluation of model performance using test set.
  - Calculation of metrics such as accuracy, precision, recall and F1-score to evaluate model performance on Sentiment Analysis.

- **Model optimization**
  - Exploration of different model variations, such as increasing the number of layers, using regularization techniques (dropout), or using more advanced neural network structures.
  - Model hyperparameters modification, such as learning rate or word embedding size, to optimize performance.
  - Cross-validation techniques esage or best-fit hyperparameter search to find the best model configuration.

## Dataset
IMDB dataset with 50,000 movie reviews for NLP and text analytics. This is a binary sentiment classification dataset, containing substantially more data than the previous benchmark datasets. A set of 25,000 highly polar movie reviews is provided for training and 25,000 for testing. Goal: Predict the number of positive and negative reviews using classification and/or deep learning algorithms.

```python
imdb_data = pd.read_csv('IMDB Dataset.csv')
print(imdb_data.shape)
```

| Review                                                                                                                                                                  | Sentiment |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------|
| One of the other reviewers has mentioned that after watching just 1 Oz episode you'll be hooked. The...                                                                  | Positive  |
| A wonderful little production. <br /><br />The filming technique is very unassuming- very old-time-B...                                                                  | Positive  |
| I thought this was a wonderful way to spend time on a too hot summer weekend, sitting in the air con...                                                                  | Positive  |
| Basically there's a family where a little boy (Jake) thinks there's a zombie in his closet & his par...                                                                  | Negative  |

## Data Preprocessing
Removing special characters, non-alphabetic characters, html tags, and stop words.

The function returns:
- **x_data**: represents the inputs (the reviews).
- **y_data**: represents the outputs (sentiments).

The second and third lines extract the "review" and "sentiment" columns from the "df" DataFrame and assign them to the "x_data" and "y_data" variables respectively. So, "x_data" represents the reviews and "y_data" represents the sentiments associated with each review.

- *y_data = y_data.replace('positive', 1)*: replaces the "positive" values ​​in the "sentiment" column with the numeric value 1.
- *y_data = y_data.replace('negative', 0)*: replaces the "negative" values ​​in the "sentiment" column with the numeric value 0.

```python
def load_dataset():
    df = pd.read_csv('IMDB Dataset.csv')
    x_data = df['review']       # recensioni: input
    y_data = df['sentiment']    # sentimenti: output

    x_data = x_data.replace({'<.*?>': ''}, regex = True)                                         # remove html tag
    x_data = x_data.replace({'[^A-Za-z]': ' '}, regex = True)                                    # remove non alphabet
    x_data = x_data.apply(lambda review: [w for w in review.split() if w not in english_stops])  # remove stop words
    x_data = x_data.apply(lambda review: [w.lower() for w in review])                            # lower case
    
    y_data = y_data.replace('positive', 1)
    y_data = y_data.replace('negative', 0)

    return x_data, y_data
```

## Dataset Split
We split the dataset into:
1. a **Training Set** of 32,000 samples. Used to tune model parameters during training

2. a **Validation Set** of 8,000 samples. Used to tune hyperparameters and monitor model performance during training

3. a **Test Set** of 10,000 samples. Used to evaluate the model's generalization ability and final performance.

To do this, we use the *train_test_split* method of Scikit-learn. Using this method, the entire dataset is **shuffled**. In the original dataset, the reviews and sentiments are in order (first positive sentiments, then negative sentiments). This way, by shuffling the data, they will be distributed equally between the two datasets.

The size of the test set is specified via the parameter test_size=0.2, which indicates that 20% of the dataset will be used for testing, while 80% will be used for training. The training data is assigned to the variables x_train and y_train, while the test data is assigned to the variables x_test and y_test.

```python
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size = 0.2, random_state=42)

# Split the original training set (x_train, y_train) into training and validation sets
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Print the shapes of the new sets
print('Train Set')
print(x_train.shape, '\n')
print('Validation Set')
print(x_val.shape, '\n')
print('Test Set')
print(x_test.shape, '\n')
```

```
Train Set
(32000,) 

Validation Set
(8000,) 

Test Set
(10000,)
```

## Word encoding and Word Padding/Truncating
A neural network only accepts **numeric data**, so it needs to encode the reviews. We use **tensorflow.keras.preprocessing.text.Tokenizer** to encode the reviews into **integers**, where each unique word is automatically indexed (using the fit_on_texts method) based on x_train.
x_train and x_test sono codificati in integers utilizzando il metodo *texts_to_sequences*.

- post: padding or truncation of words at the end of a sentence.
- pre: padding or truncation of words in front of a sentence.

```python
token = Tokenizer(lower=False)    # lower non è necessario, poichè abbiamo già reso le parole in lower case
token.fit_on_texts(x_train)
x_train = token.texts_to_sequences(x_train)
x_test = token.texts_to_sequences(x_test)
x_val = token.texts_to_sequences(x_val)

max_length = get_max_length()

x_train = pad_sequences(x_train, maxlen=max_length, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=max_length, padding='post', truncating='post')
x_val = pad_sequences(x_val, maxlen=max_length, padding='post', truncating='post')

total_words = len(token.word_index) + 1   # add 1 because of 0 padding

print('Encoded X Train\n', x_train, '\n')
print('Encoded X Test\n', x_test, '\n')
print('Encoded X Val\n', x_val, '\n')
print('Maximum review length: ', max_length)
```

```
Encoded X Train
 [[  422  2066    67 ...     0     0     0]
 [    2    23    65 ...     0     0     0]
 [ 1091  5017  2544 ... 29385   145   487]
 ...
 [    1   118     3 ...     0     0     0]
 [ 3000  5780   154 ...     0     0     0]
 [  493     9   234 ...     0     0     0]] 

Encoded X Test
 [[    1    15   334 ...     0     0     0]
 [  153    38   585 ...   840  2532   265]
 [    2     4   834 ...     0     0     0]
 ...
 [  331   591 24922 ...     0     0     0]
 [  107   131    35 ...     0     0     0]
 [    1   118  7783 ...     0     0     0]] 

Encoded X Val
 [[   39  1983   893 ...     0     0     0]
 [    2  2802  1508 ...   245 18566    13]
 [    2  8007 25097 ...   531 68617   217]
 ...
 [  504  1660   129 ...     0     0     0]
 [   11   761   748 ...  1298     1   334]
 [  106   516   694 ...     1   160    12]] 

Maximum review length:  130
```

## LSTM Neural Network Model
**LSTM Layer**: Makes decisions to keep or discard data, considering the current input, the previous output, and the previous memory (the **hidden state** that is part of the LSTM neural network architecture). In particular, the architecture contains:

- **Forget Gate**: Decides what information to keep or discard.
- **Input Gate**: Updates the state of the cells, processing the previous output, and the current input through the activation function.
- **Cell state**: Computes the new state of the cell, multiplied by the forget vector (discarded if the value is multiplied by 0), and added together with the output of the input gate to update the state.
- **Output gate**: Decides the next hidden state, and used for predictions.
- **Dense layer**: Computes the input with the weight matrix and the bias (optional) and using an activation function.

We build different model based on **different learning rates, epochs, batch sizes, different activation functions, different number of neurons and embedding space**:

| Parameter                | Values                                    |
|--------------------------|-------------------------------------------|
| Learning Rates           | 0.001, 0.01, 0.1                          |
| Epochs                   | 5, 10                                     |
| Batch Sizes              | 64, 128                                   |
| Activation Functions     | sigmoid, relu, tanh                       |
| Neurons in Output Layer  | 64, 128                                   |
| Vector Space Dimension   | 32, 64                                    |

**We built a total of 144 models for each architecture, for a total of 576 models**. By running **METTI NOME FILE**, all the models will be produced: this takes a long time, even using the gpu instead of the cpu. I personally, using an NVIDIA graphics card, even with just the first model it took around an hour and a half.

Below I show you an example of model construction, without reporting them all so that you can replicate the example quickly, choosing the parameters as input.

### First Model: Default LSTM Layer
We use nested loops to iterate over all combinations of the defined hyperparameters. 

For each combination of hyperparameters, we constructs a new Sequential LSTM model with the current set of hyperparameters. We compile the model using the Adam optimizer and binary cross-entropy loss, and we define a model checkpoint callback to save the best model based on validation accuracy.

```python
# Define hyperparameters
EMBED_DIM = [32, 64]
LSTM_OUT = [64, 128]

learning_rates = [0.001, 0.01, 0.1]
epochs = [5, 10]
batch_sizes = [64, 128]
activation_functions = ['sigmoid', 'relu', 'tanh']
```

```python
# Create the model
model = Sequential()
model.add(Embedding(total_words, embed_dim, input_length=max_length))
model.add(LSTM(lstm_out))
model.add(Dense(1, activation=activation))
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
```

Then, we proceed training the model on the training data, validating on the validation data, and evaluating the trained model on the test set to get the loss and accuracy. 

```python
# Fit the model with the specified hyperparameters
model.fit(x_train, y_train, batch_size=batch_size, epochs=num_epochs, callbacks=[checkpoint], validation_data=(x_val, y_val))

# Increment the model counter
model_counter += 1 

# Evaluate the model on the test set
loss, accuracy = model.evaluate(x_test, y_test)
```

The final step is the evaluation: we evaluate the model’s performance on the test set predicting labels for the test set and calculates F1-score and recall. We compare the current model's performance with previously recorded best metrics and updates the best model if the current model performs better.

```python
# Check if the current model has better accuracy than the previous best accuracy model
if accuracy > best_accuracy:
  best_accuracy = accuracy
  best_accuracy_model_name = model_name

# Calculate F1-score and recall for the current model
predictions = model.predict(x_test)
predicted_labels = [1 if prediction >= 0.5 else 0 for prediction in predictions]
f1 = f1_score(y_test, predicted_labels)
recall = recall_score(y_test, predicted_labels)

# Check if the current model has better F1-score than the previous best F1-score model
if f1 > best_f1_score:
  best_f1_score = f1
  best_f1_score_model_name = model_name

# Check if the current model has better recall than the previous best recall model
if recall > best_recall:
  best_recall = recall
  best_recall_model_name = model_name

# Check if the current model has better loss than the previous best loss model
if loss < best_loss:
  best_loss = loss
  best_loss_model_name = model_name
```

After iterating through all hyperparameter combinations, we finally print the best models based on accuracy, F1-score, recall, and loss.

It is possible to choose a certain parameter setting without searching for the ideal parameters in a loop; this saves a lot of time.

In addition to the default model with LSTM layer, we also build three different models explained below.

### [**the Bidirectional Layer LSTM**](https://medium.com/@anishnama20/understanding-bidirectional-lstm-for-sequential-data-processing-b83d6283befc)
It consists of two LSTM layers: one processes the input sequence in the forward direction, and the other processes it in the backward direction. This allows the network to capture information from both past and future states, providing a more comprehensive understanding of the sequence.

```python
# Create the model
model = Sequential()
model.add(Embedding(total_words, embed_dim, input_length=max_length))
model.add(Bidirectional(LSTM(lstm_out)))
model.add(Dense(1, activation=activation))
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
```
                        
### [**the Dropout Layer**](https://machinelearningmastery.com/use-dropout-lstm-networks-time-series-forecasting/) 
To prevent overfitting and improve generalization. When used with LSTM networks, the Dropout layer randomly sets a fraction of the input units to zero during each update cycle while training the model, which helps the network to learn more robust features and prevents the model from relying too heavily on specific neurons.

```python
# Create the model
model = Sequential()
model.add(Embedding(total_words, embed_dim, input_length=max_length))
model.add(LSTM(lstm_out))
model.add(Dropout(0.2)) # Dropout layer con un dropout rate di 0.2
model.add(Dense(1, activation=activation))
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
```

### [**Multi-layer or Stacked LSTM**](https://machinelearningmastery.com/stacked-long-short-term-memory-networks/)
Using multiple layers of Long Short-Term Memory (LSTM) units, known as a multi-layer or stacked LSTM, enhances the learning capability of the network.

```python
# Create the model
model = Sequential()
model.add(Embedding(total_words, embed_dim, input_length=max_length))
model.add(LSTM(lstm_out, return_sequences=True))
model.add(LSTM(lstm_out))
model.add(Dropout(0.2)) # Dropout layer con un dropout rate di 0.2
model.add(Dense(1, activation=activation))
model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=lr), loss='binary_crossentropy', metrics=['accuracy'])
```

## Plotting Results and Evaluation Metrics
Let's now compare the best models for each architecture. As an evaluation metric, we rely on **accuracy**, but we can decide to use other measures as well.

| Model Name | Accuracy | F1-Score  | Recall   | Loss     |
|------------|----------|-----------|----------|----------|
| Model 1    | 0.8747   | 0.880973  | 0.920222 | 0.333019 |
| Model 2    | 0.8691   | 0.867925  | 0.853542 | 0.710201 |
| Model 3    | 0.8697   | 0.871991  | 0.880730 | 0.333225 |
| Model 4    | 0.8675   | 0.874396  | 0.915261 | 0.543763 |

<div style="margin-bottom: 20px;">‎ </div>
<div align="center">
   <img src="https://github.com/iamluirio/lstm-sentiment-analysis/assets/118205581/62589a3c-a5c3-4e41-a60a-f4b2fa532baf" />
</div>

### ROC Curve
A [**Receiver Operating Characteristic (ROC) curve**](https://it.wikipedia.org/wiki/Receiver_operating_characteristic) is a graphical representation of the performance of a binary classification model. It illustrates the trade-off between the true positive rate (TPR) and the false positive rate (FPR) across different threshold values. The **area Under the Curve (AUC)** is a single scalar value summarizing the performance of the model. An AUC of 1 indicates a perfect model, while an AUC of 0.5 indicates a model no better than random guessing.
<div style="margin-bottom: 20px;">‎ </div>
<div align="center">
  <img src="https://github.com/iamluirio/lstm-sentiment-analysis/assets/118205581/953548bb-5619-4d69-a5c9-92f42b2d3df5" />
</div>

### Confusion Matrix
A *[*confusion matrix**](https://en.wikipedia.org/wiki/Confusion_matrix) is a performance measurement tool for classification models, particularly useful for binary and multiclass classification problems. It provides a tabular summary of the predictions made by a model compared to the actual outcomes, allowing for a more detailed analysis of how well the model is performing.

For a binary classification problem, the confusion matrix is a 2x2 table with the following structure:

|               | Predicted Positive | Predicted Negative |
|---------------|--------------------|--------------------|
| Actual Positive | True Positive (TP) | False Negative (FN) |
| Actual Negative | False Positive (FP) | True Negative (TN) |

<div style="margin-bottom: 20px;">‎ </div>
<div align="center">
  <img src="https://github.com/iamluirio/lstm-sentiment-analysis/assets/118205581/8e069ac2-1f3f-4a25-97f1-9f3227488042" />
</div>
<div style="margin-bottom: 20px;">‎ </div>












