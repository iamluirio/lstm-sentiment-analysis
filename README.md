# Sentiment Analysis task study on IMDB Reviews dataset

The IMDB dataset is a comprehensive collection of data related to movies, TV shows, and other entertainment media, curated by the Internet Movie Database (IMDb). It includes a wide range of information about these media, such as titles (names of movies, TV shows, and episodes), ratings, summaries, etc.

The IMDb dataset is often used in data analysis and machine learning projects due to its rich and detailed information. Researchers and developers use it for tasks, like **Sentiment Analysis**, that is **analyzing user reviews to determine the overall sentiment**. This is exactly what we are going to do in this project using neural approaches, exploiting the **LSTM neural network**.

[**An LSTM (Long Short-Term Memory)** neural network](https://en.wikipedia.org/wiki/Long_short-term_memory) is a type of [**recurrent neural network (RNN)**](https://en.wikipedia.org/wiki/Recurrent_neural_network) designed to better capture and utilize information over long sequences of data, which can span hundreds or thousands of time steps. LSTMs are particularly effective for tasks where the context of the data over time is crucial, such as language modeling, time series forecasting, and sequence classification.

<div align="center">
<img src="https://github.com/iamluirio/lstm-sentiment-analysis/assets/118205581/e917b27e-744b-4dba-88f4-a312caf7ab0f" />
</div>

Aim of the project is:
- Dataset:
  - Exploration and preparation of the dataset to better understand its structure and label distributions.
  - Split of the dataset into training sets and test sets to evaluate the model performance.
  - Data preprocessing: cleaning reviews by removing punctuation, stop words, and other unwanted characters. Tokenizing reviews by breaking them down into  individual words or subsequences of words (e.g., n-grams).

- LSTM model creation:
  - Definition of the LSTM model architecture.
  - Configuration of the model input to accept variable-length sequences (tokenized reviews).
  - Set of the model output as a single unit with an activation function appropriate for binary classification (for example, the sigmoid activation function).
