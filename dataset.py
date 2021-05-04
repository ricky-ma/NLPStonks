import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

N = 1989


def get_dataset(n_comp=2):
    dija_df = pd.read_csv("data/DJIA.csv")
    train = pd.DataFrame(data={'DIJA': dija_df["Adj Close"][0:int(N * 0.7)]})
    val = pd.DataFrame(data={'DIJA': dija_df["Adj Close"][int(N * 0.7):int(N * 0.9)]})
    test = pd.DataFrame(data={'DIJA': dija_df["Adj Close"][int(N * 0.9):]})
    for df in [train, val, test]:
        df['Lag-1'] = df['DIJA'].shift(1)
        df['Lag-2'] = df['DIJA'].shift(2)
        df['Lag-3'] = df['DIJA'].shift(3)
        df['Lag-4'] = df['DIJA'].shift(4)

    train_mean = train.mean()
    train_std = train.std()
    train = (train - train_mean) / train_std
    val = (val - train_mean) / train_std
    test = (test - train_mean) / train_std

    for i, filename in enumerate(os.listdir('data/embeddings')):
        with open('data/embeddings/{}'.format(filename), 'rb') as f:
            embeddings = np.load(f)
        pca = PCA(n_components=n_comp)
        embeddings = pca.fit_transform(X=embeddings)
        for c in range(n_comp):
            column = 'Top{}_{}'.format(i+1, c+1)
            train[column] = embeddings[:, c][0:int(N * 0.7)]
            val[column] = embeddings[:, c][int(N * 0.7):int(N * 0.9)]
            test[column] = embeddings[:, c][int(N * 0.9):]

    return train.dropna(), val.dropna(), test.dropna()


train_df, val_df, test_df = get_dataset(n_comp=20)


class WindowGenerator:
    def __init__(self, input_width, label_width, shift,
                 train_df=train_df, val_df=val_df, test_df=test_df, label_columns=None):
        # Store the raw data.
        self.train_df = train_df
        self.val_df = val_df
        self.test_df = test_df

        # Work out the label column indices.
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in
                                          enumerate(label_columns)}
        self.column_indices = {name: i for i, name in
                               enumerate(train_df.columns)}

        # Work out the window parameters.
        self.input_width = input_width
        self.label_width = label_width
        self.shift = shift

        self.total_window_size = input_width + shift

        self.input_slice = slice(0, input_width)
        self.input_indices = np.arange(self.total_window_size)[self.input_slice]

        self.label_start = self.total_window_size - self.label_width
        self.labels_slice = slice(self.label_start, None)
        self.label_indices = np.arange(self.total_window_size)[self.labels_slice]

    def __repr__(self):
        return '\n'.join([
            f'Total window size: {self.total_window_size}',
            f'Input indices: {self.input_indices}',
            f'Label indices: {self.label_indices}',
            f'Label column name(s): {self.label_columns}'])

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
                [labels[:, :, self.column_indices[name]] for name in self.label_columns],
                axis=-1)

        # Slicing doesn't preserve static shape information, so set the shapes
        # manually. This way the `tf.data.Datasets` are easier to inspect.
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])
        return inputs, labels

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        ds = tf.keras.preprocessing.timeseries_dataset_from_array(
            data=data,
            targets=None,
            sequence_length=self.total_window_size,
            sequence_stride=1,
            shuffle=True,
            batch_size=32)
        ds = ds.map(self.split_window)
        return ds

    @property
    def train(self):
        return self.make_dataset(self.train_df)

    @property
    def val(self):
        return self.make_dataset(self.val_df)

    @property
    def test(self):
        return self.make_dataset(self.test_df)

    @property
    def example(self):
        """Get and cache an example batch of `inputs, labels` for plotting."""
        result = getattr(self, '_example', None)
        if result is None:
            # No example batch was found, so get one from the `.train` dataset
            result = next(iter(self.test))
            # And cache it for next time
            self._example = result
        return result

    def binary_accuracy(self, model, ds):
        m = tf.keras.metrics.BinaryAccuracy()
        pred_n = []
        labels_n = []

        if ds == 'train':
            num_iter = int(len(self.train_df.index) / 32)
        elif ds == 'val':
            num_iter = int(len(self.val_df.index) / 32)
        else:
            num_iter = int(len(self.test_df.index) / 32)

        for i in range(num_iter):
            if ds == 'train':
                inputs, labels = next(iter(self.train))
            elif ds == 'val':
                inputs, labels = next(iter(self.val))
            else:
                inputs, labels = next(iter(self.test))
            predictions = model(inputs)
            for n in range(len(inputs)):
                if self.label_columns:
                    label_col_index = self.label_columns_indices.get('DIJA', None)
                else:
                    label_col_index = self.column_indices['DIJA']
                if label_col_index is None:
                    continue
                pred_n = pred_n + predictions[n, :, label_col_index].numpy().tolist()
                labels_n = labels_n + labels[n, :, label_col_index].numpy().tolist()

        delta_true = np.diff(labels_n)
        change_true = [1 if delta_true[i] >= delta_true[i - 1] else 0 for i in np.arange(1, len(delta_true))]
        delta_pred = np.diff(pred_n)
        change_pred = [1 if delta_pred[i] >= delta_pred[i - 1] else 0 for i in np.arange(1, len(delta_pred))]
        m.update_state(change_true, change_pred)
        return m.result().numpy()

    def plot(self, name, model=None, plot_col='DIJA', max_subplots=3, acc=None):
        inputs, labels = self.example
        fig = plt.figure(figsize=(12, 8))
        if acc is not None:
            name = "{}: test accuracy = {}".format(name, acc)
        fig.suptitle(name, fontsize=16)
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plt.subplot(max_n, 1, n + 1)
            plt.ylabel(f'{plot_col} [normed]')
            plt.plot(self.input_indices, inputs[n, :, plot_col_index],
                     label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index
            if label_col_index is None:
                continue

            plt.scatter(self.label_indices, labels[n, :, label_col_index],
                        edgecolors='k', label='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model(inputs)
                plt.scatter(self.label_indices, predictions[n, :, label_col_index],
                            marker='X', edgecolors='k', label='Predictions',
                            c='#ff7f0e', s=64)

            if n == 0:
                plt.legend()
        plt.xlabel('Time [day]')
        plt.show()


if __name__ == "__main__":
    train_df, val_df, test_df = get_dataset(n_comp=3)
    print(train_df)
    print(val_df)
    print(test_df)
    w1 = WindowGenerator(input_width=24, label_width=1, shift=1,
                         train_df=train_df, val_df=val_df, test_df=test_df,
                         label_columns=['DIJA'])
    print(w1)

    # Stack three slices, the length of the total window:
    example_window = tf.stack([np.array(train_df[:w1.total_window_size]),
                               np.array(train_df[100:100+w1.total_window_size]),
                               np.array(train_df[200:200+w1.total_window_size])])

    example_inputs, example_labels = w1.split_window(example_window)
    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'labels shape: {example_labels.shape}')
    w1.plot(name='Example')
