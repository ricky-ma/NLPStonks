import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from dataset import WindowGenerator


class Baseline(tf.keras.Model):
    def __init__(self, label_index=None):
        super().__init__()
        self.label_index = label_index

    def call(self, inputs):
        if self.label_index is None:
            return inputs
        result = inputs[:, :, self.label_index]
        return result[:, :, tf.newaxis]


class MultiStepLastBaseline(tf.keras.Model):
    def call(self, inputs):
        return tf.tile(inputs[:, -1:, :], [1, OUT_STEPS, 1])


class RepeatBaseline(tf.keras.Model):
    def call(self, inputs):
        return inputs


class ResidualWrapper(tf.keras.Model):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def call(self, inputs, *args, **kwargs):
        delta = self.model(inputs, *args, **kwargs)
        # The prediction for each timestep is the input
        # from the previous time step plus the delta
        # calculated by the model.
        return inputs + delta


class FeedBack(tf.keras.Model):
    def __init__(self, units, out_steps):
        super().__init__()
        self.out_steps = out_steps
        self.units = units
        self.lstm_cell = tf.keras.layers.LSTMCell(units)
        # Also wrap the LSTMCell in an RNN to simplify the `warmup` method.
        self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True)
        self.dense = tf.keras.layers.Dense(num_features)

    def warmup(self, inputs):
        # inputs.shape => (batch, time, features)
        # x.shape => (batch, lstm_units)
        x, *state = self.lstm_rnn(inputs)
        # predictions.shape => (batch, features)
        prediction = self.dense(x)
        return prediction, state

    def call(self, inputs, training=None):
        # Use a TensorArray to capture dynamically unrolled outputs.
        predictions = []
        # Initialize the lstm state
        prediction, state = self.warmup(inputs)
        # Insert the first prediction
        predictions.append(prediction)

        # Run the rest of the prediction steps
        for n in range(1, self.out_steps):
            # Use the last prediction as input.
            x = prediction
            # Execute one lstm step.
            x, state = self.lstm_cell(x, states=state, training=training)
            # Convert the lstm output to a prediction.
            prediction = self.dense(x)
            # Add the prediction to the output
            predictions.append(prediction)

        # predictions.shape => (time, batch, features)
        predictions = tf.stack(predictions)
        # predictions.shape => (batch, time, features)
        predictions = tf.transpose(predictions, [1, 0, 2])
        return predictions


def compile_and_fit(model, window, patience=5):
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                      patience=patience,
                                                      mode='min')
    model.compile(loss=tf.losses.MeanSquaredError(),
                  optimizer=tf.optimizers.Adam(),
                  metrics=[tf.metrics.MeanAbsoluteError()])
    history = model.fit(window.train, epochs=MAX_EPOCHS,
                        validation_data=window.val,
                        callbacks=[early_stopping])
    return history


def plot_experiments(metric_idx, val_loss, train_loss, train_accs, val_accs, test_accs):
    x = np.arange(len(train_loss))
    val_mae = [v[metric_idx] for v in val_loss.values()]
    test_mae = [v[metric_idx] for v in train_loss.values()]

    plt.ylabel('mean_absolute_error [DIJA, normalized]')
    plt.bar(x - 0.17, val_mae, 0.3, label='Validation')
    plt.bar(x + 0.17, test_mae, 0.3, label='Test')
    plt.xticks(ticks=x, labels=train_loss.keys(), rotation=45)
    _ = plt.legend()
    plt.show()

    train_acc = [v for v in train_accs.values()]
    val_acc = [v for v in val_accs.values()]
    test_acc = [v for v in test_accs.values()]
    plt.ylabel('binary accuracy')
    plt.bar(x - 0.23, train_acc, 0.2, label='Train')
    plt.bar(x, val_acc, 0.2, label='Validation')
    plt.bar(x + 0.23, test_acc, 0.2, label='Test')
    plt.xticks(ticks=x, labels=train_loss.keys(), rotation=45)
    _ = plt.legend()
    plt.show()


def train_and_evaluate(model, window, plot_window, name):
    if name == 'Baseline' or name == 'Last' or name == 'Repeat':
        model.compile(loss=tf.losses.MeanSquaredError(), metrics=[tf.metrics.MeanAbsoluteError()])
    else:
        compile_and_fit(model=model, window=window)
    val_performance[name] = model.evaluate(window.val)
    performance[name] = model.evaluate(window.test, verbose=0)
    train_acc[name] = window.binary_accuracy(model=model, ds='train')
    val_acc[name] = window.binary_accuracy(model=model, ds='val')
    test_acc[name] = window.binary_accuracy(model=model, ds='test')
    plot_window.plot(name=name, model=model, acc=test_acc[name])


def run_single_step_models():
    baseline = Baseline(label_index=0)
    train_and_evaluate(baseline, single_step_window, wide_window, 'Baseline')

    linear = tf.keras.Sequential([tf.keras.layers.Dense(units=1)])
    train_and_evaluate(linear, single_step_window, wide_window, 'Linear')

    # dense = tf.keras.Sequential([
    #     tf.keras.layers.Dense(units=64, activation='relu'),
    #     tf.keras.layers.Dense(units=1)
    # ])
    # train_and_evaluate(dense, single_step_window, wide_window, 'Dense')

    # multi_step_dense = tf.keras.Sequential([
    #     # Shape: (time, features) => (time*features)
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(units=32, activation='relu'),
    #     tf.keras.layers.Dense(units=1),
    #     # Add back the time dimension.
    #     # Shape: (outputs) => (1, outputs)
    #     tf.keras.layers.Reshape([1, -1]),
    # ])
    # train_and_evaluate(multi_step_dense, conv_window, conv_window, 'Multi-step dense')
    #
    # conv_model = tf.keras.Sequential([
    #     tf.keras.layers.Conv1D(filters=16,
    #                            kernel_size=(CONV_WIDTH,),
    #                            activation='relu'),
    #     tf.keras.layers.Dense(units=16, activation='relu'),
    #     tf.keras.layers.Dense(units=1),
    # ])
    # train_and_evaluate(conv_model, conv_window, wide_conv_window, 'Conv')
    #
    # lstm_model = tf.keras.models.Sequential([
    #     # Shape [batch, time, features] => [batch, time, lstm_units]
    #     tf.keras.layers.LSTM(32, return_sequences=True),
    #     # Shape => [batch, time, features]
    #     tf.keras.layers.Dense(units=1)
    # ])
    # train_and_evaluate(lstm_model, wide_window, wide_window, 'LSTM')

    metric_idx = baseline.metrics_names.index('mean_absolute_error')
    return metric_idx


def run_multi_output_models():
    baseline = Baseline(label_index=0)
    train_and_evaluate(baseline, single_step_window, wide_window, 'Baseline Multi-out')

    dense = tf.keras.Sequential([
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=num_features)
    ])
    train_and_evaluate(dense, single_step_window, wide_window, 'Dense Multi-out')

    lstm_model = tf.keras.models.Sequential([
        # Shape [batch, time, features] => [batch, time, lstm_units]
        tf.keras.layers.LSTM(32, return_sequences=True),
        # Shape => [batch, time, features]
        tf.keras.layers.Dense(units=num_features)
    ])
    train_and_evaluate(lstm_model, wide_window_multi_out, wide_window, 'LSTM Multi-out')

    residual_lstm = ResidualWrapper(
        tf.keras.Sequential([
            tf.keras.layers.LSTM(32, return_sequences=True),
            tf.keras.layers.Dense(
                num_features,
                # The predicted deltas should start small
                # So initialize the output layer with zeros
                kernel_initializer=tf.initializers.zeros())
        ]))
    train_and_evaluate(residual_lstm, wide_window_multi_out, wide_window, 'Residual LSTM Multi-out')

    metric_idx = baseline.metrics_names.index('mean_absolute_error')
    return metric_idx


def run_multi_step_models():
    last_baseline = MultiStepLastBaseline()
    train_and_evaluate(last_baseline, multi_window, multi_window, 'Last')

    repeat_baseline = RepeatBaseline()
    train_and_evaluate(repeat_baseline, multi_window, multi_window, 'Repeat')

    multi_linear_model = tf.keras.Sequential([
        # Take the last time-step.
        # Shape [batch, time, features] => [batch, 1, features]
        tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
        # Shape => [batch, 1, out_steps*features]
        tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()),
        # Shape => [batch, out_steps, features]
        tf.keras.layers.Reshape([OUT_STEPS, num_features])
    ])
    train_and_evaluate(multi_linear_model, multi_window, multi_window, 'Linear Multi')

    # multi_dense_model = tf.keras.Sequential([
    #     # Take the last time step.
    #     # Shape [batch, time, features] => [batch, 1, features]
    #     tf.keras.layers.Lambda(lambda x: x[:, -1:, :]),
    #     # Shape => [batch, 1, dense_units]
    #     tf.keras.layers.Dense(512, activation='relu'),
    #     # Shape => [batch, out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])
    # train_and_evaluate(multi_dense_model, multi_window, multi_window, 'Dense Multi')
    #
    # multi_conv_model = tf.keras.Sequential([
    #     # Shape [batch, time, features] => [batch, CONV_WIDTH, features]
    #     tf.keras.layers.Lambda(lambda x: x[:, -CONV_WIDTH:, :]),
    #     # Shape => [batch, 1, conv_units]
    #     tf.keras.layers.Conv1D(1024, activation='relu', kernel_size=CONV_WIDTH),
    #     # Shape => [batch, 1,  out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])
    # train_and_evaluate(multi_conv_model, multi_window, multi_window, 'Conv Multi')
    #
    # multi_lstm_model = tf.keras.Sequential([
    #     # Shape [batch, time, features] => [batch, lstm_units]
    #     # Adding more `lstm_units` just overfits more quickly.
    #     tf.keras.layers.LSTM(32, return_sequences=False),
    #     # Shape => [batch, out_steps*features]
    #     tf.keras.layers.Dense(OUT_STEPS * num_features, kernel_initializer=tf.initializers.zeros()),
    #     # Shape => [batch, out_steps, features]
    #     tf.keras.layers.Reshape([OUT_STEPS, num_features])
    # ])
    # train_and_evaluate(multi_lstm_model, multi_window, multi_window, 'LSTM Multi')
    #
    # feedback_model = FeedBack(units=32, out_steps=OUT_STEPS)
    # train_and_evaluate(feedback_model, multi_window, multi_window, 'AR LSTM')

    metric_idx = repeat_baseline.metrics_names.index('mean_absolute_error')
    return metric_idx


if __name__ == "__main__":
    val_performance, performance, train_acc, val_acc, test_acc = {}, {}, {}, {}, {}
    num_features = 118
    MAX_EPOCHS = 200
    CONV_WIDTH = 5
    LABEL_WIDTH = 24
    INPUT_WIDTH = LABEL_WIDTH + (CONV_WIDTH - 1)
    OUT_STEPS = 24
    single_step_window = WindowGenerator(
        input_width=1, label_width=1, shift=1, label_columns=['DIJA'])
    wide_window = WindowGenerator(
        input_width=24, label_width=24, shift=1, label_columns=['DIJA'])
    wide_window_multi_out = WindowGenerator(
        input_width=24, label_width=24, shift=1)
    conv_window = WindowGenerator(
        input_width=CONV_WIDTH, label_width=1, shift=1, label_columns=['DIJA'])
    wide_conv_window = WindowGenerator(
        input_width=INPUT_WIDTH, label_width=LABEL_WIDTH, shift=1, label_columns=['DIJA'])
    multi_window = WindowGenerator(
        input_width=24, label_width=OUT_STEPS, shift=OUT_STEPS)

    metric_index = run_single_step_models()
    # _ = run_multi_output_models()
    # _ = run_multi_step_models()
    plot_experiments(metric_index, val_performance, performance, train_acc, val_acc, test_acc)
