import os
import math
import random
import logging
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Bidirectional, Masking
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

class RulHandler:
    def __init__(self):
        self.logger = logging.getLogger()

    def prepare_y_future(self, battery_names, battery_n_cycle, y_soh, current, time, capacity_threshold=None, allow_negative_future=False, capacity=None):
        cycle_length = current.shape[1]
        battery_range_step = [x * cycle_length for x in battery_n_cycle]
        self.logger.info("battery step: %s", battery_n_cycle)
        self.logger.info("battery ranges: %s", battery_range_step)

        if capacity is None:
            battery_nominal_capacity = [float(name.split("-")[2]) for name in battery_names]
        else:
            battery_nominal_capacity = [capacity for _ in battery_names]

        current = current.ravel()
        time = time.ravel()
        capacity_integral_train = []
        a = 0
        for battery_index, b in enumerate(battery_range_step):
            self.logger.info("processing range %d - %d", a, b)
            integral_sum = 0
            pre_i = a
            for i in range(a, b, cycle_length):
                integral = np.trapz(
                    y=current[pre_i:i][current[pre_i:i] > 0],
                    x=time[pre_i:i][current[pre_i:i] > 0]
                )
                integral_sum += integral
                pre_i = i
                capacity_integral_train.append(integral_sum / battery_nominal_capacity[battery_index])
            a = b
        capacity_integral_train = np.array(capacity_integral_train)
        self.logger.info("Train integral: %s", capacity_integral_train.shape)

        y_future = []
        a = 0
        for battery_index, b in enumerate(battery_n_cycle):
            self.logger.info("processing range %d - %d", a, b)
            if capacity_threshold is None:
                index = b - 1
            else:
                index = np.argmax(y_soh[a:b] < capacity_threshold[battery_nominal_capacity[battery_index]]) + a
                if index == a:
                    index = b - 1
            self.logger.info("threshold index: %d", index)
            for i in range(a, b):
                if not allow_negative_future:
                    y = capacity_integral_train[index] - capacity_integral_train[i] if i < index else 0
                else:
                    y = capacity_integral_train[index] - capacity_integral_train[i]
                y_future.append(y)
            a = b
        y_future = np.array(y_future)
        self.logger.info("y future: %s", y_future.shape)

        y_with_future = np.column_stack((capacity_integral_train, y_future))
        self.logger.info("y with future: %s", y_with_future.shape)
        return y_with_future

    def compress_cycle(self, train_x, test_x):
        train_x[train_x == 0] = np.nan
        new_train = np.empty((train_x.shape[0], train_x.shape[2], 2))
        for i in range(train_x.shape[2]):
            for x in range(train_x.shape[0]):
                new_train[x, i, 0] = np.nanmean(train_x[x, :, i])
                new_train[x, i, 1] = np.nanstd(train_x[x, :, i])
        new_train = new_train.reshape((train_x.shape[0], train_x.shape[2] * 2))

        test_x[test_x == 0] = np.nan
        new_test = np.empty((test_x.shape[0], test_x.shape[2], 2))
        for i in range(test_x.shape[2]):
            for x in range(test_x.shape[0]):
                new_test[x, i, 0] = np.nanmean(test_x[x, :, i])
                new_test[x, i, 1] = np.nanstd(test_x[x, :, i])
        new_test = new_test.reshape((test_x.shape[0], test_x.shape[2] * 2))

        self.logger.info("new compact train x: %s, new compact test x: %s", new_train.shape, new_test.shape)
        return new_train, new_test

    def battery_life_to_time_series(self, x, n_cycle, battery_range_cycle):
        series = np.zeros((x.shape[0], n_cycle, x.shape[1]))
        a = 0
        for b in battery_range_cycle:
            for i in range(a, b):
                bounded_a = max(a, i + 1 - n_cycle)
                series[i, 0:i + 1 - bounded_a] = x[bounded_a:i + 1]
            a = b
        self.logger.info("x time series shape: %s", series.shape)
        return series

    def delete_initial(self, x, y, battery_range, soh, warmup):
        new_range = [x - warmup * (i + 1) for i, x in enumerate(battery_range)]
        battery_range = np.insert(battery_range[:-1], 0, [0])
        indexes = [int(x + i) for x in battery_range for i in range(warmup)]
        x = np.delete(x, indexes, axis=0)
        y = np.delete(y, indexes, axis=0)
        soh = np.delete(soh.flatten(), indexes, axis=0)
        self.logger.info("x with warmup: %s, y with warmup: %s", x.shape, y.shape)
        return x, y, new_range, soh

    def limit_zeros(self, x, y, battery_range, soh, limit=100):
        indexes = []
        new_range = []
        a = 0
        removed = 0
        for b in battery_range:
            zeros = np.where(y[a:b, 1] == 0)[0] + a
            indexes.extend(zeros[limit:].tolist())
            removed += len(zeros[limit:])
            new_range.append(b - removed)
            a = b
        x = np.delete(x, indexes, axis=0)
        y = np.delete(y, indexes, axis=0)
        soh = np.delete(soh.flatten(), indexes, axis=0)
        self.logger.info("x with limit: %s, y with limit: %s", x.shape, y.shape)
        return x, y, new_range, soh

    def unify_datasets(self, x, y, battery_range, soh, m_x, m_y, m_battery_range, m_soh):
        m_battery_range = m_battery_range + battery_range[-1]
        x = np.concatenate((x, m_x))
        y = np.concatenate((y, m_y))
        battery_range = np.concatenate((battery_range, m_battery_range))
        soh = np.concatenate((soh.flatten(), m_soh))

        self.logger.info("Unified x: %s, unified y : %s, unified battery range: %s", x.shape, y.shape, battery_range)
        return x, y, battery_range, soh

    class Normalization:
        def fit(self, train):
            if len(train.shape) == 1:
                self.case = 1
                self.min = min(train)
                self.max = max(train)
            elif len(train.shape) == 2:
                self.case = 2
                self.min = [min(train[:, i]) for i in range(train.shape[1])]
                self.max = [max(train[:, i]) for i in range(train.shape[1])]
            elif len(train.shape) == 3:
                self.case = 3
                self.min = [train[:, :, i].min() for i in range(train.shape[2])]
                self.max = [train[:, :, i].max() for i in range(train.shape[2])]

        def normalize(self, data):
            if self.case == 1:
                data = (data - self.min) / (self.max - self.min)
            elif self.case == 2:
                for i in range(data.shape[1]):
                    data[:, i] = (data[:, i] - self.min[i]) / (self.max[i] - self.min[i])
            elif self.case == 3:
                for i in range(data.shape[2]):
                    data[:, :, i] = (data[:, :, i] - self.min[i]) / (self.max[i] - self.min[i])
            return data

        def fit_and_normalize(self, train, test, val=None):
            self.fit(train)
            if val is not None:
                return self.normalize(train), self.normalize(test), self.normalize(val)
            else:
                return self.normalize(train), self.normalize(test)

        def denormalize(self, a):
            if self.case == 1:
                a = a * (self.max - self.min) + self.min
            elif self.case == 2:
                for i in range(a.shape[1]):
                    a[:, i] = a[:, i] * (self.max[i] - self.min[i]) + self.min[i]
            elif self.case == 3:
                for i in range(a.shape[2]):
                    a[:, :, i] = a[:, :, i] * (self.max[i] - self.min[i]) + self.min[i]
            return a

class BatteryRULPipeline:
    def __init__(self):
        self.rul_handler = RulHandler()

    def load_and_preprocess_data(self, file_path):
        data = pd.read_csv(file_path)
        for col in data.columns[1:]:
            data[col] = (data[col] - data[col].mean()) / data[col].std()
        return data

    def split_data(self, data, train_ratio=0.8):
        train_size = int(len(data) * train_ratio)
        train_data = data[:train_size]
        test_data = data[train_size:]
        return train_data, test_data

    def build_model(self, input_shape):
        model = Sequential([
            Masking(mask_value=0.0, input_shape=input_shape),
            Bidirectional(LSTM(128, return_sequences=True, dropout=0.2)),
            LSTM(64, return_sequences=False, dropout=0.2),
            Dense(32, activation='relu'),
            Dropout(0.2),
            Dense(1, activation='linear')
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        return model

    def train_model(self, model, train_X, train_y, val_X, val_y):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint('best_model.h5', save_best_only=True)
        ]
        history = model.fit(
            train_X, train_y,
            validation_data=(val_X, val_y),
            epochs=100,
            batch_size=32,
            callbacks=callbacks
        )
        return history

    def main(self, data_file):
        logging.basicConfig(level=logging.INFO)
        logging.info("Starting the RUL prediction pipeline")

        # Load data
        data = self.load_and_preprocess_data(data_file)
        train_data, test_data = self.split_data(data)

        # Prepare inputs and labels
        train_X, train_y = train_data.iloc[:, :-1].values, train_data.iloc[:, -1].values
        test_X, test_y = test_data.iloc[:, :-1].values, test_data.iloc[:, -1].values

        input_shape = (train_X.shape[1], train_X.shape[2]) if len(train_X.shape) == 3 else (train_X.shape[1], 1)

        # Build and train model
        model = self.build_model(input_shape)
        logging.info("Training the model")
        self.train_model(model, train_X, train_y, test_X, test_y)

        # Save the model
        model.save("final_model.h5")
        logging.info("Model training complete and saved.")

if __name__ == "__main__":
    pipeline = BatteryRULPipeline()
    pipeline.main("./data/rul_dataset.csv")
