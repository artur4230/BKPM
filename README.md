# Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.optimizers import Adam
import warnings
warnings.filterwarnings('ignore')

# Для байесовской оптимизации
!pip install scikit-optimize -q
from skopt import gp_minimize
from skopt.space import Real, Integer, Categorical
from skopt.utils import use_named_args
from skopt.plots import plot_convergence

# Для работы с Google Colab и Google Drive
from google.colab import drive
drive.mount('/content/drive')

import os
import glob

np.random.seed(42)
tf.random.set_seed(42)

# ============================================================================
# КЛАСС ДЛЯ ЗАГРУЗКИ И ПРЕПРОЦЕССИНГА ДАННЫХ
# ============================================================================

class CMAPSSDataLoader:
    """
    Класс для загрузки и обработки данных NASA CMAPSS
    """

    def __init__(self, data_path, dataset_number=1):
        """
        Инициализация загрузчика данных

        Параметры:
        -----------
        data_path : str
            Путь к директории с данными
        dataset_number : int
            Номер датасета (1-4)
        """
        self.data_path = data_path
        self.dataset_number = dataset_number
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def load_data(self):
        """
        Загрузка данных тренировочного и тестового наборов
        """
        # Загрузка тренировочных данных
        train_file = os.path.join(self.data_path, f'train_FD00{self.dataset_number}.txt')
        test_file = os.path.join(self.data_path, f'test_FD00{self.dataset_number}.txt')
        rul_file = os.path.join(self.data_path, f'RUL_FD00{self.dataset_number}.txt')

        # Чтение данных
        column_names = ['unit', 'cycle', 'setting1', 'setting2', 'setting3'] + \
                      [f'sensor_{i}' for i in range(1, 22)]

        self.train_data = pd.read_csv(train_file, sep='\s+', header=None, names=column_names)
        self.test_data = pd.read_csv(test_file, sep='\s+', header=None, names=column_names)
        self.true_rul = pd.read_csv(rul_file, sep='\s+', header=None, names=['RUL'])

        print(f"Тренировочные данные: {self.train_data.shape}")
        print(f"Тестовые данные: {self.test_data.shape}")

        return self.train_data, self.test_data, self.true_rul

    def add_rul_label(self, df, max_life=None):
        """
        Добавление целевой переменной RUL

        Параметры:
        -----------
        df : DataFrame
            Исходные данные
        max_life : int, optional
            Максимальный срок жизни двигателя

        Возвращает:
        -----------
        DataFrame с добавленным столбцом RUL
        """
        df_rul = df.copy()

        # Вычисление RUL для каждого двигателя
        def calculate_rul(group):
            max_cycle = group['cycle'].max()
            group['RUL'] = max_cycle - group['cycle']
            return group

        df_rul = df_rul.groupby('unit').apply(calculate_rul)

        # Ограничение максимального RUL (piece-wise linear)
        if max_life is not None:
            df_rul['RUL'] = df_rul['RUL'].clip(upper=max_life)

        return df_rul.reset_index(drop=True)

    def prepare_train_data(self, window_size=30, max_rul=125):
        """
        Подготовка тренировочных данных для LSTM

        """
        # Добавление RUL
        train_with_rul = self.add_rul_label(self.train_data, max_life=max_rul)

        # Выбор признаков 
        sensor_cols = [col for col in train_with_rul.columns if 'sensor' in col]
        feature_cols = sensor_cols  # Можно добавить setting при необходимости

        # Нормализация данных
        train_normalized = self.train_data.copy()
        train_normalized[feature_cols] = self.scaler.fit_transform(train_normalized[feature_cols])

        # Создание последовательностей
        X_seq, y_seq = [], []

        for unit_id in train_normalized['unit'].unique():
            unit_data = train_normalized[train_normalized['unit'] == unit_id]
            unit_rul = train_with_rul[train_with_rul['unit'] == unit_id]['RUL'].values

            # Формирование окон
            for i in range(len(unit_data) - window_size):
                X_seq.append(unit_data[feature_cols].values[i:i+window_size])
                y_seq.append(unit_rul[i+window_size])

        return np.array(X_seq), np.array(y_seq)

    def prepare_test_data(self, window_size=30):
        """
        Подготовка тестовых данных
        """
        # Нормализация тестовых данных
        test_normalized = self.test_data.copy()
        sensor_cols = [col for col in test_normalized.columns if 'sensor' in col]
        test_normalized[sensor_cols] = self.scaler.transform(test_normalized[sensor_cols])

        # Создание последовательностей для каждого двигателя
        X_test_seq, y_test_seq, engine_ids = [], [], []

        for unit_id in test_normalized['unit'].unique():
            unit_data = test_normalized[test_normalized['unit'] == unit_id]

            # Используем последнее окно для прогноза
            if len(unit_data) >= window_size:
                X_test_seq.append(unit_data[sensor_cols].values[-window_size:])
                engine_ids.append(unit_id)

        return np.array(X_test_seq), engine_ids

# ============================================================================
# КЛАСС LSTM МОДЕЛИ
# ============================================================================

class LSTMModel:
    """
    Класс для создания и обучения LSTM модели
    """

    def __init__(self, input_shape, hyperparams):
        """
        Инициализация модели
        """
        self.input_shape = input_shape
        self.hyperparams = hyperparams
        self.model = self._build_model()

    def _build_model(self):
        """
        Построение архитектуры LSTM модели
        """
        # Преобразование параметров в int (важно для Keras)
        lstm_units1 = int(self.hyperparams['lstm_units1'])
        lstm_units2 = int(self.hyperparams['lstm_units2'])
        dense_units1 = int(self.hyperparams['dense_units1'])
        dense_units2 = int(self.hyperparams['dense_units2'])

        model = keras.Sequential([
            # Входной слой
            layers.Input(shape=self.input_shape),

            # Первый LSTM слой
            layers.LSTM(
                units=lstm_units1,
                return_sequences=True,
                kernel_regularizer=keras.regularizers.l2(float(self.hyperparams['l2_reg']))
            ),
            layers.BatchNormalization(),
            layers.Dropout(float(self.hyperparams['dropout_rate'])),

            # Второй LSTM слой
            layers.LSTM(
                units=lstm_units2,
                return_sequences=False,
                kernel_regularizer=keras.regularizers.l2(float(self.hyperparams['l2_reg']))
            ),
            layers.BatchNormalization(),
            layers.Dropout(float(self.hyperparams['dropout_rate'])),

            # Полносвязные слои
            layers.Dense(
                units=dense_units1,
                activation='relu',
                kernel_regularizer=keras.regularizers.l2(float(self.hyperparams['l2_reg']))
            ),
            layers.Dropout(float(self.hyperparams['dropout_rate'])/2),

            layers.Dense(
                units=dense_units2,
                activation='relu'
            ),

            # Выходной слой
            layers.Dense(1, activation='linear')
        ])

        # Компиляция модели
        model.compile(
            optimizer=Adam(
                learning_rate=float(self.hyperparams['learning_rate']),
                clipvalue=0.5
            ),
            loss='mse',
            metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
        )

        return model

    def train(self, X_train, y_train, X_val, y_val, epochs=10, batch_size=32):
        """
        Обучение модели с ранней остановкой
        """
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            verbose=1
        )

        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,
            min_lr=1e-6,
            verbose=1
        )

        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        return history

    def predict(self, X):
        """
        Прогнозирование RUL
        """
        return self.model.predict(X, verbose=0).flatten()

# ============================================================================
# БАЙЕСОВСКАЯ ОПТИМИЗАЦИЯ
# ============================================================================

class BayesianOptimizer:
    """
    Класс для байесовской оптимизации гиперпараметров
    """

    def __init__(self, X_train, y_train, X_val, y_val, input_shape):
        """
        Инициализация оптимизатора
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        self.input_shape = input_shape
        self.best_score = float('inf')
        self.best_params = None
        self.results = []

        # Определение пространства поиска
        self.dimensions = [
            Integer(32, 256, name='lstm_units1'),
            Integer(16, 128, name='lstm_units2'),
            Integer(32, 128, name='dense_units1'),
            Integer(16, 64, name='dense_units2'),
            Real(1e-4, 1e-2, prior='log-uniform', name='learning_rate'),
            Real(0.1, 0.5, name='dropout_rate'),
            Real(1e-6, 1e-2, prior='log-uniform', name='l2_reg')
        ]

    def objective(self, params):
        """
        Целевая функция для оптимизации
        """
        # Преобразуем numpy типы в стандартные Python типы
        param_names = [dim.name for dim in self.dimensions]
        param_dict = {}

        for name, value in zip(param_names, params):
            if 'units' in name:
                param_dict[name] = int(value)  # Преобразуем в int для LSTM
            else:
                param_dict[name] = float(value)  # Преобразуем в float

        print(f"\nТестирование параметров: {param_dict}")

        try:
            # Создание и обучение модели
            model = LSTMModel(self.input_shape, param_dict)

            # Уменьшаем количество эпох для быстрой оптимизации
            history = model.train(
                self.X_train, self.y_train,
                self.X_val, self.y_val,
                epochs=30,  # Уменьшено для скорости
                batch_size=64
            )

            # Оценка модели
            val_pred = model.predict(self.X_val)
            score = np.sqrt(mean_squared_error(self.y_val, val_pred))

            # Сохранение лучших параметров
            if score < self.best_score:
                self.best_score = score
                self.best_params = param_dict
                print(f"Новый лучший RMSE: {score:.4f}")

            self.results.append({
                'params': param_dict,
                'score': score,
                'history': history.history
            })

            return score

        except Exception as e:
            print(f"Ошибка при обучении: {str(e)[:200]}")
            return 1000.0  # Возвращаем float

    def optimize(self, n_calls=20, random_state=42):
        """
        Запуск байесовской оптимизации
        """
        print("Начало байесовской оптимизации...")

        # Создаем обертку для совместимости с skopt
        def objective_wrapper(params):
            return self.objective(params)

        result = gp_minimize(
            func=objective_wrapper,
            dimensions=self.dimensions,
            n_calls=n_calls,
            random_state=random_state,
            verbose=True,
            n_initial_points=1,
            acq_func='EI',  # Expected Improvement
            xi=0.01
        )

        return result

# ============================================================================
# ОСНОВНОЙ ПАЙПЛАЙН
# ============================================================================
def quick_start():
    """
    Упрощенный пайплайн для быстрого старта
    """
    print("=" * 60)
    print("БЫСТРЫЙ СТАРТ - УПРОЩЕННАЯ ВЕРСИЯ")
    print("=" * 60)

    # 1. Загрузка данных
    data_path = "/content/drive/MyDrive/CMAPSSData" #УКАЖИТЕ СВОЙ ПУТЬ
    loader = CMAPSSDataLoader(data_path, dataset_number=3)
    train_data, test_data, true_rul = loader.load_data()

    # 2. Подготовка данных
    window_size = 30
    print(f"\nПодготовка данных с окном {window_size}...")
    X_train, y_train = loader.prepare_train_data(window_size=window_size, max_rul=125)
    X_test, engine_ids = loader.prepare_test_data(window_size=window_size)

    # Разделение на тренировочную и валидационную выборки
    from sklearn.model_selection import train_test_split
    X_train_split, X_val, y_train_split, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, shuffle=False  # shuffle=False для временных данных
    )

    print(f"\nРазмеры данных:")
    print(f"  X_train: {X_train_split.shape}")
    print(f"  y_train: {y_train_split.shape}")
    print(f"  X_val: {X_val.shape}")
    print(f"  y_val: {y_val.shape}")
    print(f"  X_test: {X_test.shape}")

    # 3. Проверка на небольших данных (демо-версия)
    print("\n" + "=" * 60)
    print("ТЕСТИРОВАНИЕ НА НЕБОЛЬШИХ ДАННЫХ")
    print("=" * 60)

    # Используем только часть данных для быстрого теста
    sample_size = min(1000, len(X_train_split))
    X_sample = X_train_split[:sample_size]
    y_sample = y_train_split[:sample_size]
    X_val_sample = X_val[:200]
    y_val_sample = y_val[:200]

    # 4. Базовые гиперпараметры (без оптимизации для начала)
    base_hyperparams = {
        'lstm_units1': 64,
        'lstm_units2': 32,
        'dense_units1': 32,
        'dense_units2': 16,
        'learning_rate': 0.001,
        'dropout_rate': 0.2,
        'l2_reg': 1e-3
    }

    print(f"\nБазовые гиперпараметры: {base_hyperparams}")

    # 5. Обучение базовой модели
    print("\nОбучение базовой LSTM модели...")
    base_model = LSTMModel(
        input_shape=(window_size, X_train.shape[2]),
        hyperparams=base_hyperparams
    )

    history = base_model.train(
        X_sample, y_sample,
        X_val_sample, y_val_sample,
        epochs=50,
        batch_size=32
    )

    # 6. Прогнозирование
    print("\nПрогнозирование на валидационных данных...")
    val_predictions = base_model.predict(X_val_sample)

    # 7. Оценка
    val_rmse = np.sqrt(mean_squared_error(y_val_sample, val_predictions))
    val_mae = mean_absolute_error(y_val_sample, val_predictions)

    print(f"\nРезультаты на валидационной выборке:")
    print(f"  RMSE: {val_rmse:.4f}")
    print(f"  MAE: {val_mae:.4f}")

    # 8. Базовая визуализация
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Функция потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('MSE')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.scatter(y_val_sample, val_predictions, alpha=0.5)
    plt.plot([y_val_sample.min(), y_val_sample.max()],
             [y_val_sample.min(), y_val_sample.max()], 'r--', lw=2)
    plt.title('Прогнозы vs Фактические значения')
    plt.xlabel('Фактический RUL')
    plt.ylabel('Предсказанный RUL')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return base_model, loader

def main_safe():
    """
    Безопасная версия основного пайплайна
    """
    try:
        print("\n" + "=" * 60)
        print("ЗАПУСК ПОЛНОЙ ОПТИМИЗАЦИИ")
        print("=" * 60)

        data_path = "/content/drive/MyDrive/CMAPSSData" #УКАЖИТЕ СВОЙ ПУТЬ
        loader = CMAPSSDataLoader(data_path, dataset_number=3)
        train_data, test_data, true_rul = loader.load_data()

        window_size = 30
        X_train, y_train = loader.prepare_train_data(window_size=window_size, max_rul=125)
        X_test, engine_ids = loader.prepare_test_data(window_size=window_size)

        from sklearn.model_selection import train_test_split
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, shuffle=False
        )

        # Базовая оптимизация с меньшим количеством вызовов
        optimizer = BayesianOptimizer(
            X_train_split[:2000], y_train_split[:2000],
            X_val[:500], y_val[:500],
            input_shape=(window_size, X_train.shape[2])
        )

        print("\nЗапуск байесовской оптимизации (10 итераций)...")
        optimization_result = optimizer.optimize(n_calls=3)

        print(f"\nЛучшие параметры: {optimizer.best_params}")
        print(f"Лучший RMSE на валидации: {optimizer.best_score:.4f}")

        # Обучение финальной модели с лучшими параметрами
        print("\nОбучение финальной модели...")
        final_model = LSTMModel(
            input_shape=(window_size, X_train.shape[2]),
            hyperparams=optimizer.best_params
        )

        history = final_model.train(
            X_train, y_train,
            X_val, y_val,
            epochs=30,
            batch_size=64
        )

        # Прогнозирование
        test_predictions = final_model.predict(X_test)

        # Оценка - ВАЖНО: обрезаем true_rul до размера test_predictions
        true_rul_values = true_rul['RUL'].values[:len(test_predictions)]
        test_rmse = np.sqrt(mean_squared_error(true_rul_values, test_predictions))
        test_mae = mean_absolute_error(true_rul_values, test_predictions)
        test_r2 = r2_score(true_rul_values, test_predictions)

        # ВИЗУАЛИЗАЦИЯ 1: История обучения
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.plot(history.history['loss'], label='Train Loss (MSE)')
        plt.plot(history.history['val_loss'], label='Val Loss (MSE)')
        plt.title('Функция потерь во время обучения')
        plt.xlabel('Эпоха')
        plt.ylabel('MSE')
        plt.legend()
        plt.grid(True)

        # ВИЗУАЛИЗАЦИЯ 2: MAE во время обучения
        plt.subplot(1, 3, 2)
        if 'mae' in history.history:
            plt.plot(history.history['mae'], label='Train MAE')
            plt.plot(history.history['val_mae'], label='Val MAE')
            plt.title('Средняя абсолютная ошибка (MAE)')
            plt.xlabel('Эпоха')
            plt.ylabel('MAE')
            plt.legend()
            plt.grid(True)
        else:
            plt.text(0.5, 0.5, 'MAE не записывался\nв history',
                    ha='center', va='center', transform=plt.gca().transAxes)

        # ВИЗУАЛИЗАЦИЯ 3: Прогнозы vs Фактические значения
        plt.subplot(1, 3, 3)
        plt.scatter(true_rul_values, test_predictions, alpha=0.6, s=30)

        # Идеальная линия
        min_val = min(true_rul_values.min(), test_predictions.min())
        max_val = max(true_rul_values.max(), test_predictions.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Идеальный прогноз')

        # Линия регрессии
        from scipy import stats
        slope, intercept, r_value, p_value, std_err = stats.linregress(true_rul_values, test_predictions)
        regression_line = slope * np.array([min_val, max_val]) + intercept
        plt.plot([min_val, max_val], regression_line, 'g-', lw=2,
                label=f'Линия регрессии (R²={r_value**2:.3f})')

        plt.title('Прогнозы vs Фактические значения (тестовая выборка)')
        plt.xlabel('Фактический RUL (циклы)')
        plt.ylabel('Предсказанный RUL (циклы)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

        # ВИЗУАЛИЗАЦИЯ 4: Ошибки прогнозирования
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 2, 1)
        errors = true_rul_values - test_predictions
        plt.hist(errors, bins=20, edgecolor='black', alpha=0.7)
        plt.axvline(x=0, color='r', linestyle='--', label='Нет ошибки')
        plt.axvline(x=np.mean(errors), color='g', linestyle='-',
                   label=f'Среднее: {np.mean(errors):.2f}')
        plt.title('Распределение ошибок прогнозирования')
        plt.xlabel('Ошибка (Факт - Прогноз)')
        plt.ylabel('Частота')
        plt.legend()
        plt.grid(True)

        plt.subplot(1, 2, 2)
        engine_indices = np.arange(len(test_predictions))
        plt.bar(engine_indices[:20], true_rul_values[:20], alpha=0.7,
                label='Фактический RUL', width=0.4)
        plt.bar(engine_indices[:20] + 0.4, test_predictions[:20], alpha=0.7,
                label='Прогнозируемый RUL', width=0.4)
        plt.title('Сравнение RUL для первых 20 двигателей')
        plt.xlabel('Номер двигателя')
        plt.ylabel('RUL (циклы)')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

        print(f"\nФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ НА ТЕСТОВОЙ ВЫБОРКЕ:")
        print(f"  Test RMSE: {test_rmse:.4f} циклов")
        print(f"  Test MAE:  {test_mae:.4f} циклов")
        print(f"  Test R²:   {test_r2:.4f}")
        print(f"  Средняя ошибка: {np.mean(errors):.4f} циклов")
        print(f"  Стандартное отклонение ошибок: {np.std(errors):.4f} циклов")

        # Сохранение результатов
        results_df = pd.DataFrame({
            'engine_id': engine_ids[:len(test_predictions)],
            'true_rul': true_rul_values,
            'predicted_rul': test_predictions,
            'error': errors,
            'absolute_error': np.abs(errors)
        })

        # Анализ больших ошибок
        large_errors = results_df[results_df['absolute_error'] > 20]
        if len(large_errors) > 0:
            print(f"\nВНИМАНИЕ: {len(large_errors)} двигателей с ошибкой > 20 циклов:")
            print(large_errors[['engine_id', 'true_rul', 'predicted_rul', 'error']].head())

        return final_model, optimizer.best_params, results_df

    except Exception as e:
        print(f"\nПроизошла ошибка: {e}")
        import traceback
        traceback.print_exc()
        print("Вернемся к упрощенной версии...")
        return quick_start()
# ============================================================================
# ДОПОЛНИТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ЭКСПЕРИМЕНТОВ
# ============================================================================

def analyze_feature_importance(model, X_sample, feature_names):
    """
    Анализ важности признаков с помощью градиентных методов
    """
    X_tensor = tf.convert_to_tensor(X_sample, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(X_tensor)
        predictions = model.model(X_tensor)

    gradients = tape.gradient(predictions, X_tensor)
    importance = np.mean(np.abs(gradients.numpy()), axis=(0, 1))

    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(importance_df)), importance_df['importance'])
    plt.yticks(range(len(importance_df)), importance_df['feature'])
    plt.xlabel('Важность признака')
    plt.title('Анализ важности признаков для прогнозирования RUL')
    plt.tight_layout()
    plt.show()

    return importance_df

def cross_validation_evaluation(data_path, dataset_numbers=[1, 2, 3, 4]):
    """
    Кросс-валидация по разным датасетам
    """
    results = []

    for dataset_num in dataset_numbers:
        print(f"\nОбработка датасета FD00{dataset_num}")

        loader = CMAPSSDataLoader(data_path, dataset_number=dataset_num)
        train_data, test_data, true_rul = loader.load_data()

        window_size = 30
        X_train, y_train = loader.prepare_train_data(window_size=window_size)
        X_test, engine_ids = loader.prepare_test_data(window_size=window_size)

        # Разделение данных
        X_train_split, X_val, y_train_split, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42
        )

        # Обучение модели с базовыми параметрами
        hyperparams = {
            'lstm_units1': 128,
            'lstm_units2': 64,
            'dense_units1': 64,
            'dense_units2': 32,
            'learning_rate': 0.001,
            'dropout_rate': 0.3,
            'l2_reg': 1e-3
        }

        model = LSTMModel(
            input_shape=(window_size, X_train.shape[2]),
            hyperparams=hyperparams
        )

        history = model.train(
            X_train_split, y_train_split,
            X_val, y_val,
            epochs=100,
            batch_size=64
        )

        # Прогнозирование
        test_predictions = model.predict(X_test)

        # Оценка
        test_rmse = np.sqrt(mean_squared_error(
            true_rul['RUL'].values[:len(test_predictions)],
            test_predictions
        ))

        results.append({
            'dataset': f'FD00{dataset_num}',
            'rmse': test_rmse,
            'n_train_samples': len(X_train),
            'n_test_samples': len(X_test)
        })

        print(f"  RMSE: {test_rmse:.4f}")

    return pd.DataFrame(results)

# ============================================================================
# ЗАПУСК ОСНОВНОГО ПАЙПЛАЙНА
# ============================================================================
if __name__ == "__main__":
    print("Проверка доступности GPU...")
    print(tf.config.list_physical_devices('GPU'))

    # Запускаем безопасную версию
    model, best_params, res = main_safe()

    print("\n" + "=" * 60)
    print("ПРОГРАММА УСПЕШНО ВЫПОЛНЕНА")
    print("=" * 60)
