import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential # clone_model убрал, т.к. не используется пока
from tensorflow.keras.layers import Dense, Flatten, Input # Flatten может пригодиться для других state_shape
from tensorflow.keras.optimizers import Adam
import random
import os # Для тестового блока сохранения/загрузки
import logging
import traceback

# Убедимся, что логгер доступен, даже если этот модуль используется отдельно
logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logger.setLevel(logging.INFO)
    _ch = logging.StreamHandler()
    _cf = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _ch.setFormatter(_cf)
    logger.addHandler(_ch)


class DQNAgent:
    def __init__(self,
                 state_shape: tuple, # Например, (25,) если state это вектор из 25 признаков
                 action_size: int,
                 learning_rate: float = 0.001,
                 gamma: float = 0.99,        # Коэффициент дисконтирования
                 epsilon: float = 1.0,       # Начальное значение epsilon для epsilon-greedy
                 epsilon_decay: float = 0.995, # Коэффициент уменьшения epsilon
                 epsilon_min: float = 0.01,   # Минимальное значение epsilon
                 target_update_freq: int = 100 # Обновлять target-сеть каждые N *тренировочных шагов*
                ):
        """
        Инициализация DQN Агента.

        :param state_shape: Форма входного состояния (кортеж).
        :param action_size: Количество возможных действий.
        :param learning_rate: Скорость обучения для оптимизатора Adam.
        :param gamma: Коэффициент дисконтирования будущих наград.
        :param epsilon: Начальное значение для стратегии epsilon-greedy.
        :param epsilon_decay: Множитель для уменьшения epsilon после каждого эпизода.
        :param epsilon_min: Минимальное значение epsilon.
        :param target_update_freq: Частота обновления весов target-сети (в тренировочных шагах).
        """
        if not isinstance(state_shape, tuple) or not state_shape :
            raise ValueError(f"state_shape должен быть непустым кортежем, получено: {state_shape}")
        if action_size <= 0:
            raise ValueError(f"action_size должен быть положительным, получено: {action_size}")

        self.state_shape = state_shape
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_initial = epsilon # Сохраняем начальный эпсилон для возможного сброса
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.target_update_freq = target_update_freq
        self._train_step_counter = 0 # Счетчик шагов обучения (вызовов train_step)

        # Основная Q-сеть (обучаемая)
        self.q_network = self._build_q_network()
        # Target Q-сеть (для стабилизации обучения)
        self.target_q_network = self._build_q_network()
        self.update_target_network() # Инициализируем веса target-сети такими же, как у основной

        self.optimizer = Adam(learning_rate=self.learning_rate)
        # Huber loss менее чувствителен к выбросам, чем MSE, что может быть полезно
        self.loss_function = tf.keras.losses.Huber()
        # self.loss_function = tf.keras.losses.MeanSquaredError() # Альтернатива

        logger.info(f"DQNAgent инициализирован. State shape: {self.state_shape}, Action size: {self.action_size}")
        logger.info(f"  Гиперпараметры: LR={self.learning_rate}, Gamma={self.gamma}, Epsilon_init={self.epsilon}, "
                    f"Epsilon_decay={self.epsilon_decay}, Epsilon_min={self.epsilon_min}, "
                    f"TargetUpdateFreq={self.target_update_freq} шагов обучения.")

    def _build_q_network(self) -> tf.keras.Model:
        """
        Строит и возвращает модель Q-сети.
        """
        # Простая полносвязная сеть. Можно усложнить при необходимости.
        # Input слой явно определен для лучшей читаемости summary()
        inputs = Input(shape=self.state_shape, name="input_state")
        # Если state_shape многомерный (например, для изображений или нескольких временных шагов как каналов),
        # может понадобиться Flatten() перед Dense слоями, если только Dense слои используются.
        # x = Flatten()(inputs) # Раскомментировать, если state_shape, например, (window_size, num_features)
        # x = Dense(64, activation='relu', name="dense_1")(x)
        # Для простого векторного состояния:
        x = Dense(64, activation='relu', name="dense_1")(inputs)
        x = Dense(32, activation='relu', name="dense_2")(x)
        outputs = Dense(self.action_size, activation='linear', name="output_q_values")(x) # Линейная активация для Q-значений

        model = tf.keras.Model(inputs=inputs, outputs=outputs)
        # Компиляция здесь не обязательна для ручного цикла обучения с GradientTape,
        # но может быть полезна для model.summary() или если вы захотите использовать model.fit/evaluate.
        # model.compile(optimizer=self.optimizer, loss=self.loss_function)
        return model

    def update_target_network(self):
        """
        Копирует веса из основной Q-сети в target Q-сеть.
        """
        logger.debug(f"Обновление весов Target Q-сети на шаге обучения {self._train_step_counter}.")
        self.target_q_network.set_weights(self.q_network.get_weights())

    def get_action(self, state: np.ndarray) -> int:
        """
        Выбирает действие с использованием стратегии epsilon-greedy.

        :param state: Текущее состояние среды (numpy массив).
        :return: Индекс выбранного действия (int).
        """
        if not isinstance(state, np.ndarray):
            state = np.array(state, dtype=np.float32)

        # Проверка и возможное изменение формы состояния для одного элемента батча
        if state.ndim == len(self.state_shape): # Если state уже имеет размерность state_shape (например, (25,))
            if state.shape != self.state_shape:
                 logger.error(f"Форма состояния {state.shape} не соответствует ожидаемой {self.state_shape}. Выбирается случайное действие.")
                 return np.random.choice(self.action_size)
            state_batch = np.expand_dims(state, axis=0) # (1, 25)
        elif state.ndim == len(self.state_shape) + 1 and state.shape[0] == 1: # Если state уже в форме батча из 1 элемента (1, 25)
            if state.shape[1:] != self.state_shape: # Проверяем форму признаков
                 logger.error(f"Форма признаков состояния {state.shape[1:]} не соответствует ожидаемой {self.state_shape}. Выбирается случайное действие.")
                 return np.random.choice(self.action_size)
            state_batch = state
        else:
            logger.error(f"Некорректная размерность/форма состояния: {state.shape} (ожидалась совместимая с {self.state_shape}). Выбирается случайное действие.")
            return np.random.choice(self.action_size)

        # Epsilon-greedy выбор действия
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.action_size)
            logger.debug(f"E-greedy: выбрано случайное действие {action} (epsilon={self.epsilon:.4f})")
            return action
        else:
            try:
                # Используем основную Q-сеть для выбора лучшего действия
                q_values = self.q_network(state_batch, training=False) # training=False важно при предсказании
                action = np.argmax(q_values[0].numpy()) # .numpy() для извлечения из тензора
                logger.debug(f"E-greedy: выбрано оптимальное действие {action} (Q-values: {q_values[0].numpy()})")
                return action
            except Exception as e:
                logger.error(f"Ошибка при предсказании Q-значений в get_action: {e}. Выбирается случайное действие.")
                logger.debug(traceback.format_exc())
                return np.random.choice(self.action_size)

    @tf.function # Компилируем в граф для производительности
    def _perform_train_step(self, states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf):
        """
        Выполняет один шаг градиентного спуска с использованием логики Double DQN.
        Эта функция скомпилирована TensorFlow.

        :param states_tf: Тензор состояний.
        :param actions_tf: Тензор совершенных действий.
        :param rewards_tf: Тензор наград.
        :param next_states_tf: Тензор следующих состояний.
        :param dones_tf: Тензор флагов завершения эпизода.
        :return: Значение функции потерь (скалярный тензор).
        """
        # --- Double DQN Logic ---
        # 1. Выбираем лучшее действие (a') в следующем состоянии (S') с помощью ОСНОВНОЙ сети.
        q_values_next_main_net = self.q_network(next_states_tf, training=False)
        best_actions_next_state = tf.argmax(q_values_next_main_net, axis=1, output_type=tf.int32)

        # 2. Оцениваем Q-значение этого выбранного действия (a') с помощью TARGET-сети для состояния (S').
        q_values_next_target_net = self.target_q_network(next_states_tf, training=False)
        
        # Создаем one-hot маску для выбора Q-значений, соответствующих best_actions_next_state
        best_actions_one_hot = tf.one_hot(best_actions_next_state, self.action_size, dtype=tf.float32)
        # Q_target(S', a') = Q_target(S', argmax_a'' Q_main(S', a''))
        q_value_for_best_action_from_target = tf.reduce_sum(q_values_next_target_net * best_actions_one_hot, axis=1)
        # --- End Double DQN Logic ---

        # Целевые Q-значения: R + γ * Q_target_DDQN(S', a') для нетерминальных состояний
        # dones_tf должен быть float (0.0 или 1.0) для корректного умножения
        target_q_values = rewards_tf + self.gamma * q_value_for_best_action_from_target * (1.0 - tf.cast(dones_tf, tf.float32))

        # Маска для выбора Q-значений только для тех действий, которые были совершены в states_tf
        actions_current_one_hot = tf.one_hot(actions_tf, self.action_size, dtype=tf.float32)

        with tf.GradientTape() as tape:
            # Q-значения для текущих состояний, предсказанные основной сетью
            q_values_current_main_net = self.q_network(states_tf, training=True) # training=True для обучения
            # Выбираем Q-значения для реально совершенных действий
            q_action_taken = tf.reduce_sum(q_values_current_main_net * actions_current_one_hot, axis=1)
            # Потери между предсказанными Q-значениями (для совершенных действий) и целевыми Q-значениями
            loss = self.loss_function(target_q_values, q_action_taken)

        # Вычисление и применение градиентов
        gradients = tape.gradient(loss, self.q_network.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.q_network.trainable_variables))
        
        return loss

    def train_step(self, experiences_batch: tuple) -> float | None:
        """
        Выполняет один шаг обучения на батче опыта.

        :param experiences_batch: Кортеж (states, actions, rewards, next_states, dones),
                                  каждый элемент - numpy массив.
        :return: Значение функции потерь (float) или None в случае ошибки.
        """
        states, actions, rewards, next_states, dones = experiences_batch

        # Преобразование numpy массивов в тензоры TensorFlow
        try:
            states_tf = tf.convert_to_tensor(states, dtype=tf.float32)
            # Убедимся, что actions_tf правильного типа для tf.one_hot
            actions_tf = tf.convert_to_tensor(actions, dtype=tf.int32)
            rewards_tf = tf.convert_to_tensor(rewards, dtype=tf.float32)
            next_states_tf = tf.convert_to_tensor(next_states, dtype=tf.float32)
            dones_tf = tf.convert_to_tensor(dones, dtype=tf.bool)
        except Exception as e:
            logger.error(f"Ошибка конвертации данных батча в тензоры: {e}")
            return None

        try:
            loss_tensor = self._perform_train_step(states_tf, actions_tf, rewards_tf, next_states_tf, dones_tf)
            loss_value = loss_tensor.numpy() # Получаем скалярное значение потерь
        except Exception as e:
            logger.error(f"Ошибка во время выполнения шага обучения (_perform_train_step): {e}")
            logger.debug(traceback.format_exc())
            return None

        self._train_step_counter += 1
        # Обновляем target-сеть, если пришло время
        if self._train_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss_value

    def decay_epsilon_episode(self):
        """
        Уменьшает epsilon после завершения эпизода.
        """
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
            self.epsilon = max(self.epsilon_min, self.epsilon) # Гарантируем, что не упадет ниже минимума
        logger.debug(f"Epsilon уменьшен до {self.epsilon:.4f} после эпизода.")

    def save_model(self, filepath_prefix: str):
        """
        Сохраняет веса Q-сети и Target Q-сети.

        :param filepath_prefix: Префикс пути для файлов весов (без расширения и суффикса сети).
        """
        try:
            q_path = f"{filepath_prefix}_q_network.weights.h5"
            target_q_path = f"{filepath_prefix}_target_q_network.weights.h5"
            self.q_network.save_weights(q_path)
            self.target_q_network.save_weights(target_q_path)
            logger.info(f"Веса моделей сохранены: Q-Net='{q_path}', Target-Q-Net='{target_q_path}'")
        except Exception as e:
            logger.error(f"Ошибка при сохранении весов моделей (префикс: {filepath_prefix}): {e}")

    def load_model(self, filepath_prefix: str):
        """
        Загружает веса для Q-сети и синхронизирует Target Q-сеть.

        :param filepath_prefix: Префикс пути к файлам весов (ожидается, что есть файл {prefix}_q_network.weights.h5).
        """
        try:
            q_path = f"{filepath_prefix}_q_network.weights.h5"
            # target_q_path = f"{filepath_prefix}_target_q_network.weights.h5" # Можно и target загружать, но синхронизация проще

            if os.path.exists(q_path): # Используем os.path.exists для проверки
                self.q_network.load_weights(q_path)
                self.update_target_network() # Синхронизируем target-сеть с загруженной q-сетью
                logger.info(f"Веса Q-сети успешно загружены из: '{q_path}'. Target-сеть синхронизирована.")
            else:
                logger.warning(f"Файл весов Q-сети не найден по пути: '{q_path}'. Модель не загружена.")
        except Exception as e:
            logger.error(f"Ошибка при загрузке весов моделей (префикс: {filepath_prefix}): {e}")
            logger.debug(traceback.format_exc())

# Пример использования (для тестирования самого агента)
if __name__ == '__main__':
    logger.info("--- Тестирование DQNAgent с логикой Double DQN ---")
    # Для отладки функций, обернутых в @tf.function, можно временно включить eager execution:
    # tf.config.run_functions_eagerly(True)

    test_state_shape_main = (5,) # Простой вектор состояния для теста
    test_action_size_main = 3    # Например, Hold, Buy, Sell

    try:
        agent_test = DQNAgent(state_shape=test_state_shape_main,
                              action_size=test_action_size_main,
                              learning_rate=0.001,
                              gamma=0.9,
                              epsilon=0.5,
                              epsilon_decay=0.99,
                              epsilon_min=0.01,
                              target_update_freq=5) # Обновляем target часто для теста
        logger.info("DQNAgent для теста успешно создан.")
        agent_test.q_network.summary(print_fn=logger.info) # Выведем структуру сети
    except Exception as e_init:
        logger.error(f"Ошибка при создании DQNAgent в тесте: {e_init}", exc_info=True)
        exit()

    # --- Тест get_action ---
    logger.info("\n--- Тест get_action ---")
    dummy_state_test = np.random.rand(*test_state_shape_main).astype(np.float32)
    for _ in range(5):
        action_taken = agent_test.get_action(dummy_state_test)
        logger.info(f"  get_action: состояние (форма {dummy_state_test.shape}), выбранное действие: {action_taken}")

    # --- Тест train_step ---
    logger.info("\n--- Тест train_step (с Double DQN) ---")
    batch_size_for_test = 4
    # Генерируем батч случайных данных
    states_b = np.random.rand(batch_size_for_test, *test_state_shape_main).astype(np.float32)
    actions_b = np.random.randint(0, test_action_size_main, size=batch_size_for_test)
    rewards_b = np.random.rand(batch_size_for_test).astype(np.float32) * 10 - 5 # Награды от -5 до 5
    next_states_b = np.random.rand(batch_size_for_test, *test_state_shape_main).astype(np.float32)
    dones_b = np.random.choice([True, False], size=batch_size_for_test, p=[0.2, 0.8]) # 20% терминальных
    test_batch = (states_b, actions_b, rewards_b, next_states_b, dones_b)

    # Несколько шагов обучения для проверки
    for i in range(10): # Сделаем 10 шагов обучения
        loss_val = agent_test.train_step(test_batch)
        if loss_val is not None:
            logger.info(f"  Шаг обучения {i+1}, Потери: {loss_val:.4f}, "
                        f"Счетчик шагов обучения: {agent_test._train_step_counter}, Epsilon: {agent_test.epsilon:.3f}")
        else:
            logger.error(f"  Шаг обучения {i+1}: train_step вернул None (произошла ошибка).")
        # Epsilon decay после эпизода, здесь для теста можно вызвать вручную, если хотим видеть его изменение
        # agent_test.decay_epsilon_episode() # Если бы это был конец эпизода

    # --- Тест сохранения и загрузки модели ---
    logger.info("\n--- Тест сохранения/загрузки модели (после нескольких шагов обучения) ---")
    model_test_file_prefix = "test_ddqn_agent_weights" # Без пути, сохранится в текущей директории
    agent_test.save_model(model_test_file_prefix)

    # Создаем нового агента и загружаем веса
    agent_loaded_test = DQNAgent(state_shape=test_state_shape_main, action_size=test_action_size_main)
    agent_loaded_test.load_model(model_test_file_prefix)

    # Сравним часть весов для проверки (например, первый слой)
    weights_original_q = agent_test.q_network.get_weights()[0]
    weights_loaded_q = agent_loaded_test.q_network.get_weights()[0]
    weights_original_target = agent_test.target_q_network.get_weights()[0]
    weights_loaded_target = agent_loaded_test.target_q_network.get_weights()[0]

    if np.allclose(weights_original_q, weights_loaded_q) and np.allclose(weights_original_target, weights_loaded_target):
        logger.info("  Тест сохранения/загрузки: УСПЕХ. Веса Q-сети и Target-Q-сети совпадают после загрузки.")
    else:
        logger.error("  Тест сохранения/загрузки: ОШИБКА. Веса не совпадают.")
        if not np.allclose(weights_original_q, weights_loaded_q): logger.error("    Веса Q-сети отличаются.")
        if not np.allclose(weights_original_target, weights_loaded_target): logger.error("    Веса Target-Q-сети отличаются.")


    # Очистка временных файлов после теста
    for suffix in ["_q_network.weights.h5", "_target_q_network.weights.h5"]:
        test_file_path = f"{model_test_file_prefix}{suffix}"
        if os.path.exists(test_file_path):
            try: os.remove(test_file_path); logger.debug(f"Удален тестовый файл: {test_file_path}")
            except Exception as e_del: logger.warning(f"Не удалось удалить тестовый файл {test_file_path}: {e_del}")

    logger.info("--- Тестирование DQNAgent (с Double DQN) завершено ---")