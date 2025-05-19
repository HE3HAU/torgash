import collections
import random
import numpy as np
import logging

logger = logging.getLogger(__name__)

class ReplayBuffer:
    """
    Простой буфер воспроизведения опыта для алгоритмов обучения с подкреплением.
    Хранит кортежи (state, action, reward, next_state, done).
    """
    def __init__(self, capacity: int):
        """
        Инициализирует буфер.

        :param capacity: Максимальная вместимость буфера.
                         Старые записи будут удаляться при переполнении.
        """
        if capacity <= 0:
            raise ValueError("Вместимость буфера (capacity) должна быть положительным числом.")
        self.capacity = capacity
        self.buffer = collections.deque(maxlen=self.capacity)
        logger.info(f"ReplayBuffer инициализирован с вместимостью {self.capacity}.")

    def add(self, state, action, reward: float, next_state, done: bool):
        """
        Добавляет новый опыт в буфер.

        :param state: Текущее состояние (наблюдение).
        :param action: Предпринятое действие.
        :param reward: Полученное вознаграждение.
        :param next_state: Следующее состояние (наблюдение) после действия.
        :param done: Флаг, указывающий, завершился ли эпизод после этого перехода.
        """
        experience = (state, action, reward, next_state, done)
        self.buffer.append(experience)
        logger.debug(f"В буфер добавлен опыт. Текущий размер: {len(self.buffer)}/{self.capacity}")

    def sample(self, batch_size: int):
        """
        Извлекает случайную выборку (батч) опыта из буфера.

        :param batch_size: Размер выборки.
        :return: Кортеж из пяти numpy массивов:
                 (states, actions, rewards, next_states, dones).
                 Каждый массив содержит batch_size элементов.
        :raises ValueError: Если batch_size больше текущего количества элементов в буфере.
        """
        if batch_size <= 0:
            raise ValueError("Размер выборки (batch_size) должен быть положительным числом.")
        if batch_size > len(self.buffer):
            raise ValueError(f"Размер выборки ({batch_size}) не может быть больше "
                             f"текущего количества элементов в буфере ({len(self.buffer)}).")

        batch = random.sample(self.buffer, batch_size)

        # Разделяем батч на отдельные компоненты
        # Используем np.array для удобства работы с данными в нейронных сетях
        try:
            states, actions, rewards, next_states, dones = zip(*batch)
        except ValueError as e: # Может произойти, если batch пустой (хотя проверка выше должна это предотвратить)
            logger.error(f"Ошибка при распаковке батча: {e}. Батч: {batch}")
            return None, None, None, None, None # или возбудить исключение

        # Преобразуем в numpy массивы
        # Для состояний и следующих состояний важно, чтобы они были массивами нужной формы
        # Предполагается, что state и next_state - это numpy массивы или списки, которые можно в них преобразовать
        states_np = np.array(states, dtype=np.float32)
        # Действия могут быть скалярами (например, для Discrete action space)
        actions_np = np.array(actions) # dtype будет определен автоматически, или можно задать (например, np.int64)
        rewards_np = np.array(rewards, dtype=np.float32)
        next_states_np = np.array(next_states, dtype=np.float32)
        dones_np = np.array(dones, dtype=np.bool_) # или np.uint8

        return states_np, actions_np, rewards_np, next_states_np, dones_np

    def __len__(self) -> int:
        """
        Возвращает текущее количество элементов в буфере.
        """
        return len(self.buffer)

    def is_ready(self, batch_size: int) -> bool:
        """
        Проверяет, достаточно ли в буфере опыта для формирования батча заданного размера.
        """
        return len(self.buffer) >= batch_size


# Пример использования (не будет выполняться при импорте)
if __name__ == '__main__':
    # Настройка простого логгера для теста
    test_logger = logging.getLogger("TestReplayBuffer")
    test_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    if not test_logger.hasHandlers():
        test_logger.addHandler(console_handler)
    logger.addHandler(console_handler) # Добавляем и к логгеру модуля

    buffer_capacity = 100
    batch_sz = 32
    state_shape = (4,) # Пример формы состояния

    try:
        replay_buffer = ReplayBuffer(capacity=buffer_capacity)
        test_logger.info(f"Длина буфера после создания: {len(replay_buffer)}")
        test_logger.info(f"Готов ли буфер для батча {batch_sz}? {replay_buffer.is_ready(batch_sz)}")

        # Добавление фейковых данных
        for i in range(buffer_capacity + 5): # Добавим чуть больше, чем capacity
            state = np.random.rand(*state_shape).astype(np.float32)
            action = random.randint(0, 2)
            reward = random.random()
            next_state = np.random.rand(*state_shape).astype(np.float32)
            done = random.choice([True, False])
            replay_buffer.add(state, action, reward, next_state, done)
            if (i + 1) % 10 == 0:
                 test_logger.info(f"Добавлено {i+1} элементов. Длина буфера: {len(replay_buffer)}")


        test_logger.info(f"Финальная длина буфера: {len(replay_buffer)} (должна быть {buffer_capacity})")
        test_logger.info(f"Готов ли буфер для батча {batch_sz}? {replay_buffer.is_ready(batch_sz)}")

        if replay_buffer.is_ready(batch_sz):
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_sz)
            test_logger.info(f"Пример выборки (батч размером {batch_sz}):")
            test_logger.info(f"  States shape: {states.shape}, dtype: {states.dtype}")
            test_logger.info(f"  Actions shape: {actions.shape}, dtype: {actions.dtype}, example: {actions[:3]}")
            test_logger.info(f"  Rewards shape: {rewards.shape}, dtype: {rewards.dtype}, example: {rewards[:3]}")
            test_logger.info(f"  Next States shape: {next_states.shape}, dtype: {next_states.dtype}")
            test_logger.info(f"  Dones shape: {dones.shape}, dtype: {dones.dtype}, example: {dones[:3]}")
        else:
            test_logger.warning(f"Недостаточно данных для выборки батча размером {batch_sz}")

        # Попытка сэмплировать больше, чем есть
        try:
            replay_buffer.sample(len(replay_buffer) + 1)
        except ValueError as e:
            test_logger.info(f"Успешно перехвачена ожидаемая ошибка: {e}")

        # Попытка создать буфер с неверной вместимостью
        try:
            ReplayBuffer(0)
        except ValueError as e:
            test_logger.info(f"Успешно перехвачена ожидаемая ошибка при создании буфера: {e}")


    except Exception as e:
        test_logger.error(f"Произошла ошибка в тесте: {e}", exc_info=True)