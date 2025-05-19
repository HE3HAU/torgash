import logging

class RuleBasedAgent:
    """
    Простой агент, принимающий решения на основе заданных ценовых порогов.
    """
    def __init__(self, buy_threshold: float, sell_threshold: float, logger_instance: logging.Logger = None):
        """
        Инициализирует агента.

        :param buy_threshold: Нижний ценовой порог для покупки.
        :param sell_threshold: Верхний ценовой порог для продажи.
        :param logger_instance: Экземпляр логгера для логирования действий агента.
        """
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.logger = logger_instance if logger_instance else logging.getLogger(__name__)

        if self.buy_threshold >= self.sell_threshold:
            self.logger.warning(
                f"Порог покупки ({self.buy_threshold}) выше или равен порогу продажи ({self.sell_threshold}). "
                "Это может привести к некорректной логике."
            )
        self.logger.info(
            f"RuleBasedAgent инициализирован: покупка если цена < {self.buy_threshold}, "
            f"продажа если цена > {self.sell_threshold}"
        )

    def get_decision(self, current_price: float) -> str:
        """
        Принимает решение о действии на основе текущей цены.

        :param current_price: Текущая рыночная цена торговой пары.
        :return: Строка с решением: 'buy', 'sell', или 'hold'.
        """
        if current_price is None:
            self.logger.warning("Агент: Текущая цена не определена (None), решение 'hold'.")
            return 'hold'

        if not isinstance(current_price, (int, float)):
            self.logger.error(f"Агент: Текущая цена имеет неверный тип ({type(current_price)}), решение 'hold'.")
            return 'hold'

        if current_price <= 0:
            self.logger.warning(f"Агент: Текущая цена не положительна ({current_price}), решение 'hold'.")
            return 'hold'

        if current_price < self.buy_threshold:
            self.logger.info(f"Агент: Решение 'buy' при цене {current_price} (порог < {self.buy_threshold})")
            return 'buy'
        elif current_price > self.sell_threshold:
            self.logger.info(f"Агент: Решение 'sell' при цене {current_price} (порог > {self.sell_threshold})")
            return 'sell'
        else:
            self.logger.debug(f"Агент: Решение 'hold' при цене {current_price} (между {self.buy_threshold} и {self.sell_threshold})")
            return 'hold'

# Пример использования (не будет выполняться при импорте этого модуля):
if __name__ == '__main__':
    # Настройка простого логгера для теста
    test_logger = logging.getLogger("TestAgent")
    test_logger.setLevel(logging.DEBUG)
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    if not test_logger.hasHandlers(): # Предотвращаем дублирование обработчиков при многократном запуске скрипта
        test_logger.addHandler(console_handler)

    # Создание агента
    agent = RuleBasedAgent(buy_threshold=30000.0, sell_threshold=60000.0, logger_instance=test_logger)

    # Тестирование решений
    test_prices = [25000.0, 30000.0, 45000.0, 60000.0, 65000.0, None, -100, "invalid_price"]
    for price in test_prices:
        decision = agent.get_decision(price)
        # Логирование уже происходит внутри get_decision и в __init__
        # test_logger.info(f"При цене {price}, агент решил: {decision}\n") # Это сообщение будет дублировать лог из get_decision