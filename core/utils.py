import logging
import sys
import os # Для создания директории логов

def setup_logging(log_level_numeric: int = logging.INFO,
                  log_level_string: str = "INFO", # Для вывода в лог
                  log_file: str = None,
                  log_to_console: bool = True,
                  logger_name: str = None, # Если None, настраивается корневой логгер
                  clear_existing_handlers: bool = True
                  ) -> logging.Logger:
    """
    Настраивает систему логирования.

    :param log_level_numeric: Числовой уровень логирования (e.g., logging.INFO, logging.DEBUG).
    :param log_level_string: Строковое представление уровня логирования для вывода.
    :param log_file: Путь к файлу логов. Если None, логирование в файл не производится.
    :param log_to_console: Если True, логи выводятся в консоль.
    :param logger_name: Имя логгера для настройки. Если None, настраивается корневой логгер.
                        Это позволяет настроить特定ный логгер, не затрагивая другие.
    :param clear_existing_handlers: Если True, удаляет существующие обработчики у настраиваемого логгера.
                                    Полезно для предотвращения дублирования вывода при многократных вызовах.
    :return: Настроенный экземпляр логгера.
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger() # Корневой логгер

    logger.setLevel(log_level_numeric)

    if clear_existing_handlers:
        # Удаляем существующие обработчики, чтобы избежать дублирования
        if logger.hasHandlers():
            logger.handlers.clear()
            # print(f"DEBUG: Cleared handlers for logger '{logger.name}'") # Отладочный вывод

    # Определение форматов
    # Более детальный формат для файла
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(funcName)s - %(message)s'
    )
    # Более короткий формат для консоли
    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s'
    )

    # Обработчик для вывода в консоль
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout) # Явно указываем sys.stdout
        console_handler.setFormatter(console_formatter)
        console_handler.setLevel(log_level_numeric) # Уровень для обработчика тоже важен
        logger.addHandler(console_handler)

    # Обработчик для вывода в файл
    if log_file:
        try:
            # Создаем директорию для лог-файла, если она не существует
            log_dir = os.path.dirname(log_file)
            if log_dir and not os.path.exists(log_dir): # Проверяем, что log_dir не пустая строка (если файл в текущей дир)
                os.makedirs(log_dir, exist_ok=True)
                # print(f"DEBUG: Created log directory '{log_dir}'") # Отладочный вывод

            file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
            file_handler.setFormatter(file_formatter)
            file_handler.setLevel(log_level_numeric) # Уровень для обработчика
            logger.addHandler(file_handler)
            logger.info(f"Логирование в файл настроено: {os.path.abspath(log_file)}, Уровень: {log_level_string}")
        except Exception as e:
            # Если не удалось настроить файловый лог, выводим ошибку в консоль (если она есть)
            # или используем print, если консольный логгер еще не добавлен.
            error_message = f"Не удалось создать файловый обработчик логов для {log_file}: {e}"
            if log_to_console and logger.hasHandlers(): # Проверяем, есть ли уже консольный обработчик
                logger.error(error_message)
            else:
                print(f"ERROR (setup_logging): {error_message}", file=sys.stderr)


    if not log_to_console and not log_file:
        logger.warning("Логирование не настроено ни в консоль, ни в файл.")
    elif logger.name == '' or logger.name == 'root': # Для корневого логгера
        logger.info(f"Корневой логгер настроен. Уровень: {log_level_string}.")
    else:
        logger.info(f"Логгер '{logger.name}' настроен. Уровень: {log_level_string}.")

    # Подавление слишком "шумных" логгеров от сторонних библиотек (например, ccxt)
    # Можно установить им уровень WARNING или ERROR
    # logging.getLogger('ccxt').setLevel(logging.WARNING)

    return logger


def safe_float_convert(value, default_value: float = 0.0) -> float | None:
    """
    Безопасно преобразует значение в float.
    Если преобразование невозможно, логирует предупреждение и возвращает default_value.
    Если default_value is None, то при ошибке вернет None.

    :param value: Значение для преобразования.
    :param default_value: Значение, возвращаемое по умолчанию при ошибке преобразования.
                          Если default_value is None, то при ошибке вернет None.
    :return: Преобразованное float значение, default_value или None.
    """
    if value is None:
        # Если default_value тоже None, то логично вернуть None
        # Если default_value число, то возвращаем его
        return default_value

    try:
        return float(value)
    except (ValueError, TypeError) as e:
        # Логируем только если это не просто пустая строка (которая может быть ожидаемой)
        # или если значение не None (уже обработано)
        message = None
        if isinstance(value, str) and value.strip(): # Непустая строка
            message = f"Не удалось преобразовать строку '{value}' в float: {e}."
        elif not isinstance(value, (str, int, float, type(None))): # Неожиданный тип
            message = f"Не удалось преобразовать значение '{value}' (тип: {type(value)}) в float: {e}."

        if message:
            # Используем getLogger(__name__) чтобы не зависеть от глобального логгера,
            # который может быть еще не настроен при вызове этой функции.
            # Но если setup_logging уже был вызван, он подхватит настройки.
            logging.getLogger(__name__).warning(message + f" Используется значение: {default_value}.")
        return default_value


# Пример использования (не будет выполняться при импорте)
if __name__ == '__main__':
    # Настройка тестового логгера для вывода работы utils
    utils_logger = logging.getLogger("TestUtils") # Даем имя, чтобы не конфликтовать с корневым
    utils_logger.setLevel(logging.DEBUG)
    _ch = logging.StreamHandler()
    _cf = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _ch.setFormatter(_cf)
    if not utils_logger.hasHandlers():
        utils_logger.addHandler(_ch)

    utils_logger.info("--- Тестирование setup_logging ---")

    # 1. Настройка корневого логгера в файл и консоль
    test_log_file = "temp_utils_test.log"
    # Сначала удалим старый файл, если он есть, для чистоты теста
    if os.path.exists(test_log_file):
        os.remove(test_log_file)

    # Настраиваем корневой логгер
    root_logger = setup_logging(log_level_numeric=logging.DEBUG,
                                log_level_string="DEBUG",
                                log_file=test_log_file,
                                log_to_console=True,
                                clear_existing_handlers=True) # Важно для тестов

    root_logger.debug("Это DEBUG сообщение от корневого логгера (должно быть в консоли и файле).")
    root_logger.info("Это INFO сообщение от корневого логгера.")
    logging.warning("Это WARNING сообщение через logging.warning (использует корневой логгер).")

    # Проверим, что в файле есть записи
    if os.path.exists(test_log_file):
        with open(test_log_file, 'r') as f:
            content = f.read()
            if "DEBUG сообщение" in content and "INFO сообщение" in content:
                utils_logger.info(f"Логи успешно записаны в файл {test_log_file}")
            else:
                utils_logger.error(f"Проблема с записью логов в файл {test_log_file}")
        os.remove(test_log_file) # Удаляем после теста
    else:
        utils_logger.error(f"Файл логов {test_log_file} не был создан.")


    # 2. Настройка именованного логгера (не затрагивая корневой)
    # (Сначала нужно "сбросить" корневой, чтобы он не мешал, или настроить ему более высокий уровень)
    # logging.getLogger().handlers.clear() # Очистим корневой для чистоты эксперимента
    # setup_logging(log_level_numeric=logging.CRITICAL, log_to_console=False) # "Заглушим" корневой

    # utils_logger.info("\n--- Тестирование именованного логгера ---")
    # module_logger_name = "MyModuleLogger"
    # my_module_logger = setup_logging(logger_name=module_logger_name,
    #                                  log_level_numeric=logging.INFO,
    #                                  log_level_string="INFO",
    #                                  log_to_console=True,
    #                                  clear_existing_handlers=True)
    # my_module_logger.info(f"Это сообщение от логгера '{module_logger_name}'.")
    # logging.getLogger("AnotherModule").info("Сообщение от другого модуля (должно использовать настройки корневого, если он есть, или не выводиться).")


    utils_logger.info("\n--- Тестирование safe_float_convert ---")
    test_values = [
        "123.45", 123, 0, "0.0", "-10.5",
        "abc", "", None, "  ", "1,234.56" # с запятой
    ]
    for val in test_values:
        result = safe_float_convert(val)
        utils_logger.info(f"safe_float_convert('{val if val is not None else 'None'}') -> {result} (тип: {type(result)})")

    utils_logger.info("Тест safe_float_convert с default_value=None:")
    for val in ["xyz", None]:
        result_none_default = safe_float_convert(val, default_value=None)
        utils_logger.info(f"safe_float_convert('{val if val is not None else 'None'}', default=None) -> {result_none_default} (тип: {type(result_none_default)})")


    utils_logger.info("Тестирование utils.py завершено.")