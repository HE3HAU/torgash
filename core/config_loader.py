import configparser
import os
import logging
from dotenv import load_dotenv

# Используем логгер этого модуля. Он будет настроен позже основным логгером,
# но может выводить сообщения, если что-то пойдет не так на раннем этапе.
logger = logging.getLogger(__name__)

# --- Базовая конфигурация логгера на случай, если этот модуль используется до полной настройки ---
# Это полезно для отладки самого config_loader, если он вызывается очень рано.
if not logger.hasHandlers():
    _temp_handler = logging.StreamHandler()
    _temp_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - [CONFIG_LOADER] %(message)s')
    _temp_handler.setFormatter(_temp_formatter)
    logger.addHandler(_temp_handler)
    logger.setLevel(logging.INFO) # Установим INFO, чтобы видеть сообщения о загрузке

# --- Функции для работы с конфигурацией ---

def load_dotenv_file(dotenv_path: str = None, verbose: bool = True) -> bool:
    """
    Загружает переменные из .env файла.

    :param dotenv_path: Путь к .env файлу. Если None, ищется .env в текущей директории.
    :param verbose: Если True, логирует информацию о загрузке.
    :return: True, если файл .env был найден и загружен, иначе False.
    """
    original_dotenv_path = dotenv_path
    if dotenv_path is None:
        # По умолчанию ищем .env в директории, где запущен скрипт,
        # или на один уровень выше (если скрипт в подпапке типа 'core')
        # Это более гибко, чем os.path.join(os.path.dirname(__file__), '.env'),
        # так как __file__ будет указывать на местоположение config_loader.py
        potential_paths = [
            os.path.join(os.getcwd(), '.env'),
            os.path.join(os.path.dirname(os.getcwd()), '.env') # Для случая запуска из scripts/ и т.п.
        ]
        # Добавим путь относительно самого config_loader.py, если он в корне проекта
        script_dir_env = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '.env')
        if os.path.abspath(script_dir_env) not in [os.path.abspath(p) for p in potential_paths]:
             potential_paths.append(script_dir_env)


        found_path = None
        for p_path in potential_paths:
            abs_p_path = os.path.abspath(p_path)
            if verbose:
                logger.debug(f"Проверка пути к .env: {abs_p_path}")
            if os.path.exists(abs_p_path):
                found_path = abs_p_path
                break
        dotenv_path = found_path

    if dotenv_path and os.path.exists(dotenv_path):
        loaded = load_dotenv(dotenv_path=dotenv_path, override=True) # override=True позволяет перезаписать системные переменные
        if loaded and verbose:
            logger.info(f"Переменные окружения успешно загружены из: {os.path.abspath(dotenv_path)}")
        elif not loaded and verbose:
            logger.warning(f"Файл .env найден по пути {os.path.abspath(dotenv_path)}, но load_dotenv() вернул False (возможно, пустой файл или проблема с правами).")
        return loaded
    else:
        if verbose:
            if original_dotenv_path:
                logger.warning(f".env файл не найден по указанному пути: {original_dotenv_path}.")
            else:
                logger.warning(f".env файл не найден в стандартных расположениях. API ключи и другие секреты должны быть установлены как переменные окружения ОС.")
        return False


def load_ini_config(config_file_path: str) -> configparser.ConfigParser | None:
    """
    Загружает конфигурацию из .ini файла.

    :param config_file_path: Путь к .ini файлу.
    :return: Объект ConfigParser или None в случае ошибки.
    """
    config = configparser.ConfigParser()
    if not os.path.exists(config_file_path):
        logger.error(f"Файл конфигурации {config_file_path} не найден!")
        return None
    try:
        # Читаем файл с явным указанием кодировки UTF-8
        with open(config_file_path, 'r', encoding='utf-8') as f:
            config.read_file(f)
        logger.info(f"Конфигурация успешно загружена из {config_file_path}")
        return config
    except UnicodeDecodeError as e_unicode:
        logger.error(f"Ошибка кодировки при чтении файла {config_file_path}. "
                       f"Убедитесь, что файл сохранен в UTF-8. Ошибка: {e_unicode}")
    except configparser.Error as e_parser:
        logger.error(f"Ошибка парсинга конфигурационного файла {config_file_path}: {e_parser}")
    except Exception as e:
        logger.error(f"Не удалось загрузить конфигурацию из {config_file_path}. Непредвиденная ошибка: {e}")
    return None


def get_api_keys_env_names(config: configparser.ConfigParser) -> dict:
    """
    Извлекает имена переменных окружения для API ключей из объекта ConfigParser.

    :param config: Загруженный объект ConfigParser.
    :return: Словарь с именами переменных окружения.
    """
    defaults = {
        'bybit_api_key_env': 'BYBIT_TESTNET_API_KEY',
        'bybit_api_secret_env': 'BYBIT_TESTNET_API_SECRET',
        'okx_api_key_env': 'OKX_TESTNET_API_KEY',
        'okx_api_secret_env': 'OKX_TESTNET_API_SECRET',
        'okx_api_password_env': 'OKX_TESTNET_API_PASSWORD',
    }
    if not config or 'API_KEYS_ENV_VARS' not in config:
        logger.warning("Секция [API_KEYS_ENV_VARS] отсутствует в конфигурации. "
                       "Будут использованы имена переменных окружения по умолчанию для API ключей.")
        return defaults

    section = config['API_KEYS_ENV_VARS']
    return {
        'bybit_api_key_env': section.get('BYBIT_API_KEY_ENV', defaults['bybit_api_key_env']),
        'bybit_api_secret_env': section.get('BYBIT_API_SECRET_ENV', defaults['bybit_api_secret_env']),
        'okx_api_key_env': section.get('OKX_API_KEY_ENV', defaults['okx_api_key_env']),
        'okx_api_secret_env': section.get('OKX_API_SECRET_ENV', defaults['okx_api_secret_env']),
        'okx_api_password_env': section.get('OKX_API_PASSWORD_ENV', defaults['okx_api_password_env']),
    }


def get_bot_settings(config: configparser.ConfigParser) -> dict:
    """
    Извлекает настройки бота из объекта ConfigParser.

    :param config: Загруженный объект ConfigParser.
    :return: Словарь с настройками бота.
    """
    defaults = {
        'exchange_name': 'bybit',
        'trading_pair': 'BTC/USDT',
        'buy_price_threshold': 30000.0,
        'sell_price_threshold': 68000.0,
        'order_amount_base_currency': 0.0001,
        'order_check_interval_seconds': 15,
        'main_loop_pause_seconds': 30,
    }
    if not config or 'BOT_SETTINGS' not in config:
        logger.error("Секция [BOT_SETTINGS] отсутствует в конфигурации! "
                     "Будут использованы настройки по умолчанию. Это может быть нежелательно.")
        return defaults # Возвращаем значения по умолчанию, но с ошибкой, т.к. секция важна

    section = config['BOT_SETTINGS']
    try:
        return {
            'exchange_name': section.get('EXCHANGE_NAME', defaults['exchange_name']).lower(),
            'trading_pair': section.get('TRADING_PAIR', defaults['trading_pair']).upper(),
            'buy_price_threshold': section.getfloat('BUY_PRICE_THRESHOLD', defaults['buy_price_threshold']),
            'sell_price_threshold': section.getfloat('SELL_PRICE_THRESHOLD', defaults['sell_price_threshold']),
            'order_amount_base_currency': section.getfloat('ORDER_AMOUNT_BASE_CURRENCY', defaults['order_amount_base_currency']),
            'order_check_interval_seconds': section.getint('ORDER_CHECK_INTERVAL_SECONDS', defaults['order_check_interval_seconds']),
            'main_loop_pause_seconds': section.getint('MAIN_LOOP_PAUSE_SECONDS', defaults['main_loop_pause_seconds']),
        }
    except ValueError as e:
        logger.error(f"Ошибка преобразования значения в секции [BOT_SETTINGS]: {e}. "
                       "Проверьте типы данных в config.ini. Будут использованы значения по умолчанию для проблемных ключей или всей секции.")
        # Можно вернуть defaults или попытаться спарсить оставшиеся и заменить только проблемные
        return defaults # Для простоты вернем все по умолчанию при ошибке


def get_logging_settings(config: configparser.ConfigParser) -> dict:
    """
    Извлекает настройки логирования из объекта ConfigParser.

    :param config: Загруженный объект ConfigParser.
    :return: Словарь с настройками логирования.
    """
    defaults = {
        'log_file': 'trading_bot.log',
        'log_level_str': 'INFO',
    }
    if not config or 'LOGGING' not in config:
        logger.warning("Секция [LOGGING] отсутствует в конфигурации. "
                       "Будут использованы настройки логирования по умолчанию.")
        settings = defaults
    else:
        section = config['LOGGING']
        settings = {
            'log_file': section.get('LOG_FILE', defaults['log_file']),
            'log_level_str': section.get('LOG_LEVEL', defaults['log_level_str']).upper(),
        }

    # Преобразуем строку уровня логирования в объект logging уровня
    log_level = getattr(logging, settings['log_level_str'], None)
    if not isinstance(log_level, int): # Проверка, что это валидный уровень логирования
        logger.warning(f"Некорректный уровень логирования '{settings['log_level_str']}' в конфигурации. "
                       f"Используется INFO по умолчанию.")
        log_level = logging.INFO
        settings['log_level_str'] = 'INFO' # Обновляем и строку для консистентности

    return {
        'log_file': settings['log_file'],
        'log_level_numeric': log_level, # Числовое значение уровня
        'log_level_string': settings['log_level_str'] # Строковое представление
    }


# Пример использования (не будет выполняться при импорте)
if __name__ == '__main__':
    # logger.setLevel(logging.DEBUG) # Для более детального вывода при тестировании этого модуля

    # --- Тестирование загрузки .env ---
    # Создадим временный .env для теста, если его нет
    temp_env_path = ".test_env"
    if not os.path.exists(temp_env_path):
        with open(temp_env_path, "w") as f:
            f.write("TEST_ENV_VAR_LOADER=\"Это тестовая переменная из .test_env\"\n")
            f.write("BYBIT_TESTNET_API_KEY=\"dummy_key_from_test_env\"\n")

    logger.info("\n--- Тест load_dotenv_file ---")
    load_dotenv_file(dotenv_path=temp_env_path) # Указываем путь явно
    test_var = os.getenv("TEST_ENV_VAR_LOADER")
    logger.info(f"Значение TEST_ENV_VAR_LOADER: {test_var}")
    bybit_key_test = os.getenv("BYBIT_TESTNET_API_KEY")
    logger.info(f"Значение BYBIT_TESTNET_API_KEY из .env: {bybit_key_test}")

    # Тест без указания пути (должен искать .test_env, если он в os.getcwd())
    # Для этого теста нужно, чтобы .test_env был в текущей рабочей директории
    # load_dotenv_file()
    # ...

    # --- Тестирование загрузки .ini ---
    temp_ini_path = "test_config.ini"
    config_content = """
[API_KEYS_ENV_VARS]
BYBIT_API_KEY_ENV = CUSTOM_BYBIT_KEY_NAME
BYBIT_API_SECRET_ENV = CUSTOM_BYBIT_SECRET_NAME

[BOT_SETTINGS]
EXCHANGE_NAME = bybit
TRADING_PAIR = ETH/USDT
BUY_PRICE_THRESHOLD = 2000.0
# SELL_PRICE_THRESHOLD = 4000.0 ; Закомментированный параметр для теста default
ORDER_AMOUNT_BASE_CURRENCY = 0.01
ORDER_CHECK_INTERVAL_SECONDS = 10
MAIN_LOOP_PAUSE_SECONDS = 20

[LOGGING]
LOG_FILE = test_bot.log
LOG_LEVEL = DEBUG
    """
    with open(temp_ini_path, "w", encoding="utf-8") as f:
        f.write(config_content)

    logger.info("\n--- Тест load_ini_config ---")
    loaded_config = load_ini_config(temp_ini_path)

    if loaded_config:
        logger.info("\n--- Тест get_api_keys_env_names ---")
        api_vars = get_api_keys_env_names(loaded_config)
        logger.info(f"Имена переменных для API ключей: {api_vars}")
        # Предположим, что CUSTOM_BYBIT_KEY_NAME установлен в .env или ОС
        os.environ['CUSTOM_BYBIT_KEY_NAME'] = "actual_custom_key_value"
        logger.info(f"Ключ Bybit (по имени из config): {os.getenv(api_vars['bybit_api_key_env'])}")


        logger.info("\n--- Тест get_bot_settings ---")
        bot_settings = get_bot_settings(loaded_config)
        logger.info(f"Настройки бота: {bot_settings}")
        if bot_settings.get('sell_price_threshold') == 68000.0: # Проверка значения по умолчанию
            logger.info("SELL_PRICE_THRESHOLD корректно взят по умолчанию.")


        logger.info("\n--- Тест get_logging_settings ---")
        logging_settings = get_logging_settings(loaded_config)
        logger.info(f"Настройки логирования: {logging_settings}")
        if logging_settings['log_level_numeric'] == logging.DEBUG:
            logger.info("Уровень логирования DEBUG корректно установлен.")
    else:
        logger.error("Не удалось загрузить тестовый config.ini для проведения тестов функций.")

    # Очистка временных файлов
    if os.path.exists(temp_env_path):
        os.remove(temp_env_path)
    if os.path.exists(temp_ini_path):
        os.remove(temp_ini_path)