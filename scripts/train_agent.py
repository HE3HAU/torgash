import logging
import pandas as pd
import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import re # Для извлечения номера эпизода из имени файла

# ... (импорты ваших модулей ai, core) ...
# Добавляем корневую директорию проекта в PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
if project_root not in sys.path:
    sys.path.append(project_root)

from ai.dqn_agent import DQNAgent
from ai.environment import TradingEnvironment
from ai.replay_buffer import ReplayBuffer

# ... (настройка script_logger) ...
script_logger = logging.getLogger('train_agent_script')
if not script_logger.hasHandlers():
    script_logger.setLevel(logging.INFO)
    _ch = logging.StreamHandler(sys.stdout)
    _cf = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    _ch.setFormatter(_cf)
    script_logger.addHandler(_ch)


def load_training_data(filepath: str) -> pd.DataFrame | None:
    # ... (без изменений) ...
    if not os.path.exists(filepath): script_logger.error(f"Файл не найден: {filepath}"); return None
    try:
        df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        if 'close' not in df.columns: script_logger.error(f"Нет 'close' в {filepath}"); return None
        df['close'] = pd.to_numeric(df['close'], errors='coerce'); df.dropna(subset=['close'], inplace=True)
        script_logger.info(f"Данные загружены из {filepath}, строк: {len(df)}")
        return df
    except Exception as e: script_logger.error(f"Ошибка загрузки {filepath}: {e}"); return None


def plot_training_results(rewards_history: list, loss_history: list,
                          total_episodes_trained_overall: int, # Общее число эпизодов
                          plots_dir: str,
                          plot_filename_base: str, # Базовое имя файла для графиков
                          smoothing_window_factor: float = 0.05):
    # ... (без изменений, использует total_episodes_trained_overall) ...
    if not rewards_history: script_logger.warning("История наград пуста..."); return
    os.makedirs(plots_dir, exist_ok=True)
    plot_filename_suffix = f"_totaleps{total_episodes_trained_overall}.png"
    smoothing_window = max(1, int(len(rewards_history) * smoothing_window_factor))
    plt.figure(figsize=(14, 7)); plt.plot(rewards_history, label='Награда за эпизод (тек. сеанс)', alpha=0.6, color='steelblue')
    if len(rewards_history) >= smoothing_window:
        rewards_smoothed = pd.Series(rewards_history).rolling(window=smoothing_window, min_periods=1).mean()
        plt.plot(rewards_smoothed, label=f'Награда (SMA {smoothing_window}, тек. сеанс)', color='red', linewidth=2)
    plt.title(f'История наград ({plot_filename_base}, Всего обучено эп: {total_episodes_trained_overall})')
    plt.xlabel('Эпизод (в текущем сеансе обучения)'); plt.ylabel('Общая награда за эпизод'); plt.legend(); plt.grid(True)
    rewards_plot_path = os.path.join(plots_dir, f"rewards_{plot_filename_base}{plot_filename_suffix}")
    try: plt.savefig(rewards_plot_path); script_logger.info(f"График наград сохранен: {rewards_plot_path}")
    except Exception as e: script_logger.error(f"Ошибка сохранения графика наград: {e}")
    plt.close()
    if loss_history and not all(np.isnan(l) for l in loss_history if l is not None):
        valid_losses = [l for l in loss_history if l is not None and not np.isnan(l)]; valid_loss_indices = [i for i, l in enumerate(loss_history) if l is not None and not np.isnan(l)]
        if valid_losses:
            plt.figure(figsize=(14, 7)); plt.plot(valid_loss_indices, valid_losses, label='Средний лосс за эпизод (тек. сеанс)', alpha=0.6, color='darkorange')
            loss_smoothing_window = max(1, int(len(valid_losses) * smoothing_window_factor))
            if len(valid_losses) >= loss_smoothing_window:
                loss_smoothed = pd.Series(valid_losses).rolling(window=loss_smoothing_window, min_periods=1).mean()
                plt.plot(valid_loss_indices[:len(loss_smoothed)], loss_smoothed, label=f'Лосс (SMA {loss_smoothing_window}, тек. сеанс)', color='blue', linewidth=2)
            plt.title(f'История среднего лосса ({plot_filename_base}, Всего обучено эп: {total_episodes_trained_overall})')
            plt.xlabel('Эпизод (с валидным лоссом, в текущем сеансе)'); plt.ylabel('Средний лосс'); plt.legend(); plt.grid(True)
            loss_plot_path = os.path.join(plots_dir, f"loss_{plot_filename_base}{plot_filename_suffix}")
            try: plt.savefig(loss_plot_path); script_logger.info(f"График лосса сохранен: {loss_plot_path}")
            except Exception as e: script_logger.error(f"Ошибка сохранения графика лосса: {e}")
            plt.close()
        else: script_logger.warning("Нет валидных данных для графика лосса.")
    else: script_logger.warning("История лоссов пуста или NaN.")


def run_training_loop(env: TradingEnvironment, agent: DQNAgent, replay_buffer: ReplayBuffer,
                      num_episodes_this_run: int, start_episode_num_overall: int,
                      batch_size: int, save_model_freq: int,
                      model_save_path_template: str ) -> tuple[list, list]:
    # ... (без изменений, использует start_episode_num_overall для нумерации и сохранения) ...
    script_logger.info("="*50 + f"\nПродолжение/Начало цикла тренировки ({type(agent).__name__})" +
                       f"\nЭпизодов в этом запуске: {num_episodes_this_run}, Начиная с общего эп. №: {start_episode_num_overall}" +
                       f"\nБатч: {batch_size}, Сохр. модели каждые: {save_model_freq} общих эп." +
                       f"\nEpsilon start (для этого запуска): {agent.epsilon:.4f}, decay: {agent.epsilon_decay}, min: {agent.epsilon_min}" +
                       f"\nTarget update freq (train steps): {agent.target_update_freq}"+ "\n" + "="*50)
    total_steps_overall_session = 0; session_rewards_history = []; session_avg_loss_history = []
    for i in range(num_episodes_this_run):
        current_overall_episode_num = start_episode_num_overall + i
        state, info = env.reset(); episode_reward_sum = 0.0; episode_steps = 0; current_episode_losses = []
        terminated, truncated = False, False
        script_logger.info(f"\n--- Общий Эпизод {current_overall_episode_num} (Эпсилон: {agent.epsilon:.4f}) ---")
        while not terminated and not truncated:
            total_steps_overall_session += 1; episode_steps += 1; action = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward_sum += reward; done_flag = terminated or truncated
            replay_buffer.add(state, action, reward, next_state, done_flag)
            script_logger.debug(f"  Общ.Эп {current_overall_episode_num}, Шаг {episode_steps}: A={action}, R={reward:.4f}, Эп.R={episode_reward_sum:.2f}, Done={done_flag}, NetW: {info.get('net_worth',0):.2f}")
            if replay_buffer.is_ready(batch_size):
                experiences = replay_buffer.sample(batch_size)
                loss = agent.train_step(experiences)
                if loss is not None: current_episode_losses.append(loss)
                if total_steps_overall_session % 500 == 0 and loss is not None: script_logger.info(f"    Общ.Эп {current_overall_episode_num}, Сесс.шаг {total_steps_overall_session}, Текущий лосс: {loss:.4f}")
            state = next_state
        agent.decay_epsilon_episode()
        session_rewards_history.append(episode_reward_sum)
        avg_loss_this_episode = np.mean(current_episode_losses) if current_episode_losses else np.nan
        session_avg_loss_history.append(avg_loss_this_episode)
        script_logger.info(f"Общий Эпизод {current_overall_episode_num} завершен. Награда: {episode_reward_sum:.2f}. Шагов: {episode_steps}. Ср.лосс: {avg_loss_this_episode:.4f}. Trades: {info.get('trades_in_episode',0)}. NetWorth: {info.get('net_worth',0):.2f} {env.quote_currency}")
        if current_overall_episode_num > 0 and current_overall_episode_num % save_model_freq == 0:
           current_model_path_prefix = model_save_path_template.replace("{episode_num_overall}", str(current_overall_episode_num))
           agent.save_model(current_model_path_prefix)
    script_logger.info("="*50 + "\nТекущий сеанс тренировки завершен." + (f"\nСредняя награда за текущий сеанс ({len(session_rewards_history)} эп.): {np.mean(session_rewards_history):.2f}" if session_rewards_history else "") +
                       f"\nВсего шагов в этом сеансе: {total_steps_overall_session}\n" + "="*50)
    final_model_path_prefix = model_save_path_template.replace("{episode_num_overall}", str(start_episode_num_overall + num_episodes_this_run -1) + "_final_session")
    agent.save_model(final_model_path_prefix)
    return session_rewards_history, session_avg_loss_history


def find_latest_model_episode(model_dir: str, model_base_name_pattern: str) -> tuple[int, str | None]:
    """
    Ищет последний сохраненный файл модели, соответствующий шаблону,
    и извлекает из него номер эпизода.
    Шаблон должен содержать группу для захвата номера эпизода, например, '_ep_(\d+)'
    """
    latest_episode = 0
    latest_model_prefix = None
    # Пример шаблона для извлечения номера эпизода: dqn_..._penalty10.0_..._ep_(\d+)
    # Мы ожидаем файлы типа: basename_ep_100_q_network.weights.h5 или basename_ep_100_final_session_q_network.weights.h5
    # Упростим: ищем префикс до _q_network.weights.h5
    
    # Составляем более точный regex для model_base_name_pattern
    # model_base_name_pattern уже включает параметры LR, Gamma, penalty, epsdecay.
    # Нам нужно найти число после "_ep_"
    regex_pattern = re.compile(re.escape(model_base_name_pattern) + r"_ep_(\d+)(_final_session)?_q_network\.weights\.h5")

    script_logger.debug(f"Поиск моделей в '{model_dir}' по шаблону regex: '{regex_pattern.pattern}' (на основе '{model_base_name_pattern}')")

    if not os.path.isdir(model_dir):
        script_logger.warning(f"Директория моделей '{model_dir}' не найдена. Начинаем обучение с нуля.")
        return 0, None

    for filename in os.listdir(model_dir):
        match = regex_pattern.fullmatch(filename) # Ищем полное совпадение имени файла
        if match:
            episode_num = int(match.group(1)) # Захваченное число эпизодов
            if episode_num > latest_episode:
                latest_episode = episode_num
                # Формируем префикс, который ожидает agent.load_model()
                # Это имя файла без _q_network.weights.h5 или _target_q_network.weights.h5
                # и без _final_session, если это не финальная модель сеанса
                
                # Базовое имя до _ep_
                base_prefix_for_load = model_base_name_pattern + f"_ep_{episode_num}"
                if match.group(2) == "_final_session": # Если это файл _final_session
                    base_prefix_for_load += "_final_session"
                latest_model_prefix = os.path.join(model_dir, base_prefix_for_load)
                script_logger.debug(f"  Найдена модель: {filename}, эпизод: {episode_num}, префикс для загрузки: {latest_model_prefix}")

    if latest_model_prefix:
        script_logger.info(f"Найдена последняя модель: префикс '{latest_model_prefix}', соответствующая эпизоду {latest_episode}.")
    else:
        script_logger.info(f"Сохраненные модели по шаблону '{model_base_name_pattern}' не найдены в '{model_dir}'. Начинаем обучение с нуля.")
    
    return latest_episode, latest_model_prefix


if __name__ == "__main__":
    script_logger.info("Запуск скрипта тренировки ИИ агента...")
    tf.config.run_functions_eagerly(False)
    
    # --- Общие параметры (должны быть консистентны между сеансами, если хотим продолжать ту же модель) ---
    data_filename = "binance_BTC_USDT_1h.csv"
    trading_pair_for_env = "BTC/USDT"
    historical_data_filepath = os.path.join(project_root, "data", "historical", data_filename)
    initial_balance = 1000.0
    trade_fee_env = 0.075
    
    # Параметры, определяющие МОДЕЛЬ и СТРАТЕГИЮ ОБУЧЕНИЯ (должны быть одинаковы для продолжения)
    fixed_penalty_per_trade = 10.0
    learning_r = 0.00025
    gamma_discount = 0.99
    epsilon_decay_rate = 0.999 # Важно для расчета начального эпсилона при дообучении
    epsilon_min_val = 0.01
    # Архитектура сети неявно определяется в DQNAgent, если она меняется - это новая модель.
    
    # Параметры, которые могут меняться для НОВОГО СЕАНСА дообучения
    num_episodes_this_session = 50  # Сколько ЕЩЕ эпизодов обучать
    epsilon_start_for_new_session = 1.0 # С какого эпсилона НАЧИНАТЬ этот сеанс (может быть пересчитан)

    # Параметры среды
    window_sz = 20; sma_short_p = 7; sma_long_p = 21; rsi_val_p = 14
    max_steps_in_env_episode = None 

    # Параметры буфера и обучения
    buffer_cap = 100000; batch_sz_train = 64
    target_net_update_freq = 20
    
    # Пути для сохранения
    plots_save_dir = os.path.join(project_root, "data", "training_plots")
    model_save_dir = os.path.join(project_root, "data", "models", "dqn")
    os.makedirs(model_save_dir, exist_ok=True)
    
    # ФОРМИРУЕМ БАЗОВОЕ ИМЯ МОДЕЛИ (без номера эпизода)
    # Оно должно быть одинаковым для всех сеансов обучения ОДНОЙ И ТОЙ ЖЕ модели/стратегии
    MODEL_BASE_NAME_PATTERN = (f"dqn_{trading_pair_for_env.replace('/','-')}"
                               f"_penalty{fixed_penalty_per_trade:.1f}" # Штраф часть стратегии
                               f"_lr{learning_r}_gamma{gamma_discount}" # LR и Gamma часть модели
                               f"_epsdecay{epsilon_decay_rate}") # Eps decay тоже часть стратегии

    # --- Автоматическое определение, продолжать ли обучение ---
    episodes_already_trained, model_prefix_to_load = find_latest_model_episode(model_save_dir, MODEL_BASE_NAME_PATTERN)
    
    start_episode_for_this_session = episodes_already_trained + 1
    
    if model_prefix_to_load:
        script_logger.info(f"ПРОДОЛЖЕНИЕ ОБУЧЕНИЯ. Загрузка модели: {model_prefix_to_load}")
        script_logger.info(f"Уже обучено эпизодов: {episodes_already_trained}. Начинаем с эпизода: {start_episode_for_this_session}.")
        # Рассчитываем эпсилон, с которого нужно продолжить, если хотим плавный спад
        current_epsilon = epsilon_start_for_new_session * (epsilon_decay_rate ** episodes_already_trained)
        epsilon_to_start_session_with = max(current_epsilon, epsilon_min_val)
        script_logger.info(f"Начальный эпсилон для этого сеанса будет: {epsilon_to_start_session_with:.4f}")
    else:
        script_logger.info("НАЧАЛО НОВОГО ОБУЧЕНИЯ (предыдущие модели не найдены или не соответствуют шаблону).")
        epsilon_to_start_session_with = epsilon_start_for_new_session # Обычно 1.0 для нового обучения
        # episodes_already_trained уже 0

    # Шаблон для сохранения моделей в ТЕКУЩЕМ сеансе
    # Имя файла будет включать ОБЩЕЕ количество пройденных эпизодов
    model_save_filename_template_this_run = MODEL_BASE_NAME_PATTERN + "_ep_{episode_num_overall}"
    model_save_path_template_this_run = os.path.join(model_save_dir, model_save_filename_template_this_run)
    save_model_overall_episode_freq = 100

    # --- Загрузка данных ---
    training_df = load_training_data(historical_data_filepath)

    if training_df is not None and not training_df.empty:
        try:
            env = TradingEnvironment( # ... (параметры среды без изменений)
                historical_data_df=training_df, trading_pair_symbol=trading_pair_for_env,
                initial_balance_quote=initial_balance, trade_fee_percent=trade_fee_env,
                fixed_transaction_penalty=fixed_penalty_per_trade, window_size=window_sz,
                sma_short_period=sma_short_p, sma_long_period=sma_long_p, rsi_period=rsi_val_p,
                max_steps_per_episode=max_steps_in_env_episode
            )
            script_logger.info(f"Среда: Obs shape: {env.observation_space_shape}, Action size: {env.action_space_n}")
            
            agent = DQNAgent( # Инициализируем агента с текущими параметрами LR, Gamma и стартовым эпсилоном для сеанса
                state_shape=env.observation_space_shape, action_size=env.action_space_n,
                learning_rate=learning_r, gamma=gamma_discount, epsilon=epsilon_to_start_session_with,
                epsilon_decay=epsilon_decay_rate, epsilon_min=epsilon_min_val,
                target_update_freq=target_net_update_freq
            )
            script_logger.info(f"DQNAgent инициализирован. Начальный Epsilon для этого сеанса: {agent.epsilon:.4f}")

            if model_prefix_to_load: # Если нашли модель для загрузки
                agent.load_model(model_prefix_to_load)
                # Epsilon уже установлен в epsilon_to_start_session_with при создании агента
            
            replay_buffer = ReplayBuffer(capacity=buffer_cap) # Буфер всегда создается заново
            script_logger.info(f"Буфер воспроизведения (capacity: {buffer_cap}) инициализирован.")

            session_rewards, session_losses = run_training_loop(
                env=env, agent=agent, replay_buffer=replay_buffer,
                num_episodes_this_run=num_episodes_this_session,
                start_episode_num_overall=start_episode_for_this_session,
                batch_size=batch_sz_train,
                save_model_freq=save_model_overall_episode_freq,
                model_save_path_template=model_save_path_template_this_run
            )
            
            total_episodes_trained_at_end = episodes_already_trained + num_episodes_this_session
            plot_filename_base_for_plots = MODEL_BASE_NAME_PATTERN # Для заголовков графиков
            plot_training_results(session_rewards, session_losses, total_episodes_trained_at_end,
                                  plots_save_dir, plot_filename_base_for_plots)
                              
        except ValueError as ve: script_logger.error(f"Ошибка значения: {ve}", exc_info=True)
        except Exception as e: script_logger.error(f"Непредвиденная ошибка: {e}", exc_info=True)
    else:
        script_logger.error("Не удалось загрузить данные для тренировки.")

    script_logger.info("Скрипт тренировки ИИ агента завершил работу.")