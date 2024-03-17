import json
import pickle

import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import gymnasium as gym

import warnings
warnings.filterwarnings('ignore')


def get_epsilon_greedy_action(q_values, epsilon, action_n):
    policy = np.ones(action_n) * epsilon / action_n
    max_action = np.argmax(q_values)
    policy[max_action] += 1 - epsilon

    return np.random.choice(np.arange(action_n), p=policy)


def SARSA(env, episode_n, gamma=0.99, trajectory_len=500, alpha=0.5):
    total_rewards = []  # Создаем массив для хранения общих вознаграждений для каждого эпизода

    state_n = env.observation_space.n  # Получаем количество состояний в среде
    action_n = env.action_space.n  # Получаем количество действий в среде
    qfunction = np.zeros(
        (state_n, action_n))  # Создаем Q-функцию (матрицу состояние-действие) и инициализируем её нулями

    for episode in tqdm(range(episode_n)):  # Запускаем цикл для каждого эпизода
        epsilon = 1 / (episode + 1)  # Уменьшаем параметр epsilon для epsilon-жадной стратегии с каждым эпизодом
        total_reward = 0  # Инициализируем общую сумму вознаграждений для каждого эпизода
        state = env.reset()[0]  # Сбрасываем среду и получаем начальное состояние
        action = get_epsilon_greedy_action(qfunction[state], epsilon,
                                           action_n)  # Получаем действие с использованием epsilon-жадной стратегии

        for t_n in range(
                trajectory_len):  # Запускаем цикл для каждого шага внутри эпизода (ограниченного trajectory_len)
            next_state, reward, done, _, _ = env.step(
                action)  # Выполняем выбранное действие и получаем следующее состояние, вознаграждение и флаг завершения
            next_action = get_epsilon_greedy_action(qfunction[next_state], epsilon,
                                                    action_n)  # Получаем следующее действие с использованием epsilon-жадной стратегии

            qfunction[state][action] += alpha * (reward + gamma * qfunction[next_state][next_action] - qfunction[state][
                action])  # Обновляем Q-функцию согласно формуле метода SARSA

            state = next_state  # Переходим в следующее состояние
            action = next_action  # Переходим в следующее действие
            total_reward += reward  # Добавляем вознаграждение к общей сумме вознаграждений для эпизода

            if done:  # Если эпизод завершился, выходим из цикла
                break

        total_rewards.append(
            total_reward)  # Добавляем полученное вознаграждение к общему вознаграждению текущего эпизода

    return total_rewards  # Возвращаем массив общих вознаграждений для каждого эпизода


def QLearning(env, episode_n, noisy_episode_n, gamma=0.99, t_max=500, alpha=0.5):
    state_n = env.observation_space.n  # Получаем количество состояний в среде
    action_n = env.action_space.n  # Получаем количество действий в среде

    Q = np.zeros((state_n, action_n))  # Создаем Q-функцию (матрицу состояние-действие) и инициализируем её нулями
    epsilon = 1  # Инициализируем действием рандомный шанс

    total_rewards = []  # Создаем массив для хранения общих вознаграждений для каждого эпизода
    for episode in tqdm(range(episode_n)):  # Запускаем цикл для каждого эпизода
        epsilon = 1 / (episode + 1)  # Уменьшаем параметр epsilon для epsilon-жадной стратегии с каждым эпизодом
        total_reward = 0  # Инициализируем общую сумму вознаграждений для каждого эпизода
        state, _ = env.reset()  # Инициализируем состояние и действие в среде

        for t in range(t_max):  # Запускаем цикл для каждого шага в эпизоде

            action = get_epsilon_greedy_action(Q[state], epsilon, action_n)  # Делаем действие с шансом epsilon
            next_state, reward, done, _, _ = env.step(action)  # Делаем шаг в среде и получаем результат

            Q[state][action] += alpha * (
                        reward + gamma * np.max(Q[next_state, :]) - Q[state][action])  # Обновляем Q-функцию

            total_reward += reward  # Добавляем вознаграждение к общей сумме вознаграждений для эпизода

            if done:  # Если эпизод завершён, то выходим из цикла
                break

            state = next_state  # Обновляем состояние в среде

        epsilon = max(0, epsilon - 1 / noisy_episode_n)  # Обновляем действием шанс

        total_rewards.append(total_reward)  # Добавляем вознаграждение к общей сумме вознаграждений для эпизода

    return total_rewards, Q  # Возвращаем общие вознаграждения для каждого эпизода


def main():
    env = gym.make("Taxi-v3")
    total_rewards_q, agent = QLearning(env, episode_n=500, noisy_episode_n=400, t_max=1000, gamma=0.999, alpha=0.5)

    plt.plot(total_rewards_q, label='Q-learning')
    plt.title('Comparison of learning dymanics.')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()
    plt.savefig("Results.png")
    plt.show()

    with open("agent.pickle", "wb") as f:
        pickle.dump(agent, f)


if __name__ == '__main__':
    main()
