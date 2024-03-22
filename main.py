import math
import numpy as np

# 参数设置
MAX_CARS = 20       # 每个地点的最多汽车数量
MAX_MOVE = 5        # 每次最多移动汽车的数量
DISCOUNT = 0.9      # 折扣率
RENT_REWARD = 10.0  # 租车收益
MOVE_COST = 2.0     # 移动成本

# 两个地点租车和还车的期望值
rental_lambda = [3, 4]
return_lambda = [3, 2]

# 初始化状态值和策略矩阵
state_values = np.zeros(shape=(MAX_CARS + 1, MAX_CARS + 1), dtype=np.float32)
policy = np.zeros(shape=(MAX_CARS + 1, MAX_CARS + 1), dtype=int)

# 定义动作空间，负值表示把车从第二地点到第一地点移动，正值表示把车从第一地点到第二地点移动。
actions = np.arange(-MAX_MOVE, MAX_MOVE + 1)

# 状态转移函数
def poisson_dis(lam, n):
    return (lam**n / math.factorial(n)) * np.exp(-lam)

# 动作价值函数
def action_value_function(state, action, state_values):
    reward = 0
    first_loc_cars = state[0] - action
    second_loc_cars = state[1] + action
    for first_loc_rentals in range(first_loc_cars + 1):
        for second_loc_rentals in range(second_loc_cars + 1):
            for first_loc_returns in range(MAX_CARS + 1):
                for second_loc_returns in range(MAX_CARS + 1):
                    action_value = (first_loc_rentals + second_loc_rentals) * RENT_REWARD
                    action_value -= abs(action) * MOVE_COST

                    next_first_loc_cars = min(first_loc_cars - first_loc_rentals + first_loc_returns,
                                              MAX_CARS)
                    next_second_loc_cars = min(second_loc_cars - second_loc_rentals + second_loc_returns,
                                               MAX_CARS)
                    probability = (poisson_dis(rental_lambda[0], first_loc_rentals)
                                   * poisson_dis(rental_lambda[1], second_loc_rentals)
                                   * poisson_dis(return_lambda[0], first_loc_returns)
                                   * poisson_dis(return_lambda[1], second_loc_returns))
                    reward += probability * (
                        action_value + DISCOUNT * state_values[next_first_loc_cars, next_second_loc_cars])
    return reward


if __name__ == '__main__':
    file = open('iteration_log.txt', 'w')
    # 策略迭代
    iteration = 0
    while True:
        iteration += 1

        # 复制当前的状态价值和策略矩阵
        new_state_values = np.copy(state_values)
        new_policy = np.copy(policy)

        # 遍历每个状态
        for first_loc_cars in range(MAX_CARS + 1):
            for second_loc_cars in range(MAX_CARS + 1):
                state = (first_loc_cars, second_loc_cars)

                # 当前状态下动作的价值列表
                action_value_list = []

                # 遍历所有可能的动作
                for action in actions:
                    # 定义合理的动作
                    action_doable = ((action >= 0 and (first_loc_cars >= action) and
                                      second_loc_cars + action <= MAX_CARS) or
                                     (action < 0 and second_loc_cars >= abs(action) and
                                      first_loc_cars + abs(action) <= MAX_CARS))
                    if action_doable:
                        action_value_list.append(action_value_function(state, action, new_state_values))
                    else:
                        action_value_list.append(-np.inf)
                
                #选择最大回报的动作作为新的状态值和动作
                new_state_values[first_loc_cars, second_loc_cars] = max(action_value_list)
                new_policy[first_loc_cars, second_loc_cars] = np.argmax(action_value_list) - 5
                # print(f"iteration{iteration} state{state}")

        # 计算新状态值和旧状态值的总差值
        delta = np.sum(np.abs(new_state_values - state_values))
        policy_stable = (policy == new_policy).all()
        # 将新状态值和策略矩阵复制到之前的状态值和策略矩阵
        state_values = new_state_values
        policy = new_policy

        # 输出该迭代的信息
        text = (f"iteration:{iteration}\ndelta:{delta}\nupdated state values:\n{state_values}\n"
                f"updated policy:\n{policy}\n")
        print(text)

        # 记录该迭代的信息到txt文件
        file.write(text)
        file.write('--------------------------------------------------------------------------------------------\n')

        # 如果状态值收敛并且策略稳定，则结束迭代
        if delta < 1e-4 or policy_stable:
            print("iteration finished! state values convergence")
            break
    file.close()
