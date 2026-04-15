#!/usr/bin/env python3
"""
test_env.py  ─  随机策略验证 WarehouseEnv
用途：在接入 QMIX 之前，确认环境的 reset/step/reward/done 逻辑正确。
运行：
    rosrun warehouse_astar test_env.py
"""
import random
import rospy
from warehouse_env import WarehouseEnv

def main():
    env = WarehouseEnv()
    n_episodes = 3

    for ep in range(n_episodes):
        print(f'\n{"="*50}')
        print(f'Episode {ep + 1}')
        print('='*50)

        obs = env.reset()
        print(f'初始观测 alpha: {obs[0]}')
        print(f'初始观测 beta : {obs[1]}')

        total_reward = 0.0
        done = False

        while not done:
            # 随机动作
            actions = [random.randint(0, 4) for _ in range(env.n_agents)]
            action_names = ['GOTO_A', 'GOTO_B', 'PICK_N', 'PICK_S', 'RETURN']
            print(f'\n动作: alpha={action_names[actions[0]]}, beta={action_names[actions[1]]}')

            obs, rewards, done, info = env.step(actions)
            total_reward += rewards[0]

            env.render()
            print(f'本步奖励: {rewards[0]:.1f} | 累计奖励: {total_reward:.1f}')

            if done:
                reason = '全部取完' if all(info['shelf_picked']) else '超时'
                print(f'\n>>> Episode 结束：{reason}，共 {info["step"]} 步，'
                      f'总奖励 {total_reward:.1f}')

    env.close()
    print('\n测试完成')

if __name__ == '__main__':
    main()
