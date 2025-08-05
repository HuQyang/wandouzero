from collections import Counter
import math

import random
import time

CARD_VALUE_2 = 17
BLACK_JOKER_VALUE = 20
RED_JOKER_VALUE = 30
ALL_RANKS = [3,4,5,6,7,8,9,10,11,12,13,14,17]
from douzero.env.move_generator_for_detector import MovesGener


def find_min_combinations_your_version_optimized(list_a, list_b):
    """
    基于你的算法，针对大规模数据的特别优化版本
    """
    target = Counter(list_a)
    
    # 关键优化1: 大幅减少候选数量
    useful_candidates = []
    target_keys = set(target.keys())
    
    for sub in list_b:
        sub_counter = Counter(sub)
        # 只保留对目标有实际贡献的元素
        useful_elements = {k: min(v, target.get(k, 0)) 
                          for k, v in sub_counter.items() 
                          if k in target_keys and target.get(k, 0) > 0}
        
        if useful_elements and sum(useful_elements.values()) > 0:
            useful_candidates.append((Counter(useful_elements), sub, sum(useful_elements.values())))
    
    if not useful_candidates:
        return []
    
    # 关键优化2: 按效率排序，优先选择高效候选
    useful_candidates.sort(key=lambda x: (-x[2], -len(x[0])))
    # print('useful_candidates',useful_candidates)
    
    # 关键优化3: 设置合理的搜索上限
    total_needed = sum(target.values())
    max_contribution = useful_candidates[0][2] if useful_candidates else 1
    theoretical_min = math.ceil(total_needed / max_contribution)
    
    best_len = min(len(useful_candidates), theoretical_min + 5)  # 合理上界
    best_solutions = []
    memo = {}
    
    # 关键优化4: 限制记忆化的使用范围
    memo_threshold = 50  # 只对小问题使用记忆化
    
    def smart_lower_bound(rem_counter):
        """智能下界估计"""
        if not rem_counter:
            return 0
        
        total_remaining = sum(rem_counter.values())
        return max(
            math.ceil(total_remaining / max_contribution),
            theoretical_min
        )
    
    def backtrack(start, rem_counter, path):
        nonlocal best_len, best_solutions
        
        # 成功条件
        if not rem_counter:
            cur_len = len(path)
            if cur_len < best_len:
                best_len = cur_len
                best_solutions = [path.copy()]
            elif cur_len == best_len and len(best_solutions) < 10:  # 限制解的数量
                best_solutions.append(path.copy())
            return
        
        # 提前剪枝
        if len(path) >= best_len:
            return
        
        # 有选择地使用记忆化
        remaining_size = sum(rem_counter.values())
        if remaining_size <= memo_threshold:
            key = tuple(sorted(rem_counter.items()))
            if key in memo and memo[key] <= len(path):
                return
            memo[key] = len(path)
        
        # 下界剪枝
        if len(path) + smart_lower_bound(rem_counter) >= best_len:
            return
        
        # 关键优化5: 大幅限制搜索宽度
        search_width = min(300, len(useful_candidates) - start)  # 最多只搜索30个候选
        
        for i in range(start, start + search_width):
            if i >= len(useful_candidates):
                break
                
            cnt, sub, contribution = useful_candidates[i]
            
            # 快速有效性检查
            if not any(rem_counter.get(k, 0) > 0 for k in cnt):
                continue
            
            # 可行性检查
            if all(rem_counter.get(k, 0) >= v for k, v in cnt.items()):
                new_rem = rem_counter - cnt
                new_rem = +new_rem  # 移除零值项
                
                path.append(sub)
                backtrack(i + 1, new_rem, path)
                path.pop()
                
                # 早期终止
                if best_len <= theoretical_min:
                    return
    
    backtrack(0, target, [])
    return best_solutions


if __name__ == "__main__":

    perf_start = time.time()
    count = 0
    for i in range(50): 
        hand = random.choices(ALL_RANKS, k=random.randint(13, 17))  # 减少手牌数量
        hand.sort()
        mastercards = random.sample(ALL_RANKS, k=2)
        
        mg = MovesGener(hand,mastercards)
        legal_actions = mg.gen_moves()

        min_combinations = find_min_combinations_your_version_optimized(hand, legal_actions)
    
        # combinations = find_combinations(hand, legal_actions)
        # min_combi = find_min_length_combinations(combinations)
        if len(min_combinations)==0:
            count += 1
        # print(min_combinations)
    print(count)

    perf_end = time.time()
    print(f"\n总测试耗时: {(perf_end - perf_start):.2f}秒")