from collections import Counter
import math


import random
import time

CARD_VALUE_2 = 17
BLACK_JOKER_VALUE = 20
RED_JOKER_VALUE = 30
ALL_RANKS = [3,4,5,6,7,8,9,10,11,12,13,14,17]
from douzero.env.move_generator_for_detector import MovesGener


def find_min_combinations(list_a, list_b):
    target = Counter(list_a)
    candidates = [(Counter(sub), sub) for sub in list_b]
    candidates.sort(key=lambda x: -sum(x[0].values()))
    best_len = float('inf')
    best_solutions = []

    memo = {}

    max_sub_size = max(sum(cnt.values()) for cnt, _ in candidates)

    def lower_bound(rem_counter):
        """对当前剩余 rem_counter，估计至少还需要多少个子列表。"""
        total_remaining = sum(rem_counter.values())
        return math.ceil(total_remaining / max_sub_size)

    def backtrack(start, rem_counter, path):
        nonlocal best_len, best_solutions

        if not rem_counter:
            cur_len = len(path)
            if cur_len < best_len:
                best_len = cur_len
                best_solutions = [path.copy()]
            elif cur_len == best_len:
                best_solutions.append(path.copy())
            return

        key = frozenset(rem_counter.items())
        if key in memo and memo[key] <= len(path):
            return
        memo[key] = len(path)

        if len(path) + lower_bound(rem_counter) > best_len:
            return

        # 枚举后续候选
        for i in range(start, len(candidates)):
            cnt, sub = candidates[i]

            if all(rem_counter[k] >= v for k, v in cnt.items()):
                new_rem = rem_counter - cnt
                path.append(sub)
                backtrack(i+1, new_rem, path)
                path.pop()

    # 启动搜索
    backtrack(0, target, [])
    return best_solutions

def get_min_step_small(card,mastercards):
    mg = MovesGener(card,mastercards)
    legal_actions = mg.gen_moves()

    min_combinations = find_min_combinations(card, legal_actions)
    if len(min_combinations) == 0:
        min_step = len(card) // 2
    else:
        min_step = len(min_combinations[0])
    return min_step


def find_min_combinations_large(list_a, list_b):
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
        search_width = min(200, len(useful_candidates) - start)  # 最多只搜索30个候选
        
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


def get_min_step_large(card,mastercards):
    mg = MovesGener(card,mastercards)
    legal_actions = mg.gen_moves()

    min_combinations = find_min_combinations_large(card, legal_actions)
    if len(min_combinations) == 0:
        min_step = len(card) // 2
    else:
        min_step = len(min_combinations[0])
    return min_step

# def find_all_combinations(list_a, list_b):
#     result = []
#     list_a.sort()
    
#     def backtrack(start, path):
#         if start == len(list_a):
#             result.append(path.copy())
#             return
#         for sub in list_b:
#             sub.sort()
#             end = start + len(sub)
#             if end > len(list_a):
#                 continue
#             if list_a[start:end] == sub:
#                 path.append(sub)
#                 backtrack(end, path)
#                 path.pop()
    
#     backtrack(0, [])
#     return result

# def find_combinations(list_a, list_b):
#     """
#     返回所有由 list_b 中若干子列表拼凑出 list_a（按多重集，不考虑顺序连续性）的组合。
#     """
#     target = Counter(list_a)
#     # 预处理：生成每个子列表的 Counter，并保留原列表引用
#     candidates = [(sub, Counter(sub)) for sub in list_b]
#     results = []

#     def backtrack(start, rem, path):
#         if not rem:  # 剩余要凑的元素空了
#             results.append(path.copy())
#             return
#         for j in range(start, len(candidates)):
#             sub, cnt = candidates[j]
#             # 如果子列表能被“剩余”覆盖
#             if all(rem[x] >= v for x, v in cnt.items()):
#                 new_rem = rem - cnt
#                 path.append(sub)
#                 backtrack(j+1, new_rem, path)
#                 path.pop()

#     backtrack(0, target, [])
#     return results


if __name__ == '__main__':
    
    perf_start = time.time()
    for i in range(50): 
        hand = random.choices(ALL_RANKS, k=random.randint(1,3))  # 减少手牌数量
        hand.sort()
        mastercards = random.sample(ALL_RANKS, k=2)
        
        mg = MovesGener(hand,mastercards)
        legal_actions = mg.gen_moves()

        min_combinations = find_min_combinations(hand, legal_actions)
        # combinations = find_combinations(hand, legal_actions)
        # min_combi = find_min_length_combinations(combinations)
        if len(min_combinations)==0:
            print('hand',hand)
            print('legal',legal_actions)

    perf_end = time.time()
    print(f"\n总测试耗时: {(perf_end - perf_start):.2f}秒")

    mastercards = [3,10 ]
    hand = [4, 5,5,6,7,8,9,11, 14, 14, 14,14]
    # hand = [4, 4, 14, 14,14]
    # hand = [8, 8, 7, 3, 10, 4]
    # hand = [4,5,14,14,14]
    mg = MovesGener(hand,mastercards)

    legal_actions = mg.gen_moves()
    # print('legal',len(legal_actions))
    # print('legal',(legal_actions))

    cards = [4, 5,5,6,7,8,9,11, 14, 14, 14,14]
    # moves_gener = MovesGener(cards, mastercards)
    # print(moves_gener.gen_moves(), "all moves")
    combinations_by_length = {}
    combinations = find_min_combinations(hand, legal_actions)
    # min_combi = find_min_length_combinations(combinations)

    print(combinations)


    # print(len(combinations[0]))
    # leng = 100
    # for item in combinations:
    #     if len(item) < leng:
    #         print(item)
    #         leng = len(item)

