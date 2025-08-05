from collections import Counter
import numpy as np

from douzero.env.game_alpha import GameEnv

Card2Column = {3: 0, 4: 1, 5: 2, 6: 3, 7: 4, 8: 5, 9: 6, 10: 7,
               11: 8, 12: 9, 13: 10, 14: 11, 17: 12}

NumOnes2Array = {0: np.array([0, 0, 0, 0]),
                 1: np.array([1, 0, 0, 0]),
                 2: np.array([1, 1, 0, 0]),
                 3: np.array([1, 1, 1, 0]),
                 4: np.array([1, 1, 1, 1])}

deck = []
for i in range(3, 15):
    deck.extend([i for _ in range(4)])
deck.extend([17 for _ in range(4)])
deck.extend([20, 30])

global_mastercard_values = []
ACTION_ENCODE_DIM = 67
ENCODE_DIM = 5

class Env:
    """
    Doudizhu multi-agent wrapper
    """
    def __init__(self, objective,show_action=False):
        """
        Objective is wp/adp/logadp. It indicates whether considers
        bomb in reward calculation. Here, we use dummy agents.
        This is because, in the orignial game, the players
        are `in` the game. Here, we want to isolate
        players and environments to have a more gym style
        interface. To achieve this, we use dummy players
        to play. For each move, we tell the corresponding
        dummy player which action to play, then the player
        will perform the actual action in the game engine.
        """
        self.objective = objective

        # Initialize players
        # We use three dummy player for the target position
        self.players = {}
        for position in ['landlord', 'landlord_up', 'landlord_down']:
            self.players[position] = DummyAgent(position)
        
        self.show_action = show_action

        # Initialize the internal environment
        self._env = GameEnv(self.players,show_action=show_action)

        self.infoset = None

    def reset(self):
        """
        Every time reset is called, the environment
        will be re-initialized with a new deck of cards.
        This function is usually called when a game is over.
        """
        self._env.reset()

        # Randomly shuffle the deck
        _deck = deck.copy()
        np.random.shuffle(_deck)
        self.mastercard_list = np.random.choice(range(3, 15), 2, replace=False)
        # self.mastercard_list = []
        set_global_mastercards(self.mastercard_list)
        card_play_data = {'landlord': _deck[:20],
                          'landlord_up': _deck[20:37],
                          'landlord_down': _deck[37:54],
                          'three_landlord_cards': _deck[17:20],
                          'mastercard_list': self.mastercard_list,
                          }
        for key in card_play_data:
            card_play_data[key].sort()

        # Initialize the cards
        self._env.card_play_init(card_play_data)
        self.infoset = self._game_infoset

        return get_obs(self.infoset)

    def step(self, action):
        """
        Step function takes as input the action, which
        is a list of integers, and output the next obervation,
        reward, and a Boolean variable indicating whether the
        current game is finished. It also returns an empty
        dictionary that is reserved to pass useful information.
        """
        assert action in self.infoset.legal_actions
        self.players[self._acting_player_position].set_action(action)
        self._env.step()
        self.infoset = self._game_infoset
        done = False
        reward = 0.0
        if self._game_over:
            done = True
            # reward = self._get_reward()
            reward = {
                "play": {
                    "landlord": self._get_reward("landlord") / 2,
                    "landlord_down": self._get_reward("landlord_down"),
                    "landlord_up": self._get_reward("landlord_up"),
                }
            }
            obs = None
        else:
            obs = get_obs(self.infoset)
        return obs, reward, done, {}

    # def _get_reward(self):
    #     """
    #     This function is called in the end of each
    #     game. It returns either 1/-1 for win/loss,
    #     or ADP, i.e., every bomb will double the score.
    #     """
    #     winner = self._game_winner
    #     bomb_num = self._game_bomb_num
    #     if winner == 'landlord':
    #         if self.objective == 'adp':
    #             return 2.0 ** bomb_num
    #         elif self.objective == 'logadp':
    #             return bomb_num + 1.0
    #         else:
    #             return 1.0
    #     else:
    #         if self.objective == 'adp':
    #             return -2.0 ** bomb_num
    #         elif self.objective == 'logadp':
    #             return -bomb_num - 1.0
    #         else:
    #             return -1.0

    def _get_reward(self, pos):
        winner = self._game_winner

        bomb_num = self._game_bomb_num
        bid_count = 1

        multiply = 2
        if pos != 'landlord':
            pos = 'farmer'
            multiply = 1

        if bomb_num == 0:
            r = bid_count
        elif bomb_num == 1:
            r = bid_count * (1 + bomb_num)
        else:
            r = bid_count * (2 + 2 * (bomb_num - 1))

        if pos == winner:
            return r * multiply / 24
        else:
            return -r * multiply / 24
        
    @property
    def _game_infoset(self):
        """
        Here, inforset is defined as all the information
        in the current situation, incuding the hand cards
        of all the players, all the historical moves, etc.
        That is, it contains perferfect infomation. Later,
        we will use functions to extract the observable
        information from the views of the three players.
        """
        return self._env.game_infoset

    @property
    def _game_bomb_num(self):
        """
        The number of bombs played so far. This is used as
        a feature of the neural network and is also used to
        calculate ADP.
        """
        return self._env.get_bomb_num()

    @property
    def _game_winner(self):
        """ A string of landlord/peasants
        """
        return self._env.get_winner()

    @property
    def _acting_player_position(self):
        """
        The player that is active. It can be landlord,
        landlod_down, or landlord_up.
        """
        return self._env.acting_player_position

    @property
    def _game_over(self):
        """ Returns a Boolean
        """
        return self._env.game_over

class DummyAgent(object):
    """
    Dummy agent is designed to easily interact with the
    game engine. The agent will first be told what action
    to perform. Then the environment will call this agent
    to perform the actual action. This can help us to
    isolate environment and agents towards a gym like
    interface.
    """
    def __init__(self, position):
        self.position = position
        self.action = None

    def act(self, infoset):
        """
        Simply return the action that is set previously.
        """
        assert self.action in infoset.legal_actions
        return self.action

    def set_action(self, action):
        """
        The environment uses this function to tell
        the dummy agent what to do.
        """
        self.action = action

def get_obs(infoset,new_model=True):
    """
    This function obtains observations with imperfect information
    from the infoset. It has three branches since we encode
    different features for different positions.
    
    This function will return dictionary named `obs`. It contains
    several fields. These fields will be used to train the model.
    One can play with those features to improve the performance.

    `position` is a string that can be landlord/landlord_down/landlord_up

    `x_batch` is a batch of features (excluding the hisorical moves).
    It also encodes the action feature

    `z_batch` is a batch of features with hisorical moves only.

    `legal_actions` is the legal moves

    `x_no_action`: the features (exluding the hitorical moves and
    the action features). It does not have the batch dim.

    `z`: same as z_batch but not a batch.
    """
    if new_model==True:
        return _get_obs_resnet(infoset)
    else:
        if infoset.player_position == 'landlord':
            return _get_obs_landlord(infoset)
        elif infoset.player_position == 'landlord_up':
            return _get_obs_landlord_up(infoset)
        elif infoset.player_position == 'landlord_down':
            return _get_obs_landlord_down(infoset)
        else:
            raise ValueError('')

def _get_one_hot_array(num_left_cards, max_num_cards):
    """
    A utility function to obtain one-hot endoding
    """
    one_hot = np.zeros(max_num_cards)
    one_hot[num_left_cards - 1] = 1

    return one_hot

def _cards2array(list_cards):
    """
    A utility function that transforms the actions, i.e.,
    A list of integers into card matrix. Here we remove
    the six entries that are always zero and flatten the
    the representations.
    """
    # print("list_cards in cards2array",list_cards)
    if len(list_cards) == 0:
        return np.zeros(ACTION_ENCODE_DIM, dtype=np.int8)

    matrix = np.zeros([ENCODE_DIM, 13], dtype=np.int8)
    jokers = np.zeros(2, dtype=np.int8)
    counter = Counter(list_cards)

    for card, num_times in counter.items():
            
        if card < 20:
            matrix[0:4, Card2Column[card]] = NumOnes2Array[num_times]
            if is_mastercard(card):  
                matrix[4, Card2Column[card]] = 1
        elif card == 20:
            jokers[0] = 1
        elif card == 30:
            jokers[1] = 1
    return np.concatenate((matrix.flatten('F'), jokers))

def _action_seq_list2array_lstm(action_seq_list):
    """
    A utility function to encode the historical moves.
    We encode the historical 15 actions. If there is
    no 15 actions, we pad the features with 0. Since
    three moves is a round in DouDizhu, we concatenate
    the representations for each consecutive three moves.
    Finally, we obtain a 5xACTION_ENCODE_DIM matrix, which will be fed
    into LSTM for encoding.
    """
    action_seq_array = np.zeros((len(action_seq_list), ACTION_ENCODE_DIM))
    for row, list_cards in enumerate(action_seq_list):
        action_seq_array[row, :] = _cards2array(list_cards)
    action_seq_array = action_seq_array.reshape(5, 3*ACTION_ENCODE_DIM)
    return action_seq_array


def _action_seq_list2array(action_seq_list):
    action_seq_array = np.ones((len(action_seq_list), ACTION_ENCODE_DIM)) * -1  # Default Value -1 for not using area
    for row, list_cards in enumerate(action_seq_list):
        # print("list_cards",list_cards)
        if list_cards != []:
            # print("list_cards in action seq loop",list_cards)
            # action_seq_array[row, :ACTION_ENCODE_DIM] = _cards2array(list_cards[1])
            action_seq_array[row, :ACTION_ENCODE_DIM] = _cards2array(list_cards)
    # print("action_seq_array",action_seq_array.shape)
    # action_seq_array = action_seq_array.reshape(5, 3*ACTION_ENCODE_DIM)
    return action_seq_array


# def _process_action_seq(sequence, length=15):
#     """
#     A utility function encoding historical moves. We
#     encode 15 moves. If there is no 15 moves, we pad
#     with zeros.
#     """
#     sequence = sequence[-length:].copy()
#     if len(sequence) < length:
#         empty_sequence = [[] for _ in range(length - len(sequence))]
#         empty_sequence.extend(sequence)
#         sequence = empty_sequence
#     return sequence

def _process_action_seq(sequence, length=15, new_model=True):
    sequence = sequence[-length:].copy()
    sequence = sequence[::-1]
    if len(sequence) < length:
        empty_sequence = [[] for _ in range(length - len(sequence))]
        if new_model:
            sequence.extend(empty_sequence)
        else:
            empty_sequence.extend(sequence)
            sequence = empty_sequence
    return sequence

def _get_one_hot_bomb(bomb_num):
    """
    A utility function to encode the number of bombs
    into one-hot representation.
    """
    one_hot = np.zeros(15)
    one_hot[bomb_num] = 1
    return one_hot

def _get_obs_landlord(infoset):
    """
    Obttain the landlord features. See Table 4 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)
    
    print("my action batch",my_action_batch.shape)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    landlord_up_num_cards_left_batch = np.repeat(
        landlord_up_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    landlord_down_num_cards_left_batch = np.repeat(
        landlord_down_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    landlord_up_played_cards_batch = np.repeat(
        landlord_up_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    landlord_down_played_cards_batch = np.repeat(
        landlord_down_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch, # 67
                         other_handcards_batch,# 67
                         last_action_batch,# 67
                         landlord_up_played_cards_batch,# 67
                         landlord_down_played_cards_batch,# 67
                         landlord_up_num_cards_left_batch,# 17
                         landlord_down_num_cards_left_batch,# 17
                         bomb_num_batch,# 15
                         my_action_batch)) # 67

    print("x_batch",x_batch.shape)

    
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             last_action,
                             landlord_up_played_cards,
                             landlord_down_played_cards,
                             landlord_up_num_cards_left,
                             landlord_down_num_cards_left,
                             bomb_num))
    
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    print("z_batch",z_batch.shape)
    obs = {
            'position': 'landlord',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_up(infoset):
    """
    Obttain the landlord_up features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_down'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))

    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_up',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs

def _get_obs_landlord_down(infoset):
    """
    Obttain the landlord_down features. See Table 5 in
    https://arxiv.org/pdf/2106.06135.pdf
    """
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)
    other_handcards_batch = np.repeat(other_handcards[np.newaxis, :],
                                      num_legal_actions, axis=0)

    last_action = _cards2array(infoset.last_move)
    last_action_batch = np.repeat(last_action[np.newaxis, :],
                                  num_legal_actions, axis=0)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    last_landlord_action = _cards2array(
        infoset.last_move_dict['landlord'])
    last_landlord_action_batch = np.repeat(
        last_landlord_action[np.newaxis, :],
        num_legal_actions, axis=0)
    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)
    landlord_num_cards_left_batch = np.repeat(
        landlord_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    last_teammate_action = _cards2array(
        infoset.last_move_dict['landlord_up'])
    last_teammate_action_batch = np.repeat(
        last_teammate_action[np.newaxis, :],
        num_legal_actions, axis=0)
    teammate_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)
    teammate_num_cards_left_batch = np.repeat(
        teammate_num_cards_left[np.newaxis, :],
        num_legal_actions, axis=0)

    teammate_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])
    teammate_played_cards_batch = np.repeat(
        teammate_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])
    landlord_played_cards_batch = np.repeat(
        landlord_played_cards[np.newaxis, :],
        num_legal_actions, axis=0)

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)

    x_batch = np.hstack((my_handcards_batch,
                         other_handcards_batch,
                         landlord_played_cards_batch,
                         teammate_played_cards_batch,
                         last_action_batch,
                         last_landlord_action_batch,
                         last_teammate_action_batch,
                         landlord_num_cards_left_batch,
                         teammate_num_cards_left_batch,
                         bomb_num_batch,
                         my_action_batch))
    x_no_action = np.hstack((my_handcards,
                             other_handcards,
                             landlord_played_cards,
                             teammate_played_cards,
                             last_action,
                             last_landlord_action,
                             last_teammate_action,
                             landlord_num_cards_left,
                             teammate_num_cards_left,
                             bomb_num))
    z = _action_seq_list2array(_process_action_seq(
        infoset.card_play_action_seq))
    z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    obs = {
            'position': 'landlord_down',
            'x_batch': x_batch.astype(np.float32),
            'z_batch': z_batch.astype(np.float32),
            'legal_actions': infoset.legal_actions,
            'x_no_action': x_no_action.astype(np.int8),
            'z': z.astype(np.int8),
          }
    return obs


def set_global_mastercards(mastercard_list):
    """Set the global mastercard values, ensuring jokers are not included"""
    global global_mastercard_values
    # Filter out jokers (20 and 30) from mastercard list
    global_mastercard_values = [card for card in mastercard_list if card < 20]
    # print(f"Global mastercards set to: {global_mastercard_values}")


def is_mastercard(card):
    """Check if a card is a mastercard (jokers are never mastercards)."""
    return card in global_mastercard_values and card < 20  # Jokers (20,30) are never mastercards


def _get_obs_resnet(infoset):
    num_legal_actions = len(infoset.legal_actions)
    my_handcards = _cards2array(infoset.player_hand_cards)
    my_handcards_batch = np.repeat(my_handcards[np.newaxis, :],
                                   num_legal_actions, axis=0)

    other_handcards = _cards2array(infoset.other_hand_cards)

    three_landlord_cards = _cards2array(infoset.three_landlord_cards)

    my_action_batch = np.zeros(my_handcards_batch.shape)
    for j, action in enumerate(infoset.legal_actions):
        my_action_batch[j, :] = _cards2array(action)

    landlord_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord'], 20)

    landlord_up_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_up'], 17)

    landlord_down_num_cards_left = _get_one_hot_array(
        infoset.num_cards_left_dict['landlord_down'], 17)

    landlord_played_cards = _cards2array(
        infoset.played_cards['landlord'])

    landlord_up_played_cards = _cards2array(
        infoset.played_cards['landlord_up'])

    landlord_down_played_cards = _cards2array(
        infoset.played_cards['landlord_down'])

    # bid_info_z = np.multiply(bid_info, np.ones((54, 3))).transpose((1, 0))

    bomb_num = _get_one_hot_bomb(
        infoset.bomb_num)
    bomb_num_batch = np.repeat(
        bomb_num[np.newaxis, :],
        num_legal_actions, axis=0)
    
    num_cards_left = np.hstack((
                         landlord_num_cards_left,  # 20
                         landlord_up_num_cards_left,  # 17
                         landlord_down_num_cards_left))

    num_mastercards_left = np.zeros(13)
    i = 0
    for p in ['landlord', 'landlord_up', 'landlord_down']:
        for card in infoset.played_cards[p]:
            if card in global_mastercard_values:
                i += 1

    num_mastercards_left[(8-i)-1] = 1
     
    num_cards_left = np.hstack((num_cards_left, num_mastercards_left))
  

    x_batch = np.hstack((
                        #  bid_info_batch,  # 3
                         bomb_num_batch,  # 15
                         ))
    x_no_action = np.hstack((
                            #  bid_info,
                             bomb_num, # 15
                             ))


    z = np.vstack((
                  num_cards_left,  # 67
                  my_handcards,  # 67
                  other_handcards,  # 67
                  three_landlord_cards,  # 67
                  landlord_played_cards,  # 67
                  landlord_up_played_cards,  # 67
                  landlord_down_played_cards,  # 67
                  _action_seq_list2array(_process_action_seq(infoset.card_play_action_seq, 60)) # （60,67）
                  ))

    _z_batch = np.repeat(
        z[np.newaxis, :, :],
        num_legal_actions, axis=0)
    my_action_batch = my_action_batch[:, np.newaxis, :]
    z_batch = np.concatenate((my_action_batch, _z_batch), axis=1)
    
    obs = {
        'position': infoset.player_position,
        'x_batch': x_batch.astype(np.float32),
        'z_batch': z_batch.astype(np.float32),
        'legal_actions': infoset.legal_actions,
        'x_no_action': x_no_action.astype(np.int8),
        'z': z.astype(np.int8),
    }
    return obs