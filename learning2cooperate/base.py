import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from numba import njit, jit, int64, float32, prange

nb_actions_1 = 2  # TODO make code able to handle nb_actions > 2
nb_actions_2 = 2
#payouts for player 1 and 2 in each state
payouts_1 = np.ones((nb_actions_1, nb_actions_2))
payouts_2 = np.ones((nb_actions_1, nb_actions_2))
payouts_1[0, 0] = 3
payouts_2[0, 0] = 3
payouts_1[0, 1] = 0
payouts_2[0, 1] = 4
payouts_1[1, 0] = 4
payouts_2[1, 0] = 0
payouts_1[1, 1] = 1
payouts_2[1, 1] = 1

@jit#t(int32(int32, int32))
def get_payouts(action_1 : int, action_2 : int):
    # params: actions of both players
    # return: payouts of both players
    # rtype: Vector{Float64}
    return payouts_1[action_1, action_2], payouts_2[action_1, action_2]

@njit
def get_strategy(random_strategy : np.ndarray):
    # params: random strategy
    # return: a realisation of the random strategy
    # rtype: Matrix{Int}
    shape = random_strategy.shape
    strategy = np.zeros((shape[0], shape[1]), dtype=int64)
    for i in prange(shape[0]):
        for j in prange(shape[1]):
            if np.random.rand() < random_strategy[i, j, 0]:
                strategy[i, j] = 0
            else:
                strategy[i, j] = 1
                
    return strategy

def get_action(strategy : np.ndarray, state : np.ndarray):
    # params: stategy and state
    # return: action taken according to strategy in this state
    # rtype: Int
    if len(strategy.shape) == 2:
        return strategy[state[0], state[1]]
    
    if len(strategy.shape) == 3:
        # return: action taken according to a realisation of the random strategy in this state
        return get_strategy(strategy)[state[0], state[1]]
    
    raise("The strategy doesn't have correct dimensions")

@njit
def get_all_strategies():
    # get them as probability of playing action 1, 2
    # return: an array of the 16 pure strategies (as random strategies)
    # rtype: Array{Float64, 4}
    strategies = np.ones((16, 2, 2, 2), dtype=int64)
    for i in range(16):
        strategies[i, 0, 0, 0] = np.floor(i/8)
        strategies[i, 0, 0, 1] = 1 - strategies[i, 0, 0, 0]
        strategies[i, 0, 1, 0] = np.floor((i%8)/4)
        strategies[i, 0, 1, 1] = 1 - strategies[i, 0, 1, 0]
        strategies[i, 1, 0, 0] = np.floor((i%4)/2)
        strategies[i, 1, 0, 1] = 1 - strategies[i, 1, 0, 0]
        strategies[i, 1, 1, 0] = i%2
        strategies[i, 1, 1, 1] = 1 - strategies[i, 1, 1, 0]
    
    return strategies

@njit
def get_strategy_from_mixture(mixture):
    cumsum_probas = np.cumsum(mixture)
    random_val = np.random.rand()
    indexes = cumsum_probas > random_val
    all_strats = get_all_strategies()
    return get_strategy(all_strats[np.argmax(indexes), :, :, :])



##############################################################
# Value computation
##############################################################
@njit
def get_value1(state, strategy_1, strategy_2, nb_tries, nb_iterations):
    # params: starting state, two pure strategies
    # return: mean values of the cycle TODO(explain this), starting from state
    # rtype: Vector{Float64}
    new_state = state
    total_payout_1 = 0.
    total_payout_2 = 0.

    for i in range(nb_iterations):
        action_1 = strategy_1[new_state[0], new_state[1]]
        action_2 = strategy_2[new_state[0], new_state[1]]
        new_state = np.array([action_1, action_2])
        payouts = get_payouts(action_1, action_2)
        if i>4:
            total_payout_1 += payouts[0]
            total_payout_2 += payouts[1]
    
    total_payout_1 /= ((nb_iterations-4))
    total_payout_2 /= ((nb_iterations-4))
    
    return total_payout_1, total_payout_2

@njit
def get_value2(state, strategy_1, rnd_strat_2, nb_tries, nb_iterations):
    # params: starting state, one pure and one random strategy
    # return: mean value of game after many iterations, starting from state
    # rtype: Vector{Float64}
    total_payout_1 = 0.
    total_payout_2 = 0.
    for t in prange(nb_tries):
        new_state = state.copy()
        
        for i in range(nb_iterations):
            action_1 = strategy_1[new_state[0], new_state[1]]
            action_2 = get_strategy(rnd_strat_2)[new_state[0], new_state[1]]
            new_state = np.array([action_1, action_2])
            payouts = get_payouts(action_1, action_2)
            if i>4:
                total_payout_1 += payouts[0]
                total_payout_2 += payouts[1]
    total_payout_1 /= ((nb_iterations-4) * nb_tries)
    total_payout_2 /= ((nb_iterations-4) * nb_tries)
    return total_payout_1, total_payout_2

@njit
def get_value3(state, rnd_strat_1, strategy_2, nb_tries, nb_iterations):
    # params: starting state, one pure and one random strategy
    # return: mean value of game after many iterations, starting from state
    # rtype: Vector{Float64}
    total_payout_1 = 0.
    total_payout_2 = 0.
    
    for t in prange(nb_tries):
        new_state = state.copy()
        for i in range(nb_iterations):
            action_1 = get_strategy(rnd_strat_1)[new_state[0], new_state[1]]
            action_2 = strategy_2[new_state[0], new_state[1]]
            new_state = np.array([action_1, action_2])
            payouts = get_payouts(action_1, action_2)
            if i>4:
                total_payout_1 += payouts[0]
                total_payout_2 += payouts[1]
    
    total_payout_1 /= ((nb_iterations-4) * nb_tries)
    total_payout_2 /= ((nb_iterations-4) * nb_tries)
    
    return total_payout_1, total_payout_2

@njit
def get_value4(state, rnd_strat_1, rnd_strat_2, nb_tries, nb_iterations):
    # params: starting state, two random strategies
    # return: mean value of game after many iterations, starting from state
    # rtype: Vector{Float64}
    total_payout_1 = 0.
    total_payout_2 = 0.
    for t in prange(nb_tries):
        new_state = state.copy()
        for i in range(nb_iterations):
            action_1 = get_strategy(rnd_strat_1)[new_state[0], new_state[1]]
            action_2 = get_strategy(rnd_strat_2)[new_state[0], new_state[1]]
            new_state = np.array([action_1, action_2])
            payouts = get_payouts(action_1, action_2)
            if i>4:
                total_payout_1 += payouts[0]
                total_payout_2 += payouts[1]
    
    total_payout_1 /= ((nb_iterations-4) * nb_tries)
    total_payout_2 /= ((nb_iterations-4) * nb_tries)
    return total_payout_1, total_payout_2

def get_value(state, strategy_1, strategy_2, nb_tries=20, nb_iterations=20):
    state = np.array(state)
    if len(strategy_1.shape) == 2:
        if len(strategy_2.shape) == 2:
            return get_value1(state, strategy_1, strategy_2, nb_tries, nb_iterations)
        if len(strategy_2.shape) == 3:
            return get_value2(state, strategy_1, strategy_2, nb_tries, nb_iterations)
    if len(strategy_1.shape) == 3:
        if len(strategy_2.shape) == 2:
            return get_value3(state, strategy_1, strategy_2, nb_tries, nb_iterations)
        if len(strategy_2.shape) == 3:
            return get_value4(state, strategy_1, strategy_2, nb_tries, nb_iterations)


# strategy_1 = np.ones((nb_actions_1, nb_actions_2), dtype=int)
# strategy_2 = np.ones((nb_actions_1, nb_actions_2), dtype=int)
# strategy_1[0, 0] = 0
# strategy_1[0, 1] = 1
# strategy_1[1, 0] = 1
# strategy_1[1, 1] = 1

# strategy_2[0, 0] = 0
# strategy_2[0, 1] = 1
# strategy_2[1, 0] = 1
# strategy_2[1, 1] = 1

# print(get_value([1, 1], strategy_1, strategy_2))

def get_value_in_all_states(strategy_1, strategy_2, nb_tries=20, nb_iterations=20):
    values = np.zeros((nb_actions_1, nb_actions_2, 2))
    for i in range(nb_actions_1):
        for j in range(nb_actions_2):
            value = get_value([i, j], strategy_1, strategy_2, nb_tries, nb_iterations)
            values[i, j, 0] = value[0]
            values[i, j, 1] = value[1]
    return values

# print(np.sum(get_value_in_all_states(strategy_1, strategy_2)[:,:,1]))

all_strategies = get_all_strategies()


############################################################################
# Strategy selection
############################################################################
@njit
def custom_logit(array, pivot):
    exp_arr = np.exp(array - np.max(array))
    cumsum_normed_exp = np.cumsum(exp_arr / np.sum(exp_arr))
    indexes = cumsum_normed_exp > pivot
    return np.argmax(indexes) # works since argmax returns the first highest value

def get_learned_strategy(state_frequencies, id, adv_strategy, previous_means, t, pivot,
                         eta = lambda t : np.sqrt(t)):
    # params:
    #    state_frequencies:
    #    id: equal to 1 if player 1, 2 if player 2
    #    adv_strategy: latext strategy of the other player
    #    previous_means:
    #    t: current stage
    # return: a strategy, updated means of all pure strategies
    # rtype: (Matrix{Int}, Vector{Float64})
    future_reward_per_strategy = np.zeros(16)
    for strategy_idx in prange(16):
        strategy_candidate = get_strategy(all_strategies[strategy_idx, :, :, :])
        if id==0:
            all_values = get_value_in_all_states(strategy_candidate, adv_strategy)
        else:
            all_values = get_value_in_all_states(adv_strategy, strategy_candidate)
        
        future_reward_per_strategy[strategy_idx] = np.sum(all_values[:, :, id] * state_frequencies)
    
    means = previous_means + (future_reward_per_strategy - previous_means) / (t+1)
    chosen_strategy_idx = custom_logit((eta(t) * means), pivot)
    strategy_rand = all_strategies[chosen_strategy_idx, :, :, :]
    strategy = get_strategy(strategy_rand)
    #println(strategy_rand)
    return strategy, means, chosen_strategy_idx

def get_learned_strategy_emp(state_frequencies, id, adv_strategy, previous_means, t, pivot,
                         eta = lambda t : np.sqrt(t)):
    # params:
    #    state_frequencies:
    #    id: equal to 1 if player 1, 2 if player 2
    #    adv_strategy: empirical mixed strategy of the other player
    #    t: current stage
    # return: a strategy, scores of  all pure strategies
    # rtype: (Matrix{Int}, Vector{Float64})
    future_reward_per_strategy = np.zeros(16)
    for strategy_idx in prange(16):
        strategy_candidate = get_strategy(all_strategies[strategy_idx, :, :, :])
        if id==0:
            all_values = get_value_in_all_states(strategy_candidate, adv_strategy)
        else:
            all_values = get_value_in_all_states(adv_strategy, strategy_candidate)

        future_reward_per_strategy[strategy_idx] = np.sum(all_values[:, :, id] * state_frequencies)
    
    means = previous_means + (future_reward_per_strategy - previous_means) / (t+1)
    chosen_strategy_idx = custom_logit((eta(t) * future_reward_per_strategy), pivot)
    strategy = get_strategy(all_strategies[chosen_strategy_idx, :, :, :])
    return strategy, future_reward_per_strategy, chosen_strategy_idx


def get_learned_strategy_half_memory(state_frequencies, id, adv_strategy, previous_means, t):
    # params:
    #    state_frequencies:
    #    id: equal to 1 if player 1, 2 if player 2
    #    adv_strategy: latext strategy of the other player
    #    previous_means:
    #    t: current stage
    # return: a strategy, updated means of all pure strategies
    # rtype: (Matrix{Int}, Vector{Float64})
    future_reward_per_strategy = np.zeros(4)
    strategies = np.zeros(4)
    for strategy_idx in range(4):
        if id==0:
            strategies = [1, 6, 11, 16]
        else:
            strategies = [1, 4, 13, 16]

        s_idx = strategies[strategy_idx]
        strategy_candidate = get_strategy(all_strategies[s_idx, :, :, :])
        if id==0:
            all_values = get_value_in_all_states(strategy_candidate, adv_strategy)
        else:
            all_values = get_value_in_all_states(adv_strategy, strategy_candidate)
            
        future_reward_per_strategy[strategy_idx] = sum(all_values[:, :, id] * state_frequencies)
    
    means = previous_means + (future_reward_per_strategy - previous_means) / t
    chosen_strategy_idx = strategies[custom_logit((np.sqrt(t) * means))]
    strategy_rand = all_strategies[chosen_strategy_idx, :, :, :]
    strategy = get_strategy(strategy_rand)
    #println(chosen_strategy_idx, strategy)
    return strategy, means


def render_results(means1, means2, state_appearances, all_payouts, all_states, strategy_history, 
                   best_reward = 3, *args, **kwargs):
    # plt.plot(np.cumsum(all_payouts[:, 0]), label='player 0')
    # plt.plot(np.cumsum(all_payouts[:, 1]), label='player 1')
    # plt.title('Cummulative rewards')
    # plt.legend()
    # plt.show()
    
    csum1 = np.cumsum(best_reward - all_payouts[:,:,0], axis = 1)
    csum2 = np.cumsum(best_reward - all_payouts[:,:,1], axis = 1)
    
    mean_1 = np.mean(csum1, axis=0)
    mean_2 = np.mean(csum2, axis=0)
    
    std_1 = np.std(csum1, axis=0)
    std_2 = np.std(csum2, axis=0)
    
    min1 = np.amin(csum1, axis=0)
    min2 = np.amin(csum2, axis=0)
    
    max1 = np.amax(csum1, axis=0)
    max2 = np.amax(csum2, axis=0)
    
    median1 = np.median(csum1, axis=0)
    median2 = np.median(csum2, axis=0)

    
    # fig, ax = plt.subplots(2, 1)
    plt.plot(min1, linestyle=':', label='min regret', color='black')
    plt.plot(mean_1, label='mean regret')
    plt.plot(median1, label='median regret', linestyle='--', color='black')
    plt.plot(max1, label='max regret', linestyle='-.', color='black')
    plt.fill_between(range(len(mean_1)), mean_1 - 2*std_1, mean_1 + 2*std_1, alpha=0.3, label="mean $\pm 2\sigma$")
    plt.fill_between(range(len(mean_1)), mean_1 - std_1, mean_1 + std_1, alpha=0.7, label="mean $\pm \sigma$")
    plt.title('Cumulative "regret" player 0')
    plt.xlabel("Time $t$")
    plt.legend()
    plt.show()
    
    plt.plot(min2, linestyle=':', label='min regret', color='black')
    plt.plot(mean_2, label='mean regret')
    plt.plot(median2, label='median regret', linestyle='--', color='black')
    plt.plot(max2, label='max regret', linestyle='-.', color='black')
    plt.fill_between(range(len(mean_2)), mean_2 - 2*std_2, mean_2 + 2*std_2, alpha=0.3, label="mean $\pm 2\sigma$")
    plt.fill_between(range(len(mean_2)), mean_2 - std_2, mean_2 + std_2, alpha=0.7, label="mean $\pm \sigma$")
    # plt.plot(15*np.sqrt(np.arange(len(mean_1))), label='$15\sqrt{t}$')
    plt.title('Cumulative "regret" player 1')
    plt.xlabel("Time $t$")
    plt.legend()
    plt.legend()
    plt.show()
    
    # plot:
    fig, ax = plt.subplots(1,2)
    sample_strategy = strategy_history[np.random.choice(len(strategy_history), min(len(strategy_history), 25), replace=False)]
    a = 0.1
    for i in range(len(sample_strategy)):
        ax[0].scatter(np.arange(sample_strategy.shape[1]), sample_strategy[i, :, 0].reshape(-1), alpha=a/len(sample_strategy), color='gray')
        ax[1].scatter(np.arange(sample_strategy.shape[1]), sample_strategy[i, :, 1].reshape(-1), alpha=a/len(sample_strategy), color='gray')
    # ax[0].legend()
    # ax[1].legend()
    ax[0].set_title("Strategy usage player 0")
    ax[1].set_title("Strategy usage player 1")
    ax[0].set_xlabel("Time $t$")
    ax[1].set_xlabel("Time $t$")
    
    ax[0].set_ylabel("Strategy played")

    plt.show()
    
    mean_1 = np.mean(all_payouts[:, :, 0], axis=0)
    mean_2 = np.mean(all_payouts[:, :, 1], axis=0)
    
    std_1 = np.std(all_payouts[:, :, 0], axis=0)
    std_2 = np.std(all_payouts[:, :, 1], axis=0)
    
    payout_mean_1 = np.cumsum(mean_1) / np.cumsum(np.ones(all_payouts.shape[1]))
    payout_mean_2 = np.cumsum(mean_2) / np.cumsum(np.ones(all_payouts.shape[1]))
    plt.plot(payout_mean_1, label='player 0')
    plt.fill_between(range(len(mean_1)), payout_mean_1 - 2*std_1, payout_mean_1 + 2*std_1, alpha=0.3)
    plt.plot(payout_mean_2, label='player 1')
    plt.fill_between(range(len(mean_2)), payout_mean_2 - 2*std_2, payout_mean_2 + 2*std_2, alpha=0.3)
    plt.title('Mean rewards')
    plt.legend()
    plt.show()
    
    # plt.plot(all_payouts[:, 0], label='player 0')
    # plt.plot(all_payouts[:, 1], label='player 1')
    # plt.title('Rewards per iteration')
    # plt.legend()
    # plt.show()
    
    # plt.imshow(state_apearences)#, interpolation='nearest')
    ax = sns.heatmap(state_appearances/np.sum(state_appearances), cmap='crest', linewidth=0.5, annot=True)
    plt.title('Appearance rate per state')
    plt.show()