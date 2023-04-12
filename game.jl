# using Gadfly  # for pretty plots
# using DataFrames  # to organise data, also for plots
using Plots

nb_actions_1 = 2  # TODO make code able to handle nb_actions > 2
nb_actions_2 = 2
#payouts for player 1 and 2 in each state
payouts_1 = ones(nb_actions_1, nb_actions_2)
payouts_2 = ones(nb_actions_1, nb_actions_2)
payouts_1[1, 1] = 3
payouts_2[1, 1] = 3
payouts_1[1, 2] = 0
payouts_2[1, 2] = 4
payouts_1[2, 1] = 4
payouts_2[2, 1] = 0
payouts_1[2, 2] = 1
payouts_2[2, 2] = 1

function get_payouts(action_1::Int, action_2::Int)
    # params: actions of both players
    # return: payouts of both players
    # rtype: Vector{Float64}
    return payouts_1[action_1, action_2], payouts_2[action_1, action_2]
end

get_payouts(1, 2)

strategy_1 = convert(Array{Int, 2}, ones(nb_actions_1, nb_actions_2))
strategy_1[1, 1] = 1
strategy_1[1, 2] = 2
strategy_1[2, 1] = 2
strategy_1[2, 2] = 2
strategy_2 = convert(Array{Int, 2}, ones(nb_actions_1, nb_actions_2))
strategy_2[1, 1] = 1
strategy_2[1, 2] = 2
strategy_2[2, 1] = 2
strategy_2[2, 2] = 2

function get_action(strategy::Array{Int,2}, state::Vector{Int})
    # params: stategy and state
    # return: action taken according to strategy in this state
    # rtype: Int
    return strategy[state[1], state[2]]
end

function get_action(random_strategy::Array{Float64, 3}, state::Vector{Int})
    # params: random stategy and state
    # return: action taken according to a realisation of the random strategy in this state
    # rtype: Int
    strategy = get_strategy(random_strategy)
    return get_action(strategy, state)
end

get_action(strategy_1, [1, 2])

function get_strategy(random_strategy::Array{Float64, 3})
    # params: random strategy
    # return: a realisation of the random strategy
    # rtype: Matrix{Int}
    shape = size(random_strategy)
    strategy = convert(Matrix{Int}, ones(shape[1], shape[2]))
    for i=1:shape[1]
        for j=1:shape[2]
            if rand() < random_strategy[i, j, 1]
                strategy[i, j] = 1
            else
                strategy[i, j] = 2
            end
        end
    end
    return strategy
end

function get_strategy(mixture::Vector{Float64})
    cumsum_probas = cumsum(mixture)
    random_val = rand()
    indexes = cumsum_probas .> random_val
    all_strats = get_all_strategies()
    return get_strategy(all_strats[indmax(indexes), :, :, :])
end

function get_value(state::Array{Int,1}, strategy_1::Array{Int,2}, strategy_2::Array{Int,2},
	nb_tries=20, nb_iterations=20)
    # params: starting state, two pure strategies
    # return: mean values of the cycle TODO(explain this), starting from state
    # rtype: Vector{Float64}
    new_state = state
    total_payout_1 = 0
    total_payout_2 = 0

    for i=1:nb_iterations
        action_1 = get_action(strategy_1, new_state)
        action_2 = get_action(strategy_2, new_state)
        new_state = [action_1, action_2]
        payouts = get_payouts(action_1, action_2)
        if i>4
            total_payout_1 += payouts[1]
            total_payout_2 += payouts[2]
        end
    end
    total_payout_1 /= ((nb_iterations-4) * nb_tries)
    total_payout_2 /= ((nb_iterations-4) * nb_tries)
    return total_payout_1, total_payout_2
end

function get_value(
        state::Array{Int,1}, strategy_1::Array{Int,2},
        rnd_strat_2::Array{Float64,3}, nb_tries=20, nb_iterations=20)
    # params: starting state, one pure and one random strategy
    # return: mean value of game after many iterations, starting from state
    # rtype: Vector{Float64}
    new_state = state
    total_payout_1 = 0
    total_payout_2 = 0
    for t=1:nb_tries
        for i=1:nb_iterations
            action_1 = get_action(strategy_1, new_state)
            action_2 = get_action(rnd_strat_2, new_state)
            new_state = [action_1, action_2]
            payouts = get_payouts(action_1, action_2)
            if i>4
                total_payout_1 += payouts[1]
                total_payout_2 += payouts[2]
            end
        end
    end
    total_payout_1 /= ((nb_iterations-4) * nb_tries)
    total_payout_2 /= ((nb_iterations-4) * nb_tries)
    return total_payout_1, total_payout_2
end

function get_value(
        state::Array{Int,1}, rnd_strat_1::Array{Float64,3},
        strategy_2::Array{Int,2}, nb_tries=20, nb_iterations=20)
    # params: starting state, one pure and one random strategy
    # return: mean value of game after many iterations, starting from state
    # rtype: Vector{Float64}
    new_state = state
    total_payout_1 = 0
    total_payout_2 = 0
    for t=1:nb_tries
        for i=1:nb_iterations
            action_1 = get_action(rnd_strat_1, new_state)
            action_2 = get_action(strategy_2, new_state)
            new_state = [action_1, action_2]
            payouts = get_payouts(action_1, action_2)
            if i>4
                total_payout_1 += payouts[1]
                total_payout_2 += payouts[2]
            end
        end
    end
    total_payout_1 /= ((nb_iterations-4) * nb_tries)
    total_payout_2 /= ((nb_iterations-4) * nb_tries)
    return total_payout_1, total_payout_2
end

function get_value(
        state::Array{Int,1}, rnd_strat_1::Array{Float64,3},
        rnd_strat_2::Array{Float64,3}, nb_tries=20, nb_iterations=20)
    # params: starting state, two random strategies
    # return: mean value of game after many iterations, starting from state
    # rtype: Vector{Float64}
    new_state = state
    total_payout_1 = 0
    total_payout_2 = 0
    for t=1:nb_tries
        for i=1:nb_iterations
            action_1 = get_action(rnd_strat_1, new_state)
            action_2 = get_action(rnd_strat_2, new_state)
            new_state = [action_1, action_2]
            payouts = get_payouts(action_1, action_2)
            if i>4
                total_payout_1 += payouts[1]
                total_payout_2 += payouts[2]
            end
        end
    end
    total_payout_1 /= ((nb_iterations-4) * nb_tries)
    total_payout_2 /= ((nb_iterations-4) * nb_tries)
    return total_payout_1, total_payout_2
end

get_value([1, 1], strategy_1, strategy_2)

function get_value_in_all_states(strategy_1::Array, strategy_2::Array)
    values = zeros(nb_actions_1, nb_actions_2, 2)
    for i=1:nb_actions_1
        for j=1:nb_actions_2
            value = get_value([i, j], strategy_1, strategy_2)
            values[i, j, 1] = value[1]
            values[i, j, 2] = value[2]
        end
    end
    return values
end

function get_all_strategies() # get them as probability of playing action 1, 2
    # return: an array of the 16 pure strategies (as random strategies)
    # rtype: Array{Float64, 4}
    strategies = ones(16, 2, 2, 2)
    for i=1:16
        strategies[i, 1, 1, 1] = floor((i-1)/8)
        strategies[i, 1, 1, 2] = 1 - strategies[i, 1, 1, 1]
        strategies[i, 1, 2, 1] = floor(((i-1)%8)/4)
        strategies[i, 1, 2, 2] = 1 - strategies[i, 1, 2, 1]
        strategies[i, 2, 1, 1] = floor(((i-1)%4)/2)
        strategies[i, 2, 1, 2] = 1 - strategies[i, 2, 1, 1]
        strategies[i, 2, 2, 1] = (i-1)%2
        strategies[i, 2, 2, 2] = 1 - strategies[i, 2, 2, 1]
    end
    return strategies
end

all_strategies = get_all_strategies()
# half memory for 1: 1, 6, 11, 16
# half memory for 2: 1, 4, 13, 16
get_strategy(all_strategies[13, :, :, :])

get_value_in_all_states(
    strategy_1,
    0.4 * all_strategies[12, :, :, :] + 0.6 * all_strategies[14, :, :, :])

all_values = zeros(16, 16, 4)
for strat_rnd=1:16
    for strat_rnd_2=1:16
        #println(strat_rnd)
        #println(get_strategy(all_strat[strat_rnd, :, :, :]))
        for i=1:2
            all_values[strat_rnd, strat_rnd_2, i] = get_value_in_all_states(
                get_strategy(all_strategies[strat_rnd, :, :, :]),
                get_strategy(all_strategies[strat_rnd_2, :, :, :]))[1, i, 1]
        end
        for i=1:2
            all_values[strat_rnd, strat_rnd_2, i+2] = get_value_in_all_states(
                get_strategy(all_strategies[strat_rnd, :, :, :]),
                get_strategy(all_strategies[strat_rnd_2, :, :, :]))[2, i, 1]
        end
    end
end
is_good_strategy = zeros(16, 16, 4)
strat_good = zeros(16, 16)
for i=1:16
    for j=1:16
        for s=1:4
            is_good_strategy[i, j, s] = (
                maximum(all_values[:, j, s]) == all_values[i, j, s])
        end
        strat_good[i, j] = prod(is_good_strategy[i, j, :])
    end
end
println(strat_good)
value_matrix = sum(all_values[:, :, :],dims = 3)#round(800 .+ sum(all_values[:, :, :],dims = 3), digits = 1)
println(value_matrix)
println(prod(value_matrix[:, :] .- value_matrix[10,:] .>= 0, dims=1))
println(sum(strat_good, digits=1))  # number of good strategies for each j

#find equilibrium : i is best strategy against j and j is best against i
is_eq = strat_good .* transpose(strat_good)
println(is_eq)
println(all_values[:, :, 3] .* is_eq)

function custom_logit(array::Vector{Float64})
    array = array .- maximum(array)
    exp_arr = exp.(array)
    cumsum_normed_exp = cumsum(exp_arr ./ sum(exp_arr))
    random_val = rand()
    indexes = cumsum_normed_exp .> random_val
    return findmax(indexes)[2]  # works since findmax returns the first highest value
end

function get_learned_strategy(
        state_frequencies::Matrix{Float64}, id::Int, adv_strategy::Array,
        previous_means::Vector{Float64}, t::Int)
    # params:
    #    state_frequencies:
    #    id: equal to 1 if player 1, 2 if player 2
    #    adv_strategy: latext strategy of the other player
    #    previous_means:
    #    t: current stage
    # return: a strategy, updated means of all pure strategies
    # rtype: (Matrix{Int}, Vector{Float64})
    future_reward_per_strategy = zeros(16)
    for strategy_idx=1:16
        strategy_candidate = get_strategy(all_strategies[strategy_idx, :, :, :])
        if id==1
            all_values = get_value_in_all_states(strategy_candidate, adv_strategy)
        else
            all_values = get_value_in_all_states(adv_strategy, strategy_candidate)
        end
        future_reward_per_strategy[strategy_idx] = sum(all_values[:, :, id] .* state_frequencies)
    end
    means = previous_means + (future_reward_per_strategy .- previous_means) ./ t
    chosen_strategy_idx = custom_logit((sqrt(t) .* means))
    strategy_rand = all_strategies[chosen_strategy_idx, :, :, :]
    strategy = get_strategy(strategy_rand)
    #println(strategy_rand)
    return strategy, means
end

function get_learned_strategy_emp(
        state_frequencies::Matrix{Float64}, id::Int, adv_strategy::Array, t::Int)
    # params:
    #    state_frequencies:
    #    id: equal to 1 if player 1, 2 if player 2
    #    adv_strategy: empirical mixed strategy of the other player
    #    t: current stage
    # return: a strategy, scores of  all pure strategies
    # rtype: (Matrix{Int}, Vector{Float64})
    future_reward_per_strategy = zeros(16)
    for strategy_idx=1:16
        strategy_candidate = get_strategy(all_strategies[strategy_idx, :, :, :])
        if id==1
            all_values = get_value_in_all_states(strategy_candidate, adv_strategy)
        else
            all_values = get_value_in_all_states(adv_strategy, strategy_candidate)
        end
        future_reward_per_strategy[strategy_idx] = sum(all_values[:, :, id] .* state_frequencies)
    end
    chosen_strategy_idx = custom_logit((sqrt(t) .* future_reward_per_strategy))
    strategy = get_strategy(all_strategies[chosen_strategy_idx, :, :, :])
    return strategy, future_reward_per_strategy
end

function get_learned_strategy_half_memory(
        state_frequencies::Matrix{Float64}, id::Int, adv_strategy::Array,
        previous_means::Vector{Float64}, t::Int)
    # params:
    #    state_frequencies:
    #    id: equal to 1 if player 1, 2 if player 2
    #    adv_strategy: latext strategy of the other player
    #    previous_means:
    #    t: current stage
    # return: a strategy, updated means of all pure strategies
    # rtype: (Matrix{Int}, Vector{Float64})
    future_reward_per_strategy = zeros(4)
    strategies = zeros(4)
    for strategy_idx=1:4
        if id==1
            strategies = [1, 6, 11, 16]
        else
            strategies = [1, 4, 13, 16]
        end
        s_idx = strategies[strategy_idx]
        strategy_candidate = get_strategy(all_strategies[s_idx, :, :, :])
        if id==1
            all_values = get_value_in_all_states(strategy_candidate, adv_strategy)
        else
            all_values = get_value_in_all_states(adv_strategy, strategy_candidate)
        end
        future_reward_per_strategy[strategy_idx] = sum(all_values[:, :, id] .* state_frequencies)
    end
    means = previous_means + (future_reward_per_strategy .- previous_means) ./ t
    chosen_strategy_idx = strategies[custom_logit((sqrt(t) .* means))]
    strategy_rand = all_strategies[chosen_strategy_idx, :, :, :]
    strategy = get_strategy(strategy_rand)
    #println(chosen_strategy_idx, strategy)
    return strategy, means
end


game_length = 10000
all_strategies = get_all_strategies()
means1 = zeros(16)
means2 = zeros(16)
#means1 = zeros(4)
#means2 = zeros(4)
payout_mean_1 = 0
payout_mean_2 = 0
for nb_games=1:1
    state = [1,2]
    #state = [rand(1:2), rand(1:2)]
    state_appearances = zeros(nb_actions_1, nb_actions_2)
    state_appearances[state[1], state[2]] = 1
    #random_strategy_2 = 0.4 * all_strategies[16, :, :, :] + 0.6 * all_strategies[3, :, :, :]
    random_strategy_2 = 0.5 * ones(2, 2, 2)
    # [i, j, s] : action1, action2, next_action
    random_strategy_2[1, 1, 1] = 0
    random_strategy_2[1, 1, 2] = 1
    random_strategy_2[1, 2, 1] = 0
    random_strategy_2[1, 2, 2] = 1
    random_strategy_2[2, 1, 1] = 0
    random_strategy_2[2, 1, 2] = 1
    random_strategy_2[2, 2, 1] = 0
    random_strategy_2[2, 2, 2] = 1

    #strategy_2 = get_strategy(random_strategy_2)
    strategy_1 = get_strategy(all_strategies[1, :, :, :])
    rnd_strategy_1_emp = 0.5 * ones(2, 2, 2)
    rnd_strategy_2_emp = 0.5 * ones(2, 2, 2)
    all_payouts = zeros(game_length, 2)
    all_states = ones(game_length, 2)
    means1 = zeros(16)
    means2 = zeros(16)
    #means1 = zeros(4)
    #means2 = zeros(4)
    for i=1:game_length
        action_1 = get_action(strategy_1, state)
        action_2 = get_action(strategy_2, state)
        payouts = get_payouts(action_1, action_2)
        all_payouts[i, 1] = payouts[1]
        all_payouts[i, 2] = payouts[2]
        state_appearances[action_1, action_2] += 1
        #state_frequencies = state_appearances / sum(state_appearances)
        state_frequencies = ones(2, 2) ./ 4
        strategy_1, means1 = get_learned_strategy(#_half_memory(
            state_frequencies, 1, strategy_2, means1, i)
        #strategy_1, means1 = get_learned_strategy_emp(
        #    state_frequencies, 1, rnd_strategy_2_emp, i)
        strategy_2, means2 = get_learned_strategy(#_half_memory(
            state_frequencies, 2, strategy_1, means2, i)
        #strategy_2, means2 = get_learned_strategy_emp(
        #    state_frequencies, 2, rnd_strategy_1_emp, i)
        #strategy_2 = get_strategy(random_strategy_2)
        # full info on the strategies
        #for a=1:2
        #    for b=1:2
        #        rnd_strategy_1_emp[a, b, 1] = (
        #            rnd_strategy_1_emp[a, b, 1]
        #            + (1*(strategy_1[a, b] == 1) - rnd_strategy_1_emp[a,b,1])/(i+1)
        #        )
        #        rnd_strategy_2_emp[a, b, 1] = (
        #            rnd_strategy_2_emp[a, b, 1]
        #            + (1*(strategy_2[a, b] == 1) - rnd_strategy_2_emp[a,b,1])/(i+1)
        #        )
        #    end
        #end
        # bandit info
        rnd_strategy_1_emp[action_1, action_2, 1] = (
            rnd_strategy_1_emp[action_1, action_2, 1]
            + (1*(strategy_1[action_1, action_2] == 1) - rnd_strategy_1_emp[action_1, action_2, 1])/state_appearances[action_1, action_2]
        )
        rnd_strategy_2_emp[action_1, action_2, 1] = (
            rnd_strategy_2_emp[action_1, action_2, 1]
            + (1*(strategy_2[action_1, action_2] == 1) - rnd_strategy_2_emp[action_1, action_2, 1])/state_appearances[action_1, action_2]
        )
        state = [action_1, action_2]
        all_states[i, 1] = action_1
        all_states[i, 2] = action_2
    end

    payout_mean_1 = cumsum(all_payouts[:, 1]) ./ cumsum(ones(game_length))
    payout_mean_2 = cumsum(all_payouts[:, 2]) ./ cumsum(ones(game_length))
    println(strategy_1)
    println(strategy_2)
    println(state_appearances)
    println(means1)
    array = sqrt(game_length) * means1
    array = array .- maximum(array)
    exp_arr = exp.(array)
    normed_exp = exp_arr ./ sum(exp_arr)
    println(normed_exp)
    array = sqrt(game_length) * means2
    array = array .- maximum(array)
    exp_arr = exp.(array)
    normed_exp = exp_arr ./ sum(exp_arr)
    println(normed_exp)
end
#means1 = zeros(4)
#means2 = zeros(4)
payout_mean_1 = 0
payout_mean_2 = 0
for nb_games=1:1
    state = [1,2]
    #state = [rand(1:2), rand(1:2)]
    state_appearances = zeros(nb_actions_1, nb_actions_2)
    state_appearances[state[1], state[2]] = 1
    #random_strategy_2 = 0.4 * all_strategies[16, :, :, :] + 0.6 * all_strategies[3, :, :, :]
    random_strategy_2 = 0.5 * ones(2, 2, 2)
    # [i, j, s] : action1, action2, next_action
    random_strategy_2[1, 1, 1] = 0
    random_strategy_2[1, 1, 2] = 1
    random_strategy_2[1, 2, 1] = 0
    random_strategy_2[1, 2, 2] = 1
    random_strategy_2[2, 1, 1] = 0
    random_strategy_2[2, 1, 2] = 1
    random_strategy_2[2, 2, 1] = 0
    random_strategy_2[2, 2, 2] = 1

    #strategy_2 = get_strategy(random_strategy_2)
    strategy_1 = get_strategy(all_strategies[1, :, :, :])
    rnd_strategy_1_emp = 0.5 * ones(2, 2, 2)
    rnd_strategy_2_emp = 0.5 * ones(2, 2, 2)
    all_payouts = zeros(game_length, 2)
    all_states = ones(game_length, 2)
    means1 = zeros(16)
    means2 = zeros(16)
    #means1 = zeros(4)
    #means2 = zeros(4)
    for i=1:game_length
        action_1 = get_action(strategy_1, state)
        action_2 = get_action(strategy_2, state)
        payouts = get_payouts(action_1, action_2)
        all_payouts[i, 1] = payouts[1]
        all_payouts[i, 2] = payouts[2]
        state_appearances[action_1, action_2] += 1
        #state_frequencies = state_appearances / sum(state_appearances)
        state_frequencies = ones(2, 2) ./ 4
        strategy_1, means1 = get_learned_strategy(#_half_memory(
            state_frequencies, 1, strategy_2, means1, i)
        #strategy_1, means1 = get_learned_strategy_emp(
        #    state_frequencies, 1, rnd_strategy_2_emp, i)
        strategy_2, means2 = get_learned_strategy(#_half_memory(
            state_frequencies, 2, strategy_1, means2, i)
        #strategy_2, means2 = get_learned_strategy_emp(
        #    state_frequencies, 2, rnd_strategy_1_emp, i)
        #strategy_2 = get_strategy(random_strategy_2)
        # full info on the strategies
        #for a=1:2
        #    for b=1:2
        #        rnd_strategy_1_emp[a, b, 1] = (
        #            rnd_strategy_1_emp[a, b, 1]
        #            + (1*(strategy_1[a, b] == 1) - rnd_strategy_1_emp[a,b,1])/(i+1)
        #        )
        #        rnd_strategy_2_emp[a, b, 1] = (
        #            rnd_strategy_2_emp[a, b, 1]
        #            + (1*(strategy_2[a, b] == 1) - rnd_strategy_2_emp[a,b,1])/(i+1)
        #        )
        #    end
        #end
        # bandit info
        rnd_strategy_1_emp[action_1, action_2, 1] = (
            rnd_strategy_1_emp[action_1, action_2, 1]
            + (1*(strategy_1[action_1, action_2] == 1) - rnd_strategy_1_emp[action_1, action_2, 1])/state_appearances[action_1, action_2]
        )
        rnd_strategy_2_emp[action_1, action_2, 1] = (
            rnd_strategy_2_emp[action_1, action_2, 1]
            + (1*(strategy_2[action_1, action_2] == 1) - rnd_strategy_2_emp[action_1, action_2, 1])/state_appearances[action_1, action_2]
        )
        state = [action_1, action_2]
        all_states[i, 1] = action_1
        all_states[i, 2] = action_2
    end

    payout_mean_1 = cumsum(all_payouts[:, 1]) ./ cumsum(ones(game_length))
    payout_mean_2 = cumsum(all_payouts[:, 2]) ./ cumsum(ones(game_length))
    println(strategy_1)
    println(strategy_2)
    println(state_appearances)
    println(means1)
    array = sqrt(game_length) * means1
    array = array .- maximum(array)
    exp_arr = exp.(array)
    normed_exp = exp_arr ./ sum(exp_arr)
    println(normed_exp)
    array = sqrt(game_length) * means2
    array = array .- maximum(array)
    exp_arr = exp.(array)
    normed_exp = exp_arr ./ sum(exp_arr)
    println(normed_exp)
end
#means1 = zeros(4)
#means2 = zeros(4)
payout_mean_1 = 0
payout_mean_2 = 0
for nb_games=1:1
    state = [1,2]
    #state = [rand(1:2), rand(1:2)]
    state_appearances = zeros(nb_actions_1, nb_actions_2)
    state_appearances[state[1], state[2]] = 1
    #random_strategy_2 = 0.4 * all_strategies[16, :, :, :] + 0.6 * all_strategies[3, :, :, :]
    random_strategy_2 = 0.5 * ones(2, 2, 2)
    # [i, j, s] : action1, action2, next_action
    random_strategy_2[1, 1, 1] = 0
    random_strategy_2[1, 1, 2] = 1
    random_strategy_2[1, 2, 1] = 0
    random_strategy_2[1, 2, 2] = 1
    random_strategy_2[2, 1, 1] = 0
    random_strategy_2[2, 1, 2] = 1
    random_strategy_2[2, 2, 1] = 0
    random_strategy_2[2, 2, 2] = 1

    #strategy_2 = get_strategy(random_strategy_2)
    strategy_1 = get_strategy(all_strategies[1, :, :, :])
    rnd_strategy_1_emp = 0.5 * ones(2, 2, 2)
    rnd_strategy_2_emp = 0.5 * ones(2, 2, 2)
    all_payouts = zeros(game_length, 2)
    all_states = ones(game_length, 2)
    means1 = zeros(16)
    means2 = zeros(16)
    #means1 = zeros(4)
    #means2 = zeros(4)
    for i=1:game_length
        action_1 = get_action(strategy_1, state)
        action_2 = get_action(strategy_2, state)
        payouts = get_payouts(action_1, action_2)
        all_payouts[i, 1] = payouts[1]
        all_payouts[i, 2] = payouts[2]
        state_appearances[action_1, action_2] += 1
        #state_frequencies = state_appearances / sum(state_appearances)
        state_frequencies = ones(2, 2) ./ 4
        strategy_1, means1 = get_learned_strategy(#_half_memory(
            state_frequencies, 1, strategy_2, means1, i)
        #strategy_1, means1 = get_learned_strategy_emp(
        #    state_frequencies, 1, rnd_strategy_2_emp, i)
        strategy_2, means2 = get_learned_strategy(#_half_memory(
            state_frequencies, 2, strategy_1, means2, i)
        #strategy_2, means2 = get_learned_strategy_emp(
        #    state_frequencies, 2, rnd_strategy_1_emp, i)
        #strategy_2 = get_strategy(random_strategy_2)
        # full info on the strategies
        #for a=1:2
        #    for b=1:2
        #        rnd_strategy_1_emp[a, b, 1] = (
        #            rnd_strategy_1_emp[a, b, 1]
        #            + (1*(strategy_1[a, b] == 1) - rnd_strategy_1_emp[a,b,1])/(i+1)
        #        )
        #        rnd_strategy_2_emp[a, b, 1] = (
        #            rnd_strategy_2_emp[a, b, 1]
        #            + (1*(strategy_2[a, b] == 1) - rnd_strategy_2_emp[a,b,1])/(i+1)
        #        )
        #    end
        #end
        # bandit info
        rnd_strategy_1_emp[action_1, action_2, 1] = (
            rnd_strategy_1_emp[action_1, action_2, 1]
            + (1*(strategy_1[action_1, action_2] == 1) - rnd_strategy_1_emp[action_1, action_2, 1])/state_appearances[action_1, action_2]
        )
        rnd_strategy_2_emp[action_1, action_2, 1] = (
            rnd_strategy_2_emp[action_1, action_2, 1]
            + (1*(strategy_2[action_1, action_2] == 1) - rnd_strategy_2_emp[action_1, action_2, 1])/state_appearances[action_1, action_2]
        )
        state = [action_1, action_2]
        all_states[i, 1] = action_1
        all_states[i, 2] = action_2
    end
    println(mean(all_states, dims=1))
    payout_mean_1 = cumsum(all_payouts[:, 1]) ./ cumsum(ones(game_length))
    payout_mean_2 = cumsum(all_payouts[:, 2]) ./ cumsum(ones(game_length))
    plot(payout_mean_1)
    plot(payout_mean_2)
    
    println(strategy_1)
    println(strategy_2)
    println(state_appearances)
    println(means1)
    array = sqrt(game_length) * means1
    array = array .- maximum(array)
    exp_arr = exp.(array)
    normed_exp = exp_arr ./ sum(exp_arr)
    println(normed_exp)
    array = sqrt(game_length) * means2
    array = array .- maximum(array)
    exp_arr = exp.(array)
    normed_exp = exp_arr ./ sum(exp_arr)
    println(normed_exp)
end
#means1 = zeros(4)
#means2 = zeros(4)
payout_mean_1 = 0
payout_mean_2 = 0
for nb_games=1:1
    state = [1,2]
    #state = [rand(1:2), rand(1:2)]
    state_appearances = zeros(nb_actions_1, nb_actions_2)
    state_appearances[state[1], state[2]] = 1
    #random_strategy_2 = 0.4 * all_strategies[16, :, :, :] + 0.6 * all_strategies[3, :, :, :]
    random_strategy_2 = 0.5 * ones(2, 2, 2)
    # [i, j, s] : action1, action2, next_action
    random_strategy_2[1, 1, 1] = 0
    random_strategy_2[1, 1, 2] = 1
    random_strategy_2[1, 2, 1] = 0
    random_strategy_2[1, 2, 2] = 1
    random_strategy_2[2, 1, 1] = 0
    random_strategy_2[2, 1, 2] = 1
    random_strategy_2[2, 2, 1] = 0
    random_strategy_2[2, 2, 2] = 1

    #strategy_2 = get_strategy(random_strategy_2)
    strategy_1 = get_strategy(all_strategies[1, :, :, :])
    rnd_strategy_1_emp = 0.5 * ones(2, 2, 2)
    rnd_strategy_2_emp = 0.5 * ones(2, 2, 2)
    all_payouts = zeros(game_length, 2)
    all_states = ones(game_length, 2)
    means1 = zeros(16)
    means2 = zeros(16)
    #means1 = zeros(4)
    #means2 = zeros(4)
    for i=1:game_length
        action_1 = get_action(strategy_1, state)
        action_2 = get_action(strategy_2, state)
        payouts = get_payouts(action_1, action_2)
        all_payouts[i, 1] = payouts[1]
        all_payouts[i, 2] = payouts[2]
        state_appearances[action_1, action_2] += 1
        #state_frequencies = state_appearances / sum(state_appearances)
        state_frequencies = ones(2, 2) ./ 4
        strategy_1, means1 = get_learned_strategy(#_half_memory(
            state_frequencies, 1, strategy_2, means1, i)
        #strategy_1, means1 = get_learned_strategy_emp(
        #    state_frequencies, 1, rnd_strategy_2_emp, i)
        strategy_2, means2 = get_learned_strategy(#_half_memory(
            state_frequencies, 2, strategy_1, means2, i)
        #strategy_2, means2 = get_learned_strategy_emp(
        #    state_frequencies, 2, rnd_strategy_1_emp, i)
        #strategy_2 = get_strategy(random_strategy_2)
        # full info on the strategies
        #for a=1:2
        #    for b=1:2
        #        rnd_strategy_1_emp[a, b, 1] = (
        #            rnd_strategy_1_emp[a, b, 1]
        #            + (1*(strategy_1[a, b] == 1) - rnd_strategy_1_emp[a,b,1])/(i+1)
        #        )
        #        rnd_strategy_2_emp[a, b, 1] = (
        #            rnd_strategy_2_emp[a, b, 1]
        #            + (1*(strategy_2[a, b] == 1) - rnd_strategy_2_emp[a,b,1])/(i+1)
        #        )
        #    end
        #end
        # bandit info
        rnd_strategy_1_emp[action_1, action_2, 1] = (
            rnd_strategy_1_emp[action_1, action_2, 1]
            + (1*(strategy_1[action_1, action_2] == 1) - rnd_strategy_1_emp[action_1, action_2, 1])/state_appearances[action_1, action_2]
        )
        rnd_strategy_2_emp[action_1, action_2, 1] = (
            rnd_strategy_2_emp[action_1, action_2, 1]
            + (1*(strategy_2[action_1, action_2] == 1) - rnd_strategy_2_emp[action_1, action_2, 1])/state_appearances[action_1, action_2]
        )
        state = [action_1, action_2]
        all_states[i, 1] = action_1
        all_states[i, 2] = action_2
    end

    payout_mean_1 = cumsum(all_payouts[:, 1]) ./ cumsum(ones(game_length))
    payout_mean_2 = cumsum(all_payouts[:, 2]) ./ cumsum(ones(game_length))
    println(strategy_1)
    println(strategy_2)
    println(state_appearances)
    println(means1)
    array = sqrt(game_length) * means1
    array = array .- maximum(array)
    exp_arr = exp.(array)
    normed_exp = exp_arr ./ sum(exp_arr)
    println(normed_exp)
    array = sqrt(game_length) * means2
    array = array .- maximum(array)
    exp_arr = exp.(array)
    normed_exp = exp_arr ./ sum(exp_arr)
    println(normed_exp)
end
println(get_strategy(all_strategies[9, :, :, :]))

times = cumsum(ones(game_length))
reward_plot = plot(
  layer(x=times, y=payout_mean_1, Geom.line,
  Theme(default_color=color("blue"))),
    layer(x=times, y=payout_mean_2, Geom.line,
    Theme(default_color=color("green"))),
  Guide.XLabel("Stage"), Guide.ylabel("Mean Reward"))

random_strategy_1 = 0.5 * ones(2, 2, 2)
# [i, j, s] : action1, action2, next_action
random_strategy_1[1, 1, 1] = 1
random_strategy_1[1, 1, 2] = 0
random_strategy_1[1, 2, 1] = 0
random_strategy_1[1, 2, 2] = 1
random_strategy_1[2, 1, 1] = 1
random_strategy_1[2, 1, 2] = 0
random_strategy_1[2, 2, 1] = 0
random_strategy_1[2, 2, 2] = 1

random_strategy_2 = 0.5 * ones(2, 2, 2)
random_strategy_2[1, 1, 1] = 0.8
random_strategy_2[1, 1, 2] = 0.2
random_strategy_2[1, 2, 1] = 1
random_strategy_2[1, 2, 2] = 0
random_strategy_2[2, 1, 1] = 0
random_strategy_2[2, 1, 2] = 1
random_strategy_2[2, 2, 1] = 0
random_strategy_2[2, 2, 2] = 1

for strat_idx=1:16
    if strat_idx==1
        println()
    end
    strat = get_strategy(all_strategies[strat_idx, :, :, :])
    nb_tries = 1000
    nb_iterations = 1000
    int_value = 0.
    int_value += get_value([1,1], strat, random_strategy_2, nb_tries, nb_iterations)[1]
    int_value += get_value([1,2], strat, random_strategy_2, nb_tries, nb_iterations)[1]
    int_value += get_value([2,1], strat, random_strategy_2, nb_tries, nb_iterations)[1]
    int_value += get_value([2,2], strat, random_strategy_2, nb_tries, nb_iterations)[1]
    int_value /= 4
    nb_tries = 10000
    ext_value = 0.
    for i=1:nb_tries
        value = get_value([1,1], strat, get_strategy(random_strategy_2))[1]
        value += get_value([1,2], strat, get_strategy(random_strategy_2))[1]
        value += get_value([2,1], strat, get_strategy(random_strategy_2))[1]
        value += get_value([2,2], strat, get_strategy(random_strategy_2))[1]
        ext_value += value / 4.
    end
    ext_value /= nb_tries
    println(ext_value, " ", int_value)
end

# TFT for player 2: 13
# 2, 1, 2, 2 : 5
strategy_2 = get_strategy(all_strategies[16, :, :, :])
for strat_idx=1:16
    if strat_idx==1
        println()
    end
    strat = get_strategy(all_strategies[strat_idx, :, :, :])
    value = 0.
    value += get_value([1,1], strat, strategy_2)[1]
    value += get_value([2,1], strat, strategy_2)[1]
    value += get_value([1,2], strat, strategy_2)[1]
    value += get_value([2,2], strat, strategy_2)[1]
    value /= 4
    print(value, " , ")
end


random_strategy_2 = [
    0.5, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0.5]
strat1 = get_strategy(all_strategies[1, :, :, :])
strat_p2_1 = get_strategy(all_strategies[4, :, :, :])
strat_p2_2 = get_strategy(all_strategies[13, :, :, :])
println(get_value([1,1], strat, strat_p2_2)[1])
