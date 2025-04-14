module SimplePIMC

export simulate, simulate_correlation, IChain

using LinearAlgebra, FileIO, Statistics

include("types.jl")


function generate_new_beads(ri, rm; M, τ, masses, DOF)
    r = Vector{Vector{Float64}}(undef, M - 1)
    @fastmath for i in 1:(M-1)
        r1 = i == 1 ? ri : r[i-1]
        r2 = rm
        r[i] = get_middle_unsym(r1, r2, 1, M - i; τ, masses, DOF)
    end
    return r
end

function get_middle_unsym(ri, rm, M1, M2; τ, masses, DOF)
    return randn(DOF) .* sqrt.(1 ./ ((1 / (M1 * τ) + 1 / (M2 * τ)) .* masses)) .+ (ri * M2 + rm * M1) / (M1 + M2)
end


function MC_update!(ichain::IChain; M, τ, masses, DOF, P, V)
    i = rand(1:P)
    m = i + M
    coods_init = ichain.coodinates[i, :]
    coods_exit = ichain.coodinates[mod1(m, P), :]
    new_beads = generate_new_beads(coods_init, coods_exit; M, τ, masses, DOF)
    V0 = sum(ichain.energies[mod1(n, P)] for n in i+1:m-1)
    V1s = [V(ts) for ts in new_beads]
    V1 = sum(V1s)
    accept_prob = exp(-τ * (V1 - V0))
    accept = rand() < accept_prob
    if accept
        @simd for n in 1:M-1
            @inbounds ichain.coodinates[mod1(n + i, P), :] .= new_beads[n]
            @inbounds ichain.energies[mod1(n + i, P)] = V1s[n]
        end
    end
    return ichain, accept
end


function init_chain(init_cood; P, β, V)
    v = V(init_cood)
    return IChain(β, stack([init_cood for i in 1:P], dims=1), [v for i in 1:P])
end

function simulate(; β::Float64,
    P::Int,
    masses::Vector{Float64},
    M::Int,
    equabrating_number::Int,
    simulating_number::Int,
    simulation_round::Int,
    init_cood::Vector{Float64},
    save_frequency::Int,
    verbose::Bool=false,
    V::Function,
    id::Int,
    output_directory::String
)
    τ = β / P
    DOF = length(masses)
    chain = init_chain(init_cood; P, β, V)
    print("rank $(id) equabrating\n")
    accepts = Vector{Bool}(undef, 0)
    for I in 1:equabrating_number
        accept = MC_update!(chain; P, τ, masses, DOF, M, V)[2]
        push!(accepts, accept)
        if verbose && I % 100 == 0
            print("$(sum(accepts[end-99:end]) / 100)\n")
        end
    end

    print("rank $(id) simulating\n")

    for round in 1:simulation_round
        samples = []
        for I in 1:simulating_number
            accept = MC_update!(chain; P, τ, masses, DOF, M, V)[2]
            if mod(I, save_frequency) == 0
                push!(samples, deepcopy(chain))
            end
            push!(accepts, accept)
            if verbose && I % 100 == 0
                print("$(sum(accepts[end-99:end]) / 100)\n")
            end
        end
        save(joinpath(output_directory, "output_$(id)_$(round).jld2"), Dict("samples" => samples))
        print("rank $(id) round $(round) saved\n")
    end
    print("rank $(id) finished\n")
    return
end

function sample_correlation(coodinates; F, P)
    Fs = [F(coodinates[i, :]) for i in 1:P]
    tau_correlation = vec(mean([Fs[i] ⋅ Fs[mod1(i + distance, P)] for i in 1:P, distance in 0:(P-1)], dims=1))
    return tau_correlation
end

function simulate_correlation(; β::Float64,
    P::Integer,
    masses::Vector{Float64},
    M::Integer,
    equabrating_number::Integer,
    simulating_number::Integer,
    simulation_round::Integer,
    init_cood::Vector{Float64},
    save_frequency::Int,
    verbose::Bool=false,
    V::Function, # potential
    id::Integer, # mpi rank
    F::Function, # correlation function of F
    output_directory::String
)
    τ = β / P
    DOF = length(masses)
    chain = init_chain(init_cood; P, β, V)
    print("rank $(id) equabrating\n")
    accepts = Vector{Bool}(undef, 0)
    for I in 1:equabrating_number
        accept = MC_update!(chain; P, τ, masses, DOF, M, V)[2]
        push!(accepts, accept)
        if verbose && I % 100 == 0
            print("$(sum(accepts[end-99:end]) / 100)\n")
        end
    end

    print("rank $(id) simulating\n")
    supercorrelations = []
    for round in 1:simulation_round
        correlations = []
        for I in 1:simulating_number
            accept = MC_update!(chain; P, τ, masses, DOF, M, V)[2]
            # sample
            if mod(I, save_frequency) == 0
                correlation = sample_correlation(chain.coodinates; F, P)
                push!(correlations, correlation)
            end
            push!(accepts, accept)
            if verbose && I % 100 == 0
                print("$(sum(accepts[end-99:end]) / 100)\n")
            end
        end
        push!(supercorrelations, mean(correlations))
        print("rank $(id) round $(round) saved\n")
    end

    print("rank $(id) finished\n")
    return supercorrelations

end

end
