from dimod import ExactSolver, BinaryQuadraticModel
from typing import Tuple
import dimod
import networkx as nx
import dwave_networkx as dnx
import json
import math
from dwave.embedding.chain_strength import uniform_torque_compensation
from dwave.embedding import embed_bqm, unembed_sampleset
from dwave.embedding.chain_breaks import MinimizeEnergy, discard, majority_vote, weighted_random
import time
import minorminer

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import logsumexp

def networkx_to_bqm(G, linear_node = None):
    """Convert a NetworkX graph to a binary quadratic model.
    Args:
        G (nx.Graph): The graph to be converted.
        linear_node (int, optional): If provided, the variable index for linear biases.
    Returns:
        :obj:`.BinaryQuadraticModel`
    """
    h, J = {}, {}
    n = G.number_of_nodes() - 1
    for u, v, data in G.edges(data=True):
        w = data.get('weight', 0)
        if (u == linear_node):
            h.update({v: w})
        elif (v == linear_node):
            h.update({u: w})
        else:
            J.update({(u, v) : w})      
    return BinaryQuadraticModel.from_ising(h, J)    


def is_solver_qpu(sampler):
    return (sampler != None) and ('category' in sampler.properties) and (sampler.properties['category'] == "qpu")


def bqm_from_json(file_path) -> BinaryQuadraticModel:
    """
    Reads a file in JSON format and converts it into a binary quadratic model.

    :param file_path: Path to the file containing the JSON.
    :type file_path: str
    :return: The binary quadratic model.
    :rtype: dimod.BinaryQuadraticModel
    """
    with open(file_path) as f:
        data = json.load(f)
        return BinaryQuadraticModel.from_serializable(data) 


def cannonical_ising(bqm):
    """
    Converts a binary quadratic model into an Ising Hamiltonian in cannonical form.
    The cannonical form is defined as the following:
    - The linear and quadratic biases are all non-zeroes.
    - The tuples of quadratic biases (u, v) are sorted so that u < v.
    :param bqm: The binary quadratic model to be converted.
    :type bqm: dimod.BinaryQuadraticModel
    :return: A tuple of dictionaries (h, J) representing the linear and quadratic biases of the Ising Hamiltonian.
    """
    EPSILON = 1e-6
    h, J, offset = bqm.to_ising()
    # Remove zeroes from the linear biases
    h = {i: hi for i, hi in h.items() if abs(hi) > EPSILON}
    # Remove zeroes from the quadratic biases
    J = {tuple(sorted(ij)):Jij for ij, Jij in J.items() if abs(Jij) > EPSILON}    
    return (h, J, offset)


def ising_from_edge_list(file_path) -> Tuple[dict, dict]:
    """
    Reads a file in NetworkX edge list format and converts it into an Ising Hamiltonian.

    :param file_path: Path to the file containing the NetworkX edge list.
    :type file_path: str
    :return: A tuple of (h, J) representing the linear and quadratic biases of the Ising Hamiltonian.
    :rtype: tuple(dict, dict)
    """
    G = nx.read_weighted_edgelist(file_path)
    bqm = networkx_to_bqm(G, linear_node= G.number_of_nodes() - 1)    
    return cannonical_ising(bqm)


def find_max_chain_length(embedding):
    """
    Find the maximum chain length in a given quantum annealer embedding.

    :param dict embedding: A dictionary representing the embedding, 
        where each key is a logical qubit and each value is a list of physical qubits.

    :return: The maximum length of chains in the embedding.
    :rtype: int

    The function iterates through the embedding dictionary, measuring the length
    of the list associated with each logical qubit, and returns the maximum length found.
    """
    max_length = max([ len(chain) for chain in embedding.values()], default=0)
    return max_length


def sampleset_energy(s, target_energy):
    """
    Count the number of samples in the sampleset that 
    have energy less than or equal the target_energy
    params:
    s: dimod.SampleSet
    target_energy: float
    return: tuple (total_num_occurrences with energy <= target_energy, total_num_occurrences)
    """
    a = s.record[s.record.energy <= target_energy].num_occurrences.sum()
    b = s.record.num_occurrences.sum()
    return a, b


def get_min_energy_prob(sampleset):
    """
    Get the minimum energy and the probability of finding the minimum energy (p_solve) from a sampleset.
    parameters:
        sampleset (dimod.SampleSet): The sampleset obtained from a quantum or classical solver.
    returns:
        tuple: A tuple containing the minimum energy and the probability of finding the minimum energy.
    """
    energy = sampleset.first.energy
    a, b = sampleset_energy(sampleset, energy)
    return energy, a/b


def get_heuristic_obj(optimal_obj, optimality_gap, epsilon=1e-10):
    """
    Computes the heuristic objective value given the optimal objective value and the optimality gap.
    heuristic_obj = optimal_obj + abs(optimal_obj) *  optimality_gap
    
    Parameters:
        optimal_obj (float): The optimal or best known objective value.
        optimality_gap (float): The optimality gap as a percentage.
        epsilon (float): A small value to avoid division by zero when the optimal objective is zero.
    
    Returns:
        float: The heuristic objective value.
    """
    return optimal_obj + abs(optimal_obj) * optimality_gap


def get_optimality_gap(optimal_obj, heuristic_obj, epsilon=1e-10):
    """
    Computes the optimality gap between the optimal objective value and a heuristic solution.
    optimality_gap = |(optimal_obj - heuristic_obj) / optimal_obj|
    
    Parameters:
        optimal_obj (float): The optimal or best known objective value.
        heuristic_obj (float): The objective value obtained by a heuristic or approximation algorithm.
        epsilon (float): A small value to avoid division by zero when the optimal objective is zero.
    
    Returns:
        float: The optimality gap
    """
    if abs(optimal_obj) < epsilon:
        # When the optimal objective is effectively zero, we avoid division by zero
        return abs(optimal_obj - heuristic_obj)
    else:
        return abs(optimal_obj - heuristic_obj) / abs(optimal_obj)


def calculate_tts(p_solve, total_time, confidence_level=0.99):
    """
    Calculate the Time-to-Solution (TTS) for a given probability of finding the optimal solution.
    
    :param p_solve: Probability of finding the optimal solution in one run.
    :param total_time: Total time per run in seconds.
    :param confidence_level: Desired confidence level of finding the optimal solution. Default is 0.99.
    :return: Time-to-Solution (TTS) in seconds.
    """
    eps = 1e-10
    if p_solve < -eps or p_solve > 1 + eps:
        raise ValueError("p_solve must be between 0 and 1.")
    if p_solve >= 1 - eps:
        return total_time
    if p_solve <= eps:
        return np.inf
    log_numerator = math.log(1 - confidence_level)
    log_denominator = math.log(1 - p_solve)
    
    # Avoid division by zero error
    if log_denominator == 0:
        raise ValueError("p_solve is too close to 1, leading to division by zero in logarithm calculation.")
    
    tts = (log_numerator / log_denominator) * total_time
    return tts


def chain_break_fractions(embedding, sampleset):
    """
    Calculate the fraction of chain breaks for each variable and the overall fraction.

    :param embedding: The embedding mapping each logical variable to physical spins.
    :type embedding: dict
    :param sampleset: A list of EMBEDDED samples (solutions), where each sample is a dict mapping spin index to value.
    :type sampleset: list
    :return: A dictionary with the fraction of chain breaks for each variable and the overall fraction.
    :rtype: dict
    """

    # Initialize chain break counts for variables mapped to multiple spins    
    total_chain_breaks = 0
    total_samples = 0    

    # Indices of variables in the chains (length > 1)
    chain_ind = { var:[sampleset.variables.index(s) for s in spins] for var, spins in embedding.items() if len(spins) > 1}
    chain_break_counts = {var: 0 for var in chain_ind.keys()}

    # Iterate over each sample to count chain breaks
    for record in sampleset.record:
        noc = record.num_occurrences
        sample_chain_break = False
        for var, spins in chain_ind.items():
            spin_values = [record.sample[s] for s in spins]
            if not all(spin_values[0] == s for s in spin_values):
                chain_break_counts[var] += noc
                sample_chain_break = True
        if sample_chain_break:
            total_chain_breaks += noc
        total_samples += noc

    # Calculate fractions
    chain_break_fractions = {var: count / total_samples for var, count in chain_break_counts.items()}

    # Calculate and add the overall chain break fraction
    chain_break_fractions['overall'] = total_chain_breaks / total_samples

    return chain_break_fractions


def chain_profile(emb, bqm0, sampleset, beta=1, best_energy=None, chain_break_method=discard):
    """
    ======================================================================================
    Calculate the fraction of chain breaks (fractions and probabilities) 
        for each variable and the overall fraction.
    Calculate the chain compliance factions.
    ======================================================================================

    :param embedding: The embedding mapping each logical variable to physical spins.
    :type embedding: dict
    :param sampleset: A list of samples (solutions), where each sample is a dict mapping spin index to value.
    :type sampleset: list
    :param best_energy: The energy of the best solution of ORIGINAL Ising problem (should be calculated theoretically).
    :type best_energy: float
    :return: Dictionaries of chain break fractions and chain compliance factions.
    :rtype: dict
    """

    # If type of chain_break_method is not function but a string, then convert it to function
    if isinstance(chain_break_method, str):
        chain_break_methods = {
            "majority_vote": majority_vote,
            "weighted_random": weighted_random,
            "MinimizeEnergy": MinimizeEnergy(bqm0, emb),
            "discard": discard, # Disabled as it mostly returns EMPTY sampleset
        }
        chain_break_method_name = chain_break_method
        chain_break_method = chain_break_methods[chain_break_method_name]
    else:
        if callable(chain_break_method):
            chain_break_method_name = chain_break_method.__name__
        else:
            chain_break_method_name = "MinimizeEnergy"

    unembedded_sampleset = unembed_sampleset(sampleset, emb, bqm0, chain_break_method=chain_break_method)

    if best_energy is None:
        if unembed_sampleset.record.size == 0:
            best_energy = unembedded_sampleset.first.energy

    embedded_best_energy = sampleset.first.energy # for error correction only
    
    num_cc = 0

    total_count = 0
    total_factor = 0
    total_chain_break_count = 0
    total_chain_break_factor = 0

    # Indices of variables in the chains (length > 1)
    chain_ind = { var:[sampleset.variables.index(s) for s in spins] for var, spins in emb.items() if len(spins) > 1}
    chain_break_count = {var: 0 for var in chain_ind.keys()}
    chain_break_factor = {var: 0 for var in chain_ind.keys()}

    # Iterate over each sample to count chain breaks
    for record in sampleset.record:
        noc = record.num_occurrences
        sample_chain_break = False

        Boltzmann_factor = np.exp(-beta * (record.energy - embedded_best_energy))
        z = noc * Boltzmann_factor

        for var, spins in chain_ind.items():
            spin_values = [record.sample[s] for s in spins]
            if not all(spin_values[0] == s for s in spin_values):
                chain_break_count[var] += noc
                chain_break_factor[var] += z
                sample_chain_break = True

        if sample_chain_break:
            total_chain_break_count += noc
            total_chain_break_factor += z
        else:
            num_cc += noc

        total_count += noc
        total_factor += z

    # Calculate single fractions
    chain_break_fractions = {var: count / total_count for var, count in chain_break_count.items()}
    chain_break_probabilities = {var: factor / total_factor for var, factor in chain_break_factor.items()}

    # Calculate and add the overall (global) chain break fraction and probability
    chain_break_fractions['f_overall'] = total_chain_break_count / total_count
    chain_break_probabilities['p_overall'] = total_chain_break_factor / total_factor

    # Calculate chain compliance factions
    if unembedded_sampleset.record.size > 0 and unembedded_sampleset.first.energy + 1e-6 < best_energy:
        print("unembedded_sampleset.first.energy = ", unembedded_sampleset.first.energy)
        print("theoretical best energy = ", best_energy)
        raise ValueError("The best energy is not the lowest energy in the unembedded sampleset.")

    num_cc0 = 0
    num_cc_solve = 0
    num_cc_solve99 = 0
    num_cc_solve95 = 0
    for record in unembedded_sampleset.record:
        noc = record.num_occurrences

        num_cc0 += noc
        if np.abs(record.energy - best_energy) < 1e-6: num_cc_solve += noc
        if record.energy <= best_energy * .99: num_cc_solve99 += noc
        if record.energy <= best_energy * .95: num_cc_solve95 += noc

    if num_cc != num_cc0:
        print('-'*60)
        print(f"Warning: The number of chain compliant samples is not the same as the number of samples in the unembedded sampleset.\n\
                num_cc = {num_cc}, num_cc0 = {num_cc0}, chain_break_method = {chain_break_method_name}")
        print('-'*60)

    chain_compliance_fractions = {}
    chain_compliance_fractions["num_solve"] = num_cc_solve
    chain_compliance_fractions["pcc"] = num_cc / total_count
    chain_compliance_fractions["psolve"] = num_cc_solve / total_count
    chain_compliance_fractions["psolve99"] = num_cc_solve99 / total_count
    chain_compliance_fractions["psolve95"] = num_cc_solve95 / total_count
    chain_compliance_fractions["psolve_ising"] = num_cc_solve / num_cc if num_cc > 0 else 0
    chain_compliance_fractions["psolve_ising99"] = num_cc_solve99 / num_cc if num_cc > 0 else 0
    chain_compliance_fractions["psolve_ising95"] = num_cc_solve95 / num_cc if num_cc > 0 else 0

    return chain_break_probabilities, chain_break_fractions, chain_compliance_fractions


def get_embedding_properties(bqm, embedding, embedded_bqm):
    """Calculate and return properties of the embedding."""
    h, J, offset = cannonical_ising(bqm)
    he, Je, oe = cannonical_ising(embedded_bqm)
    return {
        "dwave_chain_strength": uniform_torque_compensation(bqm, embedding),
        "max_chain_length": max((len(chain) for chain in embedding.values()), default=0),
        "physical_qubits": len(embedded_bqm.linear),
        "quadratic_terms": len(embedded_bqm.quadratic),
        "h_range": max((abs(hi) for hi in h.values()), default=0),
        "J_range": max((abs(Jij) for Jij in J.values()), default=0),
        "h_range_embedded": max((abs(hi) for hi in he.values()), default=0),
        "J_range_embedded": max((abs(Jij) for Jij in Je.values()), default=0)
    }


def qubo_graph(bqm):
    """
    Return the qubo graph of a BinaryQuadraticModel.
    """
    bqm = bqm.change_vartype(dimod.BINARY)
    qubo = {**bqm.quadratic, **{(i, i): bqm.linear[i] for i in bqm.linear}}
    return qubo


# KEY FUNCTION!!!
def solve_bqm_with_chain_break_methods(bqm, embedding, sampler, **params):
    """Solve a BQM using a D-Wave sampler with different chain break methods."""
    # if sampler has adjacency, use it, otherwise use the default Pegasus graph
    target_adjacency = sampler.adjacency if is_solver_qpu(sampler) else dnx.pegasus_graph(16)
    embedded_bqm = embed_bqm(bqm, embedding, target_adjacency, chain_strength=params.get('chain_strength', None))
    
    # Rescale the embedded BQM
    # NOTE: This step may be not necessary if sampler is DWaveSampler because of auto_scale
    if not is_solver_qpu(sampler):
        h_norm, J_norm, scalar = normalize_ising_with_bqm(embedded_bqm.linear, embedded_bqm.quadratic)
        embedded_bqm.linear.clear()
        embedded_bqm.linear.update(h_norm)
        embedded_bqm.quadratic.clear()
        embedded_bqm.quadratic.update(J_norm)

    sampleset = sampler.sample(embedded_bqm, **params)
    # print(sampleset)
    chain_break_methods = {
        "majority_vote": majority_vote,
        "weighted_random": weighted_random,
        "MinimizeEnergy": MinimizeEnergy(bqm, embedding),
        "discard": discard, # Disabled as it mostly returns EMPTY sampleset
    }
    
    results = {"embedded": sampleset} # Store the embedded sampleset as well
    
    for method_name, method in chain_break_methods.items():
        chain_break_start_time = time.time()
        # print(method)
        cb_sampleset = unembed_sampleset(sampleset, embedding, bqm, chain_break_method=method)
        chain_break_time = time.time() - chain_break_start_time
        cb_sampleset.info['timing']['chain_break_time'] = chain_break_time
        results[method_name] = cb_sampleset
    
    return results


def compute_scaling_factor(embedded_bqm, sampler=None):
    """
    Compute the scaling factor needed to fit the embedded BQM to hardware ranges.
    
    Parameters:
    -----------
    embedded_bqm : dimod.BinaryQuadraticModel
        The embedded BQM to be scaled
    sampler : DWaveSampler
        The D-Wave sampler with hardware properties
        
    Returns:
    --------
    float
        The scaling factor to apply to the BQM
    dict
        Additional information about the scaling analysis
    """
    # Default values of Advantage_system4.1 (chip id)
    # h_range = [-4, 4]
    # j_range = [-1, 1]
    # extended_j_range = [-2, 1]
    # per_qubit_coupling_range = [-18, 15]
    
    # Get the sampler hardware ranges
    if is_solver_qpu(sampler):
        h_range = sampler.properties["h_range"]
        j_range = sampler.properties["extended_j_range"]
        pqc_range = sampler.properties['per_qubit_coupling_range']
    else:
        h_range = [-4, 4]
        j_range = [-2, 1]
        pqc_range = [-18, 15]
    
    # 1. Calculate h_min, h_max, J_min, J_max
    h_values = list(embedded_bqm.linear.values())
    h_min = min(h_values) if h_values else 0
    h_max = max(h_values) if h_values else 0
    
    j_values = list(embedded_bqm.quadratic.values())
    j_min = min(j_values) if j_values else 0
    j_max = max(j_values) if j_values else 0
    
    # 2. Calculate per-qubit coupling sums
    qubit_sums = {v: sum(embedded_bqm.adj[v].values()) for v in embedded_bqm.variables}
    sum_min = min(qubit_sums.values()) if qubit_sums else 0.0
    sum_max = max(qubit_sums.values()) if qubit_sums else 0.0
    
    # Calculate scaling constraints for h and J ranges
    h_j_constraints = [
        h_min / h_range[0] if h_min < 0 and h_range[0] < 0 else 0,
        h_max / h_range[1] if h_max > 0 and h_range[1] > 0 else 0,
        j_min / j_range[0] if j_min < 0 and j_range[0] < 0 else 0,
        j_max / j_range[1] if j_max > 0 and j_range[1] > 0 else 0
    ]
    
    # Calculate scaling constraints for per-qubit coupling sums
    pqc_constraints = [
        sum_min / pqc_range[0] if sum_min < 0 and pqc_range[0] < 0 else 0,
        sum_max / pqc_range[1] if sum_max > 0 and pqc_range[1] > 0 else 0
    ]
    
    # Find the most limiting constraint
    all_constraints = h_j_constraints + pqc_constraints
    scaling_factor = max(all_constraints)
    
    # If no constraints are limiting (all zeros), use a default scaling
    if scaling_factor == 0:
        scaling_factor = 1.0
    
    # Compile and return the analysis results
    analysis = {
        "h_range": {"min": h_min, "max": h_max, "hw_range": h_range},
        "j_range": {"min": j_min, "max": j_max, "hw_range": j_range},
        "pqc_range": {"min": sum_min, "max": sum_max, "hw_range": pqc_range},
        "scaling_constraints": {
            "h_min": h_j_constraints[0],
            "h_max": h_j_constraints[1],
            "j_min": h_j_constraints[2],
            "j_max": h_j_constraints[3],
            "pqc_min": pqc_constraints[0],
            "pqc_max": pqc_constraints[1]
        },
        "limiting_factor":  "h_min" if scaling_factor == h_j_constraints[0] else
                            "h_max" if scaling_factor == h_j_constraints[1] else
                            "j_min" if scaling_factor == h_j_constraints[2] else
                            "j_max" if scaling_factor == h_j_constraints[3] else
                            "pqc_min" if scaling_factor == pqc_constraints[0] else
                            "pqc_max" if scaling_factor == pqc_constraints[1] else
                            "none"
    }
    return scaling_factor, analysis


def per_qubit_coupling_info(bqm, embedding):
    """
    Calculate the coupling information for each qubit in the binary quadratic model.
    Args:
        bqm: The binary quadratic model
        embedding: The embedding mapping each logical variable to physical spins
    Returns:
        A tuple containing:
        - A dictionary with the sum of couplings for each qubit
        - A dictionary with the number of internal couplings for each qubit
    """

    # Calculate the sum of couplings for each qubit
    qubit_couplings = {v: sum(bqm.adj[v].values()) for v in bqm.variables}
    # Count number of chain couplings per qubit
    internal_edge_count =  {v: 0 for v in bqm.variables}
    # Create a mapping from physical qubit to logical variable
    qubit_to_var = {}
    for var, chain in embedding.items():
        for qubit in chain:
            qubit_to_var[qubit] = var
    # For each quadratic interaction
    for (u, v) in bqm.quadratic:
        # Check if both qubits belong to the same logical variable chain
        if qubit_to_var.get(u) == qubit_to_var.get(v):
            internal_edge_count[u] += 1
            internal_edge_count[v] += 1

    return qubit_couplings, internal_edge_count


def scale_preserving_chain_strength(bqm, embedding, sampler=None):
    """
    Calculate the chain strength that preserves the scaling of the Ising problem.
    
    Args:
        bqm: The binary quadratic model
        embedding: The embedding mapping each logical variable to physical spins
        sampler: The D-Wave sampler
    
    Returns:
        The chain strength that preserves the scaling of the Ising problem
    """
    # Default values of Advantage_system4.1 (chip id)
    # h_range = [-4, 4]
    # j_range = [-1, 1]
    # extended_j_range = [-2, 1]
    # per_qubit_coupling_range = [-18, 15]

    # Per-qubit coupling range
    if is_solver_qpu(sampler):
        adjacency = sampler.adjacency
        j_range = sampler.properties["extended_j_range"]
        pqc_range = sampler.properties['per_qubit_coupling_range']
    else:
        adjacency = dnx.pegasus_graph(16)  # Default Pegasus graph
        j_range = [-2, 1]
        pqc_range = [-18, 15]

    # Get the scaling factor for chain strength = 0
    embedded_bqm_0 = embed_bqm(bqm, embedding, adjacency, chain_strength=0)
    scalar, andalysis = compute_scaling_factor(embedded_bqm_0, sampler)
    print(f"Scaling factor: {scalar} and limiting factor (chain_strength=0): {andalysis['limiting_factor']}")

    # Get the per-qubit coupling information
    couplings, internal_edges = per_qubit_coupling_info(embedded_bqm_0, embedding)

    # Calculate the chain strength that preserves the scaling
    chain_strength = -scalar * j_range[0]
    limiting_var_chainstrength = None
    for var, ec in internal_edges.items():
        if ec > 0:
            cbound = (couplings[var]  - pqc_range[0] * scalar)/ec
            if chain_strength > cbound:
                chain_strength = cbound
                limiting_var_chainstrength = var

    print(f"Scale-preserving chain strength: {chain_strength} Limiting qubit: {limiting_var_chainstrength}")
    return chain_strength


# Compute the chain_break probability (if embedding is provided) and p_solve probability
def analyze_sampleset(sampleset, embedding=None):
    """
    Analyze a quantum annealing sampleset to compute minimum energy, solve probability,
    and chain break probability.

    Parameters:
    -----------
    sampleset : dimod.SampleSet
        The sample set returned from a quantum annealer
    embedding : dict, optional
        Embedding used to check for chain breaks (logical variable to physical qubits mapping)

    Returns:
    --------
    min_energy : float or None
        Minimum energy found among consistent samples (None if no consistent samples)
    p_solve : float
        Probability of finding the minimum energy state among consistent samples
    chain_break_prob : float
        Probability of chain breaks in the samples
    chain_break_probs : dict
        Dictionary mapping logical variables to their chain break probabilities
    """
    # Get all samples, energies and occurrences
    samples = sampleset.record.sample
    energies = sampleset.record.energy
    occurrences = sampleset.record['num_occurrences']
    total_occurrences = occurrences.sum()

    # Initialize variables to track consistent samples
    consistent_samples_mask = None
    consistent_occurrences = 0

    # Dictionary to track chain breaks per logical variable
    chain_break_probs = {}

    # If embedding is provided, check for chain breaks
    if embedding is not None:
        # Get variable indices from the sampleset to ensure correct mapping
        variables = sampleset.variables
        var_to_idx = {var: idx for idx, var in enumerate(variables)}

        # Create a mask for consistent samples (no chain breaks)
        consistent_samples_mask = np.ones(len(samples), dtype=bool)

        # Dictionary to track chain breaks for each logical variable
        chain_breaks_count = {v: 0 for v in embedding.keys() if len(embedding[v]) > 1}

        # Check each chain in the embedding
        for v, chain in embedding.items():
            if len(chain) > 1:  # Only check if chain has more than one qubit
                for i, sample in enumerate(samples):
                    # Get values of qubits in this chain, safely handling index mapping
                    chain_indices = [var_to_idx.get(q, None) for q in chain]
                    # Get values of qubits in this chain
                    chain_values = [sample[idx] for idx in chain_indices if idx is not None]

                    if len(chain_values) <= 1:
                        continue  # Skip if we couldn't find any spins in the samples
                    # print(v, chain, chain_indices, chain_values)
                    # If not all values in the chain are the same, mark as inconsistent
                    if not all(val == chain_values[0] for val in chain_values):
                        consistent_samples_mask[i] = False
                        # print("Broken ", v, chain)
                        chain_breaks_count[v] += occurrences[i]

        # Count occurrences of consistent samples
        consistent_occurrences = occurrences[consistent_samples_mask].sum()

        # Calculate overall chain break probability
        chain_break_prob = 1.0 - (consistent_occurrences / total_occurrences)

        # Calculate chain break probability for each logical variable
        for v in chain_breaks_count.keys():
            chain_break_probs[v] = chain_breaks_count[v] / total_occurrences
    else:
        # If no embedding is provided, assume all samples are consistent
        consistent_samples_mask = np.ones(len(samples), dtype=bool)
        consistent_occurrences = total_occurrences
        chain_break_prob = 0.0

    # If there are no consistent samples, return appropriate values
    if consistent_occurrences == 0:
        return None, 0.0, 1.0, chain_break_probs

    # Get energies of consistent samples
    consistent_energies = energies[consistent_samples_mask]
    consistent_occurrences_array = occurrences[consistent_samples_mask]

    # Find minimum energy among consistent samples
    min_energy = consistent_energies.min()
    if min_energy is None:
        min_energy = np.nan

    # Calculate p_solve using only consistent samples
    min_energy_occurrences = consistent_occurrences_array[consistent_energies == min_energy].sum()
    p_solve = min_energy_occurrences / total_occurrences

    return min_energy, p_solve, chain_break_prob, chain_break_probs


# Rescale Ising to fit hardware ranges of biases and quadratic biases
def normalize_ising_with_bqm(h, J, h_range=[-4, 4], j_range=[-2, 1]):
    """
    Normalizes an Ising model using BinaryQuadraticModel.normalize.

    Parameters:
    -----------
    h : dict
        External field terms where h[v] is the external field on site v
    J : dict
        Interaction terms where J[(u,v)] is the interaction between sites u, v
    h_range : list, optional
        Desired range for h values as [min, max], default: [-4, 4]
    j_range : list, optional
        Desired range for J values as [min, max], default: [-2, 1]

    Returns:
    --------
    tuple: (normalized_h, normalized_J, scalar)
        normalized_h: The normalized external field terms
        normalized_J: The normalized interaction terms
        scalar: The scale factor applied
    """
    # Create a BinaryQuadraticModel from the Ising model
    bqm = BinaryQuadraticModel.from_ising(h, J)

    # Normalize the BQM - this returns the scaling factor directly
    scalar = bqm.normalize(bias_range=h_range, quadratic_range=j_range)

    # Extract the normalized Ising model
    normalized_h, normalized_J,_ = bqm.to_ising()

    return normalized_h, normalized_J, scalar


def get_psolve_Gibbs(h, J, beta, embedding=None):
    """
    Computes the probability of finding the minimum energy state in an Ising model
    using the Boltzmann distribution with inverse temperature beta.
    If an embedding is provided, also calculates the chain break probability
    and only considers samples without chain breaks for min energy and p_solve.

    Parameters:
    -----------
    h : dict
        External field terms where h[v] is the external field on site v
    J : dict
        Interaction terms where J[(u,v)] is the interaction between sites u, v
    beta : float
        Inverse temperature parameter (β = 1/kT)
    embedding : dict, optional
        Embedding dictionary mapping logical variables to physical qubits

    Returns:
    --------
    tuple: (min_energy, p_solve, chain_break_prob, chain_break_probs)
        min_energy: The minimum energy found (only considering consistent chains if embedding provided)
        p_solve: Probability of finding the minimum energy state (only considering consistent chains if embedding provided)
        chain_break_prob: Overall probability of chain breaks (None if no embedding provided)
        chain_break_probs: Dictionary mapping logical variables to their chain break probabilities
    """
    # Use ExactSolver to get all spin configurations and energies
    solver = ExactSolver()
    sampleset = solver.sample_ising(h, J)

    # Extract energies and samples
    energies = sampleset.record.energy
    samples = sampleset.record.sample

    # Get variable indices
    variables = sampleset.variables
    var_to_idx = {var: idx for idx, var in enumerate(variables)}

    # Initialize array to track if a configuration has consistent chains
    has_consistent_chains = np.ones(len(samples), dtype=bool)  # Default all True
    chain_break_prob = None
    has_broken_chain = None

    # Dictionary to track chain breaks per logical variable
    chain_break_probs = {}

    # Check chain consistency if embedding is provided
    if embedding is not None:
        # Initialize array to track if a configuration has a broken chain
        has_broken_chain = np.zeros(len(samples), dtype=bool)  # Default all False

        # Dictionary to track broken chains per logical variable
        chain_broken = {logical_var: np.zeros(len(samples), dtype=bool) for logical_var in embedding.keys()}

        # Check each logical variable (chain)
        for logical_var, physical_qubits in embedding.items():
            if len(physical_qubits) <= 1:
                continue  # Skip chains of length 1 (no chain to break)

            # Convert physical qubit names to indices in the sample array
            qubit_indices = [var_to_idx[qubit] for qubit in physical_qubits if qubit in var_to_idx]

            if len(qubit_indices) <= 1:
                continue  # Skip if we couldn't find the qubits in the samples

            # Check each sample
            for i, sample in enumerate(samples):
                # Extract the spin values for qubits in this chain
                chain_values = sample[qubit_indices]

                # Check if all values in the chain are the same
                if not np.all(chain_values == chain_values[0]):
                    chain_broken[logical_var][i] = True  # Mark this chain as broken for this sample
                    has_broken_chain[i] = True
                    has_consistent_chains[i] = False

    # Filter to only consider samples with consistent chains if embedding is provided
    consistent_indices = np.where(has_consistent_chains)[0]

    # If no consistent samples (should be rare), use all samples
    if len(consistent_indices) == 0:
        consistent_indices = np.arange(len(samples))

    # Calculate probabilities using only consistent chains
    consistent_energies = energies[consistent_indices]

    # Find the minimum energy (only among consistent chains if embedding provided)
    min_energy = np.min(consistent_energies)

    # Compute Boltzmann factors with shifted energies for numerical stability
    shifted_energies = energies - min_energy  # Shift all energies relative to consistent min
    log_boltzmann_factors = -beta * shifted_energies
    log_Z = logsumexp(log_boltzmann_factors)  # Full partition function

    # Find ground state indices (only among consistent chains)
    ground_state_indices = np.where(np.logical_and(
        has_consistent_chains,  # Must have consistent chains
        np.isclose(energies, min_energy)  # Must have minimum energy
    ))[0]

    # Calculate ground state probability
    if len(ground_state_indices) > 0:
        log_p_solve = logsumexp(log_boltzmann_factors[ground_state_indices]) - log_Z
        p_solve = np.exp(log_p_solve)
    else:
        p_solve = 0.0

    # Calculate chain break probability if embedding is provided
    if embedding is not None and has_broken_chain is not None:
        broken_indices = np.where(has_broken_chain)[0]
        if len(broken_indices) > 0:
            log_chain_break_prob = logsumexp(log_boltzmann_factors[broken_indices]) - log_Z
            chain_break_prob = np.exp(log_chain_break_prob)
        else:
            chain_break_prob = 0.0

        # Calculate chain break probabilities for each logical variable
        for logical_var, broken_array in chain_broken.items():
            if len(physical_qubits) <= 1:
                chain_break_probs[logical_var] = 0.0  # No chain to break
                continue

            chain_broken_indices = np.where(broken_array)[0]
            if len(chain_broken_indices) > 0:
                log_chain_var_break_prob = logsumexp(log_boltzmann_factors[chain_broken_indices]) - log_Z
                chain_break_probs[logical_var] = np.exp(log_chain_var_break_prob)
            else:
                chain_break_probs[logical_var] = 0.0

    return min_energy, p_solve, chain_break_prob, chain_break_probs
