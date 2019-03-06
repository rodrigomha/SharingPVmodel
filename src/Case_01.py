## This script run the standalone model (that depends on gamma) and compare it with the sharing solution

investment_standalone = solve_standalone(gen_per_m2, load_kw, dataids, a_max_firms, pi_s, pi_r, pi_nm)
aux = (np.sum(sol_sharing) - np.sum(investment_standalone))/np.sum(investment_standalone)
