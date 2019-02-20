investment_standalone = solve_standalone(gen_per_m2, load_kw, dataids, a_max_firms, pi_s, pi_r, pi_nm)
soluc_sharing = solve_sharing_collective(gen_per_m2, load_kw, dataids_array, a_max_firms, pi_s, pi_r)
sol_sharing = firms_investment_sharing(soluc_sharing, a_max_firms, firms)
aux = (np.sum(sol_sharing) - np.sum(investment_standalone))/np.sum(investment_standalone)
