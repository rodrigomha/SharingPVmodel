df = pd.read_csv('data/clean_2017_hourly.csv')
df_systemsize = pd.read_csv('data/pv_system_size.csv')
df.drop(columns=['pv_system_size', 'gen'])
df_merge = pd.merge(df, df_systemsize, on='dataid', how='left')
df_merge['gen_per_kW'] = (df_merge.gen/df_merge.pv_system_size)
df_summary = df_merge.groupby(['dataid']).mean()
df_merge.to_csv('clean_2017_hourly_normalized.csv', sep=',')




df = pd.read_csv('data/clean_2017_hourly_normalized.csv')
df_data = df.drop(columns=['pv_system_size', 'gen'])
diff = 1000-194
T = len(gen_per_m2[:,0])
time_data = df[df.dataid==26].localhour.values
for i in range(diff):
    rand_indices = np.random.randint(194, size=3)
    rand_gen = (gen_kw[:,rand_indices[0]] + gen_kw[:,rand_indices[1]]  + gen_kw[:,rand_indices[2]] )/3
    rand_load = (load_kw[:,rand_indices[0]] + load_kw[:,rand_indices[1]] + load_kw[:,rand_indices[2]])/3
    rand_dataid = 990000 + i
    rand_dataid_aux = np.repeat(rand_dataid, T)
    data_aux = {'dataid': rand_dataid_aux, 'localhour': time_data, 'use': rand_load, 'gen_per_kW': rand_gen }
    df_aux = pd.DataFrame(data_aux)
    print(i)
    df_data = pd.concat([df_data,df_aux], sort=False)
    print(i)
    clear_output()
print(df_data[df_data.localhour=='2017-01-01 00:00:00'])   
print(df_data.shape)