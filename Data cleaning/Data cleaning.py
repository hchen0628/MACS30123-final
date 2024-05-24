import dask.dataframe as dd

# Load the CSV data into a Dask DataFrame
df = dd.read_csv('CPS_raw.csv')

# Combined filtering for efficiency
df = df[
    (df['year'] > 1990) &
    (df['age'].between(25, 60)) &
    (~df['empstat'].isin([1, 12, 32, 33, 36])) &
    (~df['classwkr'].isin([10, 13, 14, 29])) &
    (~df['classwly'].isin([10, 13, 14, 29])) &
    (~df['ind1990'].between(940, 960)) &
    (~df['ind90ly'].between(940, 960)) &
    (df['classwkr'] != 99) & (df['classwly'] != 99) &
    (df['ind1990'] != 998) & (~df['ind1950'].isin([997, 998])) & (~df['ind50ly'].isin([997, 998])) &
    (df['ind90ly'] != 998) &
    (~((df['classwly'].between(20, 28)) & (df['fullpart'] == 0))) &
    (~((df['ind50ly'] == 0) & (df['classwly'] != 0))) &
    (~((df['classwly'] == 0) & (df['ind50ly'] != 0))) &
    (~((df['ind1950'] == 0) & (df['classwkr'] != 0))) &
    (~((df['ind1950'] != 0) & (df['classwkr'] == 0))) &
    (df['hispan'] != 901) & (df['hispan'] != 902)
]

# Rename variables
df = df.rename(columns={'ind': 'nextSect', 'indly': 'sect', 'rinc': 'wage'})

# Calculate mean wage by year for normalization
mean_wage = df.groupby('year')['wage'].mean().compute()
mean_wage.name = 'meanwage'

# Merge mean wage back to the main DataFrame and normalize
df = df.merge(mean_wage.to_frame(), on='year', how='left')
df['wage'] = df['wage'] / df['meanwage']
df = df.drop(columns=['meanwage'])  # Clean up after normalization

# Efficient recoding using dictionary mappings
map_sex = {1: 0, 2: 1}
map_marst = {1: 'married', 2: 'married', 3: 'unmarried', 7: 'unmarried'}
map_citizen = {1: 'Native', 2: 'Native', 3: 'Native', 4: 'Immigrant', 5: 'Immigrant'}
df['sex'] = df['sex'].map(map_sex)
df['marst'] = df['marst'].map(map_marst)
df['citizen'] = df['citizen'].map(map_citizen)

# Simplifying the complex mapping of 'bpl'
df['bpl'] = df['bpl'].map(lambda x: 1 if 9900 <= x <= 12090 else 2 if 15000 <= x <= 31000 else 3 if 40000 <= x <= 49900 else 4 if 50000 <= x <= 59900 else 5 if 60010 <= x <= 96000 else x)

# Recoding 'hispan' more efficiently
df['hispan'] = df['hispan'].map(lambda x: 'Not Hispanic' if x == 0 else 'Hispanic' if 100 <= x <= 612 else x)

# Applying conditions for employment based on existing columns
df['employd'] = df.apply(lambda row: (row['empstat'] == 10) and row['classwkr'].between(20, 28) and (row['ahrsworkt'].between(35, 99) or (row['usftptlw'] == 2)), axis=1)
df['employdly'] = df['classwly'].between(20, 28) & (df['fullpart'] == 1)

# Write output directly, avoiding compute
df.to_csv('CPS_cleaned-*.csv', index=False)  # Using Dask's built-in to_csv function which handles partitions automatically
