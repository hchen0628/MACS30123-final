import dask.dataframe as dd
from dask.diagnostics import ProgressBar
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.families import Poisson

# Load data using Dask
data = dd.read_csv('CPS_raw.csv')

# Filter data where sector is 1 and persist in memory for faster access
data1 = data[data['sect'] == 1].persist()

# Define the model formula
formula = 'wage ~ age + I(age**2) + sex + citizen + hispan + \
          marst + famsize + schoolyr + \
          C(educ) + C(health) + C(race) + \
          C(bpl) + C(state) + C(year)'

# Using Dask for parallel computation
with ProgressBar():
    # Compute necessary columns to a pandas DataFrame for model fitting
    data1_computed = data1.compute()

# Building the GLM model with computed pandas DataFrame
model1 = smf.glm(formula=formula, data=data1_computed, family=Poisson(link=sm.families.links.log),
                 weights=data1_computed['asecwt'], missing='drop').fit(cov_type='HC1')

# Output the model summary
print(model1.summary())
