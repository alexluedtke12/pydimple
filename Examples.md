
# Examples

## Example 1: expected density

In this example, the goal is to estimate the expected density $E_P[p(Z)]$ of a distribution $P$ ([Bickel and Ritov, 1986](https://www.jstor.org/stable/25050710)).

We'll use the following simulated data:
```python
import numpy as np
import pandas as pd

# true expected density for this simulation: 245/143 ~ 1.713
def sim_data_beta(n):
    return pd.DataFrame({'Z': np.random.beta(3,5,n)})

dat = sim_data_beta(1000)
```
The following code estimates the expected density:
```python
import pydimple
# clear the computation graph (see Example 4)
pydimple.Graph().clear()

# define a distribution and a pandas DataFrame containing iid draws
P = pydimple.Distribution(data=dat)

# define a density
dens = pydimple.Density(P, 'Z')
# take the mean of the density applied to a random draw from P
expected_density = pydimple.E(P,dens)

# estimate the expected density and compute 95% confidence interval
pydimple.estimate(expected_density) 
# Example Output:  {'est': 1.699, 'se': 0.035, 'ci': [1.631, 1.768]}
```

## Example 2: Nonparametric $R^2$

This example estimates the nonparametric $R^2$ criterion $1-\int [y-E_P(Y\mid X=x)]^2 dP(x,y)/\mathrm{Var}_P(Y)$, where $P$ is a distribution of $(X,Y)$ ([Williamson et al., 2021](https://doi.org/10.1111/biom.13392)).

We'll use the following simulated data:
```python
import numpy as np
import pandas as pd

# true R2 for this simulation: 500/1229 ~ 0.407
def sim_data_R2(n):
    X1 = np.random.uniform(-1.,1.,n)
    X2 = np.random.uniform(-1.,1.,n)
    Y = (25/9)*(X1**2) + np.random.normal(0.,1.,n)
    
    return pd.DataFrame({
        'X1': X1,
        'X2': X2,
        'Y': Y
    })

dat = sim_data_R2(500)
```
The following code estimates the nonparametric $R^2$ criterion:
```python
import pydimple
# clear the computation graph (see Example 4)
pydimple.Graph().clear()

# define a distribution and a pandas DataFrame containing iid draws
P = pydimple.Distribution(data=dat) 

# variance of Y
v = pydimple.Var(P,'Y')
# conditinal mean of Y given (X1,X2)
mu = pydimple.E(P,'Y',indep_vars=['X1','X2'])
# R2 parameter
# pydimple.RV('Y') denotes a function that takes as input (X1,X2,Y) and returns Y.
R2 = 1-pydimple.E(P,(pydimple.RV('Y')-mu)**2)/v

 # estimate R2 and compute 95% confidence interval
pydimple.estimate(R2)
# Example Output:  {'est': 0.4334, 'se': 0.0206, 'ci': [0.3930, 0.4738]}
```


## Example 3: longitudinal G-formula

This example estimates the longitudinal G-formula parameter ([Robins, 1986](https://doi.org/10.1016/0270-0255(86)90088-6); [Bang and Robins, 2005](https://doi.org/10.1111/j.1541-0420.2005.00377.x)). This parameter is defined in longitudinal settings where covariate-treatment pairs $(X_t,A_t)$ are collected at times $t=0,1,\ldots,T-1$, and then an outcome $Y$ is collected at the end of the study. For $t\in\{0,1,\ldots,T-1\}$, let $H_t$ denote the history $(X_0,A_0,X_1,A_1,\ldots,X_{t-1},A_{t-1},X_t)$ available just before treatment $A_t$ is administered. Also let $H_T=(H_{T-1},A_T,Y)$ denote all of the covariate, treatment, and outcome data. The longitudinal G-formula parameter $\psi(P)$ is defined recursively by letting $\mu_{P,T}(h_T)=y$, $\mu_{P,t}(h_t)=E_P[\mu_{P,t+1}(H_{t+1})\mid A_t=1,H_t=h_t]$ for $t=T-1,T-2,\ldots,0$, and $\psi(P)=E_P[\mu_{P,0}(H_0)]$. Under causal conditions, $\psi(P)$ gives the counterfactual mean outcome when everyone receives treatment $1$ at all time points.

We'll use the following simulated data:
```python
import numpy as np
import pandas as pd

# true longitudinal G-formula parameter for this simulation: 0.5
def sim_data_longitudinal(n):
    X0 = np.random.normal(0.,1.,n)
    A0 = np.random.binomial(1,1/(1+np.exp(-X0)),n)
    X1 = np.random.normal(0.,1.,n)
    A1 = np.random.binomial(1,1/(1+np.exp(-(X1+A0))),n)
    X2 = np.random.normal(X0*A1+A0*X1+X1*A1,1.,n)
    A2 = np.random.binomial(1,1/(1+np.exp(-(X2+A1))),n)
    Y = np.random.binomial(1,1/(1+np.exp(-(X1*A2+A1*X2+X2*A2))),n)
    
    return pd.DataFrame({
        'X0': X0,
        'A0': A0,
        'X1': X1,
        'A1': A1,
        'X2': X2,
        'A2': A2,
        'Y': Y
    })

dat = sim_data_longitudinal(1000)
```
The following code estimates the longitudinal G-formula parameter:
```python
import pydimple
# clear the computation graph (see Example 4)
pydimple.Graph().clear()

# define a distribution and a pandas DataFrame containing iid draws
P = pydimple.Distribution(data=dat) 

# number of time points
T = 3
# pydimple interprets this string as a function that takes as input H_T and returns Y
mu = 'Y'
# recursive definition of longitudinal G-formulat parameter
for t in reversed(range(T)):
    # names of variables in H_t: ['X0','X1',...,'Xt','A0','A1',...,f'A{t-1}']
    H = [f'X{j}' for j in range(t+1)]+[f'A{j}' for j in range(t)]
    # define mu_{P,t}
    mu = pydimple.E(P,dep=mu,indep_vars=H, fixed_vars={f'A{t}==1'})
# longitudinal G-formula parameter
mu = pydimple.E(P,dep=mu)

 # estimate longitudinal G-formula and compute 95% confidence interval
pydimple.estimate(mu)
# Example Output:  {'est': 0.5313, 'se': 0.0304, 'ci': [0.4717, 0.5908]}
```


## Example 4: calling ```pydimple.estimate``` more than once

### Caution: clearing the computation graph
The computation graph is automatically reset at the end of a call to ```pydimple.estimate```. This has several consequences:
1. **Redefining old Nodes and defining new Nodes:** Before you run ```pydimple.estimate``` again, you should redefine all the pydimple Nodes (Distributions, Densities, Expectations, etc.) needed to express your parameter. Not doing so may lead to errors or unexpected results.
2. **Handling interruptions:** If an execution of ```pydimple.estimate``` is interrupted (e.g., manually stopped or due to an error), the graph is not automatically cleared. This can lead to unintended consequences if previous graph states persist into subsequent operations. Therefore, if an interruption occurs, you should explicitly clear the computation graph to prevent any carryover of state:
```python
# clear the computation graph
pydimple.Graph().clear()
```
3. **Best practice:** To avoid complications from lingering graph states, it's recommended to manually clear the graph before defining your parameter and calling ```pydimple.estimate```.


### Estimating two different parameters
For example, the following code first estimates the expected marginal density $E_P[q(Y)]$ and then estimates the expected squared regression $E_P[E_P(Y|X)^2]$.
```python
import pydimple

# in case a previous call to pydimple.estimate was interrupted, clear the computation graph
pydimple.Graph().clear()

# simulate data as in Example 1:
def sim_data_beta(n):
    return pd.DataFrame({'Z': np.random.beta(3,5,n)})
dat = sim_data_beta(1000)

# run same estimation code from Example 1
P = pydimple.Distribution(data=dat)
dens = pydimple.Density(P, 'Z')
expected_density = pydimple.E(P,dens)
est1 = pydimple.estimate(expected_density) 

# clear the computation graph
pydimple.Graph().clear()

# estimate the expected squared density
# must redefine all pydimple Nodes, including P, since the graph has been cleared!
P = pydimple.Distribution(data=dat)
dens = pydimple.Density(P, 'Z')
expected_sq_density = pydimple.E(P,dens**2)
est2 = pydimple.estimate(expected_sq_density) 

print('expected density estimate:',est1)
# Example Output: {'est': 1.682, 'se': 0.034, 'ci': array([1.617, 1.748])}

print('expected squared density estimate:',est2)
# Example Output: {'est': 3.111, 'se': 0.144, 'ci': array([2.828 , 3.393])}
```

### Using pydimple in a Monte Carlo simulation

As another example, the following will estimate the expected density 10 times, based on 10 Monte Carlo replicates:
```python
import pydimple
# in case a previous call to pydimple.estimate was interrupted, clear the computation graph
pydimple.Graph().clear()

for i in range(10):
    # clear the computation graph
    pydimple.Graph().clear()
    # generate new data
    dat = sim_data_beta(1000)
    # run same estimation code from earlier code block
    P = pydimple.Distribution(data=dat)
    dens = pydimple.Density(P, 'Z')
    expected_density = pydimple.E(P,dens)
    est = pydimple.estimate(expected_density)
    # print estimation results
    print(f'estimation output on iteration{i}:',est)
```
