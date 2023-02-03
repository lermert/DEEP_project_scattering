# DEEP_project_scattering

## Description
Random media in 2-D and 3-D with a von Karman spectrum. Developed in the context of the [DEEP geothermica](http://deepgeothermal.org/home/) project by the [Swiss Seismological Service](www.seismo.ethz.ch) and [Mondaic Ltd](mondaic.com).

based on the description, formulae and recommendations from:
Carpentier, Stefan Filip Anton:
On the estimation of stochastic parameters from deep seismic reflection data and its use in delineating lower crustal structure.
Vol. 281. Utrecht University, 2007.

see also Chapter 2 of:
Sato, H., Fehler, M. C., & Maeda, T. (2012). 
Seismic wave propagation and scattering in the heterogeneous earth. Springer.

as well as:
Obermann, A., Planès, T., Larose, E., Sens-Schönfelder, C., & Campillo, M. (2013). Depth sensitivity of seismic coda waves to velocity perturbations in an elastic heterogeneous medium. Geophysical Journal International, 194(1), 372-382.


## Usage
Usage example:
```
import numpy as np
from von_karman import *
import matplotlib.pyplot as plt

# set up x, y, z vectors
# decide the sampling of the scattering medium here
x = y = z = np.linspace(0., 1000., 200)

# medium parameters
p = VonKarmanParams3D(x=x, y=y, z=z, ax=100., ay=100., az=100., sigma=0.05, nu=0.35)

# realization
m = von_karman_3d(p, np.random.default_rng(1))

# plot 2-D slice
m.sel(y=m.y.min()).T.plot()
plt.show()
```

## Roadmap
To do: bring back periodic repetition for larger areas

## Contributing
Please raise issues on any errors and bugs.
