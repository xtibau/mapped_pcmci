# Mapped-PCMCI
Version 0.1

Mapped-PCMCI is an algorithm to estimate causal networks at the grid level. It is advantageous when data is redundant (a large set of variables highly correlated), as it happens, for example, with climate data. MApped-PCMCI uses, Varimax+ to obtain latent none-redundant variables, then PCMCI to obtain the network at the latent level, and finally uses inverts Varimax+ to project the latent network again to the grid level. Its advantages are high in performance and speed. 

# Installation
You can install Mapped-PCMCI using `pip`:

`pip install git+https://github.com/xtibau/mapped-pcmci.git#egg=mapped-pcmci`

# Code example

```python
import numpy as np
from mapped_pcmci.mapped_pcmci import MappedPCMCI
data = np.random.rand(100, 50)  # Samples x Vars -> Samples x Components

# Instantiate the object
mapped = MappedPCMCI(data)

# call it
mapped()

phi = mapped.grid_phi 
```


# Contributions
Any contribution is more than welcome. If you want to collaborate in improving the algorithm, do not hesitate to contact me.  Improvements can be made by adding some tutorials with cool data, improving the algorithm that sorts the weights, or any other cool idea that you may have. 

# License
Mapped-PCMCI is a Free Software project under the GNU General Public License v3, which means all its code is available for everyone to download, examine, use, modify, and distribute, subject to the usual restrictions attached to any GPL software. If you are not familiar with the GPL, see the license.txt file for more details on license terms and other legal issues. 

# References
More information on an upcomming paper. 