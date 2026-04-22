## File Overview

### `auxf.cpp`

**Purpose**  
Auxiliary functions operating only on CPU 

**Important Functions**

| Function Name         | Description                                                              |
|------------------------|--------------------------------------------------------------------------|
| `asgns_gen_rec`     | recursively generate asgn                      |
| `update_css`   | update css for new iteration on n                |
| `new_convol_filt_array`     | do convol and                        |


---

### `chez_util.cu`

**Purpose**  
Provide utilities supporting chez_ice. Include mainly a memory pool(Workspace) and ncss.
 
**Functions**

| Function Name       | Description                                                                |
|----------------------|----------------------------------------------------------------------------|
| `make_workspace`| produce a workspace which contains all cpu and gpu memory  later we will use.           |
| `update_ncss`           | update ncss for new iteraiton n                                       |

---


### `coreset.cu`

**Purpose**  
Implementation of coreset and shuffle

**Functions**

| Function Name       | Description                                                                |
|----------------------|----------------------------------------------------------------------------|
| `shuffledChez_ICE`| shuffle version of ICE |

---

### `coreset_util.cu`

**Purpose**  
Provides utility functions for coreset_util. Include heap(for sorting by loss) and other date structures storing result of one shuffl.


---

### `gpu_utilities.cu`

**Purpose**  
funcitons used to manipulate date on GPU.


---


### `deep_ice_chez.cu`

**Purpose**  
original version of ice


---

