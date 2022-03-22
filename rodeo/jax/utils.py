r"""
Util functions for Jax kalmantv.

"""
import jax
import jax.numpy as jnp
import jax.scipy as jsp

def block_diag(array):
    r"""
    Convert an array with blocks to a block diagonal matrix.

    Args:
        array (ndarray(n_eval, n_block, n_dim, n_dim)): Array containing blocks of matrices.
    
    Returns:
        (ndarray(n_block * n_dim, n_block * n_dim)): Block diagonal matrix created from the blocks.

    """
    n_eval = array.shape[0]
    mat = jax.vmap(lambda t:
                   jsp.linalg.block_diag(*array[t]))(jnp.arange(n_eval))
        
    return mat
