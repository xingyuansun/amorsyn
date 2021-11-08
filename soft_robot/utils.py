import numpy as np
import fenics as fa
import meshio


def batch_mat_vec(sparse_matrix, vector_batch):
    """Supports batch matrix-vector multiplication for both sparse matrix and dense matrix.
    Args:
        sparse_matrix: torch tensor (k, n). Can be sparse or dense.
        vector_batch: torch tensor (b, n).
    Returns:
        vector_batch: torch tensor (b, k)
    """

    # (b, n) -> (n, b)
    matrices = vector_batch.transpose(0, 1)

    # (k, b) -> (b, k)
    return sparse_matrix.mm(matrices).transpose(1, 0)


def boundary_flag_matrix(boundary_flag):
    """something like [0,0,1,1,0] to [[0,0,1,0,0], [0,0,0,1,0]]
    """

    bc_mat = []
    for i, number in enumerate(boundary_flag):
        if number == 1:
            row = np.zeros(len(boundary_flag))
            row[i] = 1
            bc_mat.append(row)
    return np.array(bc_mat)


def save_vtu(attribute, pde, saving_path):
    """Save the solution to .vtu/.pvd format for Paraview
    """
    solution = fa.Function(pde.V)
    solution.vector()[:] = attribute
    file = fa.File(saving_path)
    solution.rename('u', 'u')
    file << solution


def parse_vtu(path):
    """Load .vtu file and parse it using meshio
    Returns
    -------
    mesh.points has shape (N, dim), meaning a collection of N points
    mesh.cells_dict contains connectivity information
    mesh.point_data contains the solution (e.g., the displacement field)
    """
    mesh = meshio.read(path)
    return mesh
