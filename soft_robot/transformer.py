import numpy as np
import fenics as fa


class PoissonRobot:
    def __init__(self, args):
        self.args = args
        self.name = 'robot'
        self._build_mesh()
        self._build_function_space()
        self._set_detailed_boundary_flags()

    def _build_mesh(self):
        self.width = 0.5
        mesh = fa.RectangleMesh(fa.Point(0, 0), fa.Point(self.width, 10), 2, 20, 'crossed')
        self.mesh = mesh

    def _build_function_space(self):
        self.V = fa.VectorFunctionSpace(self.mesh, 'P', 1)
        self.W = fa.FunctionSpace(self.mesh, 'DG', 0)
        self.num_dofs = self.V.dim()
        self.coo_dof = self.V.tabulate_dof_coordinates()

    def _set_detailed_boundary_flags(self):
        x1 = self.coo_dof[:, 0]
        x2 = self.coo_dof[:, 1]
        # [bottom, left_x, left_y, right_x, right_y]
        boundary_flags_list = [np.zeros(self.num_dofs) for i in range(5)]
        counter_left = 0
        counter_right = 0
        for i in range(self.num_dofs):
            if x2[i] < 1e-10:
                boundary_flags_list[0][i] = 1
            else:
                if x1[i] < 1e-10:
                    if counter_left % 2 == 0:
                        boundary_flags_list[1][i] = 1
                    else:
                        boundary_flags_list[2][i] = 1
                    counter_left += 1

                if x1[i] > self.width - 1e-10:
                    if counter_right % 2 == 0:
                        boundary_flags_list[3][i] = 1
                    else:
                        boundary_flags_list[4][i] = 1
                    counter_right += 1
        self.boundary_flags_list = boundary_flags_list

    def compute_areas(self):
        w = fa.Function(self.W)
        area = np.zeros(self.W.dim())
        for i in range(self.W.dim()):
            w.vector()[:] = 0
            w.vector()[i] = 1
            area[i] = fa.assemble(w * fa.dx)
        return area
