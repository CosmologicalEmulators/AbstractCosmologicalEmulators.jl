import numpy as np
import scipy
from scipy.interpolate import make_interp_spline, BSpline
import sys
import os

def save_data(filename, data):
    # Ensure directory exists if needed, but we run in the same dir
    np.savetxt(filename, data, fmt='%.16e')

def generate():
    # Write metadata
    with open("metadata.txt", "w") as f:
        f.write(f"Python version: {sys.version}\n")
        f.write(f"SciPy version: {scipy.__version__}\n")
        f.write(f"NumPy version: {np.__version__}\n")
    
    # Define a comprehensive query grid for tests (endpoints, out-of-bounds left/right, interior, knots, near-knots)
    xq_unif = np.array([-1.0, 0.0, 0.5, 1.0, 1.000001, 2.5, 4.999999, 5.0, 6.0])
    
    # 1. uniform sites, default not-a-knot
    x_unif = np.linspace(0.0, 5.0, 6)
    y_unif = x_unif**3 - 2*x_unif**2 + 3
    spl_unif = make_interp_spline(x_unif, y_unif, k=3)
    
    save_data("uniform_default_sites.txt", x_unif)
    save_data("uniform_default_ordinates.txt", y_unif)
    save_data("uniform_default_knots.txt", spl_unif.t)
    save_data("uniform_default_coefficients.txt", spl_unif.c)
    save_data("uniform_default_query.txt", xq_unif)
    save_data("uniform_default_values.txt", spl_unif(xq_unif))
    
    # Save design matrix for uniform queries (endpoints/interior only, out of bounds will raise error in BSpline.design_matrix)
    xq_valid = np.array([0.0, 0.5, 1.0, 1.000001, 2.5, 4.999999, 5.0])
    save_data("uniform_default_query_valid.txt", xq_valid)
    design_mat = BSpline.design_matrix(xq_valid, spl_unif.t, 3).toarray()
    save_data("uniform_default_design_matrix.txt", design_mat)
    
    # 2. nonuniform sites, default not-a-knot
    x_non = np.array([0.0, 0.2, 1.5, 2.8, 4.5, 5.0])
    y_non = np.sin(x_non)
    spl_non = make_interp_spline(x_non, y_non, k=3)
    xq_non = np.array([-0.5, 0.0, 0.1, 1.5, 1.500001, 2.0, 4.5, 5.0, 5.5])
    
    save_data("nonuniform_default_sites.txt", x_non)
    save_data("nonuniform_default_ordinates.txt", y_non)
    save_data("nonuniform_default_knots.txt", spl_non.t)
    save_data("nonuniform_default_coefficients.txt", spl_non.c)
    save_data("nonuniform_default_query.txt", xq_non)
    save_data("nonuniform_default_values.txt", spl_non(xq_non))
    
    xq_non_valid = np.array([0.0, 0.1, 1.5, 1.500001, 2.0, 4.5, 5.0])
    save_data("nonuniform_default_query_valid.txt", xq_non_valid)
    design_mat_non = BSpline.design_matrix(xq_non_valid, spl_non.t, 3).toarray()
    save_data("nonuniform_default_design_matrix.txt", design_mat_non)
    
    # 3. explicit strongly nonuniform simple internal knots
    t_int = np.array([1.0, 4.0])
    t_full = np.concatenate(([x_non[0]]*4, t_int, [x_non[-1]]*4))
    spl_exp = make_interp_spline(x_non, y_non, k=3, t=t_full)
    
    save_data("explicit_knots_knots.txt", spl_exp.t)
    save_data("explicit_knots_coefficients.txt", spl_exp.c)
    save_data("explicit_knots_values.txt", spl_exp(xq_non))
    
    # 4. legal double internal knot
    t_double = np.array([2.0, 2.0])
    t_double_full = np.concatenate(([x_non[0]]*4, t_double, [x_non[-1]]*4))
    spl_double = make_interp_spline(x_non, y_non, k=3, t=t_double_full)
    
    save_data("double_knots_knots.txt", spl_double.t)
    save_data("double_knots_coefficients.txt", spl_double.c)
    save_data("double_knots_values.txt", spl_double(xq_non))
    
    # 5. legal triple internal knot
    t_triple = np.array([2.0, 2.0, 2.0])
    x_non7 = np.array([0.0, 0.2, 1.5, 2.5, 2.8, 4.5, 5.0])
    y_non7 = np.sin(x_non7)
    t_triple_full = np.concatenate(([x_non7[0]]*4, t_triple, [x_non7[-1]]*4))
    spl_triple = make_interp_spline(x_non7, y_non7, k=3, t=t_triple_full)
    
    save_data("triple_knots_knots.txt", spl_triple.t)
    save_data("triple_knots_coefficients.txt", spl_triple.c)
    save_data("triple_knots_values.txt", spl_triple(xq_non))
    
    # 6. matrix values (3 distinct series)
    Y_mat = np.column_stack((y_non, x_non**2, np.exp(x_non) * np.cos(x_non)))
    spl_mat = make_interp_spline(x_non, Y_mat, k=3)
    
    save_data("matrix_sites.txt", x_non)
    save_data("matrix_ordinates.txt", Y_mat)
    save_data("matrix_coefficients.txt", spl_mat.c)
    save_data("matrix_values.txt", spl_mat(xq_non))

if __name__ == "__main__":
    generate()
