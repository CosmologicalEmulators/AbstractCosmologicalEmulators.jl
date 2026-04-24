using Pkg
Pkg.activate("test")
using AbstractCosmologicalEmulators
using OrdinaryDiffEqTsit5 # To trigger the extension
using JET

cosmo = w0waCDMCosmology()
z_num = 0.5
z_arr = [0.1, 0.5, 1.0]

println("--- JET report for D_z (Number) ---")
display(JET.report_opt(D_z, (typeof(z_num), typeof(cosmo))))

println("\n--- JET report for D_z (Vector) ---")
display(JET.report_opt(D_z, (typeof(z_arr), typeof(cosmo))))

println("\n--- JET report for f_z (Number) ---")
display(JET.report_opt(f_z, (typeof(z_num), typeof(cosmo))))

println("\n--- JET report for f_z (Vector) ---")
display(JET.report_opt(f_z, (typeof(z_arr), typeof(cosmo))))

println("\n--- JET report for D_f_z (Number) ---")
display(JET.report_opt(D_f_z, (typeof(z_num), typeof(cosmo))))
