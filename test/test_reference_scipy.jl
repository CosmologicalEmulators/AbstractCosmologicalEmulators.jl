using Test
using AbstractCosmologicalEmulators
using DelimitedFiles

@testset "SciPy Reference Verification" begin
    ref_dir = joinpath(@__DIR__, "reference", "cubic_b_spline")

    function load_txt(filename)
        return readdlm(joinpath(ref_dir, filename), Float64)[:]
    end

    function load_txt_mat(filename)
        return readdlm(joinpath(ref_dir, filename), Float64)
    end

    # Track maximum errors for IMPLEMENTATION.md
    max_knot_err = Ref(0.0)
    max_coeff_err = Ref(0.0)
    max_val_err = Ref(0.0)
    max_design_err = Ref(0.0)

    function update_errs(spl, knots, c, vals, xq, val_ref)
        max_knot_err[] = max(max_knot_err[], maximum(abs.(knot_vector(spl.basis) .- knots)))
        max_coeff_err[] = max(max_coeff_err[], maximum(abs.(spl.coefficients .- c)))
        # values are checked clamped or 0 outside, so we only compare where scipy returned a value
        max_val_err[] = max(max_val_err[], maximum(abs.(spl(xq) .- val_ref)))
    end

    @testset "1. Uniform sites, default not-a-knot" begin
        x = load_txt("uniform_default_sites.txt")
        y = load_txt("uniform_default_ordinates.txt")
        knots_scipy = load_txt("uniform_default_knots.txt")
        c_scipy = load_txt("uniform_default_coefficients.txt")
        xq = load_txt("uniform_default_query.txt")
        vals_scipy = load_txt("uniform_default_values.txt")

        # Design matrix data
        xq_valid = load_txt("uniform_default_query_valid.txt")
        design_mat = load_txt_mat("uniform_default_design_matrix.txt")

        valid_idx = (xq .>= x[1]) .& (xq .<= x[end])
        
        spl = CubicBSpline(y, x)

        update_errs(spl, knots_scipy, c_scipy, spl(xq[valid_idx]), xq[valid_idx], vals_scipy[valid_idx])

        @test knot_vector(spl.basis) ≈ knots_scipy atol=1e-13
        @test spl.coefficients ≈ c_scipy atol=1e-13
        @test spl(xq[valid_idx]) ≈ vals_scipy[valid_idx] atol=1e-13
        
        our_design = basis_matrix(spl.basis, xq_valid)
        max_design_err[] = max(max_design_err[], maximum(abs.(our_design .- design_mat)))
        @test our_design ≈ design_mat atol=1e-13
    end

    @testset "2. Nonuniform sites, default not-a-knot" begin
        x = load_txt("nonuniform_default_sites.txt")
        y = load_txt("nonuniform_default_ordinates.txt")
        knots_scipy = load_txt("nonuniform_default_knots.txt")
        c_scipy = load_txt("nonuniform_default_coefficients.txt")
        xq = load_txt("nonuniform_default_query.txt")
        vals_scipy = load_txt("nonuniform_default_values.txt")
        
        xq_valid = load_txt("nonuniform_default_query_valid.txt")
        design_mat = load_txt_mat("nonuniform_default_design_matrix.txt")

        spl = CubicBSpline(y, x)
        valid_idx = (xq .>= x[1]) .& (xq .<= x[end])
        
        update_errs(spl, knots_scipy, c_scipy, spl(xq[valid_idx]), xq[valid_idx], vals_scipy[valid_idx])

        @test knot_vector(spl.basis) ≈ knots_scipy atol=1e-13
        @test spl.coefficients ≈ c_scipy atol=1e-13
        @test spl(xq[valid_idx]) ≈ vals_scipy[valid_idx] atol=1e-13
        
        our_design = basis_matrix(spl.basis, xq_valid)
        max_design_err[] = max(max_design_err[], maximum(abs.(our_design .- design_mat)))
        @test our_design ≈ design_mat atol=1e-13
    end

    @testset "3. Explicit simple internal knots" begin
        x = load_txt("nonuniform_default_sites.txt")
        y = load_txt("nonuniform_default_ordinates.txt")
        knots_scipy = load_txt("explicit_knots_knots.txt")
        c_scipy = load_txt("explicit_knots_coefficients.txt")
        xq = load_txt("nonuniform_default_query.txt")
        vals_scipy = load_txt("explicit_knots_values.txt")

        internal = [1.0, 4.0]
        spl = CubicBSpline(y, x; internal_knots=internal)
        valid_idx = (xq .>= x[1]) .& (xq .<= x[end])
        
        update_errs(spl, knots_scipy, c_scipy, spl(xq[valid_idx]), xq[valid_idx], vals_scipy[valid_idx])

        @test knot_vector(spl.basis) ≈ knots_scipy atol=1e-13
        @test spl.coefficients ≈ c_scipy atol=1e-13
        @test spl(xq[valid_idx]) ≈ vals_scipy[valid_idx] atol=1e-13
    end

    @testset "4. Double internal knot" begin
        x = load_txt("nonuniform_default_sites.txt")
        y = load_txt("nonuniform_default_ordinates.txt")
        knots_scipy = load_txt("double_knots_knots.txt")
        c_scipy = load_txt("double_knots_coefficients.txt")
        xq = load_txt("nonuniform_default_query.txt")
        vals_scipy = load_txt("double_knots_values.txt")

        internal = [2.0, 2.0]
        spl = CubicBSpline(y, x; internal_knots=internal)
        valid_idx = (xq .>= x[1]) .& (xq .<= x[end])
        
        update_errs(spl, knots_scipy, c_scipy, spl(xq[valid_idx]), xq[valid_idx], vals_scipy[valid_idx])

        @test knot_vector(spl.basis) ≈ knots_scipy atol=1e-13
        @test spl.coefficients ≈ c_scipy atol=1e-13
        @test spl(xq[valid_idx]) ≈ vals_scipy[valid_idx] atol=1e-13
    end

    @testset "5. Triple internal knot" begin
        x = load_txt("triple_knots_knots.txt")[5:end-4] # We can't recover x from nothing, but we know it's x_non7
        x_non7 = [0.0, 0.2, 1.5, 2.5, 2.8, 4.5, 5.0]
        y = sin.(x_non7)
        knots_scipy = load_txt("triple_knots_knots.txt")
        c_scipy = load_txt("triple_knots_coefficients.txt")
        xq = load_txt("nonuniform_default_query.txt")
        vals_scipy = load_txt("triple_knots_values.txt")

        internal = [2.0, 2.0, 2.0]
        spl = CubicBSpline(y, x_non7; internal_knots=internal)
        valid_idx = (xq .>= x_non7[1]) .& (xq .<= x_non7[end])
        
        update_errs(spl, knots_scipy, c_scipy, spl(xq[valid_idx]), xq[valid_idx], vals_scipy[valid_idx])

        @test knot_vector(spl.basis) ≈ knots_scipy atol=1e-13
        @test spl.coefficients ≈ c_scipy atol=1e-13
        @test spl(xq[valid_idx]) ≈ vals_scipy[valid_idx] atol=1e-13
    end

    @testset "6. Matrix values (3 series)" begin
        x = load_txt("matrix_sites.txt")
        Y = load_txt_mat("matrix_ordinates.txt")
        c_scipy = load_txt_mat("matrix_coefficients.txt")
        xq = load_txt("nonuniform_default_query.txt")
        vals_scipy = load_txt_mat("matrix_values.txt")

        spl = CubicBSpline(Y, x)
        valid_idx = (xq .>= x[1]) .& (xq .<= x[end])
        
        # Max errors for matrix
        max_coeff_err[] = max(max_coeff_err[], maximum(abs.(spl.coefficients .- c_scipy)))
        max_val_err[] = max(max_val_err[], maximum(abs.(spl(xq[valid_idx]) .- vals_scipy[valid_idx, :])))

        @test spl.coefficients ≈ c_scipy atol=1e-13
        @test spl(xq[valid_idx]) ≈ vals_scipy[valid_idx, :] atol=1e-13
    end

    @info "Max knot error: $(max_knot_err[])"
    @info "Max coeff error: $(max_coeff_err[])"
    @info "Max val error: $(max_val_err[])"
    @info "Max design matrix error: $(max_design_err[])"
end
