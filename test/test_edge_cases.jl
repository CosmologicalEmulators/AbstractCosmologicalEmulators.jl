"""
Edge cases and additional coverage tests for AbstractCosmologicalEmulators.

Tests cover:
- Akima interpolation edge cases (boundary, extrapolation, minimal data, scalars)
- Validation utility edge cases (convert_minmax_format, safe_dict_access)
- Normalization utility edge cases (maximin/inv_maximin roundtrip)
- Description utility edge cases (missing metadata)
"""

using Test
using AbstractCosmologicalEmulators
using ForwardDiff
using Zygote

@testset "Edge Cases and Additional Coverage" begin

    # =========================================================================
    # Akima Interpolation Edge Cases
    # =========================================================================

    @testset "Akima Edge Cases: Boundary & Extrapolation" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]
        y = [0.0, 1.0, 4.0, 9.0, 16.0]  # y = x^2

        @testset "Scalar query point" begin
            # Test single point (not array)
            result_scalar = AbstractCosmologicalEmulators.akima_interpolation(y, x, 2.5)
            result_array = AbstractCosmologicalEmulators.akima_interpolation(y, x, [2.5])

            @test length(result_scalar) == 1
            @test result_scalar[1] ≈ result_array[1]
            @test isfinite(result_scalar[1])
        end

        @testset "Boundary points (query at data points)" begin
            # Query exactly at data points - should return exact values
            result = AbstractCosmologicalEmulators.akima_interpolation(y, x, x)
            @test result ≈ y rtol=1e-14

            # Test individual boundary points
            result_first = AbstractCosmologicalEmulators.akima_interpolation(y, x, [x[1]])
            result_last = AbstractCosmologicalEmulators.akima_interpolation(y, x, [x[end]])
            @test result_first[1] ≈ y[1] rtol=1e-14
            @test result_last[1] ≈ y[end] rtol=1e-14
        end

        @testset "Extrapolation behavior" begin
            # Points outside data range
            # Akima should handle extrapolation gracefully
            result_below = AbstractCosmologicalEmulators.akima_interpolation(y, x, [-0.5])
            result_above = AbstractCosmologicalEmulators.akima_interpolation(y, x, [5.0])

            # Test that it doesn't throw and returns finite values
            @test isfinite(result_below[1])
            @test isfinite(result_above[1])

            # Extrapolation should be approximately linear continuation
            # Test that extrapolated values are reasonable (not wildly different)
            @test abs(result_below[1] - y[1]) < 10.0  # Not too far from first point
            @test abs(result_above[1] - y[end]) < 20.0  # Not too far from last point
        end

        @testset "Minimum data points (n=4)" begin
            # Akima needs at least 4 points for the extended slope calculation
            x_min = [0.0, 1.0, 2.0, 3.0]
            y_min = [0.0, 1.0, 4.0, 9.0]
            x_query = [0.5, 1.5, 2.5]

            result = AbstractCosmologicalEmulators.akima_interpolation(y_min, x_min, x_query)
            @test length(result) == 3
            @test all(isfinite.(result))

            # Test gradient computation works with minimal points
            grad_fd = ForwardDiff.gradient(y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x_min, x_query)), y_min)
            grad_zy = Zygote.gradient(y -> sum(AbstractCosmologicalEmulators.akima_interpolation(y, x_min, x_query)), y_min)[1]
            @test grad_fd ≈ grad_zy rtol=1e-9
        end

        @testset "Single query point" begin
            result = AbstractCosmologicalEmulators.akima_interpolation(y, x, [2.5])
            @test length(result) == 1
            @test isfinite(result[1])

            # AD should work with single query point
            grad = Zygote.gradient(y_val -> AbstractCosmologicalEmulators.akima_interpolation(y_val, x, [2.5])[1], y)[1]
            @test all(isfinite.(grad))
        end

        @testset "Empty query (edge case)" begin
            # What happens with empty query array?
            result = AbstractCosmologicalEmulators.akima_interpolation(y, x, Float64[])
            @test length(result) == 0
            @test result isa Vector{Float64}
        end

        @testset "Dense query points (interpolation within intervals)" begin
            # Test many points between two data points
            x_dense = collect(range(1.0, 2.0, length=100))
            result = AbstractCosmologicalEmulators.akima_interpolation(y, x, x_dense)

            @test length(result) == 100
            @test all(isfinite.(result))
            # Values should be between y[2]=1 and y[3]=4
            @test all(result .>= minimum([y[2], y[3]]) - 0.5)  # Small tolerance for interpolation
            @test all(result .<= maximum([y[2], y[3]]) + 0.5)
        end
    end

    @testset "Akima Edge Cases: Matrix Version" begin
        x = [0.0, 1.0, 2.0, 3.0, 4.0]

        @testset "Single column matrix" begin
            y_matrix = reshape([0.0, 1.0, 4.0, 9.0, 16.0], 5, 1)
            x_query = [0.5, 1.5, 2.5]

            result_matrix = AbstractCosmologicalEmulators.akima_interpolation(y_matrix, x, x_query)
            result_vector = AbstractCosmologicalEmulators.akima_interpolation(y_matrix[:, 1], x, x_query)

            @test size(result_matrix) == (3, 1)
            @test result_matrix[:, 1] ≈ result_vector rtol=1e-14
        end

        @testset "Empty query with matrix" begin
            y_matrix = randn(5, 3)
            result = AbstractCosmologicalEmulators.akima_interpolation(y_matrix, x, Float64[])

            @test size(result) == (0, 3)
            @test result isa Matrix{Float64}
        end

        @testset "Boundary points with matrix" begin
            y_matrix = randn(5, 10)
            result = AbstractCosmologicalEmulators.akima_interpolation(y_matrix, x, x)

            @test size(result) == (5, 10)
            @test result ≈ y_matrix rtol=1e-14
        end

        @testset "Extrapolation with matrix" begin
            y_matrix = randn(5, 3)
            x_extrap = [-1.0, 0.0, 2.0, 5.0]

            result = AbstractCosmologicalEmulators.akima_interpolation(y_matrix, x, x_extrap)

            @test size(result) == (4, 3)
            @test all(isfinite.(result))
        end
    end

    # =========================================================================
    # Validation Utility Edge Cases
    # =========================================================================

    @testset "convert_minmax_format: All Input Types" begin
        @testset "Matrix format" begin
            minmax_matrix = [0.0 1.0; 2.0 3.0; 4.0 5.0]
            result = AbstractCosmologicalEmulators.convert_minmax_format(minmax_matrix)

            @test size(result) == (3, 2)
            @test result ≈ minmax_matrix
        end

        @testset "Vector{Vector} format" begin
            minmax_vec_vec = [[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]]
            result = AbstractCosmologicalEmulators.convert_minmax_format(minmax_vec_vec)

            @test size(result) == (3, 2)
            @test result[1, :] ≈ [0.0, 1.0]
            @test result[2, :] ≈ [2.0, 3.0]
            @test result[3, :] ≈ [4.0, 5.0]
        end

        @testset "Dict format" begin
            minmax_dict = Dict("min" => [0.0, 2.0, 4.0], "max" => [1.0, 3.0, 5.0])
            result = AbstractCosmologicalEmulators.convert_minmax_format(minmax_dict)

            @test size(result) == (3, 2)
            @test result[:, 1] ≈ [0.0, 2.0, 4.0]
            @test result[:, 2] ≈ [1.0, 3.0, 5.0]
        end

        @testset "Error cases" begin
            # Empty vector
            @test_throws ArgumentError AbstractCosmologicalEmulators.convert_minmax_format([])

            # Wrong size in nested vector
            @test_throws ArgumentError AbstractCosmologicalEmulators.convert_minmax_format([[1.0]])
            @test_throws ArgumentError AbstractCosmologicalEmulators.convert_minmax_format([[1.0, 2.0, 3.0]])

            # Missing 'max' key in dict
            @test_throws ArgumentError AbstractCosmologicalEmulators.convert_minmax_format(Dict("min" => [1.0]))

            # Mismatched lengths in dict
            @test_throws ArgumentError AbstractCosmologicalEmulators.convert_minmax_format(
                Dict("min" => [1.0, 2.0], "max" => [3.0])
            )

            # Wrong number of columns in matrix
            @test_throws ArgumentError AbstractCosmologicalEmulators.convert_minmax_format([1.0 2.0 3.0])

            # Unsupported type
            @test_throws ArgumentError AbstractCosmologicalEmulators.convert_minmax_format(42)
            @test_throws ArgumentError AbstractCosmologicalEmulators.convert_minmax_format("invalid")
        end
    end

    @testset "safe_dict_access: Nested Access & Errors" begin
        test_dict = Dict("a" => Dict("b" => Dict("c" => 42)))

        @testset "Successful nested access" begin
            @test AbstractCosmologicalEmulators.safe_dict_access(test_dict, "a", "b", "c") == 42
            @test AbstractCosmologicalEmulators.safe_dict_access(test_dict, "a", "b") == Dict("c" => 42)
            @test AbstractCosmologicalEmulators.safe_dict_access(test_dict, "a") == Dict("b" => Dict("c" => 42))
        end

        @testset "Failed access with context" begin
            # Missing key at first level
            err = try
                AbstractCosmologicalEmulators.safe_dict_access(test_dict, "x"; context="test_context")
            catch e
                e
            end
            @test isa(err, ArgumentError)
            @test contains(string(err), "x")
            @test contains(string(err), "test_context")

            # Missing key at nested level
            err = try
                AbstractCosmologicalEmulators.safe_dict_access(test_dict, "a", "x", "y"; context="nested_test")
            catch e
                e
            end
            @test isa(err, ArgumentError)
            @test contains(string(err), "a → x → y")
            @test contains(string(err), "nested_test")
        end

        @testset "Empty context" begin
            err = try
                AbstractCosmologicalEmulators.safe_dict_access(test_dict, "missing")
            catch e
                e
            end
            @test isa(err, ArgumentError)
            @test contains(string(err), "missing")
        end
    end

    # =========================================================================
    # Normalization Utility Edge Cases
    # =========================================================================

    @testset "maximin/inv_maximin: Edge Cases" begin
        @testset "Roundtrip property (identity)" begin
            # For any valid input and minmax, roundtrip should be identity
            minmax = [0.0 10.0; -5.0 5.0; 1e-6 1e6]
            input = [5.0, 0.0, 100.0]

            normalized = AbstractCosmologicalEmulators.maximin(input, minmax)
            recovered = AbstractCosmologicalEmulators.inv_maximin(normalized, minmax)

            @test recovered ≈ input rtol=1e-14
        end

        @testset "Negative ranges" begin
            minmax = [-10.0 -5.0; -2.0 2.0]
            input = [-7.5, 0.0]

            result = AbstractCosmologicalEmulators.maximin(input, minmax)
            @test result[1] ≈ 0.5
            @test result[2] ≈ 0.5

            # Inverse should recover original
            recovered = AbstractCosmologicalEmulators.inv_maximin(result, minmax)
            @test recovered ≈ input rtol=1e-14
        end

        @testset "Extreme values" begin
            # Very large ranges
            minmax = [1e-10 1e10; -1e8 1e8]
            input = [1e5, 0.0]

            normalized = AbstractCosmologicalEmulators.maximin(input, minmax)
            @test all(isfinite.(normalized))

            recovered = AbstractCosmologicalEmulators.inv_maximin(normalized, minmax)
            @test recovered ≈ input rtol=1e-6  # Larger tolerance due to extreme values
        end

        @testset "Boundary values (at min and max)" begin
            minmax = [0.0 1.0; 10.0 20.0]

            # Input at minimum
            input_min = [0.0, 10.0]
            normalized_min = AbstractCosmologicalEmulators.maximin(input_min, minmax)
            @test normalized_min ≈ [0.0, 0.0] rtol=1e-14

            # Input at maximum
            input_max = [1.0, 20.0]
            normalized_max = AbstractCosmologicalEmulators.maximin(input_max, minmax)
            @test normalized_max ≈ [1.0, 1.0] rtol=1e-14

            # Roundtrip
            @test AbstractCosmologicalEmulators.inv_maximin(normalized_min, minmax) ≈ input_min rtol=1e-14
            @test AbstractCosmologicalEmulators.inv_maximin(normalized_max, minmax) ≈ input_max rtol=1e-14
        end

        @testset "Automatic differentiation compatibility" begin
            # Test that AD works through maximin/inv_maximin
            minmax = [0.0 10.0; -5.0 5.0]
            input = [5.0, 0.0]

            # ForwardDiff through maximin
            grad_maximin = ForwardDiff.gradient(x -> sum(AbstractCosmologicalEmulators.maximin(x, minmax)), input)
            @test all(isfinite.(grad_maximin))

            # ForwardDiff through inv_maximin
            normalized = AbstractCosmologicalEmulators.maximin(input, minmax)
            grad_inv = ForwardDiff.gradient(x -> sum(AbstractCosmologicalEmulators.inv_maximin(x, minmax)), normalized)
            @test all(isfinite.(grad_inv))

            # Zygote through roundtrip
            grad_roundtrip = Zygote.gradient(x -> sum(AbstractCosmologicalEmulators.inv_maximin(
                AbstractCosmologicalEmulators.maximin(x, minmax), minmax
            )), input)[1]
            @test grad_roundtrip ≈ ones(2) rtol=1e-10  # Roundtrip derivative should be identity
        end
    end

    # =========================================================================
    # Description Utility Edge Cases
    # =========================================================================

    @testset "get_emulator_description: Missing/Partial Metadata" begin
        @testset "Minimal description (only parameters)" begin
            minimal_dict = Dict("parameters" => "θ₁, θ₂")
            # Should not throw, just print parameters
            @test_nowarn AbstractCosmologicalEmulators.get_emulator_description(minimal_dict)
        end

        @testset "No parameters (should warn)" begin
            no_params = Dict("author" => "Test Author")
            # Should warn about missing parameters
            @test_logs (:warn, r"do not know which parameters") AbstractCosmologicalEmulators.get_emulator_description(no_params)
        end

        @testset "Full description" begin
            full_dict = Dict(
                "parameters" => "Ωₘ, σ₈, h",
                "author" => "Jane Doe",
                "author_email" => "jane@example.com",
                "miscellanea" => "Trained on Planck 2018 data"
            )
            @test_nowarn AbstractCosmologicalEmulators.get_emulator_description(full_dict)
        end

        @testset "Empty dict" begin
            empty_dict = Dict{String,Any}()
            # Should warn and not throw
            @test_logs (:warn, r"do not know which parameters") AbstractCosmologicalEmulators.get_emulator_description(empty_dict)
        end

        @testset "Only author email (no author name)" begin
            # Edge case: email without author name causes KeyError
            # This is a known limitation - author_email requires author to be present
            dict_email_only = Dict("author_email" => "test@example.com")
            @test_throws KeyError AbstractCosmologicalEmulators.get_emulator_description(dict_email_only)
        end
    end

    # =========================================================================
    # Additional Validation Edge Cases
    # =========================================================================

    @testset "validate_cosmological_ranges: Edge Cases" begin
        @testset "Valid ranges" begin
            valid_ranges = [0.0 1.0; 0.2 0.4; -1.0 1.0]
            @test_nowarn AbstractCosmologicalEmulators.validate_cosmological_ranges(valid_ranges)
        end

        @testset "Invalid: min >= max" begin
            # Exactly equal
            invalid_equal = [0.0 1.0; 0.5 0.5]
            @test_throws ArgumentError AbstractCosmologicalEmulators.validate_cosmological_ranges(invalid_equal)

            # min > max
            invalid_reversed = [0.0 1.0; 1.0 0.5]
            @test_throws ArgumentError AbstractCosmologicalEmulators.validate_cosmological_ranges(invalid_reversed)
        end
    end

    @testset "validate_minmax_data: Degenerate Ranges" begin
        @testset "Degenerate range (min ≈ max)" begin
            # Range with zero width
            degenerate = [0.0 0.0; 1.0 2.0]
            err = try
                AbstractCosmologicalEmulators.validate_minmax_data(degenerate)
            catch e
                e
            end
            @test isa(err, ArgumentError)
            @test contains(string(err), "Degenerate normalization ranges")
        end

        @testset "Nearly degenerate range (very small width)" begin
            # Range with tiny width (< 1e-15)
            nearly_degenerate = [0.0 1e-16; 1.0 2.0]
            err = try
                AbstractCosmologicalEmulators.validate_minmax_data(nearly_degenerate)
            catch e
                e
            end
            @test isa(err, ArgumentError)
            @test contains(string(err), "Degenerate")
        end

        @testset "Multiple degenerate ranges" begin
            multi_degenerate = [0.0 0.0; 1.0 1.0; 2.0 3.0]
            err = try
                AbstractCosmologicalEmulators.validate_minmax_data(multi_degenerate)
            catch e
                e
            end
            @test isa(err, ArgumentError)
            # Should report all degenerate indices
            @test contains(string(err), "parameter indices")
        end
    end

end  # End of main testset
