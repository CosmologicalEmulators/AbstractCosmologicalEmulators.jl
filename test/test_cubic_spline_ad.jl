using Test
using AbstractCosmologicalEmulators
using ForwardDiff
using Zygote
using Mooncake
using DifferentiationInterface
import ADTypes: AutoForwardDiff, AutoZygote, AutoMooncake

@testset "Cubic Spline AD Rules" begin
    @testset "Vector Version" begin
        u = rand(10)
        t = collect(range(0.0, 10.0, length=10))
        t_query = collect(range(0.5, 9.5, length=5))
        
        # 1. Coefficients
        function f_coeff_u(u_in)
            h, z = AbstractCosmologicalEmulators._cubic_spline_coefficients(u_in, t)
            return sum(z)
        end
        
        grad_fd = DifferentiationInterface.gradient(f_coeff_u, AutoForwardDiff(), u)
        grad_zy = DifferentiationInterface.gradient(f_coeff_u, AutoZygote(), u)
        grad_mc = DifferentiationInterface.gradient(f_coeff_u, AutoMooncake(; config=Mooncake.Config()), u)
        
        @test grad_fd ≈ grad_zy atol=1e-10
        @test grad_fd ≈ grad_mc atol=1e-10
        
        function f_coeff_t(t_in)
            h, z = AbstractCosmologicalEmulators._cubic_spline_coefficients(u, t_in)
            return sum(z)
        end
        
        grad_fd = DifferentiationInterface.gradient(f_coeff_t, AutoForwardDiff(), t)
        grad_zy = DifferentiationInterface.gradient(f_coeff_t, AutoZygote(), t)
        grad_mc = DifferentiationInterface.gradient(f_coeff_t, AutoMooncake(; config=Mooncake.Config()), t)
        
        @test grad_fd ≈ grad_zy atol=1e-10
        @test grad_fd ≈ grad_mc atol=1e-10
        
        # 2. Eval
        h, z = AbstractCosmologicalEmulators._cubic_spline_coefficients(u, t)
        
        function f_eval_z(z_in)
            res = AbstractCosmologicalEmulators._cubic_spline_eval(u, t, h, z_in, t_query)
            return sum(res)
        end
        
        grad_fd = DifferentiationInterface.gradient(f_eval_z, AutoForwardDiff(), z)
        grad_zy = DifferentiationInterface.gradient(f_eval_z, AutoZygote(), z)
        grad_mc = DifferentiationInterface.gradient(f_eval_z, AutoMooncake(; config=Mooncake.Config()), z)
        
        @test grad_fd ≈ grad_zy atol=1e-10
        @test grad_fd ≈ grad_mc atol=1e-10
        
        function f_eval_tq(tq_in)
            res = AbstractCosmologicalEmulators._cubic_spline_eval(u, t, h, z, tq_in)
            return sum(res)
        end
        
        grad_fd = DifferentiationInterface.gradient(f_eval_tq, AutoForwardDiff(), t_query)
        grad_zy = DifferentiationInterface.gradient(f_eval_tq, AutoZygote(), t_query)
        grad_mc = DifferentiationInterface.gradient(f_eval_tq, AutoMooncake(; config=Mooncake.Config()), t_query)
        
        @test grad_fd ≈ grad_zy atol=1e-10
        @test grad_fd ≈ grad_mc atol=1e-10
        
        # 3. End-to-end
        function f_full_u(u_in)
            res = cubic_spline_interpolation(u_in, t, t_query)
            return sum(res)
        end
        
        grad_fd = DifferentiationInterface.gradient(f_full_u, AutoForwardDiff(), u)
        grad_zy = DifferentiationInterface.gradient(f_full_u, AutoZygote(), u)
        grad_mc = DifferentiationInterface.gradient(f_full_u, AutoMooncake(; config=Mooncake.Config()), u)
        
        @test grad_fd ≈ grad_zy atol=1e-10
        @test grad_fd ≈ grad_mc atol=1e-10
    end
    
    @testset "Matrix Version" begin
        u = rand(10, 3)
        t = collect(range(0.0, 10.0, length=10))
        t_query = collect(range(0.5, 9.5, length=5))
        
        # End-to-end
        function f_full_u(u_in)
            res = cubic_spline_interpolation(u_in, t, t_query)
            return sum(res)
        end
        
        grad_fd = DifferentiationInterface.gradient(f_full_u, AutoForwardDiff(), u)
        grad_zy = DifferentiationInterface.gradient(f_full_u, AutoZygote(), u)
        grad_mc = DifferentiationInterface.gradient(f_full_u, AutoMooncake(; config=Mooncake.Config()), u)
        
        @test grad_fd ≈ grad_zy atol=1e-10
        @test grad_fd ≈ grad_mc atol=1e-10
        
        # Component test for Matrix Coefficients
        function f_coeff_u_mat(u_in)
            h, z = AbstractCosmologicalEmulators._cubic_spline_coefficients(u_in, t)
            return sum(z)
        end
        
        grad_fd = DifferentiationInterface.gradient(f_coeff_u_mat, AutoForwardDiff(), u)
        grad_zy = DifferentiationInterface.gradient(f_coeff_u_mat, AutoZygote(), u)
        grad_mc = DifferentiationInterface.gradient(f_coeff_u_mat, AutoMooncake(; config=Mooncake.Config()), u)
        
        @test grad_fd ≈ grad_zy atol=1e-10
        @test grad_fd ≈ grad_mc atol=1e-10
        
        # Component test for Matrix Eval
        h, z = AbstractCosmologicalEmulators._cubic_spline_coefficients(u, t)
        
        function f_eval_z_mat(z_in)
            res = AbstractCosmologicalEmulators._cubic_spline_eval(u, t, h, z_in, t_query)
            return sum(res)
        end
        
        grad_fd = DifferentiationInterface.gradient(f_eval_z_mat, AutoForwardDiff(), z)
        grad_zy = DifferentiationInterface.gradient(f_eval_z_mat, AutoZygote(), z)
        grad_mc = DifferentiationInterface.gradient(f_eval_z_mat, AutoMooncake(; config=Mooncake.Config()), z)
        
        @test grad_fd ≈ grad_zy atol=1e-10
        @test grad_fd ≈ grad_mc atol=1e-10
    end
end