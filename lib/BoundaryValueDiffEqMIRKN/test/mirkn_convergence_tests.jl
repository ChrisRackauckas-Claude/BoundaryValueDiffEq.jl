using BoundaryValueDiffEqMIRKN, Test
using LinearAlgebra, DiffEqDevTools

include("mirkn_convergence_setup.jl")
using .MIRKNConvergenceTests

@testset "Convergence on Linear" begin
    @testset "Problem: $i" for i in (1, 2, 3, 4, 5, 6)
        prob = probArr[i]
        @testset "MIRKN$order" for order in (4, 6)
            sim = test_convergence(
                dts, prob, mirkn_solver(Val(order)); abstol = 1.0e-8, reltol = 1.0e-8
            )
            @test sim.𝒪est[:final] ≈ order atol = testTol
        end
    end
end
