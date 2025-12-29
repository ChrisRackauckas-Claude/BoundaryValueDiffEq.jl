# DAE tests for FIRK collocation methods with mass matrices

# Standard test BVDAE problem from the URI M. ASCHER and RAYMOND J. SPITERI paper
@testitem "Test FIRK solver on DAE problem 1" tags=[:dae] begin
    using BoundaryValueDiffEqFIRK, SciMLBase

    function f1!(du, u, p, t)
        e = 2.7
        du[1] = (1 + u[2] - sin(t)) * u[4] + cos(t)
        du[2] = cos(t)
        du[3] = u[4]
        du[4] = (u[1] - sin(t)) * (u[4] - e^t)
    end

    function f1(u, p, t)
        e = 2.7
        return [(1 + u[2] - sin(t)) * u[4] + cos(t), cos(t),
            u[4], (u[1] - sin(t)) * (u[4] - e^t)]
    end

    function bc1!(res, u, p, t)
        res[1] = u(0.0)[1]
        res[2] = u(0.0)[3] - 1
        res[3] = u(1.0)[2] - sin(1.0)
    end

    function bc1(u, p, t)
        return [u(0.0)[1], u(0.0)[3] - 1, u(1.0)[2] - sin(1.0)]
    end

    function bca1!(res, ua, p)
        res[1] = ua[1]
        res[2] = ua[3] - 1
    end

    function bcb1!(res, ub, p)
        res[1] = ub[2] - sin(1.0)
    end

    function bca1(ua, p)
        return [ua[1], ua[3] - 1]
    end

    function bcb1(ub, p)
        return [ub[2] - sin(1.0)]
    end

    function f1_analytic(u, p, t)
        return [sin(t), sin(t), 1.0, 0.0]
    end

    u0 = [0.0, 0.0, 1.0, 0.0]
    tspan = (0.0, 1.0)

    # Mass matrix with last row being algebraic (zeros)
    M = [1 0 0 0; 0 1 0 0; 0 0 1 0; 0 0 0 0]

    fun_iip = BVPFunction(f1!, bc1!; mass_matrix = M, analytic = f1_analytic)
    fun_oop = BVPFunction(f1, bc1; mass_matrix = M, analytic = f1_analytic)

    prob_iip = BVProblem(fun_iip, u0, tspan)
    prob_oop = BVProblem(fun_oop, u0, tspan)

    tpprob_iip = TwoPointBVProblem(
        BVPFunction(f1!, (bca1!, bcb1!); mass_matrix = M, analytic = f1_analytic),
        u0, tspan, bcresid_prototype = (zeros(2), zeros(1)))
    tpprob_oop = TwoPointBVProblem(
        BVPFunction(f1, (bca1, bcb1); mass_matrix = M, analytic = f1_analytic),
        u0, tspan, bcresid_prototype = (zeros(2), zeros(1)))

    probArr = [prob_iip, prob_oop, tpprob_iip, tpprob_oop]

    # Test with different FIRK solvers (expanded version, default)
    for prob in probArr
        for solver in (RadauIIa3(), RadauIIa5(), LobattoIIIa3())
            sol = solve(prob, solver; dt = 0.1, adaptive = false)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end

# Test DAE problem 2 - another standard BVDAE
@testitem "Test FIRK solver on DAE problem 2" tags=[:dae] begin
    using BoundaryValueDiffEqFIRK, SciMLBase

    function f2!(du, u, p, t)
        du[1] = -u[3]
        du[2] = -u[3]
        du[3] = u[2] - sin(t - 1)
    end

    function f2(u, p, t)
        return [-u[3], -u[3], u[2] - sin(t - 1)]
    end

    function bc2!(res, u, p, t)
        res[1] = u(1.0)[1]
        res[2] = u(1.0)[2]
    end

    function bc2(u, p, t)
        return [u(1.0)[1], u(1.0)[2]]
    end

    function f2_analytic(u, p, t)
        return [sin(t - 1), sin(t - 1), -cos(t - 1)]
    end

    u0 = [0.0, 0.0, 1.0]
    tspan = (0.0, 1.0)

    # Mass matrix with last row being algebraic (zeros)
    M = [1 0 0; 0 1 0; 0 0 0]

    fun_iip = BVPFunction(f2!, bc2!; mass_matrix = M, analytic = f2_analytic)
    fun_oop = BVPFunction(f2, bc2; mass_matrix = M, analytic = f2_analytic)

    prob_iip = BVProblem(fun_iip, u0, tspan)
    prob_oop = BVProblem(fun_oop, u0, tspan)

    probArr = [prob_iip, prob_oop]

    # Test with different FIRK solvers
    for prob in probArr
        for solver in (RadauIIa3(), RadauIIa5(), LobattoIIIa3())
            sol = solve(prob, solver; dt = 0.1, adaptive = false)
            @test SciMLBase.successful_retcode(sol)
        end
    end
end

# Test that regular ODEs (identity mass matrix) still work
@testitem "Test FIRK solver with identity mass matrix" tags=[:dae] begin
    using BoundaryValueDiffEqFIRK, SciMLBase, LinearAlgebra

    function f!(du, u, p, t)
        du[1] = u[2]
        du[2] = -u[1]
    end

    function f(u, p, t)
        return [u[2], -u[1]]
    end

    function bc!(res, u, p, t)
        res[1] = u(0.0)[1] - 1
        res[2] = u(1.0)[1]
    end

    function bc(u, p, t)
        return [u(0.0)[1] - 1, u(1.0)[1]]
    end

    u0 = [1.0, 0.0]
    tspan = (0.0, 1.0)

    # Explicitly specifying identity mass matrix should be the same as not specifying it
    fun_with_I = BVPFunction(f!, bc!; mass_matrix = I)
    fun_without = BVPFunction(f!, bc!)

    prob_with_I = BVProblem(fun_with_I, u0, tspan)
    prob_without = BVProblem(fun_without, u0, tspan)

    sol_with_I = solve(prob_with_I, RadauIIa5(); dt = 0.1, adaptive = false)
    sol_without = solve(prob_without, RadauIIa5(); dt = 0.1, adaptive = false)

    @test SciMLBase.successful_retcode(sol_with_I)
    @test SciMLBase.successful_retcode(sol_without)

    # Results should be identical
    @test sol_with_I.u ≈ sol_without.u atol = 1e-10
end
