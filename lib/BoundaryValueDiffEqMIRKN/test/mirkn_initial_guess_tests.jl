using BoundaryValueDiffEqMIRKN, Test

@testset "Test initial guess" begin
    function f!(ddu, du, u, p, t)
        ε = 0.1
        ddu[1] = u[2]
        ddu[2] = (-u[1] * du[2] - u[3] * du[3]) / ε
        ddu[3] = (du[1] * u[3] - u[1] * du[3]) / ε
    end
    function bc!(res, du, u, p, t)
        res[1] = u(0.0)[1]
        res[2] = u(1.0)[1]
        res[3] = u(0.0)[3] + 1
        res[4] = u(1.0)[3] - 1
        res[5] = du(0.0)[1]
        res[6] = du(1.0)[1]
    end
    u0 = [1.0, 1.0, 1.0]
    tspan = (0.0, 1.0)
    prob = SecondOrderBVProblem(f!, bc!, u0, tspan)
    sol1 = solve(prob, MIRKN4(), dt = 0.01, adaptive = false)

    guess = getindex.(getfield.(sol1.u, :x), 1)
    prob_with_guess = SecondOrderBVProblem(f!, bc!, guess, tspan)
    sol2 = solve(prob_with_guess, MIRKN4(), dt = 0.01, adaptive = false, nlsolve_kwargs = (; maxiters = 0))

    # Only test the initial guess, instead of the full solution with derivative solution, since no iterations are performed
    sol2u = getindex.(getfield.(sol2.u, :x), 1)
    @test sol2u == guess
end
