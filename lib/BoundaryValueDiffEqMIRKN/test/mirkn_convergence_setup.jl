module MIRKNConvergenceTests

using BoundaryValueDiffEqMIRKN

for order in (4, 6)
    s = Symbol("MIRKN$(order)")
    @eval mirkn_solver(::Val{$order}, args...; kwargs...) = $(s)(args...; kwargs...)
end

function f!(ddu, du, u, p, t)
    return ddu[1] = u[1]
end
function f(du, u, p, t)
    return u[1]
end
function bc!(res, du, u, p, t)
    res[1] = u(0.0)[1] - 1
    return res[2] = u(1.0)[1]
end
function bc(du, u, p, t)
    return [u(0.0)[1] - 1, u(1.0)[1]]
end
function bc_indexing!(res, du, u, p, t)
    res[1] = u[:, 1][1] - 1
    return res[2] = u[:, end][1]
end
function bc_indexing(du, u, p, t)
    return [u[:, 1][1] - 1, u[:, end][1]]
end
function bc_a!(res, du, u, p)
    return res[1] = u[1] - 1
end
function bc_b!(res, du, u, p)
    return res[1] = u[1]
end
function bc_a(du, u, p)
    return [u[1] - 1]
end
function bc_b(du, u, p)
    return [u[1]]
end
analytical_solution = (
    u0, p,
    t,
) -> [(exp(-t) - exp(t - 2)) / (1 - exp(-2)), (-exp(-t) - exp(t - 2)) / (1 - exp(-2))]
u0 = [1.0]
tspan = (0.0, 1.0)
testTol = 0.2
bvpf1 = DynamicalBVPFunction(f!, bc!, analytic = analytical_solution)
bvpf2 = DynamicalBVPFunction(f, bc, analytic = analytical_solution)
bvpf3 = DynamicalBVPFunction(f!, bc_indexing!, analytic = analytical_solution)
bvpf4 = DynamicalBVPFunction(f, bc_indexing, analytic = analytical_solution)
bvpf5 = DynamicalBVPFunction(
    f!, (bc_a!, bc_b!), analytic = analytical_solution,
    bcresid_prototype = (zeros(1), zeros(1)), twopoint = Val(true)
)
bvpf6 = DynamicalBVPFunction(
    f, (bc_a, bc_b), analytic = analytical_solution,
    bcresid_prototype = (zeros(1), zeros(1)), twopoint = Val(true)
)
probArr = [
    SecondOrderBVProblem(bvpf1, u0, tspan), SecondOrderBVProblem(bvpf2, u0, tspan),
    SecondOrderBVProblem(bvpf3, u0, tspan), SecondOrderBVProblem(bvpf4, u0, tspan),
    TwoPointSecondOrderBVProblem(bvpf5, u0, tspan),
    TwoPointSecondOrderBVProblem(bvpf6, u0, tspan),
]
dts = 1 .// 2 .^ (3:-1:1)

export probArr, dts, testTol, mirkn_solver

end
