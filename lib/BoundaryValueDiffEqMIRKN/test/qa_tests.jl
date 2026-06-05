using BoundaryValueDiffEqMIRKN, Test
using Aqua

@testset "Quality Assurance" begin
    Aqua.test_all(BoundaryValueDiffEqMIRKN)
end
