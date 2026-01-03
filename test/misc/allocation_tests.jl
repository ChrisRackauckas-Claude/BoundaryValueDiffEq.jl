@testitem "Allocation Tests" tags=[:misc] begin
    using BoundaryValueDiffEq.BoundaryValueDiffEqCore: interval, recursive_flatten!,
                                                        __maybe_matmul!
    using LinearAlgebra

    # Test that key utility functions don't allocate after warmup
    @testset "Utils allocation tests" begin
        # Test interval search - should not allocate
        mesh = collect(0.0:0.1:1.0)
        t = 0.55

        # Warmup
        interval(mesh, t)

        # Test no allocations
        allocs = @allocated interval(mesh, t)
        @test allocs == 0
    end

    @testset "Recursive flatten/unflatten allocation tests" begin
        # Test recursive_flatten! - should only allocate the output
        y = [rand(2) for _ in 1:10]
        x = zeros(20)

        # Warmup
        recursive_flatten!(x, y)

        # Test minimal allocations (the function modifies x in-place)
        allocs = @allocated recursive_flatten!(x, y)
        @test allocs == 0
    end

    @testset "Matrix-vector multiplication allocation tests" begin
        # Test __maybe_matmul! - should not allocate when given preallocated arrays
        # Using Matrix and Vector types to hit the mul! path
        A = Matrix(rand(4, 4))
        b = Vector(rand(4))
        c = Vector(zeros(4))

        # Warmup
        __maybe_matmul!(c, A, b)

        # Test no allocations for simple mul!
        allocs = @allocated __maybe_matmul!(c, A, b)
        @test allocs == 0
    end
end
