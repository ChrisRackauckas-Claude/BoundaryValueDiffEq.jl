module BoundaryValueDiffEqCore

using Adapt: adapt
using ADTypes: ADTypes, AbstractADType, AutoSparse, AutoForwardDiff, AutoFiniteDiff,
    AutoPolyesterForwardDiff
using ArrayInterface: parameterless_type
using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase, solve
using DifferentiationInterface: SecondOrder
using ForwardDiff: ForwardDiff, pickchunksize
using Integrals: Integrals, IntegralProblem
using LinearAlgebra: LinearAlgebra, mul!
using LineSearch: BackTracking
using NonlinearSolveFirstOrder: NonlinearSolveFirstOrder, NonlinearSolvePolyAlgorithm,
    GaussNewton, LevenbergMarquardt, NewtonRaphson, NonlinearSolveBase, TrustRegion
using NonlinearSolveBase: NonlinearVerbosity
using OptimizationBase: OptimizationBase, OptimizationVerbosity
using PreallocationTools: PreallocationTools, DiffCache, get_tmp
using RecursiveArrayTools: AbstractVectorOfArray, VectorOfArray, DiffEqArray
using Reexport: @reexport
using SciMLBase: SciMLBase, AbstractBVProblem, BVProblem, NonlinearFunction,
    NonlinearLeastSquaresProblem, NonlinearProblem, OptimizationFunction,
    OptimizationProblem, SecondOrderBVProblem, StandardBVProblem,
    StandardSecondOrderBVProblem, TwoPointBVProblem, TwoPointSecondOrderBVProblem,
    __solve
using SciMLLogging: SciMLLogging, Silent,
    InfoLevel, WarnLevel, @verbosity_specifier,
    None, Minimal, Standard, Detailed, All
using SciMLPublic: @public
using Setfield: @set!
using SparseArrays: sparse
using SparseConnectivityTracer: SparseConnectivityTracer, TracerLocalSparsityDetector
using SparseMatrixColorings: GreedyColoringAlgorithm
using SciMLStructures: SciMLStructures

@reexport using NonlinearSolveFirstOrder:
    AbsNormSafeBestTerminationMode, AbsNormSafeTerminationMode, AbsNormTerminationMode,
    AbsTerminationMode, AbstractAnalyticalProblem, AnalyticalProblem, ArcLengthContinuation,
    BVPFunction, BVProblem, BatchIntegralFunction, CallbackSet, CheckInit, Clocks,
    ContinuousCallback, ConvexOptimizationProblem, DAEFunction, DAEProblem, DAESolution,
    DDEFunction, DDEProblem, DampedNewtonDescent, DescentResult, DiscreteCallback,
    DiscreteFunction, DiscreteProblem, Dogleg, DynamicalBVPFunction, DynamicalDDEFunction,
    DynamicalDDEProblem, DynamicalODEFunction, DynamicalODEProblem, DynamicalSDEFunction,
    DynamicalSDEProblem, EigenvalueProblem, EigenvalueSolution, EigenvalueTarget,
    EisenstatWalkerForcing2, EnsembleAnalysis, EnsembleContext, EnsembleDistributed,
    EnsembleProblem, EnsembleSerial, EnsembleSolution, EnsembleSplitThreads, EnsembleSummary,
    EnsembleTestSolution, EnsembleThreads, FastShortcutNLLSPolyalg, GaussNewton,
    GeneralizedFirstOrderAlgorithm, GeodesicAcceleration, HomotopyNonlinearFunction,
    HomotopyPolyAlgorithm, HomotopyProblem, HomotopySweep, ImplicitDiscreteFunction,
    ImplicitDiscreteProblem, IncrementingODEFunction, IncrementingODEProblem, IntegralFunction,
    IntegralProblem, IntegralSolution, IntervalNonlinearFunction, IntervalNonlinearProblem,
    KantorovichHomotopy, LevenbergMarquardt, LinearAliasSpecifier, LinearProblem, LinearSolution,
    MultiObjectiveOptimizationFunction, NewtonDescent, NewtonRaphson, NoiseProblem,
    NonlinearFunction, NonlinearLeastSquaresProblem, NonlinearProblem, NonlinearSolution,
    NonlinearSolveBase, NonlinearSolveFirstOrder, NonlinearSolvePolyAlgorithm,
    NonlinearVerbosity, NormTerminationMode, ODEAliasSpecifier, ODEFunction, ODEInputFunction,
    ODEProblem, ODESolution, OptimizationFunction, OptimizationProblem, OptimizationSolution,
    PDENoTimeSolution, PDEProblem, PDETimeSeriesSolution, PostconditionSpace,
    PostconditionSpecifier, PseudoTransient, RODEFunction, RODEProblem, RODESolution,
    RadiusUpdateSchemes, RelNormSafeBestTerminationMode, RelNormSafeTerminationMode,
    RelNormTerminationMode, RelTerminationMode, ReturnCode, RobustMultiNewton,
    SCCNonlinearProblem, SDDEFunction, SDDEProblem, SDEFunction, SDEProblem,
    SampledIntegralProblem, SciMLBase, SecondOrderBVProblem, SecondOrderDDEProblem,
    SecondOrderODEProblem, SplitFunction, SplitODEProblem, SplitSDEFunction, SplitSDEProblem,
    SteadyStateProblem, SteadyStateSolution, SteepestDescent, TimeDomain, TraceAll, TraceMinimal,
    TraceWithJacobianConditionNumber, TrustRegion, TwoPointBVPFunction, TwoPointBVProblem,
    TwoPointDynamicalBVPFunction, TwoPointSecondOrderBVProblem, VectorContinuousCallback,
    add_saveat!, add_tstop!, addat!, addat_non_user_cache!, addsteps!, auto_dt_reset!,
    change_t_via_interpolation!, check_error, check_keywords, deleteat_non_user_cache!,
    derivative_discontinuity!, discretize, du_cache, first_tstop, full_cache, get_dt, get_du,
    get_du!, get_proposed_dt, get_rng, get_tmp_cache, has_rng, has_tstop, init,
    is_discrete_time_domain, isclock, iscontinuous, isdiscrete, isinplace, issolverstepclock,
    pop_tstop!, rand_cache, ratenoise_cache, reeval_internals_due_to_modification!, reinit!,
    remake, resize_non_user_cache!, savevalues!, set_abstol!, set_proposed_dt!, set_reltol!,
    set_rng!, set_t!, set_u!, solve, solve!, step!, supports_solve_rng, symbolic_discretize,
    terminate!, u_cache, u_modified!, user_cache, warn_compat
@reexport using SciMLBase:
    BVPFunction, BVProblem, DynamicalBVPFunction, NonlinearFunction,
    NonlinearLeastSquaresProblem, NonlinearProblem, OptimizationFunction,
    OptimizationProblem, ReturnCode, SecondOrderBVProblem, TwoPointBVProblem,
    TwoPointSecondOrderBVProblem, init, remake, solve

include("verbosity.jl")
include("types.jl")
include("solution_utils.jl")
include("utils.jl")
include("internal_problems.jl")
include("algorithms.jl")
include("abstract_types.jl")
include("alg_utils.jl")
include("default_internal_solve.jl")
include("calc_errors.jl")

function SciMLBase.__solve(
        prob::AbstractBVProblem,
        alg::AbstractBoundaryValueDiffEqAlgorithm, args...; kwargs...
    )
    cache = SciMLBase.__init(prob, alg, args...; kwargs...)
    return SciMLBase.solve!(cache)
end

export AbstractBoundaryValueDiffEqAlgorithm, BVPJacobianAlgorithm
export DefectControl, GlobalErrorControl, SequentialErrorControl, HybridErrorControl,
    NoErrorControl
export HOErrorControl, REErrorControl
export integral
export BVPVerbosity, _process_verbose_param, DEFAULT_VERBOSE

# Internal API consumed by the solver sublibraries (BoundaryValueDiffEqMIRK,
# BoundaryValueDiffEqFIRK, BoundaryValueDiffEqShooting, BoundaryValueDiffEqAscher,
# BoundaryValueDiffEqMIRKN). Marked public so the sublibraries can import these
# without ExplicitImports flagging them; not exported because they are not part
# of the user-facing API.
@public AbstractBoundaryValueDiffEqCache, AbstractErrorControl, DiffCacheNeeded,
    EvalSol, NoDiffCacheNeeded, __FastShortcutNonlinearPolyalg, __Fix3,
    __add_singular_term!, __any_sparse_ad, __build_cost, __build_solution,
    __cache_trait, __concrete_kwargs, __concrete_solve_algorithm,
    __construct_internal_problem, __default_coloring_algorithm,
    __default_nonsparse_ad, __default_sparse_ad, __default_sparsity_detector,
    __extract_mesh, __extract_problem_details, __extract_u0,
    __flatten_initial_guess, __get_bcresid_prototype, __get_non_sparse_ad,
    __has_initial_guess, __initial_guess, __initial_guess_length,
    __initial_guess_on_mesh, __internal_nlsolve_problem,
    __internal_optimization_problem, __internal_solve,
    __materialize_jacobian_algorithm, __maybe_allocate_diffcache, __maybe_matmul!,
    __needs_diffcache, __resize!, __restructure_sol, __split_kwargs,
    __tunable_part, __use_both_error_control, __vec, __vec_bc, __vec_bc!,
    __vec_f, __vec_f!, __vec_so_bc, __vec_so_bc!, _sparse_like,
    concrete_jacobian_algorithm, diff!, eval_bc_residual, eval_bc_residual!,
    get_dense_ad, interval, nodual_value, recursive_flatten, recursive_flatten!,
    recursive_flatten_twopoint!, recursive_unflatten!, safe_similar, _unwrap_val

end
