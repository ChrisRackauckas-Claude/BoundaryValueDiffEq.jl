using SafeTestsets, InteractiveUtils

@info sprint(InteractiveUtils.versioninfo)

# Standard sublibrary test groups (Core / QA). The root test/runtests.jl
# activates this sublibrary and sets BVDE_TEST_GROUP to the standard group name.
# Core (and All) run the functional tests plus the Aqua quality checks, matching
# the previous ReTestItems behavior where the Core leg ran every test item
# (including the :qa-tagged Aqua item). QA runs only the Aqua checks.
const GROUP = get(ENV, "BVDE_TEST_GROUP", "All")

@info "Running tests for group: $(GROUP)"

if GROUP in ("Core", "All")
    @time @safetestset "Convergence on Linear" include("mirkn_convergence_tests.jl")
    @time @safetestset "Example problem from paper" include("mirkn_example_paper_tests.jl")
    @time @safetestset "Test initial guess" include("mirkn_initial_guess_tests.jl")
end

# Aqua runs in Core/All (the old Core leg ran the :qa item) and in QA.
if GROUP in ("Core", "QA", "All")
    @time @safetestset "Quality Assurance" include("qa_tests.jl")
end
