using SciMLTesting
using BoundaryValueDiffEqMIRK
using Test

run_qa(
    BoundaryValueDiffEqMIRK;
    ei_kwargs = (;
        # External internals with no public replacement:
        #   - StandardBVProblem: SciMLBase-owned problem type, not public.
        all_explicit_imports_are_public = (;
            ignore = (:StandardBVProblem,),
        ),
        # SciMLStructures interface (Tunable/canonicalize/isscimlstructure) is not
        # marked public.
        all_qualified_accesses_are_public = (;
            ignore = (:Tunable, :canonicalize, :isscimlstructure),
        ),
    ),
)
