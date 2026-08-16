using SciMLTesting
using BoundaryValueDiffEqMIRKN
using Test

run_qa(
    BoundaryValueDiffEqMIRKN;
    ei_kwargs = (;
        # External internals with no public replacement:
        #   - StandardSecondOrderBVProblem: SciMLBase-owned problem type, not public.
        all_explicit_imports_are_public = (;
            ignore = (:StandardSecondOrderBVProblem,),
        ),
    ),
)
