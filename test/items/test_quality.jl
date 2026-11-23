@testitem "Aqua quality checks" tags = [:quality] begin
    using Aqua
    Aqua.test_all(
        Surfacelets;
        persistent_tasks = false,  # system resource check can fail in constrained envs
    )
end

@testitem "ExplicitImports" tags = [:quality] begin
    using ExplicitImports
    # Public-ness is only known from Julia 1.11 (`Base.ispublic`); older versions
    # report every non-exported name as non-public.
    public_checks = VERSION >= v"1.11"
    # Loaded package extensions are checked too; they may use Surfacelets's internals.
    internal = public_checks ? Tuple(n for n in names(Surfacelets; all = true) if !Base.ispublic(Surfacelets, n)) : ()
    test_explicit_imports(
        Surfacelets;
        all_explicit_imports_are_public = public_checks ? (; ignore = internal) : false,
        all_qualified_accesses_are_public = public_checks ? (; ignore = (:ESTIMATE, :MEASURE, :UNALIGNED, :ReshapedArray, :broadcasted, :Broadcasted, internal...)) : false,
    )
end

@testitem "Code complexity" tags = [:quality] begin
    using CodeComplexity
    # No definition in src/ or ext/ may exceed these limits: CodeComplexity's
    # defaults (10 and 15), or the package's most complex definition where that
    # is higher.
    root = pkgdir(Surfacelets)
    dirs = filter(isdir, joinpath.(root, ("src", "ext")))
    for (metric, limit) in ((CyclomaticComplexity(), 11), (CognitiveComplexity(), 21))
        over = [
            (relpath(f.path, root), m.line, m.name, m.value)
                for dir in dirs for f in measure_directory(metric, dir; recursive = true)
                for m in f.functions if m.value > limit
        ]
        @test isempty(over)
    end
end
