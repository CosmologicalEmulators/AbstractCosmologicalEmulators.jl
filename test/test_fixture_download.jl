@testset "Pinned reference fixture restoration" begin
    manifest = joinpath(@__DIR__, "fixtures.sha256")
    firstline = first(readlines(manifest))
    expected, name = split(firstline, "  "; limit=2)
    mktempdir() do tmp
        root = joinpath(tmp, "checkout")
        selected = joinpath(tmp, "selected.sha256")
        write(selected, firstline * "\n")
        ReferenceFixtures.ensure_fixtures(root; manifest=selected)
        dest = joinpath(root, split(name, '/')...)
        @test bytes2hex(open(ReferenceFixtures.sha256, dest)) == expected
        @test !isdir(joinpath(root, "src"))
        # A second run verifies local bytes without needing another download.
        @test ReferenceFixtures.ensure_fixtures(root; manifest=selected) === nothing
        write(dest, "local change")
        @test_throws ErrorException ReferenceFixtures.ensure_fixtures(root; manifest=selected)
        @test read(dest, String) == "local change"
        rm(dest)
        write(selected, repeat("0", 64) * "  " * name * "\n")
        @test_throws ErrorException ReferenceFixtures.ensure_fixtures(root; manifest=selected)
        @test !isfile(dest)
        for bad in ("../outside.txt", "/absolute.txt", "test/../outside.txt", "src/source.jl")
            write(selected, repeat("0", 64) * "  " * bad * "\n")
            @test_throws ErrorException ReferenceFixtures.ensure_fixtures(root; manifest=selected)
        end
    end
end

@testset "Complete fixture archive restoration" begin
    manifest = joinpath(@__DIR__, "fixtures.sha256")
    entries = [split(line, "  "; limit=2) for line in eachline(manifest)]
    names = last.(entries)
    @test length(unique(names)) == length(names)
    @test all(endswith(name, ".txt") for name in names)
    @test count(name -> startswith(name, "test/reference/lux_mlp/"), names) == 9
    mktempdir() do root
        # Start without any fixture files, not just without one representative.
        @test ReferenceFixtures.ensure_fixtures(root; manifest) === nothing
        for (expected, name) in entries
            dest = joinpath(root, split(name, '/')...)
            @test bytes2hex(open(ReferenceFixtures.sha256, dest)) == expected
        end
        restored = sort!([relpath(joinpath(dir, file), root) for
            (dir, _, files) in walkdir(root) for file in files])
        @test restored == sort!([joinpath(split(name, '/')...) for name in names])
        @test !isdir(joinpath(root, "src"))
        # All listed bytes already exist: this path must not need the archive.
        @test ReferenceFixtures.ensure_fixtures(root; manifest) === nothing
    end
end
