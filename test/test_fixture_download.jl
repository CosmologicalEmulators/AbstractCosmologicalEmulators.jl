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
