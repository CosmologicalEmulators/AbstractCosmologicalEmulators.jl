# Intel-coretemp laptop acceptance harness. Pin externally (e.g. taskset -c 2),
# use an absolute --project, and pass the output JSON filename as ARGS[1].
using BenchmarkTools, JSON, LinearAlgebra, ForwardDiff, DifferentiationInterface, Mooncake
using AbstractCosmologicalEmulators, OrdinaryDiffEqTsit5, SciMLSensitivity, Integrals, FastGaussQuadrature
import ADTypes: AutoMooncake
BLAS.set_num_threads(1)
const bg=Base.get_extension(AbstractCosmologicalEmulators,:BackgroundCosmologyExt)
const sensors=[joinpath(d,f) for d in readdir("/sys/class/hwmon";join=true)
    if isfile(joinpath(d,"name")) && strip(read(joinpath(d,"name"),String))=="coretemp"
    for f in readdir(d) if occursin(r"^temp\d+_input$",f)]
temperature()=maximum(parse(Float64,strip(read(p,String)))/1000 for p in sensors)
function measure(b)
    GC.gc()
    deadline=time()+180
    while temperature()>55
        time()<deadline || error("Thermal cooldown failed; reject timings")
        sleep(1)
    end
    before=temperature(); t=run(b); after=temperature()
    (;before,after,samples=length(t.times),valid=abs(after-before)<=5 && after<=60 && length(t.times)==50,
      minimum_us=minimum(t).time/1000,median_us=median(t).time/1000,allocations=t.allocs,bytes=t.memory)
end
function main()
    z=collect(range(0.,5.;length=50));x=[.31,.67,.06,-.9,.1,.01]
    kw(x)=(;mν=(x[3]/6,x[3]/3,x[3]/2),w0=x[4],wa=x[5],Ωk0=x[6])
    ef=x->bg.E_z(z,x[1],x[2];kw(x)...)
    rf=x->bg.r_z(z,x[1],x[2];kw(x)...,order=100)
    r9=x->bg.r_z(z,x[1],x[2];kw(x)...,order=9)
    df=x->bg.D_f_z(z,x[1],x[2];kw(x)...,reltol=1e-6,abstol=1e-8)
    backend=AutoMooncake(;config=Mooncake.Config())
    loss=x->sum(first(df(x)));prep=prepare_gradient(loss,backend,x)
    gf=x->DifferentiationInterface.gradient(loss,prep,backend,x)
    @assert isapprox(gf(x),ForwardDiff.gradient(loss,x);rtol=1e-6,atol=1e-8)
    rloss=x->sum(rf(x));rp=prepare_gradient(rloss,backend,x)
    rg=x->DifferentiationInterface.gradient(rloss,rp,backend,x)
    cases=Any[("E",ef,x),("r9",r9,x),("r100",rf,x),("Df",df,x),("gradD",gf,x),("gradr100",rg,x)]
    for n in (16,128), k in (1,8)
        sites=collect(range(0.,6.;length=n));query=collect(range(.03,5.97;length=n+7))
        u=reduce(hcat,[sin.(sites.*(1+j/10)) for j in 1:k])
        let spline_plan=CubicBSplinePlan(sites,query), values=u
            sloss=v->sum(abs2,spline_plan(v))
            sp=prepare_gradient(sloss,backend,values)
            sg=v->DifferentiationInterface.gradient(sloss,sp,backend,v)
            push!(cases,("spline_$(n)x$k",sg,values))
        end
    end
    cheb_plan=prepare_chebyshev_plan(0.,2.,128)
    for k in (1,8)
        u=reshape(sin.(collect(1.:129k)),129,k)
        push!(cases,("cheb_batch$k",v->chebyshev_decomposition(cheb_plan,v),u))
    end
    benches=map(cases) do (name,f,v)
        f(v)
        b=@benchmarkable $f($v) samples=50 evals=1 seconds=10 gctrial=false gcsample=false
        run(b; samples=name=="gradr100" ? 1 : 50)
        (name,b)
    end
    measure(last(first(benches))) # compile measurement wrapper before accepted rows
    rows=[]
    for (name,b) in benches
        # The baseline 50-sample distance-gradient window heated 55 -> 90 C.
        # Retain allocation evidence, but do not retry/tune its timing gate.
        row = if name=="gradr100"
            t=run(b; samples=1)
            (;name,allocation_only=true,samples=length(t.times),allocations=t.allocs,bytes=t.memory)
        else
            merge((;name),measure(b))
        end
        push!(rows,row)
        println(row);flush(stdout)
        open(ARGS[1],"w") do io
            JSON.print(io,(;source=pathof(AbstractCosmologicalEmulators),rows))
        end
        name=="gradr100" || row.valid || error("Thermal rejection; retain evidence but no timing claim")
    end
end
main()
