# =============================================================================
# Thread configuration
# =============================================================================

# Requested host task budget for B-spline coefficient solves. Zero means
# automatic: the whole default Julia pool, resolved when a solve runs so that
# precompilation does not freeze a thread count. Only `set_threads` writes it.
const _HOST_THREADS = Threads.Atomic{Int}(0)

"""
    set_threads(n::Integer)

Use up to `n` threads in AbstractCosmologicalEmulators. This single setting
controls both:

- FFTW: the thread count of FFT plans created afterwards, for example by
  `prepare_chebyshev_plan`. Existing plans keep the setting they were created with.
- B-spline host solves: the task budget for large dense `Float32`/`Float64`
  coefficient solves in `bspline_coefficients`, plan evaluation and their
  Zygote/Mooncake pullbacks.

Julia's default thread pool is fixed at startup, so the B-spline budget is
`min(n, Threads.nthreads(:default))`. Start Julia with `--threads=N` or
`JULIA_NUM_THREADS=N` to make workers available. FFTW receives `n` as given.
`set_threads(1)` requests serial host solves and serial newly created FFT plans.
Small batches and vectors stay serial. Other kernels retain their serial paths;
public input adapters may first convert containers to dense floating-point arrays.

If `set_threads` has not been called, large coefficient solves may use the whole
default pool and FFTW keeps its existing setting. When you already parallelize
over independent chains or tasks, call `set_threads(1)` to avoid oversubscription.
Configure before starting concurrent work.
"""
function set_threads(n::Integer)
    1 <= n <= typemax(Cint) ||
        throw(ArgumentError("set_threads requires 1 ≤ n ≤ $(typemax(Cint)); got $n."))
    requested = Int(n)
    FFTW.set_num_threads(requested)
    _HOST_THREADS[] = requested
    return nothing
end

"""
    set_fft_threads(n::Integer)

Deprecated; use [`set_threads`](@ref), which configures FFTW and the B-spline
host task budget together.
"""
function set_fft_threads(n::Integer)
    Base.depwarn(
        "`set_fft_threads` is deprecated; use `set_threads`, which also sets " *
        "the B-spline host thread budget.",
        :set_fft_threads,
    )
    return set_threads(n)
end

# Tasks a host solve may use right now: snapshot once per solve.
function _host_thread_budget()
    requested = _HOST_THREADS[]
    available = Threads.nthreads(:default)
    return requested == 0 ? available : min(requested, available)
end
