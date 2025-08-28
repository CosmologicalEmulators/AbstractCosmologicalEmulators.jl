function maximin(input, minmax)
    result = @views @.  (input - minmax[:,1]) ./ (minmax[:,2] - minmax[:,1])
    return result
end

function inv_maximin(input, minmax)
    result = @views @. input * (minmax[:,2] - minmax[:,1]) + minmax[:,1]
    return result
end

function get_emulator_description(input_dict::AbstractDict{String})
    if haskey(input_dict, "parameters")
        println("The parameters the model has been trained are, in the following order: "*input_dict["parameters"]*".")
    else
        @warn "We do not know which parameters were included in the emulators training space. Use this trained emulator with caution!"
    end
    if haskey(input_dict, "author")
        println("The emulator has been trained by "*input_dict["author"]*".")
    end
    if haskey(input_dict, "author_email")
        println(input_dict["author"]*" email is "*input_dict["author_email"]*".")
    end
    if haskey(input_dict, "miscellanea")
        println(input_dict["miscellanea"])
    end
    return nothing
end

function get_emulator_description(emu::AbstractTrainedEmulators)
    if haskey(emu.Description, "emulator_description")
        get_emulator_description(emu.Description["emulator_description"])
    else
        @warn "No emulator description found!"
    end
    return nothing
end
