
isexpr(x, head) = isa(x, Expr) && x.head == head

macro parent(expr::Expr)
  @assert isexpr(expr, :type)
  tname = expr.args[2]
  tfields = filter(x->!isexpr(x,:line), expr.args[3].args)
  quote
    $(esc(tname)) = $tfields
  end
end


macro inherit(expr::Expr, parents::Symbol...)
  @assert isexpr(expr, :type)
  for parent in parents
    append!(expr.args[end].args, eval(parent))
  end
  quote
    $expr
  end
end



abstract Animal

@parent type CatOrDog
  age::Int
  weight::Float64
end

kilo(x) = x.weight * 0.4535

@inherit type Cat <: Animal end CatOrDog
@inherit type Dog <: Animal end CatOrDog

abstract Country
type USA <: Country end
type UK <: Country end

incharge(a::Animal, ::USA) = "Mike"
incharge(a::Animal, ::UK) = "The Queen?"

play{T<:Animal}(a1::T, a2::T) = "no toys? boring"
play{T<:Animal}(a1::T, a2::T, toys::Int...) = "playing with Ints: $toys"
play{T<:Animal}(a1::T, a2::T, toys::String...) = "playing with Strings: $toys"

play(a1::Cat, a2::Dog, args...) = play(a2, a1, args...)
play(a1::Dog, a2::Cat, args...) = "Grrrrrrrrrrrrr.... meow"

cat = Cat(10, 8.0)
dog = Dog(20, 15.0);

# u = 0
# spikes = [1, 4, 8, 15]
# tstep = 0.1

# α(q, s) = q / (τs - τr) * exp(-s / τs)

# current = 0.
# for t in tstep:tstep:30.
#   du = 
#   ut+1 = (1-lambda) * u
#   du = lambda * u
# end




# # -----------------------------------------------------------------------
# # general algo for each timestep:

# foreach neuron
#   step!(neuron)  # this should decay u towards urest, as well as setting n.fired=false... 
#                   # also decay q (:= total current (pulse)) from all synapses towards 0, then add it to u
# end

# while true
#   didfire = false
#   foreach neuron
#     if u >= threshold
      
#       # reset neuron
#       didfire = true
#       fired=true
#       u = u_refractory

#       # transmit pulse
#       foreach n.synapses
#         pn = synapse.postneuron
#         pulse = basepulse * syn.weight 
#         pn.q += pulse

#         # don't increate u if it already fired
#         if !pn.fired
#           pn.u += pulse
#         end
#       end
#     end
#   end
# end

# # now that we broke out, everything has stepped forward, fired, and transmitted

