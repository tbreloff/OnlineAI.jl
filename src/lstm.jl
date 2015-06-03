
# implements long short term memory as a pluggable node within any network
# see http://www.deeplearning.net/tutorial/lstm.html for some additional background
# and http://www.overcomplete.net/papers/nn2012.pdf for a generalized version

# a memory gate is just a perceptron that takes in [xₜ ; cₜ₋₁ ; hₜ₋₁] as its
# inputs and applies a sigmoid activation.  This "allows passage" of the values when activated.
# note: we assume bias is already included in x
# note: w should have length nin+2

type MemoryGate <: Node
	perceptron::Perceptron
	memblock
end

# TODO: allow cellstate to be a vector?

# the "state" of the memory block
type MemoryCell <: Node
	perceptron::Perceptron # could be a layer??
	memblock
end


type MemoryBlock <: Node
	inputGate::MemoryGate
	forgetGate::MemoryGate
	outputGate::MemoryGate
	lastOutput::Float64
end



function buildMemoryGate(w::VecF, memblock::MemoryBlock)
	MemoryGate(Perceptron(w, SigmoidActivation()), memblock)
end
