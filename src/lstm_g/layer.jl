
# See: http://www.overcomplete.net/papers/nn2012.pdf
# Gist: LSTM-g (Generalized Long Short Term Memory) is a more general version of LSTM
#       which can be easily used in alternative network configurations, including 
#       hierarchically stacking.  Connections are gated, as opposed to the activations.

# Important methods: GatedLayers can be `connect`ed together, and `gate`d by another layer.
#   connect(layer_from, layer_to, method = FULLY_CONNECTED)  # or ELEMENTWISE, or SELF_TO_OTHERS
#   gate(gatinglayer, layer_from, layer_to)

"This is the core object... the LSTM-g layer."
immutable GatedLayer
end

