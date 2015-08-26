module OnlineAITests

import OnlineAI
import OnlineStats
import FactCheck
FactCheck.clear_results()

sev = OnlineStats.log_severity()
OnlineStats.log_severity!(OnlineStats.ErrorSeverity)  # turn off most logging

include("nnet_tests.jl")
include("ensembles_test.jl")

# put logging back the way it was
OnlineStats.log_severity!(sev)

FactCheck.exitstatus()

end # module
