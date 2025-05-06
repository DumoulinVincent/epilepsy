NEURON {
    POINT_PROCESS HomeoPlasticSyn
    RANGE tau, eta, target_rate, weight, max_weight, min_weight
    NONSPECIFIC_CURRENT i
}

PARAMETER {
    tau = 1000 (ms)        : Time constant for post rate decay
    eta = 0.001 (/ms)      : Learning rate
    target_rate = 5 (Hz)   : Desired firing rate
    weight = 0.01          : Initial synaptic weight
    max_weight = 0.1
    min_weight = 0.001
}

ASSIGNED {
    v (mV)
    i (nA)
    post_trace (Hz)
    last_post_spike (ms)
}

INITIAL {
    post_trace = 0
    last_post_spike = -1e9
}

BREAKPOINT {
    i = 0  : No direct current, plasticity only
}

NET_RECEIVE (w) {
    if (flag == 0) {
        : Presynaptic spike
        post_trace = post_trace * exp(-(t - last_post_spike)/tau)
        
        if (post_trace < target_rate) {
            weight = weight + eta * (target_rate - post_trace)
            if (weight > max_weight) {
                weight = max_weight
            }
        }
    } else if (flag == 1) {
        : Postsynaptic spike
        post_trace = post_trace * exp(-(t - last_post_spike)/tau) + 1
        last_post_spike = t
        printf("Postsynaptic spike received at time %g\n", t)
    }
}
