NEURON {
    POINT_PROCESS STDPsyn
    RANGE taupre, taupost, Apre, Apost, w, wmax, wmin, e, tau
    NONSPECIFIC_CURRENT i
    NET_RECEIVE (weight, last_pre, last_post)
}

PARAMETER {
    taupre = 20 (ms)   : Time constant for pre
    taupost = 20 (ms)  : Time constant for post
    Apre = 0.01        : Potentiation amplitude
    Apost = -0.012     : Depression amplitude
    w = 0.005          : Initial weight
    wmax = 0.01        : Max weight
    wmin = 0           : Min weight
    e = 0 (mV)         : Reversal potential
    tau = 2 (ms)       : Decay of synaptic current
}

ASSIGNED {
    v (mV)
    i (nA)
}

STATE {
    g (uS)
}

INITIAL {
    g = 0
}

BREAKPOINT {
    SOLVE state METHOD cnexp
    i = g*(v - e)
}

DERIVATIVE state {
    g' = -g/tau
}

NET_RECEIVE(weight, last_pre, last_post) {
    : Event flag (1 = presynaptic spike, 2 = postsynaptic spike)
    if (flag == 0) {   : Presynaptic spike
        g = g + w
        : update weight based on last post
        w = w + Apost*exp((t - last_post)/taupost)
        if (w > wmax) { w = wmax }
        if (w < wmin) { w = wmin }
        last_pre = t
    } else if (flag == 1) {  : Postsynaptic spike
        : update weight based on last pre
        w = w + Apre*exp((last_pre - t)/taupre)
        if (w > wmax) { w = wmax }
        if (w < wmin) { w = wmin }
        last_post = t
    }
}
