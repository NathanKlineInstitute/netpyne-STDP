: $Id: nsloc_v2.mod,v 1.7 2013/06/20  salvad $
: from nrn/src/nrnoc/netstim.mod
: modified to use as proprioceptive units in arm2dms model

NEURON	{
  ARTIFICIAL_CELL NSLOC_V2
  RANGE interval, number, start, xloc, yloc, zloc, id, type, subtype, fflag, mlenmin, mlenmax, checkInterval, ispike, version
  RANGE noise
  THREADSAFE : only true if every instance has its own distinct Random
  POINTER donotuse
}

PARAMETER {
	interval	= 100 (ms) <1e-9,1e9>: time between spikes (msec)
	number	= 3000 <0,1e9>	: number of spikes (independent of noise)
	start		= 1 (ms)	: start of first spike
	noise		= 0 <0,1>	: amount of randomness (0.0 - 1.0)
    xloc = -1
    yloc = -1
    zloc = -1         : location
    id = -1
    type = -1
    subtype = -1
    fflag           = 1             : don't change -- indicates that this is an artcell
    check_interval = 1.0 (ms) : time between checking if interval has changed
    ispike = 0
    version = 2
}

ASSIGNED {
	event (ms)
	last_interval (ms)
	interval_diff (ms)
	transition
	on
	donotuse
}

PROCEDURE seed(x) {
	set_seed(x)
}

INITIAL {
	on = 0 : off
	ispike = 0
	if (noise < 0) {
		noise = 0
	}
	if (noise > 1) {
		noise = 1
	}
	if (start >= 0 && number > 0) {
		on = 1
		: randomize the first spike so on average it occurs at
		: start + noise*interval
		event = start + invl(interval) - interval*(1. - noise)
		: but not earlier than 0
		if (event < 0) {
			event = 0
		}
		net_send(event, 3)
	}
}

PROCEDURE init_sequence(t(ms)) {
	if (number > 0) {
		on = 1
		event = 0
		ispike = 0
		last_interval = interval
	}
}

FUNCTION invl(mean (ms)) (ms) {
	if (mean <= 0.) {
		mean = .01 (ms) : I would worry if it were 0.
	}
	if (noise == 0) {
		invl = mean
	}else{
		invl = (1. - noise)*mean + noise*mean*erand()
	}
}
VERBATIM
double nrn_random_pick(void* r);
void* nrn_random_arg(int argpos);
ENDVERBATIM

FUNCTION erand() {
VERBATIM
	if (_p_donotuse) {
		/*
		:Supports separate independent but reproducible streams for
		: each instance. However, the corresponding hoc Random
		: distribution MUST be set to Random.negexp(1)
		*/
		_lerand = nrn_random_pick(_p_donotuse);
	}else{
		/* only can be used in main thread */
		if (_nt != nrn_threads) {
hoc_execerror("multithread random in NetStim"," only via hoc Random");
		}
ENDVERBATIM
		: the old standby. Cannot use if reproducible parallel sim
		: independent of nhost or which host this instance is on
		: is desired, since each instance on this cpu draws from
		: the same stream
		erand = exprand(1)
VERBATIM
	}
ENDVERBATIM
}

PROCEDURE noiseFromRandom() {
VERBATIM
 {
	void** pv = (void**)(&_p_donotuse);
	if (ifarg(1)) {
		*pv = nrn_random_arg(1);
	}else{
		*pv = (void*)0;
	}
 }
ENDVERBATIM
}

PROCEDURE next_invl() {
	if (number > 0) {
		event = invl(interval)
	}
}

NET_RECEIVE (w) {
	if (flag == 0) { : external event
		if (w > 0 && on == 0) { : turn on spike sequence
			: but not if a netsend is on the queue
			init_sequence(t)
			: randomize the first spike so on average it occurs at
			: noise*interval (most likely interval is always 0)
			next_invl()
			event = event - interval*(1. - noise)
			net_send(event, 1)
		}else if (w < 0) { : turn off spiking definitively
			on = 0
		}
	}
	if (flag == 3) { : from INITIAL
		if (on == 1) { : but ignore if turned off by external event
			init_sequence(t)
			net_send(0, 1)
			net_send(2*check_interval,4)
		}
	}
	if (flag == 1 && on == 1) {
		if (interval_diff > 0) {
			: There was a change on interval so trigger at interval_diff instead
			net_send(interval_diff, 1)
			interval_diff = 0
		} else {
			ispike = ispike + 1
			net_event(t)
			next_invl()
			transition = 0
			if (on == 1) {
				net_send(event, 1)
			}
		}
	}
	if (flag == 5 && on == 1 && transition == 1) {
		if (interval_diff > 0) {
			: There was a change on interval so trigger at interval_diff instead
			net_send(interval_diff, 1)
			interval_diff = 0
		} else {
			ispike = ispike + 1
	        net_event(t)
			next_invl()
			if (on == 1) {
				net_send(event, 5)
			}
		}
    }
	if (flag == 4 && on == 1) { : check if interval has changed
		if (interval < last_interval) { :if (2*interval < event || interval > 2*event) {
			interval_diff = 0
			next_invl()
			transition = 1
			net_send(0, 5) : Send a spike right away
		}
		if (interval > last_interval) {
			interval_diff = interval - last_interval
		}
		last_interval = interval
		net_send(check_interval, 4) : next check interval event
	}
    if (flag == 1 && on == 0) { : For external PMd inputs - .event(timeStamp, 1) in server.py
        net_event(t)
    }
}

COMMENT
Presynaptic spike generator
---------------------------

This mechanism has been written to be able to use synapses in a single
neuron receiving various types of presynaptic trains.  This is a "fake"
presynaptic compartment containing a spike generator.  The trains
of spikes can be either periodic or noisy (Poisson-distributed)

Parameters;
   noise: 	between 0 (no noise-periodic) and 1 (fully noisy)
   interval: 	mean time between spikes (ms)
   number: 	number of spikes (independent of noise)

Written by Z. Mainen, modified by A. Destexhe, The Salk Institute

Modified by Michael Hines for use with CVode
The intrinsic bursting parameters have been removed since
generators can stimulate other generators to create complicated bursting
patterns with independent statistics (see below)

Modified by Michael Hines to use logical event style with NET_RECEIVE
This stimulator can also be triggered by an input event.
If the stimulator is in the on==0 state (no net_send events on queue)
 and receives a positive weight
event, then the stimulator changes to the on=1 state and goes through
its entire spike sequence before changing to the on=0 state. During
that time it ignores any positive weight events. If, in an on!=0 state,
the stimulator receives a negative weight event, the stimulator will
change to the on==0 state. In the on==0 state, it will ignore any ariving
net_send events. A change to the on==1 state immediately fires the first spike of
its sequence.

ENDCOMMENT
