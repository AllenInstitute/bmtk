import numpy as np
import scipy.interpolate as sinterp
import scipy.integrate as spi
import warnings
import scipy.optimize as sopt
import scipy.stats as sps


def generate_renewal_process(t0, t1, renewal_distribution):
    last_event_time = t0
    curr_interevent_time = float(renewal_distribution())
    event_time_list = []
    while last_event_time+curr_interevent_time <= t1:
        event_time_list.append(last_event_time+curr_interevent_time)
        curr_interevent_time = float(renewal_distribution())
        last_event_time = event_time_list[-1]
        
    return event_time_list 


def generate_poisson_process(t0, t1, rate):
    if rate is None:
        raise ValueError('Rate cannot be None')
    if rate > 10000:
        warnings.warn('Very high rate encountered: %s' % rate)

    try: 
        assert rate >= 0
    except AssertionError: 
        raise ValueError('Negative rate (%s) not allowed' % rate)
    
    try: 
        assert rate < np.inf
    except AssertionError: 
        raise ValueError('Rate (%s) must be finite' % rate) 

    if rate == 0:
        return []
    else:
        return generate_renewal_process(t0, t1, sps.expon(0, 1./rate).rvs)


def generate_inhomogenous_poisson(t_range, y_range, seed=None):
    if not seed == None: np.random.seed(seed) 
    spike_list = []
    for tl, tr, y in zip(t_range[:-1], t_range[1:], y_range[:-1]):
        spike_list += generate_poisson_process(tl, tr, y) 
    return spike_list


def generate_poisson_rescaling(t, y, seed=None):
    y = np.array(y)
    t = np.array(t)
    assert not np.any(y < 0)
    f = sinterp.interp1d(t, y, fill_value=0, bounds_error=False)
    return generate_poisson_rescaling_function(lambda y, t: f(t), t[0], t[-1], seed=seed)
    

def generate_poisson_rescaling_function(f, t_min, t_max, seed=None):
    def integrator(t0, t1):
        return spi.odeint(f, 0, [t0, t1])[1][0]
    
    if not seed == None:
        np.random.seed(seed)
    
    spike_train = []
    while t_min < t_max:
        e0 = np.random.exponential()

        def root_function(t):
            return e0 - integrator(t_min, t)
        
        try:
            with warnings.catch_warnings(record=True) as w:
                result = sopt.root(root_function, .1)
            assert result.success
        except AssertionError:
            if not e0 < integrator(t_min, t_max):
                assert Exception
            else:
                break

        t_min = result.x[0]
        spike_train.append(t_min)

    return np.array(spike_train)
