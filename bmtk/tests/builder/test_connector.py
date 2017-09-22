from bmtk.builder import connector


def test_fnc_params():
    con_fnc = connector.create(connector=lambda x, p: x**p)
    assert(con_fnc(2, 3) == 2**3)


def test_fnc_noparams():
    con_fnc = connector.create(connector=lambda x, p, a:x**p+a, a=10)
    assert(con_fnc(2, 3) == 2**3+10)


def test_literal():
    con_fnc = connector.create(connector=100.0)
    assert(con_fnc() == 100.0)

    con_fnc1 = connector.create(connector=101.0, a=10, b='10')  # parameters in literals should be ignored
    assert(con_fnc1() == 101.0)


def test_list():
    con_fnc = connector.create(connector=['a', 'b', 'c'])
    assert(con_fnc == ['a', 'b', 'c'])

    con_fnc1 = connector.create(connector=[100, 200, 300], p1=1, p2='2', p34=(3,4))
    assert(con_fnc1 == [100, 200, 300])


def test_dict():
    con_fnc = connector.create(connector={'a': 1, 'b': 'b', 'c': [5, 6]})
    assert('a' in con_fnc())
    assert('b' in con_fnc())
    assert('c' in con_fnc())

    con_fnc = connector.create(connector={'a': 1, 'b': 'b', 'c': [5, 6]}, p1='p1', p2=2)
    assert('a' in con_fnc())
    assert('b' in con_fnc())
    assert('c' in con_fnc())


#test_dict()
#test_connector_fnc_params()
#test_connector_fnc_noparams()