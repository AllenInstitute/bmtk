import pytest

from bmtk.builder.id_generator import IDGenerator


def test_generator():
    generator = IDGenerator()
    assert(generator.next() == 0)
    assert(generator.next() == 1)
    assert(generator.next() == 2)


def test_get_ids():
    generator = IDGenerator()
    assert(generator.get_ids(1) == [0])
    assert(generator.get_ids(size=3) == [1, 2, 3])
    assert(generator.get_ids(size=0) == [])


def test_call():
    generator = IDGenerator()
    assert(len(generator(100)) == 100)
    assert(len(generator(N=10)) == 10)
    assert(isinstance(generator(), int))

    with pytest.raises(ValueError):
        generator(1, 2, 3)

    with pytest.raises(ValueError):
        generator(bad_arg=0)

    with pytest.raises(ValueError):
        generator(a=0, b=1)


def test_generator_initval():
    generator = IDGenerator(101)
    assert(generator.next() == 101)
    assert(generator.next() == 102)
    assert(generator.next() == 103)


def test_contains():
    generator = IDGenerator(init_val=10)
    gids = [generator.next() for _ in range(10)]
    assert(len(gids) == 10)
    assert(10 in generator)
    assert(19 in generator)
    assert(20 not in generator)

    generator.remove_id(54678)
    assert(54678 in generator)
    assert(54677 not in generator)
    assert(54679 not in generator)
    generator.remove_id(54677)
    assert(54677 in generator)


def test_remove():
    generator = IDGenerator(init_val=101)
    assert(generator.next() == 101)
    generator.remove_id(102)
    generator.remove_id(104)
    generator.remove_id(106)
    assert(generator.next() == 103)
    assert(generator.next() == 105)
    assert(generator.next() == 107)


if __name__ == '__main__':
    # test_generator()
    # test_contains()
    # test_call()
    # test_get_ids()
    test_contains()
