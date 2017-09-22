import pytest

from bmtk.builder.id_generator import IDGenerator

def test_generator():
    generator = IDGenerator()
    assert(generator.next() == 0)
    assert(generator.next() == 1)
    assert(generator.next() == 2)


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


def test_remove():
    generator = IDGenerator(init_val=101)
    assert(generator.next() == 101)
    generator.remove_id(102)
    generator.remove_id(104)
    generator.remove_id(106)
    assert(generator.next() == 103)
    assert(generator.next() == 105)
    assert(generator.next() == 107)
