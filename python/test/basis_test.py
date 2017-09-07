import unittest
import numpy
import irlib.basis


class TestStringMethods(unittest.TestCase):

    def test_upper(self):
        self.assertEqual('foo'.upper(), 'FOO')

    def test_isupper(self):
        self.assertTrue('FOO'.isupper())
        self.assertFalse('Foo'.isupper())

    def test_split(self):
        s = 'hello world'
        self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        with self.assertRaises(TypeError):
            s.split(2)

if __name__ == '__main__':
    unittest.main()
    b = irlib.basis.basis_f(100.0, 10)
    b.compute_Tnl(numpy.array([0, 1, 2]))

#https://cmake.org/pipermail/cmake/2012-May/050120.html
