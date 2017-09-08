import unittest
import numpy
from basis import *
import scipy.integrate as integrate

class TestStringMethods(unittest.TestCase):
    def __init__():
        max_dim = 1000
        self.b = basis_f(100.0, max_dim)
        #self.b.compute_Tnl(numpy.array([0, 1, 2]))

    def test_norm(self):
        for l in range(self.b.dim()):
            self.assertTrue(numpy.abs(integrate.quad(lambda x: self.b.value(x,l)**2, -1.0, 1.0, epsabs=1e-6, limit=400)[0]-1) < 1e-8)

    #def test_upper(self):
        #self.assertEqual('foo'.upper(), 'FOO')
#
    #def test_isupper(self):
        #self.assertTrue('FOO'.isupper())
        #self.assertFalse('Foo'.isupper())
#
    #def test_split(self):
        #s = 'hello world'
        #self.assertEqual(s.split(), ['hello', 'world'])
        # check that s.split fails when the separator is not a string
        #with self.assertRaises(TypeError):
            #s.split(2)

if __name__ == '__main__':
    unittest.main()
    #b = basis_f(100.0, 10)
    #b.compute_Tnl(numpy.array([0, 1, 2]))

#https://cmake.org/pipermail/cmake/2012-May/050120.html
