import numpy as np
import tensorflow as tf
from tensorflow import keras
import unittest

import a1

class TestBasic(unittest.TestCase):
    def test_q1(self):
        image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                          [[  2,   2,  20], [250, 255, 255], [127, 127, 127]]])                          
        target = {'resolution': (2, 3), 
                  'dark_pixels': (3, 3, 2)}
        result = a1.image_statistics(image, 10)
        self.assertEqual(result['resolution'], target['resolution'])
        self.assertEqual(result['dark_pixels'], target['dark_pixels'])

    def test_q2(self):
        image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                          [[  2,   2,   2], [250, 255, 255], [127, 127, 127]]])                          
        target = np.array([[[250,   2,   2], [  0, 255,   2]], \
                           [[  2,   2,   2], [250, 255, 255]]])                          

        result = a1.bounding_box(image, (0, 0), (1, 1))
        np.testing.assert_array_equal(result, target)

    def test_q3(self):
        model = a1.build_deep_nn(45, 34, 3, 2, (40, 20), (0, 0.5), 3, 'sigmoid')
        self.assertTrue(isinstance(model.layers[0], keras.layers.Flatten))
        self.assertEqual(model.layers[0].output_shape, (None, 4590))
        self.assertTrue(isinstance(model.layers[1], keras.layers.Dense))
        self.assertEqual(model.layers[1].output_shape, (None, 40))
        self.assertTrue(isinstance(model.layers[2], keras.layers.Dense))
        self.assertEqual(model.layers[2].output_shape, (None, 20))
        self.assertTrue(isinstance(model.layers[3], keras.layers.Dropout))
        self.assertTrue(isinstance(model.layers[4], keras.layers.Dense))
        self.assertEqual(model.layers[4].output_shape, (None, 3))
        self.assertEqual(model.layers[1].get_config()['activation'],'relu')
        self.assertEqual(model.layers[2].get_config()['activation'],'relu')
        self.assertEqual(model.layers[4].get_config()['activation'],'sigmoid')

if __name__ == "__main__":
    unittest.main()

