import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout

# Task 1 (2 marks)
def image_statistics(image, darkness):
     """Return a dictionary with the following statistics about the image. Assume that 
     the image is a colour image with three channels.
     - resolution: a tuple of the form (number_rows, number_columns).
     - dark_pixels: a tuple of tree elements, one per channel, where each element 
          shows the number of channel values lower than the given darkness value.
     >>> image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                           [[  2,   2,  20], [250, 255, 255], [127, 127, 127]]])                          
     >>> image_statistics(image, 10)
     {'resolution': (2, 3), 'dark_pixels': (3, 3, 2)}
     """
     resolution = image.shape[:-1]
     dark_pixels = tuple(np.sum(image[:,:,i] < darkness) for i in range(image.shape[2]))

     return {'resolution': resolution, 'dark_pixels': dark_pixels}

# Task 2 (2 marks)
def bounding_box(image, top_left, bottom_right):
     """Return an extract of the image determined by the bounding box, where the bounding box
     is the (row, column) positions of the pixels at the top left and bottom right of the box.
     >>> image = np.array([[[250,   2,   2], [  0, 255,   2], [  0,   0, 255]], \
                           [[  2,   2,   2], [250, 255, 255], [127, 127, 127]]])
     >>> bounding_box(image, (0, 0), (1, 1))
     array([[[250,   2,   2],
             [  0, 255,   2]],
     <BLANKLINE>
            [[  2,   2,   2],
             [250, 255, 255]]])
     """
     return image[top_left[0]:bottom_right[0]+1, top_left[1]:bottom_right[1]+1]

# Task 3 (2 marks)
def build_deep_nn(rows, columns, channels, num_hidden, hidden_sizes, dropout_rates,
                  output_size, output_activation):
     """Return a Keras neural model that has the following layers:
     - a Flatten layer with input shape (rows, columns, channels)
     - as many hidden layers as specified by num_hidden
       - hidden layer number i is of size hidden_sizes[i] and activation 'relu'
       - if dropout_rates[i] > 0, then hidden layer number i is followed
         by a dropout layer with dropout rate dropout_rates[i]
     - a final layer with size output_size and activation output_activation
     >>> model = build_deep_nn(45, 34, 3, 2, (40, 20), (0, 0.5), 3, 'sigmoid')
     >>> model.summary()
     Model: "sequential"
     _________________________________________________________________
      Layer (type)                Output Shape              Param #   
     =================================================================
      flatten (Flatten)           (None, 4590)              0         
     <BLANKLINE>
      dense (Dense)               (None, 40)                183640    
     <BLANKLINE>
      dense_1 (Dense)             (None, 20)                820       
     <BLANKLINE>
      dropout (Dropout)           (None, 20)                0         
     <BLANKLINE>
      dense_2 (Dense)             (None, 3)                 63        
     <BLANKLINE>
     =================================================================
     Total params: 184523 (720.79 KB)
     Trainable params: 184523 (720.79 KB)
     Non-trainable params: 0 (0.00 Byte)
     _________________________________________________________________

     >>> model.layers[1].get_config()['activation']
     'relu'
     >>> model.layers[2].get_config()['activation']
     'relu'
     >>> model.layers[4].get_config()['activation']
     'sigmoid'

     """
     model = Sequential()
     model.add(Flatten(input_shape=(rows, columns, channels)))
    
     for i in range(num_hidden):
        model.add(Dense(hidden_sizes[i], activation='relu'))
        if dropout_rates[i] > 0:
            model.add(Dropout(dropout_rates[i]))
    
     model.add(Dense(output_size, activation=output_activation))
    
     return model

if __name__ == "__main__":
     import doctest
     doctest.testmod()
