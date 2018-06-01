from keras.layers import *
from keras.models import *
from layer_dropout import LayerDropout

def model():
    inputs = Input((32,32,3))
    x0 = Conv2D(32, 3, padding='same', activation='relu')(inputs)
    x0 = MaxPooling2D()(x0)
    x = Conv2D(32, 3, padding='same', activation='relu')(x0)
    x = Conv2D(32, 3, padding='same', activation='relu')(x)
    x = LayerDropout(0.1,residual_add=False)([x,x0])
    x = Flatten()(x)
    x = Dense(5,activation='softmax')(x)

    return Model(inputs=inputs,outputs=x)

model=model()
x=np.random.random((1000,32,32,3))
y=np.random.random((1000,5))
model.compile('adam',loss='categorical_crossentropy',metrics=['acc'])
model.fit(x,y)