from keras.layers import Input,Conv2D,Dropout,BatchNormalization,Activation,Reshape,Concatenate,Conv2DTranspose
from keras.models import Model,Sequential
from keras.applications import vgg16
def UNet(nClasses,input_height,input_width):
    assert input_height%32==0
    assert input_width%32==0
    img_input=Input(shape=(input_height,input_width,3))
    #编码器
    vgg_streamlined=vgg16.VGG16(
        include_top=False,
        weights="imagenet",
        input_tensor=img_input,
    )
    assert isinstance(vgg_streamlined,Model)
    #解码器
    #连接层
    x=vgg_streamlined.get_layer(name="block5_conv3").output
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    x = Conv2D(1024, (3, 3), activation='relu', padding='same')(x)
    x = Dropout(0.5)(x)
    #Decoder1
    x=Conv2DTranspose(512,(2,2),strides=2,padding="valid")(x)
    x=Concatenate(axis=-1)([vgg_streamlined.get_layer(name="block4_conv3").output,x])
    x=Conv2D(512,(3,3),padding="same",activation="relu")(x)
    x = Conv2D(512, (3, 3), padding="same", activation="relu")(x)
    x=BatchNormalization()(x)

    #Decoder2
    x=Conv2DTranspose(256,2,2,padding="valid")(x)
    x=Concatenate(axis=-1)([vgg_streamlined.get_layer(name="block3_conv3").output,x])
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(256, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    #Decoder3
    x = Conv2DTranspose(128, 2, 2, padding="valid")(x)
    x = Concatenate(axis=-1)([vgg_streamlined.get_layer(name="block2_conv2").output, x])
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(128, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    #Decoder4
    x = Conv2DTranspose(64, 2, 2, padding="valid")(x)
    x = Concatenate(axis=-1)([vgg_streamlined.get_layer(name="block1_conv2").output, x])
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = Conv2D(64, (3, 3), padding="same", activation="relu")(x)
    x = BatchNormalization()(x)

    #segmentation mask
    x=Conv2D(nClasses,(1,1),padding="same")(x)
    x=BatchNormalization()(x)
    x=Activation("relu")(x)
    x=Reshape((-1,nClasses))(x)
    x=Activation("softmax")(x)
    model=Model(inputs=img_input,outputs=x)
    return model
if __name__=="__main__":
    m=UNet(15,320,320)
    from keras.utils import  plot_model
    print(len(m.layers))
    plot_model(m,show_shapes=True,to_file="model_unet.png")
    m.summary()