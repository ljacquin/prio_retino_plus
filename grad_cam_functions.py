import numpy as np
import matplotlib.cm as cm
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.utils import load_img, img_to_array, array_to_img


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()


def superimpose_images(original_img, heatmap, alpha=0.4):
  
    # Convert image to array
    img_arr = img_to_array(original_img)

    # Rescale heatmap to a range 0-255
    heatmap = np.uint8(255 * heatmap)

    # Use jet colormap to colorize heatmap
    jet = cm.get_cmap("jet")

    # Use RGB values of the colormap
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    # Create an image with RGB colorized heatmap
    jet_heatmap = array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_arr.shape[1], img_arr.shape[0]))
    jet_heatmap = img_to_array(jet_heatmap)

    superimposed_img_arr = jet_heatmap * alpha + img_arr

    return superimposed_img_arr


def compute_grad_cam(cnn_model, original_img, resized_img, last_conv_layer_name):

    xception_base = cnn_model.get_layer('xception')
    cnn_model_ = xception_base.output
    cnn_model_ = cnn_model.get_layer('global_average_pooling2d')(cnn_model_)
    cnn_model_ = cnn_model.get_layer('fc1')(cnn_model_)
    cnn_model_ = cnn_model.get_layer('dropout1')(cnn_model_)
    cnn_model_ = cnn_model.get_layer('fc2')(cnn_model_)
    cnn_model_ = cnn_model.get_layer('dropout2')(cnn_model_)
    cnn_model_ = cnn_model.get_layer('flatten')(cnn_model_)
    cnn_model_ = cnn_model.get_layer('fc3')(cnn_model_)
    cnn_model_ = cnn_model.get_layer('fc4')(cnn_model_)

    cnn_model_ = Model(inputs=cnn_model.get_layer('xception').get_layer('input_1').input, outputs=cnn_model_)

    img_arr = img_to_array(resized_img)
    img_arr = np.expand_dims(img_arr, axis=0)/255

    # Remove last layer's softmax
    cnn_model_.layers[-1].activation = None

    # Generate class activation heatmap
    heatmap = make_gradcam_heatmap(img_arr, cnn_model_, last_conv_layer_name)

    superimpose_img_ = superimpose_images(original_img, heatmap, alpha=0.4)
    # return superimposed images
    return superimpose_img_

