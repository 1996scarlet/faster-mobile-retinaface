import mxnet as mx
import numpy as np
from mxnet.contrib import onnx as onnx_mxnet
 
prefix, epoch = "./R50", 0
sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
mx.model.save_checkpoint("./R49", 0, sym, arg_params, aux_params)

exit(0)

# sym = './M25-symbol.json'
# params = './M25-0000.params'
# input_shape = (1,3,112,112)
# # Path of the output file
# onnx_file = './mxnet_exported_retina_face_mobile_net_25.onnx'
# converted_model_path = onnx_mxnet.export_model(sym, params, [input_shape], np.float32, onnx_file)

print(mx.visualization.print_summary(sym))

model = mx.mod.Module(sym, label_names=None)
model.bind(data_shapes=[('data', (1, 3, 640, 640))], for_training=False)
model.set_params(arg_params, aux_params)

print(model.get_params()[0].keys())
# mobilenet0_conv0_weight
# print(model.get_params()[0]['face_rpn_landmark_pred_stride8_bias'])
    #   "op": "Softmax", 
    #   "axis": "1", 

# mmtoir -f mxnet -n M25-symbol.json -w M25-0000.params -d resnet152 --inputShape 3,112,112
