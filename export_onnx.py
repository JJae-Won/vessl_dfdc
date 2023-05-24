import argparse
import os
import re
import time

import torch
import pandas as pd
import onnx
import onnxruntime
import numpy as np

from kernel_utils import VideoReader, FaceExtractor, confident_strategy, predict_on_video_set
from training.zoo.classifiers import DeepFakeClassifier


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Predict test videos")
    arg = parser.add_argument
    arg('--weights-dir', type=str, default="weights", help="path to directory with checkpoints")
    arg('--model', required=True, help="checkpoint file")
    arg('--input-size', default=380, help= "model's input size")
    args = parser.parse_args()

    path = os.path.join(args.weights_dir, args.model)
    model = DeepFakeClassifier(encoder="tf_efficientnet_b7_ns")
    print("loading state dict {}".format(path))
    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict({re.sub("^module.", "", k): v for k, v in state_dict.items()}, strict=True)
    model.eval()
    del checkpoint

    print('model loaded')
    input_size = args.input_size
    x = torch.randn(1, 3, input_size, input_size, requires_grad=True)
    torch_out = model(x)

    torch.onnx.export(model,
                      x,
                      "test.onnx",
                      export_params=True,
                      opset_version=11,
                      do_constant_folding=True,
                      input_names = ['input'],
                      output_names = ['output'])

    print("onnx exported")
    
    
    onnx_model = onnx.load('test.onnx')
    onnx.checker.check_model(onnx_model)

    ort_session = onnxruntime.InferenceSession("test.onnx")

    def to_numpy(tensor):
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

    # ONNX 런타임에서 계산된 결과값
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x)}
    ort_outs = ort_session.run(None, ort_inputs)

    # ONNX 런타임과 PyTorch에서 연산된 결과값 비교
    np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)

    print("Exported model has been tested with ONNXRuntime, and the result looks good!")
    
    frames_per_video = 32
    video_reader = VideoReader()
    video_read_fn = lambda x: video_reader.read_frames(x, num_frames=frames_per_video)
    face_extractor = FaceExtractor(video_read_fn)
    
    strategy = confident_strategy
    stime = time.time()

    test_videos = sorted([x for x in os.listdir(args.test_dir) if x[-4:] == ".mp4"])
    print("Predicting {} videos".format(len(test_videos)))
    predictions = predict_on_video_set(face_extractor=face_extractor, input_size=input_size, models=models,
                                       strategy=strategy, frames_per_video=frames_per_video, videos=test_videos,
                                       num_workers=6, test_dir=args.test_dir)
    submission_df = pd.DataFrame({"filename": test_videos, "label": predictions})
    submission_df.to_csv(args.output, index=False)
    print("Elapsed:", time.time() - stime)
