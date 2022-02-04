import argparse

import gradio as gr
import torch
from PIL import Image
from torchvision.transforms import Compose, ConvertImageDtype, Normalize, PILToTensor, Resize
from torchvision.transforms.functional import InterpolationMode

from holocron import models


def main(args):

    model = models.rexnet1_3x(pretrained=True).eval()

    preprocessor = Compose([
        Resize(model.default_cfg['input_shape'][1:], interpolation=InterpolationMode.BILINEAR),
        PILToTensor(),
        ConvertImageDtype(torch.float32),
        Normalize(model.default_cfg['mean'], model.default_cfg['std'])
    ])

    def predict(input):
        input = Image.fromarray(input.astype('uint8'), 'RGB')
        input = preprocessor(input)
        with torch.inference_mode():
            prediction = torch.nn.functional.softmax(model(input.unsqueeze(0))[0], dim=0)
        return {class_name: float(conf) for class_name, conf in zip(model.default_cfg['classes'], prediction)}

    image = gr.inputs.Image()
    outputs = gr.outputs.Label(num_top_classes=3)

    interface = gr.Interface(
        fn=predict,
        inputs=[image],
        outputs=outputs,
        title="Holocron: image classification demo",
        article=("<p style='text-align: center'><a href='https://github.com/frgfm/Holocron'>" "Github Repo</a> | "
                 "<a href='https://frgfm.github.io/Holocron/'>Documentation</a></p>"),
        live=True,
        theme="huggingface",
        layout="horizontal",
    )

    interface.launch(server_port=args.port, show_error=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Holocron image classification demo',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--port", type=int, default=8001, help="Port on which the webserver will be run")
    args = parser.parse_args()

    main(args)
