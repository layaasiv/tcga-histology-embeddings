#!/usr/bin/env python

from PIL import Image
import numpy as np
import pandas as pd
import os
from tifffile import imread
from tiler import Tiler
from torchvision import transforms
import torch
import timm
import argparse
import sys
import logging
import time

class ImageCropTileFilter:
    def __init__(self, imageLoc, hf_token: str):
        self.img = imread(imageLoc)
        self.h, self.w, self.channels = self.img.shape
        self.hf_token = hf_token
        self.tile_encoder = None
        self.transform = None
        self.cancer_type = imageLoc.split("/")[-3]
        self.image_file_name = imageLoc.split("/")[-1].split(".")[0] + '/'
        self.subid = self.image_file_name.split("_")[1].split("/")[0]
        self.tsv_file_name = imageLoc.split("/")[-2]

    def crop(self):
        nrows, h_rem = divmod(self.h, 256)
        ncols, w_rem = divmod(self.w, 256)

        y = int(self.h) - h_rem
        x = int(self.w) - w_rem

        self.cropped = self.img[:y, :x, :]
        self.cropped_h, self.cropped_w, self.cropped_d = self.cropped.shape

    def other_pixel_var(self, tile):
        self.u, count_unique = np.unique(tile, return_counts=True)
        tile_1d = tile.ravel()
        self.per_5 = np.percentile(tile_1d, 5)
        self.per_50 = np.percentile(tile_1d, 50)

    def load_gp_tile_encoder(self):
        os.environ["HF_TOKEN"] = self.hf_token
        self.tile_encoder = timm.create_model("hf_hub:prov-gigapath/prov-gigapath", pretrained=True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(self.device)
        self.tile_encoder.to(self.device)

        self.transform = transforms.Compose(
            [
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )

    def filter_and_save(self):
        self.load_gp_tile_encoder()
        self.crop()

        tiler = Tiler(data_shape=self.cropped.shape,
                    tile_shape=(256, 256, 3),
                    channel_dimension=None)
        
        x = -256
        y = 0
        for _, tile in tiler.iterate(self.cropped):
            x += 256

            if x == self.cropped_w:
                x = 0
                y += 256


            tile_pos = str(x) + "x_" + str(y) + "y"
            self.other_pixel_var(tile=tile)
            if self.u[0] < 135 and self.u[-1] >= 255 and self.per_5 < 162 and self.per_50 < 225:
                # Convert the NumPy tile array directly into a PIL Image
                tile_image = Image.fromarray(tile)
                # Apply the transformations directly on the PIL Image object
                gp_input = self.transform(tile_image.convert("RGB")).unsqueeze(0)
                self.tile_encoder.eval()
                with torch.no_grad():
                    self.model_output = self.tile_encoder(gp_input.to(self.device)).squeeze()

                    t_np = self.model_output.cpu().detach().numpy()  # convert to Numpy array
                    df = pd.DataFrame(t_np)  # convert to a dataframe
                    df_transposed = df.transpose()
                    df_transposed['submitter_id'] = self.subid
                    df_transposed['cancer_type'] = self.cancer_type
                    df_transposed['tile_position'] = tile_pos
                    df_transposed.to_csv("/home/exacloud/gscratch/CEDAR/sivakuml/ellrott-proj/" + self.cancer_type + "-emb/" + self.tsv_file_name + ".tsv",
                                            sep="\t",
                                            mode='a',
                                            index=False, header=False)  # append row to existing tsv
            else:
                continue 


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Process images and filter tiles for cancer type.")
    parser.add_argument('--image_directory', "-id", type=str, help='Path to the directory containing images',
                        required=True)
    parser.add_argument("--hftoken", "-hf", type=str, help="Hugging Face token", required=True)
    parser.add_argument("--logfilename", "-lf", type=str, help="Logfile name (or path + logfile name)", required=True)

    # Parse arguments
    args = parser.parse_args()

    hugging_face_token = args.hftoken
    logfilename = args.logfilename

    # Configure logging
    logging.basicConfig(filename=logfilename + '.log', level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Process images in the provided directory
    
    # for cancer_type in os.listdir(args.image_directory):
    #     cancer_path = os.path.join(args.image_directory, cancer_type)  # path to original cancer type directory

    for image in os.listdir(args.image_directory):
        image_path = os.path.join(args.image_directory, image)  # path to original tcga wsi
        logging.info(f"Processing {image_path}...")

        og_img = ImageCropTileFilter(image_path, hf_token=hugging_face_token)
        og_img.filter_and_save()
        logging.info(f"Done processing {image_path}...")
