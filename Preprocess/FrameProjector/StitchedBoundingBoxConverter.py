import argparse
import vrProjector
import numpy as np 
import cv2 as cv
import os


def main():
	parser = argparse.ArgumentParser(description='Reproject bounding boxes')
	parser.add_argument('--source', required=True, help='Source Metadata File')
	parser.add_argument('--cubeMapDim', required=True, help='output images width in pixels')
	parser.add_argument('--dirPath', required=True, help='Path to directory of frames')
	args = parser.parse_args()
	
	# takes in the filepath of a text file containing the bounding boxes of objects in the cubemap projection of a frame.
	# It then reprojects the bounding boxes to the equirectangular projection of the frame and saves the reprojected bounding boxes to a text file.
	out = vrProjector.CubemapProjection()
	out.initImages(int(args.cubeMapDim), int(args.cubeMapDim))

	source = vrProjector.EquirectangularProjection()
	source.loadImage(args.dirPath+"/frame0.jpg")
	out.reprojectToEquirectangular(args.source, source, int(args.cubeMapDim), int(args.cubeMapDim))

if __name__ == "__main__":
    main()