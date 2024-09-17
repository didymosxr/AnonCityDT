These scripts are used to generate a new transformed COCO Dataset from a given one.
To be more precise, new images with a tunnel effect are created and the corresponding 
annotations are adjusted.

Please note, that in this current version, only the COCO format for object detection 
is allowed, see also,

https://cocodataset.org/#format-data

and that only the polygon format of the segmentation masks is supported (an RLE version is in progress).

An example usage would be

python tunnel_transform_run.py -i <Path-to-image-folder> -j <Path-to-Annotation-Json-file> -o <Path-to-output-directory> -s <Scaling-of-the-tunnel-effect>
