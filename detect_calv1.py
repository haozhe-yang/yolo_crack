import argparse
import time
from pathlib import Path

import cv2
import os
import torch
import colorsys
import torch.backends.cudnn as cudnn
import numpy as np
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadImages, LoadStreams
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized, TracedModel
import torch.utils.data as data

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, not opt.no_trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    if trace:
        model = TracedModel(model, device, opt.img_size)

    if half:
        model.half()  # to FP16

    # Second-stage classifier
    classify = False
    if classify:
        modelc = load_classifier(name='resnet101', n=2)  # initialize
        modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model']).to(device).eval()

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Warmup
        if device.type != 'cpu' and (old_img_b != img.shape[0] or old_img_h != img.shape[2] or old_img_w != img.shape[3]):
            old_img_b = img.shape[0]
            old_img_h = img.shape[2]
            old_img_w = img.shape[3]
            for i in range(3):
                model(img, augment=opt.augment)[0]

        # Inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]
        t2 = time_synchronized()

        # Apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t3 = time_synchronized()

        # Apply Classifier
        if classify:
            pred = apply_classifier(pred, modelc, img, im0s)

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0s.shape).round()

                # Process each detected object
                for *xyxy, conf, cls in reversed(det):
                    # Extract object coordinates
                    x1, y1, x2, y2 = [int(c) for c in xyxy]
                    # Crop the object from the image
                    object_img = im0[y1:y2, x1:x2]
                    # Extract the hue values of the image and save the result as a matrix
                    hue_matrix = np.zeros((object_img.shape[0], object_img.shape[1]))
                    # Iterate over the pixels in the object image
                    for x in range(object_img.shape[1]):
                        for y in range(object_img.shape[0]):
                            # Get the pixel value
                            pixel_value = object_img[y, x]
                            hue = hue_from_rgb(pixel_value)
                            hue_matrix[y, x] = hue
                    # Mark the data that matches the ROI
                    mark_roi = np.logical_or(np.logical_and(hue_matrix > 0, hue_matrix < 70), hue_matrix > 300)
                    # Find the row indices of all elements greater than 200
                    cracal = np.where(np.any(mark_roi, axis=1))[0]
                    # Deduplicate and sort row index
                    cracal = np.unique(cracal)
                    # Find the dividing line
                    if any(np.all(hue_matrix == 0, axis=1)):
                        dvline = np.where(np.all(hue_matrix == 0, axis=1))[0]
                        if len(dvline) > 1:
                            # If there is more than one element in dvline, raise an error
                            print("Error: dvline contains more than one element")
                            # sys.exit()
                        if len(dvline) == 1:
                            if cracal[0] < dvline[0] < cracal[-1]:
                                # Calculate the crack depth if the dividing line is within the ROI
                                cradep = max(abs(cracal[-1] - dvline[0]), abs(dvline[0] - cracal[0])) * depuni
                            else:
                                # Calculate the crack depth if the dividing line is outside the ROI
                                cradep = (cracal[-1] - cracal[0]) * depuni
                        else:
                            # Calculate the crack depth if there is no dividing line
                            cradep = (cracal[-1] - cracal[0]) * depuni
                        # Print the calculated crack depth
                        crapri = f"crack_depth : {cradep:.1f} mm"
                        print("depth: ", '%.1f' % cradep, "mm")

                    # Save the object image to a file
                    filename = f'object_{i}_{int(conf * 100)}.png'
                    cv2.imwrite(os.path.join(opt.crop_output, filename), object_img)

                    # Print results
                    for c in det[:, -1].unique():
                        n = (det[:, -1] == c).sum()  # detections per class
                        s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                        if save_txt:  # Write to file
                            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                            line = (cls, *xywh, conf) if opt.save_conf else (cls, *xywh)  # label format
                            # (cls, *xywh, conf, crapri) if opt.save_conf else (cls, *xywh, crapri)

                            with open(txt_path + '.txt', 'a') as f:
                                f.write(('%g ' * len(line)).rstrip() % line + '\n')

                        if save_img or view_img:  # Add bbox to image
                            # label = f'{names[int(cls)]} {conf:.2f} \n Depth: {cradep:.1f} mm'
                            label = f'{names[int(cls)]} Depth:{cradep:.1f}mm'
                            plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)


                    # Write results to a file
                    if save_txt:
                        xywh = [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]
                        line = (img_path.split('/')[-1].replace('.png', ''), conf, *xywh, cls)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%s %.2f %.2f %.2f %.2f %d\n' % line))
            else:
                print(f'No detections found for {source}')

                # Save results (image with detections)
            if save_img:
                if dataset.mode == 'image':
                    cv2.imwrite(save_path, im0)
                    print(f" The image with the result is saved in: {save_path}")

    if save_txt or save_img:
        s = f"\n{len(list(save_dir.glob('labels/*.txt')))} labels saved to {save_dir / 'labels'}" if save_txt else ''
        #print(f"Results saved to {save_dir}{s}")

    print(f'Done. ({time.time() - t0:.3f}s)')


# 使用yolov7框架进行检测，并调用crop_objects函数提取目标区域
def detect_and_crop(model, image_path, output_folder):
    # 加载图像并进行检测
    image = cv2.imread(image_path)
    detections = model.detect(image)

    # 裁剪目标区域
    crop_objects(image, detections, output_folder)

# Define a function to convert RGB to Hue
def hue_from_rgb(rgb):
    r, g, b = rgb
    h, s, v = colorsys.rgb_to_hsv(r/255.0, g/255.0, b/255.0)
    return h * 360

# Define the depth per unit pixel as 12/110
depuni = 12 / 110


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='test', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--crop_output', type=str, default='output')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.25, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='IOU threshold for NMS')
    parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='cs', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--no-trace', action='store_true', help='don`t trace model')
    opt = parser.parse_args()
    print(opt)
    #check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['yolov7.pt']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()