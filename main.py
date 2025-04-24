
import typer
import cv2 as cv
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
from tqdm import tqdm 
from pathlib import Path
import csv

@dataclass
class Targets:
    objects: Dict[str, List[Tuple]] = field(default_factory=dict)

    def add_object(self, object_id, tuple_data):
        if object_id not in self.objects:
            self.objects[object_id] = []
        self.objects[object_id].append(tuple_data)

    def get_tuples(self, object_id):
        return self.objects.get(object_id, [])

    def write_to_csv(self, output_file_path: str):
        max_frames = max(len(tuples) for tuples in self.objects.values())

        with open(output_file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            
            header = ['frame']
            for object_id in self.objects.keys():
                header.extend([f'{object_id}_u', f'{object_id}_v'])
            writer.writerow(header)

            for frame in range(max_frames):
                row = [frame]
                for object_id in self.objects.keys():
                    u, v = self.objects[object_id][frame] if frame < len(self.objects[object_id]) else ('', '')
                    row.extend([u, v])
                writer.writerow(row)



def select_target(image: np.array):
    bbox_list = []
    while True:
        bbox = cv.selectROI("Tracking", image, showCrosshair=True, fromCenter=True)

        if bbox[2] > 0 and bbox[3] > 0:
            draw_bbox(image, bbox, (0,255,0))
            bbox_list.append(bbox)

        key = cv.waitKey(0) & 0xFF

        if key == 27:  # 27 est le code ASCII de la touche Échap
            break

    return bbox_list

def draw_bbox(image, bbox, color):
    p1 = (int(bbox[0]), int(bbox[1]))
    p2 = (int(bbox[0] + bbox[2]),
        int(bbox[1] + bbox[3]))
    cv.rectangle(image, p1, p2, color, 2, 2)

#"/Volumes/LM2/Public Folders/Ferraut Samuel/Video_originale/N=0/0103secondpump/N=0 (1).MP4"
#'/Volumes/lm2/Public Folders/Ferraut Samuel/Video_taille_réduite/N=1/0802mainpump/N=1_main (17)_r.MP4'

def main(video_path: str="/Users/yanis/Desktop/N=1 (1).MP4", write_csv: bool=True, visualize: bool=True, downsampling: int = None):
    video = cv.VideoCapture(video_path)
    video_path = Path(video_path)
    output_folder = video_path.parents[0] / "output_csv"
    output_folder.mkdir(parents=True, exist_ok=True)
    output_file = output_folder / (video_path.stem + ".csv")

    total_frames = int(video.get(cv.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv.CAP_PROP_FPS)
    height = video.get(cv.CAP_PROP_FRAME_HEIGHT)
    print(f"VIDEO INFO: \n Time (s): {total_frames / fps} \n FPS : {fps} \n Frame number: {total_frames} \n hauteur : {height} \n")

    cv.namedWindow("Tracking", cv.WINDOW_NORMAL)
    tracker_list = [] 
    frame_id = 0
    my_targets = Targets()

    for _ in tqdm(range(total_frames)):
        is_frame, frame = video.read()
        if is_frame:
            
            if downsampling is not None:
                original_height, original_width = frame.shape[:2]
                new_width = original_width // downsampling
                new_height = original_height // downsampling
                frame = cv.resize(frame, (new_width, new_height), interpolation=cv.INTER_AREA)

            if frame_id == 0:
                bbox_list = select_target(frame)
                if not visualize:
                     cv.destroyAllWindows()
                for object_id, bbox in enumerate(bbox_list):
                    tracker_list.append(cv.TrackerKCF.create())
                    tracker_list[-1].init(frame, bbox)
                    bbox_position = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                    my_targets.add_object(object_id=object_id, tuple_data=bbox_position)

            for object_id, tracker in enumerate(tracker_list):
                detected, bbox = tracker.update(frame)
                if detected:
                    bbox_position = (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)
                    my_targets.add_object(object_id, bbox_position)
                    if visualize:
                        draw_bbox(frame, bbox, (0,0,255))
                else:
                    my_targets.add_object(object_id, (-1, -1))

            if visualize:
                cv.imshow("Tracking", frame)
                k = cv.waitKey(1) & 0xff
                if k == 27 : break
            
            frame_id += 1
    
    if write_csv:    
        my_targets.write_to_csv(str(output_file))

if __name__ == "__main__":
    typer.run(main)


  

