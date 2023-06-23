import os
import cv2
import math
import pickle
import dataclasses
import numpy as np
from xml.etree import ElementTree


# # this is the original mapping, we only care for some so the rest are cast to 0
# legacy_complex_event_mapping = {
#     "joining": 1,
#     "interacting": 2,
#     "moving": 3,
#     "split up": 4,
#     "leaving object": 5,
#     "fighting": 6,
#     "no_event": 7,
#     "leaving victim": 8,
# }

complex_event_mapping = {
    "interacting": 1,
    "moving": 2,
    "no_event": 0,
    "joining": 0,
    "split up": 0,
    "leaving object": 0,
    "fighting": 0,
    "leaving victim": 0,
}

simple_event_mapping = {
    "active": 0,
    "inactive": 1,
    "walking": 2,
    "running": 3,
}

inverse_simple_event_mapping = {
    0: "active",
    1: "inactive",
    2: "walking",
    3: "running",
}


@dataclasses.dataclass
class BoundingBox:
    orientation: int
    height: int
    width: int
    x_position: int
    y_position: int
    simple_event: str
    cropped_img: np.ndarray

    def get_features_from_bb(self):
        return list(
            map(
                int,
                [
                    self.height,
                    self.width,
                    self.x_position,
                    self.y_position,
                    self.orientation,
                ],
            )
        )

    def print_bb_info(self):
        print(
            f"x_position: {self.x_position}, y_position: {self.y_position}, height: {self.height}, width: {self.width}\n"
            f"orientation: {self.orientation}, simple_event: {self.simple_event}, cropped_img_shape: {self.cropped_img.shape}\n"
        )

    @classmethod
    def from_dict(cls, dictionary):
        return cls(
            height=dictionary["h"],
            width=dictionary["w"],
            x_position=dictionary["xc"],
            y_position=dictionary["yc"],
            orientation=dictionary["orientation"],
            simple_event=dictionary["simple_event"],
            cropped_img=dictionary["cropped_img"],
        )


@dataclasses.dataclass
class Group:
    subgroups: dict[tuple[int, int], str]


# this is referred to as "full" because it includes info about all groups of people
# interacting within the frame, as opposed to the Interaction class just below which
# describes only one pair of people having some sort of interaction
@dataclasses.dataclass
class FullCaviarFrame:
    bounding_boxes: dict[int, BoundingBox]
    group: Group


# class describing an interaction between two people
# includes both their bounding boxes and the complex event describing their interaction
@dataclasses.dataclass
class Interaction:
    bb1: BoundingBox
    bb2: BoundingBox
    complex_event: str

    def get_images_from_frame(self):
        return [self.bb1.cropped_img, self.bb2.cropped_img]

    def get_features_from_frame(self):
        return self.bb1.get_features_from_bb() + self.bb2.get_features_from_bb()

    def get_CE_from_frame(self):
        return complex_event_mapping[self.complex_event]

    def get_SEs_from_frame(self):
        return [
            simple_event_mapping[self.bb1.simple_event],
            simple_event_mapping[self.bb2.simple_event],
        ]


class CaviarVisionDataset:
    def __init__(self, dataset_root, window_size, window_stride):
        self.dataset_root = dataset_root
        self.window_size = window_size
        self.window_stride = window_stride

        (
            self.input_images,
            self.bb_features,
            self.simple_event_labels,
            self.complex_event_labels,
        ) = self.get_vision_dataset()

    def get_vision_dataset(self):
        # this function simply checks if the current configuration has been generated in the past
        # if it has, it is simply loaded to avoid duplicating computation, otherwise it is generated
        file_name = f"vision_window_size({self.window_size})_window_stride({self.window_stride})"
        file_path = os.path.join("caviar_deepproblog/data/vision_data", file_name)

        if os.path.exists(file_path):
            print(f"file '{file_path}' exists, loading now...")
            with open(file_path, "rb") as load_file:
                dataset = pickle.load(load_file)
            return (
                dataset["input_images"],
                dataset["bb_features"],
                dataset["se_labels"],
                dataset["ce_labels"],
            )
        else:
            print(f"file '{file_path}' doesn't exist, generating now...")
            self.raw_sequences = self.load_caviar_data()
            return self.generate_vision_dataset()

    def load_caviar_data(self):
        caviar_root = self.dataset_root

        # define the mapping between the .xml files and corresponding videos
        filename_to_videoname = {}
        with open(os.path.join(caviar_root, "video2annotation.csv"), "r") as csvfile:
            for row in csvfile.readlines():
                video_name, xml_name = row.split(",")
                filename_to_videoname[xml_name[:-1]] = video_name

        # Parse the different xml files. We are going to use the bounding boxes of
        # different persons to predict the labels so store the boxes and the labels
        raw_sequences = []
        for xml_counter, xml_file in enumerate(filename_to_videoname.keys()):
            # if xml_file == "ms3ggt.xml":
            video_name = filename_to_videoname[xml_file]
            video_path = os.path.join(caviar_root, video_name)
            video = cv2.VideoCapture(video_path)
            print(
                f"{xml_counter + 1}/{len(filename_to_videoname.keys())} - current video: {video_name}"
            )

            # read all frames efrom the .xml file
            with open(os.path.join(caviar_root, xml_file), "r") as input_file:
                frames = ElementTree.parse(input_file).findall("frame")

            # parse the frame: get info on all bounding boxes + group complex events
            # All of the this code is directly based on the xml structure and
            # should never crash but the type checker obviously has no idea and
            # thinks all of this information might not exist.
            # TODO: Add all runtime checks but for now just ignore the lines
            frame_objects = []
            for frame_idx, frame in enumerate(frames):
                frame_objects.append(
                    self.parse_frame(
                        video,
                        frame,
                        frame_idx,
                        show=False,
                    )
                )

            # Now somehow we need to pass this data though a sequence
            # absorbing model which means we need constant feature dimensionality
            # per frame. What we do is for each group that performs a complex event
            # we extract its complete history. So if in the video there are three
            # seperate groups then we will create three seperate sequences each of
            # which holds the complete history of the bounding boxes of two people

            # First get the unique groups (each group is 2 people)
            unique_groups = set()
            for frame in frame_objects:
                unique_groups.update(frame.group.subgroups.keys())

            # For each group get the complete history (i.e. the bounding boxes
            # of the two people since the appeared in the video) and at each time
            # associate the frame with a label being either no_event, i.e. no
            # complex event in the frame or some complex event identifier
            group_raw_streams = []
            for group_id in unique_groups:
                # TODO: potentially use the ones with more than 2 bounding boxes
                if len(group_id) == 2:
                    group_raw_stream = []

                    # Go and extract the history of each group. In each frame if both
                    # objects from the group exist keep their bounding boxes and
                    # retrieve the frame label for those two people
                    for frame in frame_objects:
                        if all(
                            object_id in frame.bounding_boxes.keys()
                            for object_id in group_id
                        ):
                            group_raw_stream.append(
                                Interaction(
                                    frame.bounding_boxes[group_id[0]],
                                    frame.bounding_boxes[group_id[1]],
                                    frame.group.subgroups.get(group_id, "no_event"),
                                )
                            )

                    group_raw_streams.append(group_raw_stream)

            # Add the raw streams from the filename to the total raw_sequences list
            raw_sequences.extend(group_raw_streams)

        return raw_sequences

    def generate_vision_dataset(self):
        """
        Generates a dataset consisting of input_images, bounding box features, and
        complex event labels from videos within the CAVIAR dataset

        :param dataset_root:
            the root of the CAVIAR dataset to extract 27 raw sequences of frames (videos)
        :param window_size:
            The number of frames indicating the length of the sequences used as a feature
            map (a window_size=50 is 2sec worth of video given 25frames/s)
        :param window_stride: The stride of the sliding window

        :returns:
            input_images: np.array of size (#examples, window_size, 2)
            bb_features: np.array of size (#examples, window_size, num_features)
            CE_labels: np.array of size (#examples, window_size, 1)

            for window_size=50, window_stride=10 these would be:
                input_images: (637 examples, 50 frames/sequence, 2 images/frame)
                bb_features: (637 examples, 50 frames/sequence, 12 features/frame)
                CE_labels: (637 examples, 50 frames/sequence, 1 CE/frame)
        """

        sequences = []
        window_size = self.window_size
        window_stride = self.window_stride

        # generate sequences of length window_size with window_stride
        for raw_sequence in self.raw_sequences:
            sequences.extend(
                [
                    raw_sequence[i * window_stride : (i * window_stride) + window_size]
                    for i in range(math.ceil(len(raw_sequence) / window_stride))
                ]
            )

        # filter out sequences with are not equal in length to window_size
        # (remainders from the end of a sequence)
        sequences = filter(lambda sequence: len(sequence) == window_size, sequences)

        input_images = []
        bb_features = []
        se_labels = []  # simple event labels
        ce_labels = []  # complex event labels

        # for each of the sequences of length window_size, go to each sequence
        # and transform the frame into a list of features (10 features - 5 for
        # each person) and a label for a complex event
        for sequence in sequences:
            input_images.append([frame.get_images_from_frame() for frame in sequence])
            bb_features.append([frame.get_features_from_frame() for frame in sequence])
            se_labels.append([frame.get_SEs_from_frame() for frame in sequence])
            ce_labels.append([frame.get_CE_from_frame() for frame in sequence])

        # save data in a file so that it won't have to be generated again in
        # the future if the settings (window_size, window_stride) are the same
        file_name = f"vision_window_size({window_size})_window_stride({window_stride})"
        file_path = os.path.join("caviar_deepproblog/data/vision_data", file_name)
        print(f"file '{file_path}' generated, saving now...")

        with open(file_path, "wb") as save_file:
            save_data = {
                "input_images": input_images,
                "bb_features": bb_features,
                "se_labels": se_labels,
                "ce_labels": ce_labels,
            }
            pickle.dump(save_data, save_file)

        return (
            input_images,
            bb_features,
            se_labels,
            ce_labels,
        )

    def get_cropped_frame(self, video, frame_idx, bounding_box):
        height = int(bounding_box["h"])
        width = int(bounding_box["w"])
        xcenter = int(bounding_box["xc"])  # x of bounding box center
        ycenter = int(bounding_box["yc"])  # y of bounding box center

        x_bottom_left = xcenter - width // 2  # bottom left corner x
        y_bottom_left = ycenter - height // 2  # bottom left corner y

        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, entire_frame = video.read()

        cropped_frame = entire_frame[
            y_bottom_left : y_bottom_left + height,
            x_bottom_left : x_bottom_left + width,
            :,
        ]

        return cropped_frame

    def show_frame_and_bbs(self, video, frame_idx, bounding_boxes):
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        _, entire_frame = video.read()
        print(f"Entire frame shape: {entire_frame.shape}\n")
        cv2.imshow("frame", entire_frame)

        all_bbs = []
        bb_ses = ""

        for bb_id, bb_info in bounding_boxes.items():
            print(f"Bounding box {bb_id}")
            bb_info.print_bb_info()
            all_bbs.append(
                cv2.copyMakeBorder(
                    bb_info.cropped_img,  # original image
                    0,  # padding top
                    85 - bb_info.cropped_img.shape[0],  # padding bottom
                    0,  # padding left
                    83 - bb_info.cropped_img.shape[0],  # padding right
                    cv2.BORDER_CONSTANT,  # border type: constant color
                    value=[255, 255, 255],  # color: white
                )
            )
            bb_ses += bb_info.simple_event + " "

        if all_bbs:
            concatenated = np.concatenate(all_bbs, axis=1)
            cv2.namedWindow(bb_ses, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(bb_ses, concatenated.shape[1], concatenated.shape[0])
            cv2.imshow(bb_ses, concatenated)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def parse_frame(self, video, frame, frame_idx, show=True):
        # get bounding box info for every object in the frame
        bounding_boxes = {}
        for object in frame.find("objectlist").findall("object"):
            # get information about the bounding box fom the .xml
            bb_id = int(object.attrib["id"])
            bb_info = object.find("box").attrib
            bb_info["orientation"] = int(object.find("orientation").text)
            bb_info["simple_event"] = (
                object.find("hypothesislist").find("hypothesis").find("movement").text
            )
            bb_info["cropped_img"] = self.get_cropped_frame(
                video,
                frame_idx,
                object.find("box").attrib,
            )

            bounding_boxes[bb_id] = BoundingBox.from_dict(bb_info)

        # Parse the groups in the frame if they exist and register the complex events
        groups = {}
        for group in frame.find("grouplist").findall("group"):
            groups[tuple(map(int, group.find("members").text.split(",")))] = (
                group.find("hypothesislist").find("hypothesis").find("situation").text
            )

        if show:
            self.show_frame_and_bbs(video, frame_idx, bounding_boxes)

        return FullCaviarFrame(bounding_boxes, Group(groups))


if __name__ == "__main__":
    caviar_root = r"C:\Users\Vasilis Manginas\Downloads\caviar_videos"

    visioN = CaviarVisionDataset(
        dataset_root=caviar_root,
        window_size=24,
        window_stride=24,
    )

    print()
