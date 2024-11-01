import numpy as np
from PIL import Image
import cv2
import torch
from .dwpose import DWposeDetector, draw_pose_simple
import math

detector = DWposeDetector()
detector = detector.to(f"cuda")


# 0 鼻 1 脖根 2 左肩 3 左肘 4 左腕 5 右肩 6 右肘 7 右腕
# 8 左胯 9 左膝 10左踝 11 右胯 12 右膝 13右踝
# 14 左眼 15 右眼 16 左耳 17右耳
class TreeNode:
    def __init__(self, point):
        self.x = point[0]
        self.y = point[1]
        self.new_x = point[0]
        self.new_y = point[1]
        self.children = []
        self.parent = None
        self.scale = 1

    def add_child(self, child):
        self.children.append(child)
        child.parent = self

def get_dis(node):
    # todo 肢体缺失
    if not node.parent:
        return
    dis = ((node.x - node.parent.x) ** 2 + (node.y - node.parent.y) ** 2) ** 0.5
    return dis

def get_scale(node, ref_node):
    for child1, child2 in zip(node.children, ref_node.children):
        dis1 = get_dis(child1)
        dis2 = get_dis(child2)
        child1.scale = dis2 / dis1
        get_scale(child1, child2)

def adjust_coordinates(node):
    # node.new_x += offset[0]
    # node.new_y += offset[1]

    if node.parent:
        # 和父亲距离
        dx = node.x - node.parent.x
        dy = node.y - node.parent.y
        # scale
        dx *= node.scale
        dy *= node.scale
        # 新坐标
        new_x = node.parent.new_x + dx
        new_y = node.parent.new_y + dy
        # 仿射
        center = (node.parent.new_x, node.parent.new_y)
        M = cv2.getRotationMatrix2D(center, 0, 1.0)
        new_coordinates = np.dot(np.array([[new_x, new_y, 1]]), M.T)
        # update
        node.new_x, node.new_y = new_coordinates[0, :2]

    for child in node.children:
        adjust_coordinates(child)


def build_tree(pose):

    bodies = pose["bodies"]["candidate"]

    # todo 手，眼睛
    # TODO, 有些节点为空,数值边界
    nodes = [None] * 18
    root = TreeNode(bodies[1])
    nodes[1] = root

    # 脖子到肩膀鼻子腰
    for i in [0, 2, 5, 8, 11]:
        nodes[i] = TreeNode(bodies[i])
        root.add_child(nodes[i])

    # 脸
    for i in [14, 15, 16, 17]:
        nodes[i] = TreeNode(bodies[i])
        nodes[0].add_child(nodes[i])

    # 左臂
    nodes[3] = TreeNode(bodies[3])
    nodes[2].add_child(nodes[3])

    nodes[4] = TreeNode(bodies[4])
    nodes[3].add_child(nodes[4])

    # 右臂
    nodes[6] = TreeNode(bodies[6])
    nodes[5].add_child(nodes[6])

    nodes[7] = TreeNode(bodies[7])
    nodes[6].add_child(nodes[7])

    # 左腿
    nodes[9] = TreeNode(bodies[9])
    nodes[8].add_child(nodes[9])

    nodes[10] = TreeNode(bodies[10])
    nodes[9].add_child(nodes[10])

    # 右腿
    nodes[12] = TreeNode(bodies[12])
    nodes[11].add_child(nodes[12])

    nodes[13] = TreeNode(bodies[13])
    nodes[12].add_child(nodes[13])

    # 手 2 21 2, 0右
    # print ('hands==',pose['hands'])
    # input('x')
    hand_nodes = []
    for single_hand in pose["hands"]:
        single_hand_nodes = [None] * 21
        single_hand_nodes[0] = TreeNode(single_hand[0])
        for i in range(5):
            for j in range(4):
                idx = i * 4 + j + 1
                single_hand_nodes[idx] = TreeNode(single_hand[idx])
                if j == 0:
                    # print('idx==',idx)
                    single_hand_nodes[0].add_child(single_hand_nodes[idx])
                else:
                    single_hand_nodes[idx - 1].add_child(single_hand_nodes[idx])
        hand_nodes.append(single_hand_nodes)

    nodes[7].add_child(hand_nodes[0][0])
    nodes[4].add_child(hand_nodes[1][0])
    nodes = nodes + hand_nodes[0] + hand_nodes[1]

    # print("nodes num without face and eyes", len(nodes))

    # 脸
    faces = pose["faces"][0]  # 1 68 2
    face_nodes = [None] * 68

    # TODO， 鼻子嘴巴这些可以平均
    for i in range(68):
        if i < 36 or i >= 48:
            face_nodes[i] = TreeNode(faces[i])
            nodes[0].add_child(face_nodes[i])

    # 眼睛
    for i in range(6):
        face_nodes[36 + i] = TreeNode(faces[36 + i])
        nodes[14].add_child(face_nodes[36 + i])

        face_nodes[36 + i + 6] = TreeNode(faces[36 + i + 6])
        nodes[15].add_child(face_nodes[36 + i + 6])
    nodes = nodes + face_nodes
    # print("nodes num with face and eyes", len(nodes))
    return nodes

# 算algin想要变成ref的缩放比例
def get_scales(ref_pose, align_pose, with_hand=True):
    scales = []
    ref_nodes = build_tree(ref_pose)
    align_nodes = build_tree(align_pose)

    get_scale(align_nodes[1], ref_nodes[1])
    for align_node in align_nodes:
        scales.append(align_node.scale)

    #两只胳膊scale应当一样,不然有几率会越拉越长
    pairs =[[2,5],[3,6],[4,7],[8,11],[9,12],[10,13],[14,15],[16,17]]
    for i,j in pairs:
        s = (scales[i] + scales[j] ) / 2
        scales[i] = s
        scales[j] = s
    
    #手可以根据肢体长度scale ,不然初始状态影响很大
    scales[18:60] = [(scales[8] + scales[7])/2] * 42

    #眼睛
    pairs =[[60,66],[61,67],[62,68],[63,69],[64,70],[65,71]]
    for i,j in pairs:
        s = (scales[i] + scales[j] ) / 2
        scales[i] = s
        scales[j] = s

    # check for without hand-pose error
    if with_hand:
        for i, scale in enumerate(scales):
            if i >= 39 and i <= 59:
                
                # three situations
                if scales[i] == float('inf'):
                    print("inf_idx", i)
                    scales[i] = scales[i+1]
                if math.isnan(scales[i]):
                    print("Nan_idx", i)
                    scales[i] = scales[i+1] 
                if scales[i] == 0:
                    scales[i] = scales[i+1] 

                scales[i] = scales[i] / 1  # scaling for whole hand-pose points


    # scales[0] = 0.7  # 鼻
    # scales[2] = 1  # 左肩
    # scales[3] = 0  # 左肘
    # scales[4] = 0  # 左腕
    # scales[5] = 1  # 右肩
    # scales[6] = 0  # 右肘
    # scales[7] = 0  # 右腕

    # scales[16] = 1 # 左耳
    # scales[17] = 1 # 右耳

    return scales


# 在pose的基础上缩放成ref的尺寸
def align_frame(pose, ref_pose, scales, offset):
    nodes = build_tree(pose)
    for node, scale in zip(nodes, scales):
        node.scale = scale
        
    adjust_coordinates(nodes[1])
    new_pose = []
    for node in nodes:
        new_pose.append([node.new_x + offset[0], node.new_y + offset[1]])
    return new_pose


def draw_new_pose(pose, subset, H, W, with_face, with_hand):
    bodies = pose[:18]
    hands = [pose[18:39],pose[39:60]]
    faces = [pose[60:128]]
    data = {
        "bodies": {"candidate": bodies, "subset": subset},
        "hands": hands,
        "faces": faces,
    }
    result = draw_pose_simple(data, H, W, with_face, with_hand)
    return result


def get_pose(image):
    result, pose_data = detector(image, only_eye=False)
    # candidate = pose_data["bodies"]["candidate"]
    #subset = pose_data["bodies"]["subset"]
    return pose_data, result

# tensor-HWC-RGB-0~1  to numpy-HWC-BGR-0~255
def tensor2numpy(input):
    output = (input*255).numpy().astype(np.uint8)[..., ::-1] 
    return output

# numpy-HWC-RGB-0~255 to tensor-HWC-RGB-0~1  
def numpt2tensor(input):
    output = input.astype(np.float32) / 255.0
    return torch.from_numpy(output)


class align_pose():
    @classmethod
    def INPUT_TYPES(s):
        return {"required": 
                    {"target_img": ("IMAGE", ),
                     "current_img": ("IMAGE",),
                     "input_imgs": ("IMAGE", ),},
                "optional":
                    {"with_face": ("BOOLEAN", {"default": True}),
                     "with_hand": ("BOOLEAN", {"default": True}),
                     "auto_offset": ("BOOLEAN", {"default": True}),
                     "offset_x": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step":0.1}),
                     "offset_y": ("FLOAT", {"default": 0, "min": -1, "max": 1, "step":0.1}),}
                }
    
    CATEGORY = "align_pose_jee"

    FUNCTION = "handle_imgs"
    RETURN_TYPES = ("IMAGE",)

    # input images are torch.tensors with shape [B, H, W, C], sequence RGB, and value 0~1
    def handle_imgs(self, target_img, current_img, input_imgs, with_face, with_hand, auto_offset, offset_x, offset_y):
        assert target_img.shape[0] == 1, "The shape[0] of target_img must be 1"
        assert current_img.shape[0] == 1, "The shape[0] of current_img must be 1"
        assert target_img.shape[1:3] == current_img.shape[1:3] == input_imgs.shape[1:3], "The H and W of all the imgs must be equal"
        H, W = target_img.shape[1:3]

        current_img, target_img = current_img[0], target_img[0]
        target_img, current_img, input_imgs = list(map(lambda x:tensor2numpy(x), [target_img, current_img, input_imgs]))

        target_pose, _ = get_pose(target_img)
        current_pose, _ = get_pose(current_img)
        scales = get_scales(target_pose, current_pose, with_hand)

        target_nodes = build_tree(target_pose)
        current_nodes = build_tree(current_pose)
        if auto_offset:
            offset = [target_nodes[1].x - current_nodes[1].x, target_nodes[1].y - current_nodes[1].y]
        else:
            offset = (offset_x, offset_y)

        results = []
        for frame in input_imgs:
            pose, _ = get_pose(frame)
            subset = pose["bodies"]["subset"]
            new_pose = align_frame(pose, target_pose, scales, offset)

            result = draw_new_pose(new_pose, subset, H, W, with_face, with_hand)
            result = numpt2tensor(result)[None,]
            results.append(result)
        results = torch.cat(results, dim=0)

        return (results,)

NODE_CLASS_MAPPINGS = {
    "Align_Pose": align_pose,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Align_Pose": "align pose",
}
