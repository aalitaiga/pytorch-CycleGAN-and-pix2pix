import random
import numpy as np
import torch
from torch.autograd import Variable

class ImagePool():
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if self.pool_size > 0:
            self.num_imgs = 0
            self.images = []
            self.joints = []

    def query(self, images, joints):
        if self.pool_size == 0:
            return images, joints
        return_images = []
        return_joints = []
        for image, joint in zip(images.data, joints.data):
            image = torch.unsqueeze(image, 0)
            joint = torch.unsqueeze(joint, 0)
            if self.num_imgs < self.pool_size:
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                self.joints.append(joint)
                return_images.append(image)
                return_joints.append(joint)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, self.pool_size-1)
                    tmp = self.images[random_id].clone()
                    tmp2 = self.joints[random_id].clone()
                    self.images[random_id] = image
                    self.joints[random_id] = joint
                    return_images.append(tmp)
                    return_joints.append(tmp2)
                else:
                    return_images.append(image)
                    return_joints.append(joint)
        return_images = Variable(torch.cat(return_images, dim=0))
        return_joints = Variable(torch.cat(return_joints, dim=0))
        return return_images, return_joints
