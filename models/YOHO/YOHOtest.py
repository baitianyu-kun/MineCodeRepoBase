import numpy as np

phi = (1 + np.sqrt(5)) / 2  # 黄金比例
vertices = np.array([[0, 1, phi], [0, 1, -phi], [0, -1, phi], [0, -1, -phi],
                     [1, phi, 0], [1, -phi, 0], [-1, phi, 0], [-1, -phi, 0],
                     [phi, 0, 1], [phi, 0, -1], [-phi, 0, 1], [-phi, 0, -1]
                     ])


def rotation_matrix(axis, theta):
    """
    生成绕某个坐标轴旋转一定角度的矩阵
    """
    axis = np.asarray(axis)
    theta = np.asarray(theta)
    axis = axis / np.sqrt(np.dot(axis, axis))
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


matrices = []
for i in range(3):
    axis = vertices[i]
    for j in range(2):
        angle = j * np.pi / 3
        matrices.append(rotation_matrix(axis, angle))

for i in range(3):
    for j in range(4):
        if i == 0:
            angle = (2 * j + 1) * np.pi / 5
            matrices.append(rotation_matrix(vertices[i], angle))
        else:
            axis1 = vertices[i]
            axis2 = vertices[(i + j) % 3 + 3]
            matrices.append(
                np.dot(rotation_matrix(axis1, 2 * np.pi / 5), rotation_matrix(axis2, (2 * j + 1) * np.pi / 3)))