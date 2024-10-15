import numpy as np
import pyvista
import scipy
import torch
import trimesh

def read_vtk(in_file):
    """
    read .vtk POLYDATA file
    Parameters:
        in_file (str) file path
    Output: 
        data (dict)  'vertices', 'faces', 'curv', 'sulc', ...
    """

    polydata = pyvista.read(in_file)
 
    n_faces = polydata.n_faces
    vertices = np.array(polydata.points)  # get vertices coordinate
    
    # only for triangles polygons data
    faces = np.array(polydata.GetPolys().GetData())  # get faces connectivity
    assert len(faces)/4 == n_faces, "faces number is not consistent!"
    faces = np.reshape(faces, (n_faces,4))
    
    data = {'vertices': vertices,
            'faces': faces
            }
    
    #cell_data = polydata.cell_data
    #for key, value in cell_data.items():
    #    if value.dtype == 'uint32':
    #        data[key] = np.array(value).astype(np.int64)
    #    elif  value.dtype == 'uint8':
    #        data[key] = np.array(value).astype(np.int32)
    #    else:
    #        data[key] = np.array(value)

    point_data = polydata.point_data
    for key, value in point_data.items():
        if value.dtype == 'uint32':
            data[key] = np.array(value).astype(np.int64)
        elif  value.dtype == 'uint8':
            data[key] = np.array(value).astype(np.int32)
        else:
            data[key] = np.array(value)

    return data

def write_vtk(in_dic, mode, file):
    """
    Write .vtk POLYDATA file
    
    in_dic: dictionary, vtk data
    mode: vertices or faces
    file: string, output file name
    """
    assert 'vertices' in in_dic, "output vtk data does not have vertices!"
    assert 'faces' in in_dic, "output vtk data does not have faces!"
    assert mode in ['vertices','faces']
    
    vertices = in_dic['vertices']
    faces = in_dic['faces']
    surf = pyvista.PolyData(vertices, faces)
    
    del in_dic['vertices']
    del in_dic['faces']
    for key, value in in_dic.items():
        if mode == 'vertices':
            surf.point_data[key] = value
        elif mode == 'faces':
            surf.cell_data[key] = value

    surf.save(file, binary=False)

def norm(x):
    return np.linalg.norm(x, axis=-1)

def cross(vec_A, vec_B):
    return np.cross(vec_A, vec_B, axis=-1)

def dot(vec_A, vec_B):
    return np.sum(vec_A * vec_B, axis=-1)

def face_coords(verts, faces):
    coords = verts[faces]  # (Nf,3,3)
    return coords

def project_to_tangent(vecs, unit_normals):
    dots = dot(vecs, unit_normals)
    return vecs - unit_normals * dots[...,np.newaxis]

def normalize(x, divide_eps=1e-6, highdim=False):
    """
    Computes norm^2 of an array of vectors. Given (shape,d), returns (shape) after norm along last dimension
    """
    if(len(x.shape) == 1):
        raise ValueError("called normalize() on single vector of dim " +
                         str(x.shape) + " are you sure?")
    if(not highdim and x.shape[-1] > 4):
        raise ValueError("called normalize() with large last dimension " +
                         str(x.shape) + " are you sure?")
    return x / (norm(x) + divide_eps)[...,np.newaxis]

def face_normals(verts, faces, normalized=True):
    coords = face_coords(verts, faces)
    vec_A = coords[:, 1, :] - coords[:, 0, :]
    vec_B = coords[:, 2, :] - coords[:, 0, :]
    raw_normal = cross(vec_A, vec_B)

    if normalized:
        return normalize(raw_normal)

    return raw_normal

def neighborhood_normal(points):
    (_, _, vh) = np.linalg.svd(points, full_matrices=False)
    normal = vh[:,2,:]
    return normal / np.linalg.norm(normal,axis=-1, keepdims=True)

def mesh_vertex_normals(verts, faces):
    face_n = face_normals(verts, faces)
    vertex_normals = np.zeros(verts.shape)
    for i in range(3):
        np.add.at(vertex_normals, faces[:,i], face_n)

    vertex_normals = vertex_normals / np.linalg.norm(vertex_normals,axis=-1,keepdims=True)

    return vertex_normals

def find_knn(source, target, k, largest=False):
    """
    Input:
        source (Ns,3)
        target (Nt,3)
        k (int)
        largest (bool, optional): _description_. Defaults to False.
    Output:
        _type_: _description_
    """
    S, _ = source.shape
    T, _ = target.shape
    
    s = np.sum(source**2, axis=1)
    t = np.sum(target**2, axis=1)
    st = np.matmul(s, t.transpose(1,0))
    s = np.repeat(s[np.newaxis,:], repeats=T, axis=0)
    t = np.repeat(t[np.newaxis,:], repeats=S, axis=0)
    
    dist = s.transpose(1,0) + t - 2*st
    dist_torch = torch.from_numpy(dist)
    dists_torch, indices_torch = torch.topk(dist_torch, k=k, largest=largest)
    
    return dists_torch.numpy(), indices_torch.numpy()
    
def vertex_normals(verts, faces, n_neighbors_cloud=30):
    """
    Input:
        verts (Nv,3)
        faces (Nf,3)
        n_neighbors_cloud (int) optional
    Output:
        normals (Nv,3)
    """
    # point cloud
    if faces.size == 0:    
        _, neigh_inds = find_knn(verts, verts, n_neighbors_cloud, omit_diagonal=True, method='cpu_kd')
        neigh_points = verts[neigh_inds,:]
        neigh_points = neigh_points - verts[:,np.newaxis,:]
        normals = neighborhood_normal(neigh_points)
    # mesh
    else:
        normals = mesh_vertex_normals(verts, faces)

        # if any are NaN, wiggle slightly and recompute
        bad_normals_mask = np.isnan(normals).any(axis=1, keepdims=True)
        if bad_normals_mask.any():
            bbox = np.amax(verts, axis=0) - np.amin(verts, axis=0)
            scale = np.linalg.norm(bbox) * 1e-4
            wiggle = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5) * scale
            wiggle_verts = verts + bad_normals_mask * wiggle
            normals = mesh_vertex_normals(wiggle_verts, faces)

        # if still NaN assign random normals (probably means unreferenced verts in mesh)
        bad_normals_mask = np.isnan(normals).any(axis=1)
        if bad_normals_mask.any():
            normals[bad_normals_mask,:] = (np.random.RandomState(seed=777).rand(*verts.shape)-0.5)[bad_normals_mask,:]
            normals = normals / np.linalg.norm(normals, axis=-1)[:,np.newaxis]

    if np.any(np.isnan(normals)): 
        raise ValueError("NaN normals :(")

    return normals

def toNP(x):
    """
    Really, definitely convert a torch tensor to a numpy array
    """
    return x.detach().to(torch.device('cpu')).numpy()

# Numpy sparse matrix to pytorch
def sparse_np_to_torch(A):
    Acoo = A.tocoo()
    values = Acoo.data
    indices = np.vstack((Acoo.row, Acoo.col))
    shape = Acoo.shape
    return torch.sparse.FloatTensor(torch.LongTensor(indices), torch.FloatTensor(values), torch.Size(shape)).coalesce()

# Pytorch sparse to numpy csc matrix
def sparse_torch_to_np(A):
    if len(A.shape) != 2:
        raise RuntimeError("should be a matrix-shaped type; dim is : " + str(A.shape))

    indices = toNP(A.indices())
    values = toNP(A.values())

    mat = scipy.sparse.coo_matrix((values, indices), shape=A.shape).tocsc()

    return mat

def compute_dice(pred, gt, num_class):
    """
    Input:
        pred (Nv,)
        gt (Nv,)
        parcel_idx (int)
    """
    pred = pred.cpu().numpy()
    gt = gt.cpu().numpy()
    
    dice = np.zeros(num_class)
    for i in range(num_class):
        gt_indices = np.where(gt == i)[0]
        pred_indices = np.where(pred == i)[0]
        dice[i] = 2 * len(np.intersect1d(gt_indices, pred_indices))/(len(gt_indices) + len(pred_indices))
    return np.mean(dice)

def coords_normalize(coords):
    """
    Input:
        coords (Nv,3)
    """
    centroid = np.mean(coords, axis=0, keepdims=True)
    coords = coords - centroid
    r = np.max(np.linalg.norm(coords, axis=1))
    coords = coords / r
    return coords

def coords_normalize_torch(coords):
    """
    Input:
        coords (Nv,3)
    """
    centroid = torch.mean(coords, dim=0)
    coords = coords - centroid
    r = torch.max(torch.norm(coords, dim=1))
    coords = coords / r
    return coords

def label_transfer(labels):
    par_fs_label = np.sort(np.unique(labels))
    par_dic = {}
    for i in range(len(par_fs_label)):
        par_dic[par_fs_label[i]] = i
    
    for i in range(len(labels)):
        labels[i] = par_dic[labels[i]]

    return labels
