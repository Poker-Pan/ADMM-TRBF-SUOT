from scipy import stats
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
import math, os, time, torch, copy, sys, random, inspect, psutil, gc, scipy, trimesh


import pprint as pp
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
from matplotlib import cm
import scipy.io as scio
from mpl_toolkits.mplot3d import Axes3D
from torch.linalg import cholesky
import torch.distributions as td
from scipy.spatial import Delaunay
from sklearn.neighbors import KDTree
from tqdm import tqdm


class Utilize(object):
    def __init__(self, Key_Para): 
        super(Utilize, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']

    def make_file(self):
        root = os.getcwd()
        path = root + '/' + self.Key_Para['File_name']
        if not os.path.exists(path):
            os.makedirs(path)
            if self.Key_Para['type_print'] == 'True':
                pass
            elif self.Key_Para['type_print'] == 'False':
                sys.stdout = open(self.Key_Para['File_name'] + '/' + str(self.Key_Para['File_name']) + '-Code-Print.txt', 'w')
            else:
                print('There need code!')
            print('************' + str(self.Key_Para['File_name']) + '************')

    def print_key(self, keyword):
        print('************Key-Word************')
        pp.pprint(keyword)
        print('************************************')

    def setup_seed(self, seed):
        torch.manual_seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed_all(seed)

    def mu_0_mu_1(self):
        mu_0, mu_1 = np.array([self.Key_Para['mu_0']]), np.array([self.Key_Para['mu_1']])
        self.Key_Para['mu_0'], self.Key_Para['mu_1'] = mu_0, mu_1

    def sigma_0_sigma_1(self):
        sigma_0 = self.Key_Para['sigma_0']
        sigma_1 = self.Key_Para['sigma_1']

        self.Key_Para['sigma_0'], self.Key_Para['sigma_1'] = sigma_0, sigma_1

    def Time_Space(self):
        Time = np.array(self.Key_Para['Time']).reshape(1,-1)
        Space = np.array(self.Key_Para['Space']).reshape(1,-1).repeat(self.Dim_space, axis=0)
        
        self.Key_Para['Time'], self.Key_Para['Space'] = Time, Space


class Gnerate_node(object):
    def __init__(self, Key_Para):
        super(Gnerate_node, self).__init__()
        self.Key_Para = Key_Para
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']
        self.Time = Key_Para['Time']
        self.Space = Key_Para['Space']
        self.Num_Nodes_t = Key_Para['Num_Nodes_t']
        self.Num_Nodes_all_space = Key_Para['Num_Nodes_all_space']
        self.type_node = Key_Para['type_node']

    def forward(self):
        if self.type_node == 'Load':
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                file_name = type_surface

                load_data = scio.loadmat('./ADMM-MUOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Torus':
                file_name = type_surface

                load_data = scio.loadmat('./ADMM-MUOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Opener':
                file_name = type_surface

                load_data = scio.loadmat('./ADMM-MUOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Ellipsoid':
                file_name = type_surface

                load_data = scio.loadmat('./ADMM-MUOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'Peanut':
                file_name = type_surface

                load_data = scio.loadmat('./ADMM-MUOT/Data_Set/surface/' + file_name + '.mat')
                nodes_space = load_data['nodes']
                elements = (load_data['triangles'] - 1).astype(int)
                normal = load_data['normal']
                self.Key_Para['nodes_center'] = np.array([0.5, 0.5, 0.5])

            elif type_surface == 'cow':
                file_name = type_surface

                load_data = trimesh.load('./ADMM-MUOT/Data_Set/surface/' + file_name + '.obj')
                nodes_space = np.array(load_data.vertices[:, [0, 2, 1]])
                elements = load_data.faces
                normal = np.array(load_data.vertex_normals[:, [0, 2, 1]])

            elif type_surface == 'armadillo':
                file_name = type_surface

                load_data = trimesh.load('./ADMM-MUOT/Data_Set/surface/' + file_name + '.obj')
                nodes_space = np.array(load_data.vertices[:, [0, 2, 1]])
                elements = load_data.faces
                normal = np.array(load_data.vertex_normals[:, [0, 2, 1]])

            elif type_surface == 'face':
                file_name = type_surface

                load_data = trimesh.load('./ADMM-MUOT/Data_Set/surface/' + file_name + '.obj')
                nodes_space = np.array(load_data.vertices)
                elements = load_data.faces
                normal = np.array(load_data.vertex_normals)

            elif type_surface == 'Airplane':
                file_name = type_surface

                load_data = trimesh.load('./ADMM-MUOT/Data_Set/surface/' + file_name + '.obj')
                nodes_space = np.array(load_data.vertices)
                elements = load_data.faces
                normal = np.array(load_data.vertex_normals)

            elif type_surface == 'fish':
                file_name = type_surface

                load_data = trimesh.load('./ADMM-MUOT/Data_Set/surface/' + file_name + '.obj')
                nodes_space = np.array(load_data.vertices)
                elements = load_data.faces
                normal = np.array(load_data.vertex_normals)




            self.Key_Para['Num_Nodes_all_space'] = nodes_space.shape[0]          
            nodes = nodes_space
            t = np.linspace(self.Time[0, 0], self.Time[0, 1], self.Num_Nodes_t)
            self.Key_Para['dt'] = (self.Time[0, 1] - self.Time[0, 0]) / (self.Num_Nodes_t - 1)
            self.Key_Para['nodes_normal'] = normal

        elif self.type_node == 'Generate':
            print('There need code!')

        return t, nodes, elements

    def oprater_nodes(self, nodes, elements):
        pt = nodes
        trg = elements

        npt = pt.shape[0]
        ntrg = trg.shape[0]
        normalVec = np.zeros((ntrg, 3))
        trgCenter = np.zeros((ntrg, 3))
        trgArea = np.zeros((ntrg, 1))
        ptArea = np.zeros((npt, 1))

        for i in range(trg.shape[0]):
            p1, p2, p3 = trg[i, 0], trg[i, 1], trg[i, 2]
            v1, v2, v3 = pt[p1, :], pt[p2, :], pt[p3, :]
            v12 = (v2 - v1).reshape(1, -1)
            v31 = (v1 - v3).reshape(1, -1)
            n = np.cross(v12, -v31, axis=1)
            trgCenter[i, :] = np.mean(np.stack([v1, v2, v3]), axis=0)
            normalVec[i, :] = n / np.linalg.norm(n)
            trgArea[i] = 1 / 2 * np.linalg.norm(n)

            ptArea[p1] = ptArea[p1] + trgArea[i] / 3
            ptArea[p2] = ptArea[p2] + trgArea[i] / 3
            ptArea[p3] = ptArea[p3] + trgArea[i] / 3

        self.Key_Para['tri_normal'] = normalVec
        self.Key_Para['nodes_area'] = ptArea
        self.Key_Para['tri_area'] = trgArea
        self.Key_Para['tri_center'] = trgCenter

    def eigenvectors(self, nodes, elements):
        eigenvector = np.zeros((nodes.shape[0], 3, 3))
        for i in range(nodes.shape[0]):
            id_x, _ = np.where(elements==i)
            sub_elements = elements[id_x, :]
            id_y = np.unique(sub_elements.reshape(-1), axis=0)
            nodes_local = nodes[id_y, :]

            c_nodes = np.mean(nodes_local, axis=0)
            P = np.dot((nodes_local - c_nodes).T, (nodes_local - c_nodes))
            eigen_value = np.linalg.eig(P)[0]
            eigen_vector = np.linalg.eig(P)[1]
            idx = np.flip(np.argsort(eigen_value))
            eigen_value = np.flip(np.sort(eigen_value))
            eigen_vector = eigen_vector[:, idx]
            eigenvector[i, :, :] = eigen_vector
        
        self.Key_Para['eigenvector'] = eigenvector


class ADMM_solver(object):
    def __init__(self, Key_Para):
        super(ADMM_solver, self).__init__()
        self.Key_Para = Key_Para

    def solver_rho_m_f(self, t, nodes, elements, cur_rho, cur_m, cur_f, cur_rho_bar, cur_m_bar, cur_f_bar, cur_p, cur_q, cur_r):
        def select_rho(rho):
            id_1 = np.where(np.abs(rho.imag)<1e-8)
            out = rho[id_1]
            out = np.max(out.real)
            out = np.max((out, 0))
            return out
        
        time_begin = time.time()
        id_UOT = self.Key_Para['type_UOT'] * (1 / self.Key_Para['eta'])
        for i in range(t.shape[0]):
            alpha = self.Key_Para['alpha'][:, i]
            item1 = cur_rho_bar[:, i] - cur_p[:, i]
            item2 = cur_m_bar[:, i, :] - cur_q[:, i, :]
            item3 = cur_f_bar[:, i] - cur_r[:, i]

            a = (alpha).reshape(-1, 1)
            b = ((2 - alpha*item1)).reshape(-1, 1)
            c = ((1/alpha - 2*item1)).reshape(-1, 1)
            d = ((-item1/alpha - 0.5*(np.linalg.norm(item2, axis=1)**2))).reshape(-1, 1)
            e = (0*alpha).reshape(-1, 1)
            f = (0*alpha).reshape(-1, 1)
            if self.Key_Para['type_UOT'] == 1:
                a = a
                b = b + ((4*np.ones_like(alpha))*id_UOT).reshape(-1, 1)
                c = c + ((((4/alpha)*id_UOT) + (8/alpha) - (4*item1)) * id_UOT).reshape(-1, 1)
                d = d + (((4/(alpha**2)) + ((8/(alpha**2))*id_UOT) - ((4/alpha)*item1*id_UOT) - ((8/alpha)*item1) - (item3**2)) * id_UOT).reshape(-1, 1)
                e = e + (((4/(alpha**3)*id_UOT) - (8/(alpha**2)*item1*id_UOT) - (4/(alpha**2)*item1) - ((2/alpha)*(np.linalg.norm(item2, axis=1)**2)) - ((2/alpha)*(item3**2))) * id_UOT).reshape(-1, 1)
                f = f + (((-(4/(alpha**3))*id_UOT*item1) - ((2/(alpha**2))*(np.linalg.norm(item2, axis=1)**2)*id_UOT) - ((1/(alpha**2))*(item3**2))) * id_UOT).reshape(-1, 1)


            coef_poly = np.hstack((a, b, c, d, e, f))
            roots = np.apply_along_axis(np.roots, 1, coef_poly)
            roots = np.apply_along_axis(select_rho, 1, roots)
            cur_rho[:, i] = roots.reshape(-1)
            rho = cur_rho[:, i]
            
            cur_m[:, i, :] = np.dot(np.diag((alpha * rho) / (1 + alpha * rho)), item2)
            cur_f[:, i] = self.Key_Para['type_UOT'] * (((alpha*rho) / (2*id_UOT + alpha*rho + 1e-10)) * item3)
        
        # print('Solve_rho_time:', time.time() - time_begin, 's')
        return cur_rho, cur_m, cur_f

    def solver_rho_m_f_bar(self, t, nodes, elements, cur_rho, cur_m, cur_f, cur_rho_bar, cur_m_bar, cur_f_bar, cur_p, cur_q, cur_r):
        time_begin = time.time()
        N, M = t.shape[0], nodes.shape[0]
        F1 = np.zeros((N*M, 1))
        F2 = np.zeros((N*M, 1))
        F3 = np.zeros((N*M, 1))

        coef_RBF = self.Key_Para['coef_RBF']
        coef_RBF_x, coef_RBF_y, coef_RBF_z = coef_RBF[0], coef_RBF[1], coef_RBF[2]
        for i in range(N):
            if i == 0:
                # F1[(i*M):((i+1)*M)] = -0.5*2*(( -(cur_rho[:, i+2] + cur_p[:, i+2]) + 4*(cur_rho[:, i+1] + cur_p[:, i+1]) - 3*(cur_rho[:, i] + cur_p[:, i]) ) * (1/(2*self.Key_Para['dt']))).reshape(-1, 1) 
                F1[(i*M):((i+1)*M)] = -0.5*2*(( (cur_rho[:, i+1] + cur_p[:, i+1]) - (cur_rho[:, i] + cur_p[:, i]) ) * (1/(self.Key_Para['dt']))).reshape(-1, 1) 
                F2[(i*M):((i+1)*M)] = 0.5*2*(coef_RBF_x.dot(cur_m[:, i, 0:1] + cur_q[:, i, 0:1]) + (coef_RBF_y.dot(cur_m[:, i, 1:2] + cur_q[:, i, 1:2])) + (coef_RBF_z.dot(cur_m[:, i, 2:3] + cur_q[:, i, 2:3]))).reshape(-1, 1)
                F3[(i*M):((i+1)*M)] = 0.5*2*(cur_f[:, i] + cur_r[:, i]).reshape(-1, 1)
                F1[(i*M):((i+1)*M)] = F1[(i*M):((i+1)*M)] + 2*((cur_rho_bar[:, i] - (cur_rho[:, i] + cur_p[:, i])) * (1/(self.Key_Para['dt']))).reshape(-1, 1)
            elif i == N-1:
                # F1[(i*M):((i+1)*M)] = -0.5*2*(( 3*(cur_rho[:, i] + cur_p[:, i]) - 4*(cur_rho[:, i-1] + cur_p[:, i-1]) + (cur_rho[:, i-2] + cur_p[:, i-2])) * (1/(2*self.Key_Para['dt']))).reshape(-1, 1) 
                F1[(i*M):((i+1)*M)] = -0.5*2*(( (cur_rho[:, i] + cur_p[:, i]) - (cur_rho[:, i-1] + cur_p[:, i-1]) ) * (1/(self.Key_Para['dt']))).reshape(-1, 1) 
                F2[(i*M):((i+1)*M)] = 0.5*2*(coef_RBF_x.dot(cur_m[:, i, 0:1] + cur_q[:, i, 0:1]) + (coef_RBF_y.dot(cur_m[:, i, 1:2] + cur_q[:, i, 1:2])) + (coef_RBF_z.dot(cur_m[:, i, 2:3] + cur_q[:, i, 2:3]))).reshape(-1, 1)
                F3[(i*M):((i+1)*M)] = 0.5*2*(cur_f[:, i] + cur_r[:, i]).reshape(-1, 1)
                F1[(i*M):((i+1)*M)] = F1[(i*M):((i+1)*M)] - 2*((cur_rho_bar[:, i] - (cur_rho[:, i] + cur_p[:, i])) * (1/(self.Key_Para['dt']))).reshape(-1, 1)
            else:
                F1[(i*M):((i+1)*M)] = -(((cur_rho[:, i+1] + cur_p[:, i+1]) - (cur_rho[:, i-1] + cur_p[:, i-1])) * (1/(2*self.Key_Para['dt']))).reshape(-1, 1)
                F2[(i*M):((i+1)*M)] = (coef_RBF_x.dot(cur_m[:, i, 0:1] + cur_q[:, i, 0:1]) + (coef_RBF_y.dot(cur_m[:, i, 1:2] + cur_q[:, i, 1:2])) + (coef_RBF_z.dot(cur_m[:, i, 2:3] + cur_q[:, i, 2:3]))).reshape(-1, 1)
                F3[(i*M):((i+1)*M)] = (cur_f[:, i] + cur_r[:, i]).reshape(-1, 1)

        F_all = (F1 - F2 + self.Key_Para['type_UOT']*F3).reshape(N, M).T
        R = np.dot(F_all, self.Key_Para['eigenvector_inv'].T)

        U = np.zeros((M, N))
        for i in range(N):
            U[:, i] = self.Key_Para['temp_inv'][i].dot(R[:, i])
        lambda_h = np.dot(U, self.Key_Para['eigenvector'].T)


        lambda_grad_t = np.zeros((M, N))
        lambda_grad_h = np.zeros((M, N, nodes.shape[1]))
        for i in range(N):
            if i != 0 and i != N-1:
                lambda_grad_t[:, i] = (lambda_h[:, i+1] - lambda_h[:, i-1]) * (1/(2*self.Key_Para['dt']))
                cur_rho_bar[:, i] = (cur_rho[:, i] + cur_p[:, i]) + lambda_grad_t[:, i]
            sub_lambda_grad_h = np.hstack((coef_RBF_x.dot(lambda_h[:, i:i+1]), coef_RBF_y.dot(lambda_h[:, i:i+1]), coef_RBF_z.dot(lambda_h[:, i:i+1])))
            cur_m_bar[:, i, :] = (cur_m[:, i, :] + cur_q[:, i, :]) + sub_lambda_grad_h
            lambda_grad_h[:, i, :] = sub_lambda_grad_h
        cur_f_bar = self.Key_Para['type_UOT'] * ((cur_f + cur_r) + lambda_h)
        # print('Solve_m_time:', time.time() - time_begin, 's')
        return cur_rho_bar, cur_m_bar, cur_f_bar, lambda_h, lambda_grad_t, lambda_grad_h

    def solver_p_q_r(self, t, nodes, elements, cur_rho, cur_m, cur_f, cur_rho_bar, cur_m_bar, cur_f_bar, cur_p, cur_q, cur_r):
        k = 1
        time_begin = time.time()
        cur_p = cur_p + k*(cur_rho - cur_rho_bar)
        cur_q = cur_q + k*(cur_m - cur_m_bar)
        cur_r = cur_r + k*(cur_f - cur_f_bar)
        # print('Solve_r_time:', time.time() - time_begin, 's')
        return cur_p, cur_q, cur_r


class Train_loop(object):
    def __init__(self, Key_Para, gen_Nodes, admm_solver, compute, plot_result):
        super(Train_loop, self).__init__()
        self.Key_Para = Key_Para
        self.gen_Nodes = gen_Nodes
        self.admm_solver = admm_solver
        self.compute = compute
        self.plot_result = plot_result

    def train(self):
        def Sphere_Surface_Gaussian(self, nodes, mu, sigma):      
            coefficient = 100
            f = coefficient * np.exp(-(np.linalg.norm(nodes - mu, axis=1)**2)/sigma)
            f = f / np.sum(f*self.Key_Para['nodes_area'][:, 0])
            return f

        def RBF_Tangent_Plane_Poly(self, nodes, elements):
            def RBF_basis(x, y, xi, yi, c=1):
                dis = np.sqrt((x - xi)**2 + (y - yi)**2)
                # Gaussian
                phi = np.exp(-c**2*((x - xi)**2 + (y - yi)**2))
                phix = (-c**2*np.exp(-c**2*((x - xi)**2 + (y - yi)**2))*(2*x - 2*xi))
                phiy = (-c**2*np.exp(-c**2*((x - xi)**2 + (y - yi)**2))*(2*y - 2*yi))
                phixx = 2*c**2*np.exp(-c**2*((x - xi)**2 + (y - yi)**2))*(2*c**2*x**2 - 4*c**2*x*xi + 2*c**2*xi**2 - 1)
                phixy = c**4*np.exp(-c**2*((x - xi)**2 + (y - yi)**2))*(2*x - 2*xi)*(2*y - 2*yi)
                phiyx = c**4*np.exp(-c**2*((x - xi)**2 + (y - yi)**2))*(2*x - 2*xi)*(2*y - 2*yi)
                phiyy = 2*c**2*np.exp(-c**2*((x - xi)**2 + (y - yi)**2))*(2*c**2*y**2 - 4*c**2*y*yi + 2*c**2*yi**2 - 1)
                phi_gradient = np.array([[phix, phiy]])
                phi_laplace = np.array([[phixx, phixy],
                                        [phiyx, phiyy]])

                # C4-Wendland
                # if dis == 0 or dis >= 1:
                #     phi = 0
                #     phi_gradient = np.array([[0, 0]])
                #     phi_laplace = np.array([[0, 0],
                #                             [0, 0]])
                # else:   
                #     phi = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3)
                #     phix = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*(35*c**2*(2*x - 2*xi) + (9*c*(2*x - 2*xi))/((x - xi)**2 + (y - yi)**2)**(1/2)) + (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*x - 2*xi)*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/((x - xi)**2 + (y - yi)**2)**(1/2)
                #     phiy = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*(35*c**2*(2*y - 2*yi) + (9*c*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2)) + (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*y - 2*yi)*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/((x - xi)**2 + (y - yi)**2)**(1/2)
                #     phixx = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*((18*c)/((x - xi)**2 + (y - yi)**2)**(1/2) + 70*c**2 - (9*c*(2*x - 2*xi)**2)/(2*((x - xi)**2 + (y - yi)**2)**(3/2))) + (6*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/((x - xi)**2 + (y - yi)**2)**(1/2) - (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*x - 2*xi)**2*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/(2*((x - xi)**2 + (y - yi)**2)**(3/2)) + (6*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*x - 2*xi)*(35*c**2*(2*x - 2*xi) + (9*c*(2*x - 2*xi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (15*c**2*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**4*(2*x - 2*xi)**2*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/(2*((x - xi)**2 + (y - yi)**2))
                #     phixy = (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*y - 2*yi)*(35*c**2*(2*x - 2*xi) + (9*c*(2*x - 2*xi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) - (9*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*(2*x - 2*xi)*(2*y - 2*yi))/(2*((x - xi)**2 + (y - yi)**2)**(3/2)) + (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*x - 2*xi)*(35*c**2*(2*y - 2*yi) + (9*c*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (15*c**2*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**4*(2*x - 2*xi)*(2*y - 2*yi)*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/(2*((x - xi)**2 + (y - yi)**2)) - (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*x - 2*xi)*(2*y - 2*yi)*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/(2*((x - xi)**2 + (y - yi)**2)**(3/2))
                #     phiyx = (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*y - 2*yi)*(35*c**2*(2*x - 2*xi) + (9*c*(2*x - 2*xi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) - (9*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*(2*x - 2*xi)*(2*y - 2*yi))/(2*((x - xi)**2 + (y - yi)**2)**(3/2)) + (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*x - 2*xi)*(35*c**2*(2*y - 2*yi) + (9*c*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (15*c**2*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**4*(2*x - 2*xi)*(2*y - 2*yi)*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/(2*((x - xi)**2 + (y - yi)**2)) - (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*x - 2*xi)*(2*y - 2*yi)*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/(2*((x - xi)**2 + (y - yi)**2)**(3/2))
                #     phiyy = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*((18*c)/((x - xi)**2 + (y - yi)**2)**(1/2) + 70*c**2 - (9*c*(2*y - 2*yi)**2)/(2*((x - xi)**2 + (y - yi)**2)**(3/2))) + (6*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/((x - xi)**2 + (y - yi)**2)**(1/2) - (3*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*y - 2*yi)**2*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/(2*((x - xi)**2 + (y - yi)**2)**(3/2)) + (6*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**5*(2*y - 2*yi)*(35*c**2*(2*y - 2*yi) + (9*c*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (15*c**2*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**4*(2*y - 2*yi)**2*(35*c**2*((x - xi)**2 + (y - yi)**2) + 18*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 3))/(2*((x - xi)**2 + (y - yi)**2))
                #     phi_gradient = np.array([[phix, phiy]])
                #     phi_laplace = np.array([[phixx, phixy],
                #                             [phiyx, phiyy]])

                # C8-Wendland
                # if dis == 0 or dis >= 1:
                #     phi = 0
                #     phi_gradient = np.array([[0, 0]])
                #     phi_laplace = np.array([[0, 0],
                #                             [0, 0]])
                # else:   
                #     phi = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**8*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1)
                #     phix = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**8*(25*c**2*(2*x - 2*xi) + 48*c**3*(2*x - 2*xi)*((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(2*x - 2*xi))/((x - xi)**2 + (y - yi)**2)**(1/2)) + (4*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*x - 2*xi)*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)**(1/2)
                #     phiy = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**8*(25*c**2*(2*y - 2*yi) + 48*c**3*(2*y - 2*yi)*((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2)) + (4*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*y - 2*yi)*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)**(1/2)
                #     phixx = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**8*(96*c**3*((x - xi)**2 + (y - yi)**2)**(1/2) + (8*c)/((x - xi)**2 + (y - yi)**2)**(1/2) + 50*c**2 - (2*c*(2*x - 2*xi)**2)/((x - xi)**2 + (y - yi)**2)**(3/2) + (24*c**3*(2*x - 2*xi)**2)/((x - xi)**2 + (y - yi)**2)**(1/2)) + (8*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)**(1/2) - (2*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*x - 2*xi)**2*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)**(3/2) + (8*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*x - 2*xi)*(25*c**2*(2*x - 2*xi) + 48*c**3*(2*x - 2*xi)*((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(2*x - 2*xi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (14*c**2*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*(2*x - 2*xi)**2*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)
                #     phixy = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**8*((24*c**3*(2*x - 2*xi)*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2) - (2*c*(2*x - 2*xi)*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(3/2)) + (4*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*y - 2*yi)*(25*c**2*(2*x - 2*xi) + 48*c**3*(2*x - 2*xi)*((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(2*x - 2*xi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*x - 2*xi)*(25*c**2*(2*y - 2*yi) + 48*c**3*(2*y - 2*yi)*((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (14*c**2*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*(2*x - 2*xi)*(2*y - 2*yi)*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2) - (2*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*x - 2*xi)*(2*y - 2*yi)*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)**(3/2)
                #     phiyx = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**8*((24*c**3*(2*x - 2*xi)*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2) - (2*c*(2*x - 2*xi)*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(3/2)) + (4*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*y - 2*yi)*(25*c**2*(2*x - 2*xi) + 48*c**3*(2*x - 2*xi)*((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(2*x - 2*xi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*x - 2*xi)*(25*c**2*(2*y - 2*yi) + 48*c**3*(2*y - 2*yi)*((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (14*c**2*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*(2*x - 2*xi)*(2*y - 2*yi)*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2) - (2*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*x - 2*xi)*(2*y - 2*yi)*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)**(3/2)
                #     phiyy = (c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**8*(96*c**3*((x - xi)**2 + (y - yi)**2)**(1/2) + (8*c)/((x - xi)**2 + (y - yi)**2)**(1/2) + 50*c**2 - (2*c*(2*y - 2*yi)**2)/((x - xi)**2 + (y - yi)**2)**(3/2) + (24*c**3*(2*y - 2*yi)**2)/((x - xi)**2 + (y - yi)**2)**(1/2)) + (8*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)**(1/2) - (2*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*y - 2*yi)**2*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)**(3/2) + (8*c*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**7*(2*y - 2*yi)*(25*c**2*(2*y - 2*yi) + 48*c**3*(2*y - 2*yi)*((x - xi)**2 + (y - yi)**2)**(1/2) + (4*c*(2*y - 2*yi))/((x - xi)**2 + (y - yi)**2)**(1/2)))/((x - xi)**2 + (y - yi)**2)**(1/2) + (14*c**2*(c*((x - xi)**2 + (y - yi)**2)**(1/2) - 1)**6*(2*y - 2*yi)**2*(32*c**3*((x - xi)**2 + (y - yi)**2)**(3/2) + 25*c**2*((x - xi)**2 + (y - yi)**2) + 8*c*((x - xi)**2 + (y - yi)**2)**(1/2) + 1))/((x - xi)**2 + (y - yi)**2)
                #     phi_gradient = np.array([[phix, phiy]])
                #     phi_laplace = np.array([[phixx, phixy],
                #                             [phiyx, phiyy]])

                # thin-plate spline
                # if dis == 0:
                #     phi = 0
                #     phi_gradient = np.array([[0, 0]])
                #     phi_laplace = np.array([[0, 0],
                #                             [0, 0]])
                # else:
                #     phi = np.log(((x - xi)**2 + (y - yi)**2)**(1/2))*((x - xi)**2 + (y - yi)**2)**2
                #     phix = ((2*x - 2*xi)*((x - xi)**2 + (y - yi)**2))/2 + 2*np.log(((x - xi)**2 + (y - yi)**2)**(1/2))*(2*x - 2*xi)*((x - xi)**2 + (y - yi)**2)
                #     phiy = ((2*y - 2*yi)*((x - xi)**2 + (y - yi)**2))/2 + 2*np.log(((x - xi)**2 + (y - yi)**2)**(1/2))*(2*y - 2*yi)*((x - xi)**2 + (y - yi)**2)
                #     phixx = (x - xi)**2 + (y - yi)**2 + 2*np.log(((x - xi)**2 + (y - yi)**2)**(1/2))*(2*x - 2*xi)**2 + 4*np.log(((x - xi)**2 + (y - yi)**2)**(1/2))*((x - xi)**2 + (y - yi)**2) + (3*(2*x - 2*xi)**2)/2
                #     phixy = (3*(2*x - 2*xi)*(2*y - 2*yi))/2 + 2*np.log(((x - xi)**2 + (y - yi)**2)**(1/2))*(2*x - 2*xi)*(2*y - 2*yi)
                #     phiyx = (3*(2*x - 2*xi)*(2*y - 2*yi))/2 + 2*np.log(((x - xi)**2 + (y - yi)**2)**(1/2))*(2*x - 2*xi)*(2*y - 2*yi)
                #     phiyy = (x - xi)**2 + (y - yi)**2 + 2*np.log(((x - xi)**2 + (y - yi)**2)**(1/2))*(2*y - 2*yi)**2 + 4*np.log(((x - xi)**2 + (y - yi)**2)**(1/2))*((x - xi)**2 + (y - yi)**2) + (3*(2*y - 2*yi)**2)/2
                #     phi_gradient = np.array([[phix, phiy]])
                #     phi_laplace = np.array([[phixx, phixy],
                #                             [phiyx, phiyy]])


                # IMQ
                # b = -1
                # phi = (((x - xi)**2 + (y - yi)**2)*c**2 + 1)**(b/2)
                # phix = (b*c**2*(((x - xi)**2 + (y - yi)**2)*c**2 + 1)**(b/2 - 1)*(2*x - 2*xi))/2
                # phiy = (b*c**2*(((x - xi)**2 + (y - yi)**2)*c**2 + 1)**(b/2 - 1)*(2*y - 2*yi))/2
                # phixx = b*c**2*(((x - xi)**2 + (y - yi)**2)*c**2 + 1)**(b/2 - 1) + (b*c**4*(((x - xi)**2 + (y - yi)**2)*c**2 + 1)**(b/2 - 2)*(2*x - 2*xi)**2*(b/2 - 1))/2
                # phixy = (b*c**4*(((x - xi)**2 + (y - yi)**2)*c**2 + 1)**(b/2 - 2)*(2*x - 2*xi)*(2*y - 2*yi)*(b/2 - 1))/2
                # phiyx = (b*c**4*(((x - xi)**2 + (y - yi)**2)*c**2 + 1)**(b/2 - 2)*(2*x - 2*xi)*(2*y - 2*yi)*(b/2 - 1))/2
                # phiyy = b*c**2*(((x - xi)**2 + (y - yi)**2)*c**2 + 1)**(b/2 - 1) + (b*c**4*(((x - xi)**2 + (y - yi)**2)*c**2 + 1)**(b/2 - 2)*(2*y - 2*yi)**2*(b/2 - 1))/2
                # phi_gradient = np.array([[phix, phiy]])
                # phi_laplace = np.array([[phixx, phixy],
                #                         [phiyx, phiyy]])

                return phi, phi_gradient, phi_laplace


            def Ploy_basis(s, t, x, y, i):
                if i == 2:
                    ploy = np.array([1, x, y, x**2, x*y, y**2])
                    ploy_gradient = np.array([[0, 1, 0, 2*x, y, 0], 
                                            [0, 0, 1, 0, x, 2*y]])
                    ploy_laplace = np.array([[0, 0, 0, 2, 0, 2]])
                elif i == 1:
                    ploy = np.array([1, x, y])
                    ploy_gradient = np.array([[0, 1, 0], 
                                            [0, 0, 1]])
                    ploy_laplace = np.array([[0, 0, 0]])
                elif i == 0:
                    ploy = np.array([1])
                    ploy_gradient = np.array([[0, 0]])
                    ploy_laplace = np.array([[0]])
                return ploy, ploy_gradient, ploy_laplace

            def Condition_number(A):
                eigenvalues = np.linalg.eigvals(A.todense())
                return np.abs(eigenvalues).max() / np.abs(eigenvalues).min()

            def Compute_matrix(id_y, new_nodes_local, new_nodes_center, c=1):
                if self.Key_Para['kappa_T'] != 0:
                    id_poly = 1
                    id_poly_m = np.int32(0.5*((id_poly + 1)+1)*(id_poly + 1))
                    A_rbf = scipy.sparse.lil_matrix((len(id_y), len(id_y)))
                    F_rbf_x = np.zeros((len(id_y), 1))
                    F_rbf_y = np.zeros((len(id_y), 1))
                    F_rbf_laplace = np.zeros((len(id_y), 1))
                    B_rbf = scipy.sparse.lil_matrix((len(id_y), id_poly_m))
                    FB_rbf_x = np.zeros((id_poly_m, 1))
                    FB_rbf_y = np.zeros((id_poly_m, 1))
                    FB_rbf_laplace = np.zeros((id_poly_m, 1))
                    for j in range(len(id_y)):
                        phi, _, _ = RBF_basis(new_nodes_local[:, 0], new_nodes_local[:, 1], new_nodes_local[j, 0], new_nodes_local[j, 1], c)
                        A_rbf[j, :] = phi
                        _, phi_gradient, phi_laplace = RBF_basis(new_nodes_center[0, 0], new_nodes_center[0, 1], new_nodes_local[j, 0], new_nodes_local[j, 1], c)
                        phix, phiy = phi_gradient[0, 0], phi_gradient[0, 1]
                        phixx, phixy = phi_laplace[0, 0], phi_laplace[0, 1]
                        phiyx, phiyy = phi_laplace[1, 0], phi_laplace[1, 1]
                        
                        F_rbf_x[j, 0] = phix
                        F_rbf_y[j, 0] = phiy
                        F_rbf_laplace[j, 0] = phixx + phiyy

                        ploy, _, _ = Ploy_basis(0, 0, new_nodes_local[j, 0], new_nodes_local[j, 1], i=id_poly) 
                        B_rbf[j, :] = ploy

                    _, ploy_gradient, ploy_laplace = Ploy_basis(0, 0, new_nodes_center[0, 0], new_nodes_center[0, 1], i=id_poly)
                    FB_rbf_x = ploy_gradient[0, :].reshape(-1, 1)
                    FB_rbf_y = ploy_gradient[1, :].reshape(-1, 1)
                    FB_rbf_laplace = ploy_laplace.reshape(-1, 1)

                    new_A = scipy.sparse.vstack([scipy.sparse.hstack([A_rbf, B_rbf]), scipy.sparse.hstack([B_rbf.T, scipy.sparse.lil_matrix((id_poly_m, id_poly_m))])]).tolil()
                    new_F_rbf_x = np.vstack([F_rbf_x, FB_rbf_x])
                    new_F_rbf_y = np.vstack([F_rbf_y, FB_rbf_y])
                    new_F_rbf_laplace = np.vstack([F_rbf_laplace, FB_rbf_laplace])
                    cond_sub_A = Condition_number(new_A)


                elif self.Key_Para['kappa_T'] == 0:
                    A_rbf = scipy.sparse.lil_matrix((len(id_y), len(id_y)))
                    F_rbf_x = np.zeros((len(id_y), 1))
                    F_rbf_y = np.zeros((len(id_y), 1))
                    F_rbf_laplace = np.zeros((len(id_y), 1))
                    for j in range(len(id_y)):
                        phi, _, _ = RBF_basis(new_nodes_local[:, 0], new_nodes_local[:, 1], new_nodes_local[j, 0], new_nodes_local[j, 1], c)
                        A_rbf[j, :] = phi
                        _, phi_gradient, phi_laplace = RBF_basis(new_nodes_center[0, 0], new_nodes_center[0, 1], new_nodes_local[j, 0], new_nodes_local[j, 1], c)
                        phix, phiy = phi_gradient[0, 0], phi_gradient[0, 1]
                        phixx, phixy = phi_laplace[0, 0], phi_laplace[0, 1]
                        phiyx, phiyy = phi_laplace[1, 0], phi_laplace[1, 1]
                        
                        F_rbf_x[j, 0] = phix
                        F_rbf_y[j, 0] = phiy
                        F_rbf_laplace[j, 0] = phixx + phiyy

                    cond_sub_A = Condition_number(A_rbf)
                    new_A, new_F_rbf_x, new_F_rbf_y, new_F_rbf_laplace = A_rbf, F_rbf_x, F_rbf_y, F_rbf_laplace
                return new_A, new_F_rbf_x, new_F_rbf_y, new_F_rbf_laplace, cond_sub_A

            def bisection_method(a, b, id_y, new_nodes_local, new_nodes_center, tol, max_iter):
                def func(epsilon):
                    _, _, _, _, kappa = Compute_matrix(id_y, new_nodes_local, new_nodes_center, c=epsilon)
                    kappa_T_sub = 1/(self.Key_Para['kappa_T'])
                    return kappa, np.log(kappa / kappa_T_sub)
                
                ka, fa = func(a)
                kb, fb = func(b)
                
                if fa * fb > 0:
                    w = 0
                
                for i in range(max_iter):
                    mid = (a + b) / 2
                    kf, fmid = func(mid)
                    # print(i, a, b, ka, kb, kf)
                    
                    if np.abs(fmid) < tol or (b - a) / 2 < tol:
                        return mid
                    
                    if fa * fmid < 0:
                        b = mid
                        fb = fmid
                    else:
                        a = mid
                        fa = fmid
                
                return (a + b) / 2



            M = nodes.shape[0]
            normal = self.Key_Para['nodes_normal']
            A_sub_coef_x = scipy.sparse.lil_matrix((nodes.shape[0], nodes.shape[0]))
            A_sub_coef_y = scipy.sparse.lil_matrix((nodes.shape[0], nodes.shape[0]))
            A_sub_coef_z = scipy.sparse.lil_matrix((nodes.shape[0], nodes.shape[0]))
            A_coef_laplce = scipy.sparse.lil_matrix((nodes.shape[0], nodes.shape[0]))
            cond_A = np.zeros((M, 1))
            all_c = np.zeros((M, 1))
            for i in tqdm(range(M)):
            # for i in range(M):
                if self.Key_Para['id_node'] == -1:
                    kdtree = KDTree(nodes, leaf_size=2)
                    _, id_y = kdtree.query(nodes[i, :].reshape(1, -1), k=10)
                    id_y = id_y[0]
                    nodes_local = nodes[id_y, :]
                elif self.Key_Para['id_node'] == 0:
                    id_x, _ = np.where(elements==i)
                    sub_elements = elements[id_x, :]
                    id_y = np.unique(sub_elements.reshape(-1), axis=0)
                    nodes_local = nodes[id_y, :]
                elif self.Key_Para['id_node'] == 1:
                    id_x, _ = np.where(elements==i)
                    sub_elements = elements[id_x, :]
                    id_y = np.unique(sub_elements.reshape(-1), axis=0)
                    id_y2 = []
                    for j in range(len(id_y)):
                        id_x, _ = np.where(elements==id_y[j])
                        sub_elements = elements[id_x, :]
                        temp = np.unique(sub_elements.reshape(-1), axis=0)
                        id_y2 = np.hstack((id_y2, temp))
                    id_y = np.unique(id_y2).astype(int)
                    nodes_local = nodes[id_y, :]
                else:
                    dis = np.sqrt(np.sum((nodes - nodes[i,:])** 2, axis=1))
                    id_dis = np.argsort(dis)
                    id_y = id_dis[0:self.Key_Para['id_node']]
                    nodes_local = nodes[id_y, :]


                nodes_center = nodes[i, :]
                nodes_normal = normal[i, :]
                if nodes_normal[2] != 0:
                    t1 = np.array([[1, 0, (-nodes_normal[0]*(1-nodes_center[0]) + nodes_center[1]*nodes_normal[1]) / nodes_normal[2] + nodes_center[2]]]).T
                    t2 = np.array([[0, 1, (-nodes_normal[1]*(1-nodes_center[1]) - nodes_center[0]*nodes_normal[0]) / nodes_normal[2] + nodes_center[2]]]).T
                else:
                    t1 = np.array([[1, 0, 1]]).T
                    t2 = np.array([[0, 1, 1]]).T

                t1 = t1 - np.dot(nodes_normal.reshape(1, -1), t1) * nodes_normal.reshape(-1, 1)
                t2 = t2 - np.dot(nodes_normal.reshape(1, -1), t2) * nodes_normal.reshape(-1, 1) - (np.dot(t2.T, t1) / np.dot(t1.T, t1)) * t1
                t1 = t1 / np.linalg.norm(t1)
                t2 = t2 / np.linalg.norm(t2)
                R = np.hstack((t1, t2, nodes_normal.reshape(-1, 1))).T

                new_nodes_local = np.dot(nodes_local, R.T)[:, 0:2]
                new_nodes_center = np.dot(nodes_center.reshape(1, -1), R.T)[:, 0:2]

                end_c = 1
                if self.Key_Para['kappa_T'] != 0:
                    end_c = bisection_method((1e-10), 1/(1e-2), id_y, new_nodes_local, new_nodes_center, 1e-16, 100)
                new_A, new_F_rbf_x, new_F_rbf_y, new_F_rbf_laplace, cond_A_end = Compute_matrix(id_y, new_nodes_local, new_nodes_center, c=end_c)
                cond_A[i, 0] = cond_A_end
                all_c[i, 0] = end_c
                
                temp1 = scipy.sparse.linalg.inv(new_A).dot(new_F_rbf_x).reshape(1, -1)[:, 0:len(id_y)]
                temp2 = scipy.sparse.linalg.inv(new_A).dot(new_F_rbf_y).reshape(1, -1)[:, 0:len(id_y)]
                temp = np.dot(np.vstack([temp1, temp2]).T, R[0:2, :]).T

                A_sub_coef_x[np.array([i]).repeat(len(id_y)).tolist(), id_y] = temp[0, :]
                A_sub_coef_y[np.array([i]).repeat(len(id_y)).tolist(), id_y] = temp[1, :]
                A_sub_coef_z[np.array([i]).repeat(len(id_y)).tolist(), id_y] = temp[2, :]
                A_coef_laplce[np.array([i]).repeat(len(id_y)).tolist(), id_y] = scipy.sparse.linalg.inv(new_A).dot(new_F_rbf_laplace).reshape(1, -1)[:, 0:len(id_y)]


            # eigen_value = scipy.sparse.linalg.eigs(A_coef_laplce, k=(M-2))[0]
            # print(np.real(eigen_value).max())
            # plt.figure()
            # plt.scatter(np.real(eigen_value), np.imag(eigen_value), s=1)
            # plt.scatter(0, 0, c='r', s=1)
            # plt.savefig(self.Key_Para['File_name'] + '/' + 'eigen_value.png')
            return A_sub_coef_x, A_sub_coef_y, A_sub_coef_z, A_coef_laplce

        def Fast_matrix(self, nodes, elements):
            coef_RBF_laplce = self.Key_Para['coef_RBF_laplce']
            N, M = t.shape[0], nodes.shape[0]
            m = np.linspace(0, N-1, N).astype(int)
            gamma = -(2 - 2*np.cos(m * np.pi * self.Key_Para['dt']))
            gamma = gamma / (self.Key_Para['dt']**2)
            eigenvector = np.cos(np.dot((m * self.Key_Para['dt']).reshape(-1, 1), m.reshape(1, -1)) * np.pi).T
            eigenvector_inv = np.linalg.inv(eigenvector)

            temp_inv = []
            for i in range(N):
                temp = gamma[i] * scipy.sparse.eye(M) + coef_RBF_laplce - self.Key_Para['type_UOT']*scipy.sparse.eye(M) + 1e-6*scipy.sparse.eye(M)
                temp_inv.append(scipy.sparse.linalg.inv(temp))

            self.Key_Para['gamma'] = gamma
            self.Key_Para['eigenvector'] = eigenvector
            self.Key_Para['eigenvector_inv'] = eigenvector_inv
            self.Key_Para['temp_inv'] = temp_inv



        t, nodes, elements = self.gen_Nodes.forward()
        self.gen_Nodes.oprater_nodes(nodes, elements)
        self.gen_Nodes.eigenvectors(nodes, elements)
        
        if os.path.exists('ADMM-MUOT/Matrix-OD/Matrix-OD-TPM/A_sub_coef_x_' + self.Key_Para['type_surface'] + '.mtx'):
            A_sub_coef_x = scipy.io.mmread('ADMM-MUOT/Matrix-OD/Matrix-OD-TPM/A_sub_coef_x_' + self.Key_Para['type_surface'] + '.mtx')
            A_sub_coef_y = scipy.io.mmread('ADMM-MUOT/Matrix-OD/Matrix-OD-TPM/A_sub_coef_y_' + self.Key_Para['type_surface'] + '.mtx')
            A_sub_coef_z = scipy.io.mmread('ADMM-MUOT/Matrix-OD/Matrix-OD-TPM/A_sub_coef_z_' + self.Key_Para['type_surface'] + '.mtx')
            A_coef_laplce = scipy.io.mmread('ADMM-MUOT/Matrix-OD/Matrix-OD-TPM/A_coef_laplce_' + self.Key_Para['type_surface'] + '.mtx')
            A_sub_coef = [A_sub_coef_x, A_sub_coef_y, A_sub_coef_z]
            self.Key_Para['coef_RBF'] = A_sub_coef
            self.Key_Para['coef_RBF_laplce'] = A_coef_laplce
        else:
        # if 1:
            A_sub_coef_x, A_sub_coef_y, A_sub_coef_z, A_coef_laplce = RBF_Tangent_Plane_Poly(self, nodes, elements)
            A_sub_coef = [A_sub_coef_x, A_sub_coef_y, A_sub_coef_z]
            self.Key_Para['coef_RBF'] = A_sub_coef
            self.Key_Para['coef_RBF_laplce'] = A_coef_laplce
            scipy.io.mmwrite('ADMM-MUOT/Matrix-OD/Matrix-OD-TPM/A_sub_coef_x_' + self.Key_Para['type_surface'] + '.mtx', A_sub_coef_x)
            scipy.io.mmwrite('ADMM-MUOT/Matrix-OD/Matrix-OD-TPM/A_sub_coef_y_' + self.Key_Para['type_surface'] + '.mtx', A_sub_coef_y)
            scipy.io.mmwrite('ADMM-MUOT/Matrix-OD/Matrix-OD-TPM/A_sub_coef_z_' + self.Key_Para['type_surface'] + '.mtx', A_sub_coef_z)
            scipy.io.mmwrite('ADMM-MUOT/Matrix-OD/Matrix-OD-TPM/A_coef_laplce_' + self.Key_Para['type_surface'] + '.mtx', A_coef_laplce)

        Fast_matrix(self, nodes, elements)
        self.Key_Para['alpha'] = (self.Key_Para['alpha'])*np.ones((nodes.shape[0], t.shape[0]))
        cur_rho = np.zeros((nodes.shape[0], t.shape[0]))
        cur_m = np.zeros((nodes.shape[0], t.shape[0], nodes.shape[1]))
        cur_f = np.zeros((nodes.shape[0], t.shape[0]))
        cur_rho_bar = np.zeros((nodes.shape[0], t.shape[0]))
        cur_m_bar = np.zeros((nodes.shape[0], t.shape[0], nodes.shape[1]))
        cur_f_bar = np.zeros((nodes.shape[0], t.shape[0]))
        cur_p = np.zeros((nodes.shape[0], t.shape[0]))
        cur_q = np.zeros((nodes.shape[0], t.shape[0], nodes.shape[1]))
        cur_r = np.zeros((nodes.shape[0], t.shape[0]))
        if self.Key_Para['type_surface'] == 'Sphere':
            cur_rho_bar[:, 0] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_0'], self.Key_Para['sigma_0']).reshape(-1)
            cur_rho_bar[:, -1] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_1'], self.Key_Para['sigma_1']).reshape(-1)
            # cur_rho_bar[:, 0] = 0.375*Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_1'], self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.375*Sphere_Surface_Gaussian(self, nodes, np.array([1.0, 0.5, 0.5]), self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.375*Sphere_Surface_Gaussian(self, nodes, np.array([0.5, 0.0, 0.5]), self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.375*Sphere_Surface_Gaussian(self, nodes, np.array([0.5, 1.0, 0.5]), self.Key_Para['sigma_1']).reshape(-1)
            # cur_rho_bar[:, 0] = 0.1875*Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_1'], self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.1875*Sphere_Surface_Gaussian(self, nodes, np.array([1.0, 0.5, 0.5]), self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.1875*Sphere_Surface_Gaussian(self, nodes, np.array([0.5, 0.0, 0.5]), self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.1875*Sphere_Surface_Gaussian(self, nodes, np.array([0.5, 1.0, 0.5]), self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.1875*Sphere_Surface_Gaussian(self, nodes, np.array([0.5 + np.sqrt(1/8), 0.5 + np.sqrt(1/8), 0.5]), self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.1875*Sphere_Surface_Gaussian(self, nodes, np.array([0.5 - np.sqrt(1/8), 0.5 - np.sqrt(1/8), 0.5]), self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.1875*Sphere_Surface_Gaussian(self, nodes, np.array([0.5 + np.sqrt(1/8), 0.5 - np.sqrt(1/8), 0.5]), self.Key_Para['sigma_1']).reshape(-1) + \
            #                     0.1875*Sphere_Surface_Gaussian(self, nodes, np.array([0.5 - np.sqrt(1/8), 0.5 + np.sqrt(1/8), 0.5]), self.Key_Para['sigma_1']).reshape(-1)
        elif self.Key_Para['type_surface'] == 'cow':
            self.Key_Para['mu_0'] = np.array(nodes[1260, :]).reshape(1, -1)
            self.Key_Para['mu_1'] = np.array(nodes[288, :]).reshape(1, -1)
            cur_rho_bar[:, 0] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_0'], self.Key_Para['sigma_0']).reshape(-1)
            cur_rho_bar[:, -1] = 0.25*Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_1'], self.Key_Para['sigma_1']).reshape(-1) + \
                                0.25*Sphere_Surface_Gaussian(self, nodes, np.array(nodes[1152, :]).reshape(1, -1), self.Key_Para['sigma_1']).reshape(-1) + \
                                0.25*Sphere_Surface_Gaussian(self, nodes, np.array(nodes[2238, :]).reshape(1, -1), self.Key_Para['sigma_1']).reshape(-1) + \
                                0.25*Sphere_Surface_Gaussian(self, nodes, np.array(nodes[2283, :]).reshape(1, -1), self.Key_Para['sigma_1']).reshape(-1)
        elif self.Key_Para['type_surface'] == 'armadillo':
            self.Key_Para['mu_0'] = np.array(nodes[520, :]).reshape(1, -1)
            self.Key_Para['mu_1'] = np.array(nodes[1363, :]).reshape(1, -1)
            cur_rho_bar[:, 0] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_0'], self.Key_Para['sigma_0']).reshape(-1)
            cur_rho_bar[:, -1] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_1'], self.Key_Para['sigma_1']).reshape(-1)
        elif self.Key_Para['type_surface'] == 'face':
            self.Key_Para['mu_0'] = np.array(nodes[4853, :]).reshape(1, -1)
            self.Key_Para['mu_1'] = np.array(nodes[4756, :]).reshape(1, -1)
            cur_rho_bar[:, 0] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_0'], self.Key_Para['sigma_0']).reshape(-1)
            cur_rho_bar[:, -1] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_1'], self.Key_Para['sigma_1']).reshape(-1)
        elif self.Key_Para['type_surface'] == 'Airplane':
            self.Key_Para['mu_0'] = np.array(nodes[1043, :]).reshape(1, -1)
            self.Key_Para['mu_1'] = np.array(nodes[47, :]).reshape(1, -1)
            cur_rho_bar[:, 0] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_0'], self.Key_Para['sigma_0']).reshape(-1)
            cur_rho_bar[:, -1] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_1'], self.Key_Para['sigma_1']).reshape(-1)
        elif self.Key_Para['type_surface'] == 'fish':
            self.Key_Para['mu_0'] = np.array(nodes[5718, :]).reshape(1, -1)
            self.Key_Para['mu_1'] = np.array(nodes[6531, :]).reshape(1, -1)
            cur_rho_bar[:, 0] = 0.5*Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_0'], self.Key_Para['sigma_0']).reshape(-1) + \
                                0.5*Sphere_Surface_Gaussian(self, nodes, np.array(nodes[5886, :]).reshape(1, -1), self.Key_Para['sigma_0']).reshape(-1)
            cur_rho_bar[:, -1] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_1'], self.Key_Para['sigma_1']).reshape(-1)
        else:
            cur_rho_bar[:, 0] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_0'], self.Key_Para['sigma_0']).reshape(-1)
            cur_rho_bar[:, -1] = Sphere_Surface_Gaussian(self, nodes, self.Key_Para['mu_1'], self.Key_Para['sigma_1']).reshape(-1)

        loss_all = []
        old_rho = copy.deepcopy(cur_rho)
        self.Key_Para['loss_name'] = ['W2', 'W2_bar', 'ADMM1', 'ADMM2', 'Norm_rho_rho_bar', 'Minus_rho']
        for ep_k in range(self.Key_Para['epochs_k']):
            print('Epochs: %d' %ep_k)
            self.Key_Para['ep_k'] = ep_k

            
            for _ in range(1):
                cur_rho, cur_m, cur_f = self.admm_solver.solver_rho_m_f(t, nodes, elements, cur_rho, cur_m, cur_f, cur_rho_bar, cur_m_bar, cur_f_bar, cur_p, cur_q, cur_r)
                w2 = self.compute.compute_w2(cur_rho, cur_m, cur_f)
                admm1 = self.compute.compute_admm1(t, nodes, cur_rho, cur_m, cur_f, cur_rho_bar, cur_m_bar, cur_f_bar, cur_p, cur_q, cur_r)

                cur_rho_bar, cur_m_bar, cur_f_bar, lambda_h, lambda_grad_t, lambda_grad_h = self.admm_solver.solver_rho_m_f_bar(t, nodes, elements, cur_rho, cur_m, cur_f, cur_rho_bar, cur_m_bar, cur_f_bar, cur_p, cur_q, cur_r)
                w2_bar = self.compute.compute_w2_bar(cur_rho_bar, cur_m_bar, cur_f_bar)
                admm2 = self.compute.compute_admm2(t, nodes, cur_rho, cur_m, cur_f, cur_rho_bar, cur_m_bar, cur_f_bar, cur_p, cur_q, cur_r)

                norm_rho_rho_bar = self.compute.compute_rho_minus_rho_bar(t, nodes, cur_rho, cur_rho_bar)
                minus_rho = self.compute.compute_rho_minus_rho(t, nodes, cur_rho, old_rho)
                old_rho = copy.deepcopy(cur_rho)

            cur_p, cur_q, cur_r = self.admm_solver.solver_p_q_r(t, nodes, elements, cur_rho, cur_m, cur_f, cur_rho_bar, cur_m_bar, cur_f_bar, cur_p, cur_q, cur_r)
            loss_all.append([w2, w2_bar, admm1, admm2, norm_rho_rho_bar, minus_rho])
            

            if self.Key_Para['type_pre_plot'] == 'True':
                if ep_k % 50 == 0:
                    self.plot_result.plot_rho(cur_rho, self.Key_Para['ep_k'])
                    # self.plot_result.plot_rho_bar(cur_rho_bar, self.Key_Para['ep_k'])
                    # self.plot_result.plot_m(cur_m, self.Key_Para['ep_k'])
                    # self.plot_result.plot_m_bar(cur_m_bar, self.Key_Para['ep_k'])
                    # self.plot_result.plot_p(cur_p, self.Key_Para['ep_k'])
                    # self.plot_result.plot_q(cur_q, self.Key_Para['ep_k'])
                    # self.plot_result.plot_lambda(lambda_h, self.Key_Para['ep_k'])
                    # self.plot_result.plot_lambda_grad_t(lambda_grad_t, self.Key_Para['ep_k'])
                    # self.plot_result.plot_lambda_grad_h(lambda_grad_h, self.Key_Para['ep_k'])
                    if self.Key_Para['type_UOT'] == 1:
                        # self.plot_result.plot_f(cur_f, self.Key_Para['ep_k'])
                        # self.plot_result.plot_f_bar(cur_f_bar, self.Key_Para['ep_k'])
                        # self.plot_result.plot_r(cur_r, self.Key_Para['ep_k'])
                        w = 1


        return cur_rho, cur_m, cur_f, cur_rho_bar, cur_m_bar, cur_f_bar, lambda_h, lambda_grad_t, lambda_grad_h, cur_p, cur_q, cur_r, loss_all


class Compute(object):
    def __init__(self, Key_Para):
        super(Compute, self).__init__()
        self.Key_Para = Key_Para

    def compute_w2(self, rho, m, f):
        item1 = np.dot(np.diag(self.Key_Para['nodes_area'][:, 0]), (np.nan_to_num(np.linalg.norm(m, axis=2)**2 / rho) * self.Key_Para['dt']))
        item2 = np.dot(np.diag(self.Key_Para['nodes_area'][:, 0]), (np.nan_to_num(f**2 / rho) * self.Key_Para['dt']))
        _id = np.where(rho<=1e-8)
        item1[_id] = np.zeros_like(item1[_id])
        item2[_id] = np.zeros_like(item2[_id])
        Pre_W_dis = np.sum(item1) + np.sum(item2)
        print('Predicte-W2: ', Pre_W_dis)
        return Pre_W_dis

    def compute_w2_bar(self, rho, m, f):
        item1 = np.dot(np.diag(self.Key_Para['nodes_area'][:, 0]), (np.nan_to_num(np.linalg.norm(m, axis=2)**2 / rho) * self.Key_Para['dt']))
        item2 = np.dot(np.diag(self.Key_Para['nodes_area'][:, 0]), (np.nan_to_num(f**2 / rho) * self.Key_Para['dt']))
        _id = np.where(rho<=1e-8)
        item1[_id] = np.zeros_like(item1[_id])
        item2[_id] = np.zeros_like(item2[_id])
        Pre_W_dis = np.sum(item1) + np.sum(item2)
        print('Predicte-W2_bar: ', Pre_W_dis)
        return Pre_W_dis

    def compute_admm1(self, t, nodes, rho, m, f, rho_bar, m_bar, f_bar, p, q, r):
        item1 = np.sum(np.dot(np.diag(self.Key_Para['nodes_area'][:, 0]), (np.nan_to_num(np.linalg.norm(m, axis=2)**2 / rho) * self.Key_Para['dt'])))
        item2 = np.sum(np.dot(np.diag(self.Key_Para['nodes_area'][:, 0]), (np.nan_to_num(f**2 / rho) * self.Key_Para['dt'])))
        temp = (self.Key_Para['alpha']/2) * ((rho-rho_bar+p)**2 + (np.linalg.norm(m-m_bar+q, axis=2))**2 + (f-f_bar+r)**2)
        item3 = np.sum(np.dot(np.diag(self.Key_Para['nodes_area'][:, 0]), (temp * self.Key_Para['dt'])))
        Pre_W_dis = item1 + item2 + item3
        print('Predicte-admm1: ', Pre_W_dis)
        return Pre_W_dis

    def compute_admm2(self, t, nodes, rho, m, f, rho_bar, m_bar, f_bar, p, q, r):
        temp = (self.Key_Para['alpha']/2) * ((rho-rho_bar+p)**2 + (np.linalg.norm(m-m_bar+q, axis=2))**2 + (f-f_bar+r)**2)
        item3 = np.sum(np.dot(np.diag(self.Key_Para['nodes_area'][:, 0]), (temp * self.Key_Para['dt'])))
        Pre_W_dis = item3
        print('Predicte-admm2: ', Pre_W_dis)
        return Pre_W_dis

    def compute_rho_minus_rho_bar(self, t, nodes, rho, rho_bar):
        temp = np.linalg.norm(rho - rho_bar)
        norm_rho = temp
        print('Predicte-rho_minus_rho_bar: ', norm_rho)
        return norm_rho
    
    def compute_rho_minus_rho(self, t, nodes, cur_rho, old_rho):
        temp = np.linalg.norm(cur_rho - old_rho)
        norm_rho = temp
        print('Predicte-rho_minus_rho: ', norm_rho)
        return norm_rho


class Plot_result(object):
    def __init__(self, Key_Para, gen_Nodes):
        super(Plot_result, self).__init__()
        self.Key_Para = Key_Para
        self.gen_Nodes = gen_Nodes
        self.Dim_time = Key_Para['Dim_time']
        self.Dim_space = Key_Para['Dim_space']
        self.Time = Key_Para['Time']
        self.Space = Key_Para['Space']

    def plot_rho(self, rho, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                if T >= 10:
                    new_T = np.linspace(0, T-1, 13)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, len(new_T), i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    # ax.view_init(elev=90, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))
                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=25, azim=45)
                        # ax.view_init(elev=90, azim=45)
                        ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        ax.set_box_aspect([1, 1, 1])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'Torus':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                if T >= 10:
                    new_T = np.linspace(0, T-1, 13)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, len(new_T), i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=45)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([10, 10, 2])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))
                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=60, azim=45)
                        # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                        ax.set_box_aspect([10, 10, 2])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'Opener':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                if T >= 10:
                    new_T = np.linspace(0, T-1, 13)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 13, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=45, azim=45)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([10, 8, 2])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))
                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=45, azim=45)
                        # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                        ax.set_box_aspect([10, 8, 2])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'Ellipsoid':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=60, azim=0)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([6, 3, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))
                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=60, azim=0)
                        # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                        ax.set_box_aspect([6, 3, 1])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'Peanut':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                if T >= 10:
                    new_T = np.linspace(0, T-1, 13)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, len(new_T), i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=45, azim=45)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                    ax.set_box_aspect([10, 4, 4])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))
                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=45, azim=45)
                        # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 3].min(), cur_nodes[:, 3].max())
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        # ax.set_xlabel('x'), ax.set_ylabel('y'), ax.set_zlabel('z')
                        ax.set_box_aspect([10, 4, 4])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'cow':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)

                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)

                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=20, azim=-45)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1.5, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=20, azim=-45)
                        # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        ax.set_box_aspect([1, 1.5, 1])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'armadillo':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)

                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)

                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=-80, azim=95)
                    ax.set_box_aspect([12, 12, 10])
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=-80, azim=95)
                        ax.set_box_aspect([12, 12, 10])
                        # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'face':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)

                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)

                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=85, azim=170)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1.5, 1, 1.5])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=85, azim=170)
                        # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                        ax.set_box_aspect([1.5, 1, 1.5])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'Airplane':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)

                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)

                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=70, azim=145)
                    ax.set_box_aspect([3, 4, 1])
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=70, azim=145)
                        ax.set_box_aspect([3, 4, 1])
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()

            elif type_surface == 'fish':
                T = rho.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)

                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)

                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=-55, azim=180)
                    ax.set_box_aspect([18, 6, 8])
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

                if 1:
                    sub_T = 5
                    new_T = np.linspace(0, rho.shape[1]-1, sub_T)
                    new_T = np.ceil(new_T).astype(int)


                    for i in range(1, sub_T+1):
                        fig = plt.figure(figsize=(6, 6))
                        cur_nodes = nodes
                        cur_rho = rho[:, new_T[i-1]]

                        surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                        color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                        ax = fig.add_subplot(1, 1, 1, projection='3d')
                        surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                        surf.set_array(color_rho[:, 0])
                        ax.view_init(elev=-55, azim=180)
                        ax.set_box_aspect([18, 6, 8])
                        ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])

                        ax.grid(None)
                        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
                        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))

                        cb = fig.colorbar(surf, ax=[ax])
                        cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                        scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                        scale1 = []
                        for j in range(0, 5):
                            scale1.append('{:.3f}'.format(scale[j]))
                        cb.set_ticklabels(scale1)
                        plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-sub-rho' + str(i) + '.png', dpi=300)
                        plt.close()


        elif self.Dim_space > 3:
            print('There need code')

    def plot_m(self, m, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = m.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 3.6))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = m[:, new_T[i-1], 0]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = m[:, new_T[i-1], 1]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i+10, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    # plt.title('t=%1.3f' %(t[new_T[i-1]]))

                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = m[:, new_T[i-1], 2]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i+20, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    # plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-m-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_f(self, f, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = f.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)


                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = f[:, i-1]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-f-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_rho_bar(self, rho_bar, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = rho_bar.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)


                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = rho_bar[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-rho_bar-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_m_bar(self, m_bar, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = m_bar.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 3.6))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = m_bar[:, new_T[i-1], 0]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = m_bar[:, new_T[i-1], 1]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i+10, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    # plt.title('t=%1.3f' %(t[new_T[i-1]]))

                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = m_bar[:, new_T[i-1], 2]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i+20, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    # plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-m_bar-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_f_bar(self, f_bar, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = f_bar.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)


                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = f_bar[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-f_bar-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_p(self, p, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = p.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)


                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = p[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-p-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_q(self, q, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = q.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 3.6))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = q[:, new_T[i-1], 0]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = q[:, new_T[i-1], 1]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i+10, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    # plt.title('t=%1.3f' %(t[new_T[i-1]]))

                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = q[:, new_T[i-1], 2]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i+20, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    # plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-q-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_r(self, r, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = r.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)


                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = r[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-r-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_lambda(self, lambda_h, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = lambda_h.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)


                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = lambda_h[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-lambda-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

            elif type_surface == 'cow':
                T = lambda_h.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()
                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)

                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)

                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = lambda_h[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=20, azim=-45)
                    # ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1.5, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-lambda_h-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()


        elif self.Dim_space > 3:
            print('There need code')

    def plot_lambda_grad_t(self, lambda_t, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = lambda_t.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)


                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 1.2))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = lambda_t[:, new_T[i-1]]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(1, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-lambda_grad_t-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_lambda_grad_h(self, lambda_grad_h, ite=''):
        if self.Dim_space == 3:
            type_surface = self.Key_Para['type_surface']
            if type_surface == 'Sphere':
                T = lambda_grad_h.shape[1]
                t, nodes, elements = self.gen_Nodes.forward()

                t = np.linspace(self.Time[0, 0], self.Time[0, 1], T)
                if T >= 10:
                    new_T = np.linspace(0, T-1, 10)
                    new_T = np.ceil(new_T).astype(int)


                fig = plt.figure(figsize=(int(3*len(new_T)), 3.6))
                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = lambda_grad_h[:, new_T[i-1], 0]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    plt.title('t=%1.3f' %(t[new_T[i-1]]))

                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = lambda_grad_h[:, new_T[i-1], 1]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i+10, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    # plt.title('t=%1.3f' %(t[new_T[i-1]]))

                for i in range(1, len(new_T)+1):
                    cur_nodes = nodes
                    cur_rho = lambda_grad_h[:, new_T[i-1], 2]

                    surface_rho = (cur_rho[elements[:, 0]] + cur_rho[elements[:, 1]] + cur_rho[elements[:, 2]]) / 3
                    color_rho = ((surface_rho - surface_rho.min()) / (surface_rho - surface_rho.min()).max()).reshape(-1, 1)

                    ax = fig.add_subplot(3, 10, i+20, projection='3d')
                    surf = ax.plot_trisurf(cur_nodes[:, 0], cur_nodes[:, 1], cur_nodes[:, 2], triangles=elements, vmin=0, vmax=1, antialiased=False)
                    surf.set_array(color_rho[:, 0])
                    ax.view_init(elev=25, azim=45)
                    ax.set_xlim3d(0, 1), ax.set_ylim3d(0, 1), ax.set_zlim3d(cur_nodes[:, 2].min(), cur_nodes[:, 2].max())
                    ax.set_xticks([]),  ax.set_yticks([]), ax.set_zticks([])
                    ax.set_box_aspect([1, 1, 1])
                    
                    cb = fig.colorbar(surf, ax=[ax])
                    cb.set_ticks(np.linspace(surf.norm.vmin, surf.norm.vmax, num=5))
                    scale = np.linspace(surface_rho.min(), surface_rho.max(), num=5)
                    scale1 = []
                    for j in range(0, 5):
                        scale1.append('{:.3f}'.format(scale[j]))
                    cb.set_ticklabels(scale1)
                    # plt.title('t=%1.3f' %(t[new_T[i-1]]))

                plt.savefig(self.Key_Para['File_name'] + '/' + 'Fig-lambda_grad_h-' + str(ite) + '.png', bbox_inches='tight', dpi=300)
                plt.close()

        elif self.Dim_space > 3:
            print('There need code')

    def plot_loss(self, all_sub_loss):
        num = len(all_sub_loss)
        num_loss = len(all_sub_loss[0])
        all_loss = np.zeros((num, num_loss + 1))
        for i in range(num):
            for j in range(1, num_loss + 1):
                all_loss[i, j] = all_sub_loss[i][j-1]
            all_loss[i, 0] = np.sum(all_loss[i, 1:])

        plt.figure(figsize=(4*all_loss.shape[1], 3))
        for i in range(all_loss.shape[1]):
            if i == 0:
                plt.subplot(1, all_loss.shape[1], i+1)
                plt.plot(all_loss[:, i], color='r')
                plt.title('all_loss')
            else:
                plt.subplot(1, all_loss.shape[1], i+1)
                plt.plot(all_loss[:, i])
                plt.title(self.Key_Para['loss_name'][i-1])
        plt.savefig(self.Key_Para['File_name'] + '/' + 'loss.png')
        plt.close()

        np.save(self.Key_Para['File_name'] + '/' + 'all_loss.npy', all_loss)

    def plot_norm_rho_rho_bar(self, all_sub_loss):
        num = len(all_sub_loss)
        num_loss = len(all_sub_loss[0])
        all_loss = np.zeros((num, num_loss + 1))
        for i in range(num):
            for j in range(1, num_loss + 1):
                all_loss[i, j] = all_sub_loss[i][j-1]
            all_loss[i, 0] = np.sum(all_loss[i, 1:])

        plt.figure(figsize=(6, 6))
        plt.plot(all_loss[:, -2], color='blue')
        plt.xlabel('epochs')
        plt.ylabel(r'$log^{\|\rho-\bar{\rho}\|_2}$')
        # plt.title('Norm_rho_rho_bar')
        plt.savefig(self.Key_Para['File_name'] + '/' + 'Norm_rho_rho_bar.png')
        plt.close()
        print('Norm_rho_rho_bar:', all_loss[-1, -2])

    def plot_rho_minus_rho(self, all_sub_loss):
        num = len(all_sub_loss)
        num_loss = len(all_sub_loss[0])
        all_loss = np.zeros((num, num_loss + 1))
        for i in range(num):
            for j in range(1, num_loss + 1):
                all_loss[i, j] = all_sub_loss[i][j-1]
            all_loss[i, 0] = np.sum(all_loss[i, 1:])

        plt.figure(figsize=(6, 6))
        plt.plot(all_loss[:, -1], color='blue')
        plt.xlabel('epochs')
        plt.ylabel(r'$log^{\|\rho^{end}-\rho^{end-1}\|_2}$')
        # plt.title('Minus_rho')
        plt.savefig(self.Key_Para['File_name'] + '/' + 'Minus_rho.png')
        plt.close()
        print('Minus_rho:', all_loss[-1, -1])


def main(Key_Para):
    utilize = Utilize(Key_Para)
    utilize.make_file()
    utilize.setup_seed(1)
    utilize.mu_0_mu_1()
    utilize.sigma_0_sigma_1()
    utilize.Time_Space()
    utilize.print_key(Key_Para)



    gen_Nodes = Gnerate_node(Key_Para)
    admm_solver = ADMM_solver(Key_Para)
    compute = Compute(Key_Para)
    plot_result = Plot_result(Key_Para, gen_Nodes)
    train_loop = Train_loop(Key_Para, gen_Nodes, admm_solver, compute, plot_result)
    rho, m, f, rho_bar, m_bar, f_bar, lambda_h, lambda_grad_t, lambda_grad_h, p, q, r, loss_all = train_loop.train()

    plot_result.plot_rho(rho)
    plot_result.plot_m(m)
    plot_result.plot_f(f)
    plot_result.plot_rho_bar(rho_bar)
    plot_result.plot_m_bar(m_bar)
    plot_result.plot_f_bar(f_bar)
    plot_result.plot_p(p)
    plot_result.plot_q(q)
    plot_result.plot_r(r)
    plot_result.plot_lambda(lambda_h)
    plot_result.plot_lambda_grad_t(lambda_grad_t)
    plot_result.plot_lambda_grad_h(lambda_grad_h)
    plot_result.plot_loss(loss_all)
    plot_result.plot_norm_rho_rho_bar(loss_all)
    plot_result.plot_rho_minus_rho(loss_all)



if __name__== "__main__" :
    time_begin = time.time()
    time_now = time.strftime("%Y%m%d-%H%M", time.localtime())
    name = os.path.basename(sys.argv[0])
    File_name = time_now + '-' + name[:-3]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(0)
    torch.set_default_dtype(torch.float64)

    test = 'Sphere'
    if test == 'Sphere':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 13
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 1.0], [0.5, 0.5, 0.0]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Sphere'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0

    elif test == 'Ellipsoid':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 0.5 + np.sqrt(1/24)], \
                     [0.5, 0.5, 0.5 - np.sqrt(1/24)]  
        sigma_0, sigma_1 = 0.025, 0.025
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Ellipsoid'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0

    elif test == 'Peanut':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 13
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5 + 0.25*np.sqrt(1+np.sqrt(1.2)), 0.5, 0.5], \
                     [0.5 - 0.25*np.sqrt(1+np.sqrt(1.2)), 0.5, 0.5],
        sigma_0, sigma_1 = 0.025, 0.025
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Peanut'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0

    elif test == 'Torus':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 13
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5 + np.sqrt(1/8), 0.5 + np.sqrt(1/8), 0.5], \
                     [0.5 - np.sqrt(1/8), 0.5 - np.sqrt(1/8), 0.5]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 7
        alpha = 1
        kappa_T = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Torus'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0

    elif test == 'Opener':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 13
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5 + np.sqrt((3 + np.sqrt(9 + 60*np.sqrt(1/60)))/30), 0.5, 0.5], \
                     [0.5 - np.sqrt((3 + np.sqrt(9 + 60*np.sqrt(1/60)))/30), 0.5, 0.5]
        sigma_0, sigma_1 = 0.025, 0.025
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 7
        alpha = 1
        kappa_T = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Opener'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0

    elif test == 'Sphere_U1':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 13
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 1.0], [0., 0.5, 0.5]
        sigma_0, sigma_1 = 0.025, 0.025
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Sphere'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 1

    elif test == 'Sphere_U2':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 13
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 1.0], [0., 0.5, 0.5]
        sigma_0, sigma_1 = 0.025, 0.025
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Sphere'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 1

    elif test == 'Sphere_U3':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 13
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 1.0], [0., 0.5, 0.5]
        sigma_0, sigma_1 = 0.025, 0.025
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Sphere'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 1

    elif test == 'Sphere_U4':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 13
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 1.0], [0., 0.5, 0.5]
        sigma_0, sigma_1 = 0.025, 0.025
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 0

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Sphere'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 1

    elif test == 'face':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 0.0], [0.5, 0.5, 1.0]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 1e-4

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'face'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0

    elif test == 'armadillo':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 0.0], [0.5, 0.5, 1.0]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 1e-4

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'armadillo'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0

    elif test == 'cow':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 50
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 0.0], [0.5, 0.5, 1.0]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 1e-4

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'cow'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0

    elif test == 'Airplane':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 0.0], [0.5, 0.5, 1.0]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 1e-4

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'Airplane'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0

    elif test == 'fish':
        Dim_time = 1
        Dim_space = 3
        Num_Nodes_t = 10
        Num_Nodes_all_space = 1
        mu_0, mu_1 = [0.5, 0.5, 0.0], [0.5, 0.5, 1.0]
        sigma_0, sigma_1 = 0.05, 0.05
        Time, Space = [0.0, 1.0], [0.0, 1.0]
        eta = 2

        epochs_k = 200
        id_node = 0
        alpha = 1
        kappa_T = 1e-4

        type_print = 'False'           # 'True'      'False' 
        type_surface = 'fish'  
        type_node = 'Load'         # 'Load'  'Generate' 
        type_pre_plot = 'True'   # 'True'  'False'  
        type_UOT = 0







    File_name = File_name + '_' + test
    Key_Parameters = {
        'test': test,
        'File_name': File_name,
        'Dim_time': Dim_time, 
        'Dim_space': Dim_space,
        'Num_Nodes_t': Num_Nodes_t, 
        'Num_Nodes_all_space': Num_Nodes_all_space,
        'mu_0': mu_0, 
        'mu_1': mu_1, 
        'sigma_0': sigma_0, 
        'sigma_1': sigma_1, 
        'Time': Time, 
        'Space': Space,
        'eta': eta,
        'epochs_k': epochs_k, 
        'id_node': id_node,
        'alpha': alpha,
        'kappa_T': kappa_T,
        'type_print': type_print,
        'type_node': type_node, 
        'type_surface': type_surface,
        'type_pre_plot': type_pre_plot,
        'type_UOT': type_UOT, 
            }

    main(Key_Parameters)
    print('Runing_time:', time.time() - time_begin, 's')





