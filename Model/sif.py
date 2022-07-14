#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import numpy as np
import torch.nn.functional as F

def roll_pitch_yaw_to_rotation_matrices(roll_pitch_yaw):
    cosines = torch.cos(roll_pitch_yaw)
    sines = torch.sin(roll_pitch_yaw)
    cx, cy, cz = cosines.unbind(-1)
    sx, sy, sz = sines.unbind(-1)
    rotation = torch.stack(
        [cz * cy, cz * sy * sx - sz * cx, cz * sy * cx + sz * sx,
         sz * cy, sz * sy * sx + cz * cx, sz * sy * cx - cz * sx,
         -sy, cy * sx, cy * cx], -1
    )
    rotation = torch.reshape(rotation, [rotation.shape[0], -1, 3, 3])
    return rotation

def sample_quadric_surface(quadric, center, samples):
    samples = samples - center.unsqueeze(2)
    homogeneous_sample_coords = F.pad(samples, [0, 1], "constant", 1)
    half_distance = torch.matmul(quadric, homogeneous_sample_coords.transpose(-1, -2))
    half_distance = half_distance.transpose(-1, -2)
    algebraic_distance = torch.sum(homogeneous_sample_coords * half_distance, -1, keepdim=True)
    return algebraic_distance

def decode_covariance_roll_pitch_yaw(radius, invert=False):
    d = 1.0 / (radius[..., :3] + 1e-8) if invert else radius[..., :3]
    diag = torch.diag_embed(d)
    rotation = roll_pitch_yaw_to_rotation_matrices(radius[..., 3:6])
    return torch.matmul(torch.matmul(rotation, diag), rotation.transpose(-1, -2))

def sample_cov_bf(center, radius, samples):
    diff = samples - center.unsqueeze(2)
    x, y, z = diff.unbind(-1)

    inv_cov = decode_covariance_roll_pitch_yaw(radius, invert=True)
    inv_cov = torch.reshape(inv_cov, [inv_cov.shape[0], -1, 1, 9])
    c00, c01, c02, _, c11, c12, _, _, c22 = inv_cov.unbind(-1)
    dist = (x * (c00 * x + c01 * y + c02 * z)
            + y * (c01 * x + c11 * y + c12 * z)
            + z * (c02 * x + c12 * y + c22 * z))
    dist = torch.exp(-0.5 * dist)
    return dist.unsqueeze(-1)

def compute_shape_element_influences(quadrics, centers, radii, samples):
    sampled_quadrics = sample_quadric_surface(quadrics, centers, samples)
    sampled_rbfs = sample_cov_bf(centers, radii, samples)
    return sampled_quadrics, sampled_rbfs

def homogenize(m):
  m = F.pad(m, [0, 1, 0, 1], "constant", 0)
  m[..., -1, -1] = 1
  return m

def _unflatten(config, vector):
    return torch.split(vector, [1, 3, 6, config['model']['implicit_parameter_length']], -1)

class StructuredImplicit(object):
    def __init__(self, config, constant, center, radius, iparam, net=None):
        self.config = config
        self.implicit_parameter_length = config['model']['implicit_parameter_length']
        self.element_count = config['model']['element_count']
        self.sym_element_count = config['model']['sym_element_count']

        self.constants = constant
        self.radii = radius
        self.centers = center
        self.iparams = iparam
        self.effective_element_count = self.element_count + self.sym_element_count
        self.device = constant.device
        self.batch_size = constant.size(0)
        self.net = net
        self._packed_vector = None
        self._analytic_code = None
        self._all_centers = None

    @classmethod
    def from_packed_vector(cls, config, packed_vector, net):
        constant, center, radius, iparam = _unflatten(config, packed_vector)
        return cls(config, constant, center, radius, iparam, net)

    @classmethod
    def from_activation(cls, config, activation, net):
        constant, center, radius, iparam = _unflatten(config, activation)
        constant = -torch.abs(constant)
        radius_var = torch.sigmoid(radius[..., :3])
        radius_var = 0.15 * radius_var
        radius_var = radius_var * radius_var
        max_euler_angle = np.pi / 4.0
        radius_rot = torch.clamp(radius[..., 3:], -max_euler_angle, max_euler_angle)
        radius = torch.cat([radius_var, radius_rot], -1)
        new_center = center / 2
        return cls(config, constant, new_center, radius, iparam, net)

    def _tile_for_symgroups(self, elements):
        sym_elements = elements[:, :self.sym_element_count, ...]
        elements = torch.cat([elements, sym_elements], 1)
        return elements

    def _generate_symgroup_samples(self, samples):
        samples = samples.unsqueeze(1).expand(-1, self.element_count, -1, -1)
        sym_samples = samples[:, :self.sym_element_count].clone()
        sym_samples *= torch.tensor([-1, 1, 1], dtype=torch.float32, device=self.device)
        effective_samples = torch.cat([samples, sym_samples], 1)
        return effective_samples

    def compute_world2local(self):
        tx = torch.eye(3, device=self.device).expand(self.batch_size, self.element_count, -1, -1)
        centers = self.centers.unsqueeze(-1)
        tx = torch.cat([tx, -centers], -1)
        lower_row = torch.tensor([0., 0., 0., 1.], device=self.device).expand(self.batch_size, self.element_count, 1, -1)
        tx = torch.cat([tx, lower_row], -2)

        rotation = roll_pitch_yaw_to_rotation_matrices(self.radii[..., 3:6]).inverse()
        diag = 1.0 / (torch.sqrt(self.radii[..., :3] + 1e-8) + 1e-8)
        scale = torch.diag_embed(diag)

        tx3x3 = torch.matmul(scale, rotation)
        return torch.matmul(homogenize(tx3x3), tx)

    def implicit_values(self, local_samples):
        iparams = self._tile_for_symgroups(self.iparams)
        values = self.net.eval_implicit_parameters(iparams, local_samples)
        return values

    @property
    def all_centers(self):
        if self._all_centers is None:
            sym_centers = self.centers[:, :self.sym_element_count].clone()
            sym_centers[:, :, 0] *= -1  # reflect across the YZ plane
            self._all_centers = torch.cat([self.centers, sym_centers], 1)
        return self._all_centers

    def class_at_samples(self, samples, apply_class_transfer=True):
        effective_constants = self._tile_for_symgroups(self.constants)
        effective_centers = self._tile_for_symgroups(self.centers)
        effective_radii = self._tile_for_symgroups(self.radii)

        effective_samples = self._generate_symgroup_samples(samples)
        constants_quadrics = torch.zeros(self.batch_size, self.effective_element_count, 4, 4, device=self.device)
        constants_quadrics[:, :, -1:, -1] = effective_constants

        per_element_constants, per_element_weights = compute_shape_element_influences(
            constants_quadrics, effective_centers, effective_radii, effective_samples
        )

        effective_world2local = self._tile_for_symgroups(self.compute_world2local())
        local_samples = torch.matmul(F.pad(effective_samples, [0, 1], "constant", 1),
                                     effective_world2local.transpose(-1, -2))[..., :3]
        implicit_values = self.implicit_values(local_samples)

        residuals = 1 + implicit_values
        local_decisions = per_element_constants * per_element_weights * residuals
        local_weights = per_element_weights
        sdf = torch.sum(local_decisions, 1)
        if apply_class_transfer:
            sdf = torch.sigmoid(100 * (sdf + 0.07))

        return sdf, (local_decisions, local_weights)

    @property
    def vector(self):
        if self._packed_vector is None:
            self._packed_vector = torch.cat([self.constants, self.centers, self.radii, self.iparams], -1)
        return self._packed_vector

    @property
    def analytic_code(self):
        if self._analytic_code is None:
            self._analytic_code = torch.cat([self.constants, self.centers, self.radii], -1)
        return self._analytic_code

    def unbind(self):
        return [StructuredImplicit.from_packed_vector(self.config, self.vector[i:i+1], self.net)
                for i in range(self.vector.size(0))]

    def __getitem__(self, item):
        return StructuredImplicit.from_packed_vector(self.config, self.vector[item], self.net)

    def dict(self):
        return {'constant': self.constants, 'radius': self.radii, 'center': self.centers, 'iparam': self.iparams}

