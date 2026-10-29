"""Differentiable models checked against the public NumPy observation models."""
from __future__ import annotations
import torch


def multiply(a, b):
    av, aw, bv, bw = a[..., :3], a[..., 3:], b[..., :3], b[..., 3:]
    return torch.cat((aw*bv+bw*av+torch.cross(av, bv, dim=-1), aw*bw-(av*bv).sum(-1, keepdim=True)), -1)


def quaternion(rotvec):
    angle = torch.sqrt((rotvec*rotvec).sum(-1, keepdim=True)+1e-24)
    return torch.cat((.5*torch.sinc(angle/(2*torch.pi))*rotvec, torch.cos(angle/2)), -1)


def rotation(q):
    q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True)
    x, y, z, w = q.unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w),
                        2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w),
                        2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def skew(v):
    x, y, z = v.unbind(-1); zero = torch.zeros_like(x)
    return torch.stack((zero, -z, y, z, zero, -x, -y, x, zero), -1).reshape(*v.shape[:-1], 3, 3)


class TensorModel:
    def __init__(self, model, device="cpu", dtype=torch.float64):
        self.model, self.spec, self.device, self.dtype = model, model.spec, device, dtype
        self.scale = self.tensor(self.spec.error_scale)
        self.weights = 1/self.tensor(self.spec.measurement_scale)**2
        if self.spec.name == "motor_speed":
            self.ad, self.bd = self.tensor(model.Ad), self.tensor(model.Bd)
        if self.spec.name == "cooperative_pose":
            self.points = self.tensor(model.points)

    def tensor(self, value):
        return torch.tensor(value, dtype=self.dtype, device=self.device)

    def flow_cart(self, state, control):
        _, velocity, theta, omega = state.unbind(-1)
        sine, cosine = theta.sin(), theta.cos()
        temp = (control[..., 0]+.05*omega.square()*sine)/1.1
        angular = (9.81*sine-cosine*temp)/(.5*(4/3-.1*cosine.square()/1.1))
        linear = temp-.05*angular*cosine/1.1
        return torch.stack((velocity, linear, omega, angular), -1)

    def predict(self, state, control):
        dt = self.spec.dt
        if self.spec.name == "motor_speed":
            return state@self.ad.T+control@self.bd.T
        if self.spec.name == "cartpole_output":
            k1 = self.flow_cart(state, control)
            k2 = self.flow_cart(state+dt*k1/2, control)
            k3 = self.flow_cart(state+dt*k2/2, control)
            k4 = self.flow_cart(state+dt*k3, control)
            return state+dt*(k1+2*k2+2*k3+k4)/6
        q = multiply(state[..., 3:7], quaternion(dt*state[..., 10:13]))
        q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        return torch.cat((state[..., :3]+dt*state[..., 7:10], q, state[..., 7:]), -1)

    def retract(self, state, delta):
        if self.spec.name != "cooperative_pose":
            return state+delta
        q = multiply(state[..., 3:7], quaternion(delta[..., 3:6]))
        q = q/torch.linalg.vector_norm(q, dim=-1, keepdim=True)
        return torch.cat((state[..., :3]+delta[..., :3], q, state[..., 7:10]+delta[..., 6:9],
                          state[..., 10:13]+delta[..., 9:12]), -1)

    def error(self, truth, estimate):
        if self.spec.name == "motor_speed":
            return truth-estimate
        if self.spec.name == "cartpole_output":
            diff = truth-estimate
            return torch.stack((diff[..., 0], diff[..., 1], torch.atan2(diff[..., 2].sin(), diff[..., 2].cos()), diff[..., 3]), -1)
        inverse = torch.cat((-estimate[..., 3:6], estimate[..., 6:7]), -1)
        relative = multiply(inverse, truth[..., 3:7])
        relative = torch.where(relative[..., 3:]<0, -relative, relative)
        norm = torch.sqrt(relative[..., :3].square().sum(-1, keepdim=True)+1e-24)
        angle = 2*torch.atan2(norm, relative[..., 3:])
        orientation = relative[..., :3]*(angle/norm)
        return torch.cat((truth[..., :3]-estimate[..., :3], orientation, truth[..., 7:]-estimate[..., 7:]), -1)

    def measure_and_jacobian(self, state):
        batch_shape = state.shape[:-1]
        if self.spec.name == "motor_speed":
            return state[..., 1:2], self.tensor([[0., 1.]]).expand(*batch_shape, 1, 2)
        if self.spec.name == "cartpole_output":
            return state[..., [0, 2]], self.tensor([[1., 0., 0., 0.], [0., 0., 1., 0.]]).expand(*batch_shape, 2, 4)
        matrix = rotation(state[..., 3:7])
        points = state[..., None, :3]+torch.einsum("...ij,pj->...pi", matrix, self.points)
        x, y, z = points.unbind(-1)
        # Domain loss/validation rejects negative depth; clamp only avoids NaN in proposals.
        z = z.clamp(min=.01)
        measurement = torch.stack((600*x/z+320, 600*y/z+240), -1).flatten(-2)
        zero = torch.zeros_like(z)
        projection = torch.stack((600/z, zero, -600*x/z.square(), zero, 600/z, -600*y/z.square()), -1).reshape(*points.shape[:-1], 2, 3)
        oriented = -torch.einsum("...ij,pjk->...pik", matrix, skew(self.points))
        hrot = projection@oriented
        h = torch.cat((projection, hrot, torch.zeros((*projection.shape[:-1], 6), dtype=state.dtype, device=state.device)), -1)
        return measurement, h.flatten(-3, -2)

    def groups(self, errors):
        if self.spec.name == "cooperative_pose":
            return torch.linalg.vector_norm(errors.reshape(*errors.shape[:-1], 4, 3), dim=-1)
        return errors.abs()

    def domain_penalty(self, state):
        if self.spec.name == "cooperative_pose":
            return torch.relu(.3-state[..., 2]).square()+torch.relu(torch.linalg.vector_norm(state[..., :3], dim=-1)-20).square()
        if self.spec.name == "cartpole_output":
            return torch.relu(state[..., 0].abs()-2.4).square()+torch.relu(state[..., 2].abs()-.785).square()
        return torch.relu(state.abs()-6).square().sum(-1)
