import os
import copy
import torch

def export_policy_as_jit(policy_latent, action, normalizer, path, filename="policy.pt"):
    policy_exporter = TorchPolicyExporter(policy_latent, action, normalizer)
    policy_exporter.export(path, filename)


def export_policy_as_onnx(policy_latent, action, normalizer, path, filename="policy.onnx"):
    policy_exporter = OnnxPolicyExporter(policy_latent, action, normalizer)
    policy_exporter.export(path, filename)


def export_transformer_policy_as_onnx(transf_policy, normalizer, path, filename="policy.onnx"):
    os.makedirs(path, exist_ok=True)
    policy_exporter = OnnxTransformerPolicyExporter(transf_policy, normalizer)
    policy_exporter.export(path, filename)


class TorchPolicyExporter(torch.nn.Module):
    def __init__(self, policy_latent, action, normalizer=None):
        super().__init__()
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.policy_latent = copy.deepcopy(policy_latent)
        self.action = copy.deepcopy(action)

    def forward(self, x):
        return self.action(self.policy_latent(self.normalizer(x)))

    @torch.jit.export
    def reset(self):
        pass

    def export(self, path, filename):
        os.makedirs(path, exist_ok=True)
        path = os.path.join(path, filename)
        self.to("cpu")
        traced_script_module = torch.jit.script(self)
        traced_script_module.save(path)


class OnnxPolicyExporter(torch.nn.Module):
    def __init__(self, policy_latent, action, normalizer=None):
        super().__init__()
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.policy_latent = copy.deepcopy(policy_latent)
        self.action = copy.deepcopy(action)

    def forward(self, x):
        return self.action(self.policy_latent(self.normalizer(x)))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.policy_latent[0].in_features)
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=True,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )


class OnnxTransformerPolicyExporter(torch.nn.Module):
    def __init__(self, transf_policy, normalizer):
        super().__init__()
        if normalizer:
            self.normalizer = copy.deepcopy(normalizer)
        else:
            self.normalizer = torch.nn.Identity()
        self.transformer_encoder = copy.deepcopy(transf_policy.transformer_enc)
        self.policy_latent = copy.deepcopy(transf_policy.policy_latent_net)
        self.action_net = copy.deepcopy(transf_policy.action_mean_net)
        self.num_states = transf_policy.num_states
        self.seq_len = transf_policy.seq_len
        self.feature_dim = transf_policy.feature_dim
        self.pooling = transf_policy.pooling

    def forward(self, input_x, mask=None):
        """
        input_x: (batch_size, num_obs)
        num_obs = normalized(state_dim + action_dim) + seq_len * feature_dim
        """
        input_x, _ = self.normalizer(input_x, mask)
        state = input_x[:, :self.num_states]
        features = input_x[:, self.num_states:].view(-1, self.seq_len, self.feature_dim)
        state_features = torch.cat((state.unsqueeze(1).repeat(1, self.seq_len, 1), features), dim=2)
        encoded_features = self.transformer_encoder(state_features, src_key_padding_mask=(mask != 1))
        if self.pooling == 'mean':
            encoded_features = torch.mean(encoded_features, dim=1)
        else:
            encoded_features = torch.max(encoded_features, dim=1)[0]
        policy_input = torch.cat((state, encoded_features), dim=1)
        return self.action_net(self.policy_latent(policy_input))

    def export(self, path, filename):
        self.to("cpu")
        obs = torch.zeros(1, self.num_states + self.seq_len * self.feature_dim)
        mask = torch.ones(1, self.seq_len, dtype=torch.int64)
        torch.onnx.export(
            self,
            (obs, mask),
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=False,
            input_names=["obs", "mask"],
            output_names=["actions"],
            dynamic_axes={}
        )
