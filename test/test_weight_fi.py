import torch

from pytorchfi.core import FaultInjection as pfi_core

from test.util_test import CIFAR10_set_up_custom


class TestWeightFi:
    """Testing focuses on weight perturbations."""

    def setup_class(self):
        torch.manual_seed(0)

        batch_size = 1
        workers = 1
        channels = 3
        img_size = 32
        use_gpu = False

        self.model, self.dataset = CIFAR10_set_up_custom(batch_size, workers)
        dataiter = iter(self.dataset)
        self.images, self.labels = dataiter.next()

        self.model.eval()
        with torch.no_grad():
            self.golden_output = self.model(self.images)

        self.p = pfi_core(
            self.model,
            batch_size,
            input_shape=[channels, img_size, img_size],
            layer_types=[torch.nn.Conv2d, torch.nn.Linear],
            use_cuda=use_gpu,
        )

    def test_single_weight_fi_cpu(self):
        layer_i = [1]
        k = [15]
        c_i = [20]
        h_i = [2]
        w_i = [3]
        inj_value_base = 10000.0
        inj_value_i = [inj_value_base]

        corrupt_model = self.p.declare_weight_fault_injection(
            layer_num=layer_i, k=k, dim1=c_i, dim2=h_i, dim3=w_i, value=inj_value_i
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        assert not torch.all(corrupt_output.eq(self.golden_output))

        corrupt_model = self.p.declare_weight_fault_injection(
            layer_num=layer_i,
            k=k,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=[0.01388985849916935],
        )

        corrupt_model.eval()
        with torch.no_grad():
            uncorrupted_output = corrupt_model(self.images)

        assert torch.all(uncorrupted_output.eq(self.golden_output))

        corrupt_model = self.p.declare_weight_fault_injection(
            layer_num=layer_i,
            k=k,
            dim1=c_i,
            dim2=h_i,
            dim3=w_i,
            value=[inj_value_base * 2],
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output_2 = corrupt_model(self.images)

        assert not torch.all(corrupt_output_2.eq(self.golden_output))
        assert torch.all(corrupt_output_2.eq(corrupt_output_2))

    def test_single_weight_fi_no_error_cpu(self):
        layer_i = [4]
        k = [153]
        c_i = [254]
        h_i = [0]
        w_i = [0]
        inj_value_i = [10000.0]

        corrupt_model = self.p.declare_weight_fault_injection(
            layer_num=layer_i, k=k, dim1=c_i, dim2=h_i, dim3=w_i, value=inj_value_i
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        assert torch.all(corrupt_output.eq(self.golden_output))

    def test_multi_weight_fi_cpu(self):
        layer_i = [1, 2, 5]
        k = [15, 12, 1]
        c_i = [20, 8, 1]
        h_i = [2, 1, None]
        w_i = [3, 1, None]
        inj_value_base = 10000.0
        inj_value_i = [inj_value_base, inj_value_base, inj_value_base]

        corrupt_model = self.p.declare_weight_fault_injection(
            layer_num=layer_i, k=k, dim1=c_i, dim2=h_i, dim3=w_i, value=inj_value_i
        )

        corrupt_model.eval()
        with torch.no_grad():
            corrupt_output = corrupt_model(self.images)

        assert not torch.all(corrupt_output.eq(self.golden_output))
