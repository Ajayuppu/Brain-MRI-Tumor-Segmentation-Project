"""U-Net size and output shape."""

import torch

from brain_mri_segmentation.model import EXPECTED_PARAMETER_COUNT, UNet, count_trainable_parameters


def test_parameter_count_matches_the_study():
    assert count_trainable_parameters(UNet(base_channels=32)) == EXPECTED_PARAMETER_COUNT == 7_763_041


def test_output_is_one_logit_channel_at_input_resolution():
    with torch.no_grad():
        output = UNet(base_channels=8).eval()(torch.zeros(2, 3, 256, 256))
    assert output.shape == (2, 1, 256, 256)


def test_checkpoint_keys_match_the_original_study_layout():
    keys = UNet().state_dict().keys()
    assert {"downs.0.0.weight", "ups.0.weight", "decs.0.0.weight", "head.weight"} <= set(keys)
