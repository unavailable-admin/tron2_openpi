import dataclasses

import torch

from openpi.models_pytorch.ttt_fast_weight import TTTConfig
from openpi.models_pytorch.ttt_fast_weight import TTTFastWeightStack
from openpi.models_pytorch.ttt_fast_weight import TTTSession
from openpi.models_pytorch.ttt_fast_weight import TTTUpdateMode
from openpi.models_pytorch.ttt_fast_weight import fast_mlp_forward
from openpi.models_pytorch.ttt_fast_weight import update_fast_state


def make_stack() -> TTTFastWeightStack:
    return TTTFastWeightStack(
        TTTConfig(
            width=8,
            fast_hidden_size=16,
            layer_count=2,
            register_tokens=2,
            initialization_seed=7,
        )
    )


def autograd_reference_update(state, keys, values, learning_rate):
    differentiable = tuple(tensor.detach().requires_grad_(True) for tensor in state)
    predictions = fast_mlp_forward(keys, *differentiable)
    loss = torch.nn.functional.mse_loss(predictions.float(), values.float())
    gradients = torch.autograd.grad(loss, differentiable)
    return tuple(
        (parameter - learning_rate * gradient).detach()
        for parameter, gradient in zip(differentiable, gradients, strict=True)
    )


def test_gate_zero_is_exact_and_slow_weights_have_no_gradient() -> None:
    stack = make_stack()
    session = TTTSession(stack)
    request = session.start_request(
        "observation-0",
        TTTUpdateMode.UPDATE_ONCE,
        2,
        use_register_tokens=True,
        gate_zero=True,
    )
    hidden = torch.randn(1, 3, 8)
    output = request.process_layer(0, hidden)

    torch.testing.assert_close(output, hidden, rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in stack.parameters())
    assert all(not parameter.requires_grad for parameter in stack.parameters())


def test_update_once_counts_layers_not_denoise_passes() -> None:
    stack = make_stack()
    session = TTTSession(stack)
    request = session.start_request(
        "observation-0",
        TTTUpdateMode.UPDATE_ONCE,
        2,
        use_register_tokens=False,
    )
    hidden = torch.randn(1, 3, 8)
    for pass_index in range(10):
        request.begin_denoise_pass(pass_index)
        for layer_index in range(2):
            hidden = request.process_layer(layer_index, hidden)
    request.finish()

    assert request.timing.update_count == 2
    assert request.timing.apply_count == 20


def test_every_pass_updates_each_layer() -> None:
    stack = make_stack()
    request = TTTSession(stack).start_request(
        "observation-0",
        TTTUpdateMode.EVERY_PASS,
        2,
        use_register_tokens=True,
    )
    hidden = torch.randn(1, 3, 8)
    for pass_index in range(3):
        request.begin_denoise_pass(pass_index)
        for layer_index in range(2):
            hidden = request.process_layer(layer_index, hidden)

    assert request.timing.update_count == 6
    assert request.timing.apply_count == 6
    assert torch.isfinite(hidden).all()


def test_five_step_every_pass_counts_90_updates_and_applies() -> None:
    stack = TTTFastWeightStack(
        TTTConfig(width=8, fast_hidden_size=16, layer_count=18, register_tokens=2)
    )
    request = TTTSession(stack).start_request(
        "observation-0",
        TTTUpdateMode.EVERY_PASS,
        18,
        use_register_tokens=True,
    )
    hidden = torch.randn(1, 3, 8)
    for pass_index in range(5):
        request.begin_denoise_pass(pass_index)
        for layer_index in range(18):
            hidden = request.process_layer(layer_index, hidden)
    assert request.timing.update_count == 90
    assert request.timing.apply_count == 90


def test_functional_grad_matches_autograd_and_compiles() -> None:
    layer = make_stack().layers[0]
    hidden = torch.randn(1, 3, 8)
    tokens = layer._tokens_with_registers(hidden, True)  # noqa: SLF001
    keys = layer.key_projection(tokens).detach()
    values = layer.value_projection(tokens).detach()
    state = layer.initial_state().tensors()
    expected = autograd_reference_update(state, keys, values, layer.config.inner_learning_rate)
    actual = update_fast_state(state, keys, values, layer.config.inner_learning_rate)
    for actual_tensor, expected_tensor in zip(actual, expected, strict=True):
        torch.testing.assert_close(actual_tensor, expected_tensor)

    compiled_update = torch.compile(update_fast_state, backend="eager", fullgraph=True)
    compiled = compiled_update(state, keys, values, layer.config.inner_learning_rate)
    for compiled_tensor, expected_tensor in zip(compiled, expected, strict=True):
        torch.testing.assert_close(compiled_tensor, expected_tensor)


def test_flat_tensor_state_threading_and_fp32_master() -> None:
    config = dataclasses.replace(make_stack().config, fast_state_dtype="float32")
    stack = TTTFastWeightStack(config).to(dtype=torch.bfloat16)
    session = TTTSession(stack)
    flat_states = session.flat_states()
    assert all(tensor.dtype == torch.float32 for tensor in flat_states)
    replacements = tuple(tensor + 1 for tensor in flat_states)
    session.replace_flat_states(replacements)
    for actual, expected in zip(session.flat_states(), replacements, strict=True):
        torch.testing.assert_close(actual, expected)


def test_layer_states_are_independent_and_reset_is_reproducible() -> None:
    stack = make_stack()
    session = TTTSession(stack)
    initial_first_layer = session.states[0].detached_clone()
    initial_second_layer = session.states[1].detached_clone()
    request = session.start_request(
        "observation-0",
        TTTUpdateMode.UPDATE_ONCE,
        1,
        use_register_tokens=True,
    )
    request.process_layer(0, torch.randn(1, 3, 8))

    assert not torch.equal(session.states[0].second_weight, initial_first_layer.second_weight)
    torch.testing.assert_close(session.states[1].first_weight, initial_second_layer.first_weight)
    assert session.states[0].first_weight.data_ptr() != session.states[1].first_weight.data_ptr()

    session.reset()
    torch.testing.assert_close(session.states[0].first_weight, initial_first_layer.first_weight)
    torch.testing.assert_close(session.states[1].first_weight, initial_second_layer.first_weight)


def test_duplicate_observation_does_not_update_twice() -> None:
    stack = make_stack()
    session = TTTSession(stack)
    hidden = torch.randn(1, 3, 8)
    first_request = session.start_request(
        "same-observation",
        TTTUpdateMode.UPDATE_ONCE,
        1,
        use_register_tokens=False,
    )
    first_request.process_layer(0, hidden)
    first_request.finish()
    after_first_update = session.states[0].detached_clone()

    replay_request = session.start_request(
        "same-observation",
        TTTUpdateMode.UPDATE_ONCE,
        1,
        use_register_tokens=False,
    )
    replay_request.process_layer(0, hidden)
    replay_request.finish()

    assert replay_request.timing.update_count == 0
    torch.testing.assert_close(session.states[0].first_weight, after_first_update.first_weight)
