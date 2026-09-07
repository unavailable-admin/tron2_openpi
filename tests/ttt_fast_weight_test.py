import torch

from openpi.models_pytorch.ttt_fast_weight import TTTConfig
from openpi.models_pytorch.ttt_fast_weight import TTTFastWeightStack
from openpi.models_pytorch.ttt_fast_weight import TTTSession
from openpi.models_pytorch.ttt_fast_weight import TTTUpdateMode


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

    assert not torch.equal(session.states[0].first_weight, initial_first_layer.first_weight)
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
