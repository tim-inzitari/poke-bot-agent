import multiprocessing as mp

from poke_agent.worker_pool import bind_progress_queue, emit_game_progress, iter_with_live_progress


def test_emit_game_progress_puts_on_bound_queue():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    bind_progress_queue(queue)
    emit_game_progress()
    emit_game_progress(result=0, our_seat=0)
    bind_progress_queue(None)

    assert queue.get(timeout=1) == 1
    assert queue.get(timeout=1) == {"result": 0, "our_seat": 0}


def test_iter_with_live_progress_drains_while_iterating():
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    updates: list[int] = []

    class FakeProgress:
        def update(self, n: int) -> None:
            updates.append(n)

    def source():
        queue.put(1)
        yield "a"
        queue.put({"result": 0, "our_seat": 0})
        yield "b"

    collected = list(
        iter_with_live_progress(
            source(),
            queue,
            FakeProgress(),
            on_game=lambda _msg: None,
        )
    )
    assert collected == ["a", "b"]
    assert updates == [1, 1]
