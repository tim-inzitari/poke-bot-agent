from pathlib import Path
from unittest.mock import call, patch

from scripts.finalize_latest20_specialist_corpora import grant_publish_reader


def test_publish_reader_gets_recursive_and_default_acl() -> None:
    root = Path("/archive/latest20/specialist-corpora")
    with (
        patch(
            "scripts.finalize_latest20_specialist_corpora.shutil.which",
            return_value="/usr/bin/setfacl",
        ),
        patch(
            "scripts.finalize_latest20_specialist_corpora.subprocess.run"
        ) as run,
    ):
        grant_publish_reader(root, "admin")
    assert run.call_args_list == [
        call(
            ["/usr/bin/setfacl", "-R", "-m", "u:admin:rX", str(root)],
            check=True,
        ),
        call(
            ["/usr/bin/setfacl", "-m", "d:u:admin:rX", str(root)],
            check=True,
        ),
    ]


def test_missing_setfacl_defers_to_managed_group_permission_step() -> None:
    with (
        patch(
            "scripts.finalize_latest20_specialist_corpora.shutil.which",
            return_value=None,
        ),
        patch(
            "scripts.finalize_latest20_specialist_corpora.subprocess.run"
        ) as run,
    ):
        grant_publish_reader(Path("/archive/latest20"), "admin")
    run.assert_not_called()


def test_empty_publish_reader_is_a_noop() -> None:
    with patch(
        "scripts.finalize_latest20_specialist_corpora.subprocess.run"
    ) as run:
        grant_publish_reader(Path("/archive/latest20"), "")
    run.assert_not_called()
