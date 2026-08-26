from unittest.mock import MagicMock, patch

import main as app_main


def test_default_mode_starts_chat() -> None:
    with (
        patch.object(
            app_main,
            "run_chat",
        ) as mocked_chat,
        patch.object(
            app_main,
            "unload_embedding_model",
        ),
    ):
        app_main.main([])

    mocked_chat.assert_called_once()


def test_chat_flag_starts_chat() -> None:
    with (
        patch.object(
            app_main,
            "run_chat",
        ) as mocked_chat,
        patch.object(
            app_main,
            "unload_embedding_model",
        ),
    ):
        app_main.main(
            ["--chat"]
        )

    mocked_chat.assert_called_once()


def test_ingest_flag_runs_incremental_ingestion() -> None:
    with (
        patch.object(
            app_main,
            "run_ingestion",
        ) as mocked_ingestion,
        patch.object(
            app_main,
            "unload_embedding_model",
        ),
    ):
        app_main.main(
            ["--ingest"]
        )

    mocked_ingestion.assert_called_once_with(
        reset_database=False
    )


def test_reset_flag_runs_full_ingestion() -> None:
    with (
        patch.object(
            app_main,
            "run_ingestion",
        ) as mocked_ingestion,
        patch.object(
            app_main,
            "unload_embedding_model",
        ),
    ):
        app_main.main(
            ["--reset"]
        )

    mocked_ingestion.assert_called_once_with(
        reset_database=True
    )
def test_run_chat_warms_up_embedding_and_chat_models() -> None:
    with (
        patch.object(
            app_main,
            "initialize_database",
        ),
        patch.object(
            app_main,
            "warm_up_embedding_model",
        ) as mocked_embedding_warmup,
        patch.object(
            app_main,
            "RAGAssistant",
        ) as mocked_assistant_class,
        patch.object(
            app_main,
            "run_chat_session",
        ),
    ):
        mocked_assistant = (
            mocked_assistant_class
            .return_value
            .__enter__
            .return_value
        )

        app_main.run_chat()

    mocked_embedding_warmup.assert_called_once()
    mocked_assistant.warm_up.assert_called_once()

def test_stats_flag_displays_statistics() -> None:
    with (
        patch.object(
            app_main,
            "show_stats",
        ) as mocked_stats,
        patch.object(
            app_main,
            "unload_embedding_model",
        ),
    ):
        app_main.main(
            ["--stats"]
        )

    mocked_stats.assert_called_once()
    
    
def test_sync_flag_runs_incremental_sync() -> None:
    with (
        patch.object(
            app_main,
            "run_sync",
        ) as mocked_sync,
        patch.object(
            app_main,
            "unload_embedding_model",
        ),
    ):
        app_main.main(
            ["--sync"]
        )

    mocked_sync.assert_called_once()


def test_run_sync_displays_sync_result() -> None:
    mock_result = MagicMock()

    mock_result.new_files = (
        "new.txt",
    )
    mock_result.modified_files = ()
    mock_result.deleted_files = ()
    mock_result.unchanged_files = ()
    mock_result.inserted_chunks = 1
    mock_result.deleted_chunks = 0
    mock_result.has_changes = True

    with (
        patch.object(
            app_main,
            "synchronize_knowledge_base",
            return_value=mock_result,
        ) as mocked_sync,
        patch.object(
            app_main,
            "format_sync_result",
            return_value=[
                "SYNC RESULT",
            ],
        ),
        patch(
            "builtins.print",
        ) as mocked_print,
    ):
        app_main.run_sync()

    mocked_sync.assert_called_once()

    assert any(
        call.args
        and "SYNC RESULT" in str(
            call.args[0]
        )
        for call in mocked_print.call_args_list
    )