from src.evaluation import (
    EvaluationCase,
    RAGCaseType,
    RAGQualityCase,
)


# ==========================================================
# Retrieval Benchmark
# ==========================================================


RETRIEVAL_EVALUATION_CASES = [
    # STM32
    EvaluationCase(
        question="STM32 nedir?",
        expected_sources=(
            "stm32_notes.txt",
        ),
    ),
    EvaluationCase(
        question="STM32 hangi çevre birimlerine sahiptir?",
        expected_sources=(
            "stm32_notes.txt",
        ),
    ),
    EvaluationCase(
        question="PWM motor hızını nasıl kontrol eder?",
        expected_sources=(
            "stm32_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "DC motor hız kontrolünde "
            "duty cycle nasıl kullanılır?"
        ),
        expected_sources=(
            "stm32_notes.txt",
        ),
    ),

    # PID
    EvaluationCase(
        question=(
            "PID kontrolcüsü hangi "
            "bileşenlerden oluşur?"
        ),
        expected_sources=(
            "pid_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "PID kontrolünde integral "
            "terimin görevi nedir?"
        ),
        expected_sources=(
            "pid_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "PID kontrolünde türev "
            "terimi neyi dikkate alır?"
        ),
        expected_sources=(
            "pid_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "Çizgi takip sisteminde "
            "hata nedir?"
        ),
        expected_sources=(
            "pid_notes.txt",
            "stm32_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "PID çıktısı çizgi takip "
            "aracında nasıl kullanılabilir?"
        ),
        expected_sources=(
            "pid_notes.txt",
        ),
    ),

    # RAG
    EvaluationCase(
        question="RAG sistemi nasıl çalışır?",
        expected_sources=(
            "rag_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "Retrieval-Augmented Generation nedir?"
        ),
        expected_sources=(
            "rag_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "RAG neden kaynak temelli "
            "cevap üretir?"
        ),
        expected_sources=(
            "rag_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "RAG sürecinin temel "
            "aşamaları nelerdir?"
        ),
        expected_sources=(
            "rag_notes.txt",
        ),
    ),

    # SQLite
    EvaluationCase(
        question=(
            "SQLite neden yerel veri "
            "depolamada kullanılabilir?"
        ),
        expected_sources=(
            "sqlite_notes.txt",
        ),
    ),
    EvaluationCase(
        question="SQLite nedir?",
        expected_sources=(
            "sqlite_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "SQLite ayrı bir veritabanı "
            "sunucusu gerektirir mi?"
        ),
        expected_sources=(
            "sqlite_notes.txt",
        ),
    ),

    # Foundry Local
    EvaluationCase(
        question="Foundry Local ne işe yarar?",
        expected_sources=(
            "foundry_local_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "Foundry Local ile modeller "
            "nerede çalıştırılır?"
        ),
        expected_sources=(
            "foundry_local_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "Foundry Local neden offline AI "
            "uygulamaları için uygundur?"
        ),
        expected_sources=(
            "foundry_local_notes.txt",
        ),
    ),

    # Cross-document
    EvaluationCase(
        question=(
            "Çizgi takip aracında sensör bilgisinden "
            "motor hızlarına kadar kontrol süreci "
            "nasıl ilişkilidir?"
        ),
        expected_sources=(
            "pid_notes.txt",
            "stm32_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "Yerel RAG uygulamasında belgeler ve "
            "embeddingler hangi yapıda saklanabilir?"
        ),
        expected_sources=(
            "sqlite_notes.txt",
            "rag_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "Yerel çalışan bir RAG sisteminde model "
            "ve veri depolama bileşenleri nasıl "
            "birlikte kullanılabilir?"
        ),
        expected_sources=(
            "foundry_local_notes.txt",
            "sqlite_notes.txt",
            "rag_notes.txt",
        ),
    ),
]


# Backward-compatible alias.
EVALUATION_CASES = (
    RETRIEVAL_EVALUATION_CASES
)


# ==========================================================
# End-to-End RAG Quality Benchmark
# ==========================================================


RAG_QUALITY_CASES = [
    RAGQualityCase(
        name="Supported STM32 definition",
        case_type=RAGCaseType.SUPPORTED,
        question="STM32 nedir?",
        expected_sources=(
            "stm32_notes.txt",
        ),
        expect_fallback=False,
        expect_rewrite=False,
        expected_confidence_levels=(
            "medium",
            "high",
        ),
    ),

    RAGQualityCase(
        name="Supported PWM question",
        case_type=RAGCaseType.SUPPORTED,
        question=(
            "STM32'nin PWM özelliği "
            "ne işe yarar?"
        ),
        expected_sources=(
            "stm32_notes.txt",
        ),
        expect_fallback=False,
        expect_rewrite=False,
        expected_confidence_levels=(
            "medium",
            "high",
        ),
    ),

    RAGQualityCase(
        name="Supported SQLite definition",
        case_type=RAGCaseType.SUPPORTED,
        question="SQLite nedir?",
        expected_sources=(
            "sqlite_notes.txt",
        ),
        expect_fallback=False,
        expect_rewrite=False,
        expected_confidence_levels=(
            "medium",
            "high",
        ),
    ),

    RAGQualityCase(
        name="Supported Foundry Local",
        case_type=RAGCaseType.SUPPORTED,
        question="Foundry Local ne işe yarar?",
        expected_sources=(
            "foundry_local_notes.txt",
        ),
        expect_fallback=False,
        expect_rewrite=False,
        expected_confidence_levels=(
            "medium",
            "high",
        ),
    ),

    RAGQualityCase(
        name="Unsupported geography question",
        case_type=RAGCaseType.UNSUPPORTED,
        question="Fransa'nın başkenti nedir?",
        expected_sources=(),
        expect_fallback=True,
        expect_rewrite=False,
        expected_confidence_levels=(
            "low",
        ),
    ),

    RAGQualityCase(
        name="Unsupported astronomy question",
        case_type=RAGCaseType.UNSUPPORTED,
        question=(
            "Ay ile Dünya arasındaki "
            "ortalama uzaklık nedir?"
        ),
        expected_sources=(),
        expect_fallback=True,
        expect_rewrite=False,
        expected_confidence_levels=(
            "low",
        ),
    ),

    RAGQualityCase(
        name="STM32 PWM follow-up rewrite",
        case_type=RAGCaseType.FOLLOW_UP,
        setup_questions=(
            "STM32 nedir?",
        ),
        question="Peki PWM ne işe yarar?",
        expected_sources=(
            "stm32_notes.txt",
        ),
        expect_fallback=False,
        expect_rewrite=True,
        expected_confidence_levels=(
            "medium",
            "high",
        ),
    ),

    RAGQualityCase(
        name="SQLite server follow-up rewrite",
        case_type=RAGCaseType.FOLLOW_UP,
        setup_questions=(
            "SQLite nedir?",
        ),
        question=(
            "Peki ayrı bir sunucu "
            "gerektirir mi?"
        ),
        expected_sources=(
            "sqlite_notes.txt",
        ),
        expect_fallback=False,
        expect_rewrite=True,
        expected_confidence_levels=(
            "medium",
            "high",
        ),
    ),

    RAGQualityCase(
        name="Standalone question after history",
        case_type=RAGCaseType.SUPPORTED,
        setup_questions=(
            "PID kontrol nedir?",
        ),
        question="STM32 nedir?",
        expected_sources=(
            "stm32_notes.txt",
        ),
        expect_fallback=False,
        expect_rewrite=False,
        expected_confidence_levels=(
            "medium",
            "high",
        ),
    ),
]