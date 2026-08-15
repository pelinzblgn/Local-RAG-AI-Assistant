from src.evaluation import EvaluationCase


EVALUATION_CASES = [
    # ======================================================
    # STM32
    # ======================================================

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
        question="DC motor hız kontrolünde duty cycle nasıl kullanılır?",
        expected_sources=(
            "stm32_notes.txt",
        ),
    ),

    # ======================================================
    # PID / LINE FOLLOWING
    # ======================================================

    EvaluationCase(
        question="PID kontrolcüsü hangi bileşenlerden oluşur?",
        expected_sources=(
            "pid_notes.txt",
        ),
    ),
    EvaluationCase(
        question="PID kontrolünde integral terimin görevi nedir?",
        expected_sources=(
            "pid_notes.txt",
        ),
    ),
    EvaluationCase(
        question="PID kontrolünde türev terimi neyi dikkate alır?",
        expected_sources=(
            "pid_notes.txt",
        ),
    ),
    EvaluationCase(
        question="Çizgi takip sisteminde hata nedir?",
        expected_sources=(
            "pid_notes.txt",
            "stm32_notes.txt",
        ),
    ),
    EvaluationCase(
        question="PID çıktısı çizgi takip aracında nasıl kullanılabilir?",
        expected_sources=(
            "pid_notes.txt",
        ),
    ),

    # ======================================================
    # RAG
    # ======================================================

    EvaluationCase(
        question="RAG sistemi nasıl çalışır?",
        expected_sources=(
            "rag_notes.txt",
        ),
    ),
    EvaluationCase(
        question="Retrieval-Augmented Generation nedir?",
        expected_sources=(
            "rag_notes.txt",
        ),
    ),
    EvaluationCase(
        question="RAG neden kaynak temelli cevap üretir?",
        expected_sources=(
            "rag_notes.txt",
        ),
    ),
    EvaluationCase(
        question="RAG sürecinin temel aşamaları nelerdir?",
        expected_sources=(
            "rag_notes.txt",
        ),
    ),

    # ======================================================
    # SQLITE
    # ======================================================

    EvaluationCase(
        question="SQLite neden yerel veri depolamada kullanılabilir?",
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
        question="SQLite ayrı bir veritabanı sunucusu gerektirir mi?",
        expected_sources=(
            "sqlite_notes.txt",
        ),
    ),

    # ======================================================
    # FOUNDRY LOCAL
    # ======================================================

    EvaluationCase(
        question="Foundry Local ne işe yarar?",
        expected_sources=(
            "foundry_local_notes.txt",
        ),
    ),
    EvaluationCase(
        question="Foundry Local ile modeller nerede çalıştırılır?",
        expected_sources=(
            "foundry_local_notes.txt",
        ),
    ),
    EvaluationCase(
        question="Foundry Local neden offline AI uygulamaları için uygundur?",
        expected_sources=(
            "foundry_local_notes.txt",
        ),
    ),

    # ======================================================
    # CROSS-DOCUMENT / HARDER CASES
    # ======================================================

    EvaluationCase(
        question=(
            "Çizgi takip aracında sensör bilgisinden motor "
            "hızlarına kadar kontrol süreci nasıl ilişkilidir?"
        ),
        expected_sources=(
            "pid_notes.txt",
            "stm32_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "Yerel RAG uygulamasında belgeler ve embeddingler "
            "hangi yapıda saklanabilir?"
        ),
        expected_sources=(
            "sqlite_notes.txt",
            "rag_notes.txt",
        ),
    ),
    EvaluationCase(
        question=(
            "Yerel çalışan bir RAG sisteminde model ve veri "
            "depolama bileşenleri nasıl birlikte kullanılabilir?"
        ),
        expected_sources=(
            "foundry_local_notes.txt",
            "sqlite_notes.txt",
            "rag_notes.txt",
        ),
    ),
]